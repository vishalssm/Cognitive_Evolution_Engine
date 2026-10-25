"""The reflective reasoning engine.

One engine, one knowledge base, one actuator bank.  Every surface (dashboard,
headless node, CLI) reads this same object, so there is no possibility of two
components drifting into disagreeing about what the rover learned.

Each telemetry frame walks a fixed cognitive pipeline:

    SENSE  ->  DETECT  ->  COGNITION  ->  ACT  ->  VERIFY

``COGNITION`` is the fork.  If a stored predicate fits the anomaly the engine
takes **Working 2 (recall)** and reuses what it already knows.  If nothing fits
-- or the best fit is too weak to trust -- the engine takes **Working 1
(induction)** and commits a new rule to the knowledge base.  Either way the
chosen action is dispatched to the actuators and judged afterwards.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence

from . import config
from .actions import PRIMITIVES
from .actuators import ActuatorBank
from .features import AXIS_BY_KEY, Condition, FeatureVector, band_status
from .knowledge import KnowledgeBase

ANOMALY_THRESHOLD = 0.18     # normalised deviation that counts as an anomaly
COOLDOWN_FRAMES = 2          # frames to let an intervention bite before re-deciding
VERIFY_DELAY = 2             # frames before judging the last action
# A drivetrain spinning up from rest legitimately passes through every "stalled"
# reading on its way to cruising.  Judging those frames teaches the engine that
# pulling away from a standstill is a fault, so the first moments after a stream
# opens are observed for baseline rather than reasoned about.
WARMUP_FRAMES = 8
# Only the drivetrain channels are excused during warm-up.  Excusing every axis
# would swallow a fault injected in the first second -- several hazards last
# fewer frames than the warm-up window itself -- which would hide real failures.
WARMUP_EXEMPT_AXES = frozenset({"wheel_rpm_avg", "imu_accel_x", "kin_residual"})

# Stage descriptors are shipped to the dashboard so the working of the model is
# explained in the UI itself rather than only in the code.
PIPELINE_STAGES: List[Dict[str, str]] = [
    {
        "key": "sense",
        "title": "SENSE",
        "desc": "Decode the frame and re-express each channel as a distance from its healthy operating band.",
    },
    {
        "key": "detect",
        "title": "DETECT",
        "desc": "Channels that left the band become an anomaly signature; the drivetrain coupling is checked for slip.",
    },
    {
        "key": "cognition",
        "title": "COGNITION",
        "desc": "Recall a stored rule that fits (Working 2). If none fits, synthesise a new one (Working 1).",
    },
    {
        "key": "act",
        "title": "ACT",
        "desc": "Dispatch the chosen action to the vehicle actuators, which measurably change the telemetry.",
    },
    {
        "key": "verify",
        "title": "VERIFY",
        "desc": "Re-measure the same channels. Help raises the rule's confidence; failure demotes it.",
    },
]

WORKING_DESCRIPTIONS = {
    "induce": (
        "Working 1 - Induction",
        "Nothing in memory explained this failure, so CEE generalised the deviating "
        "channels into a symbolic predicate, chose the primitive whose forward model "
        "best reduces the residual, and wrote the pair into the knowledge graph as a "
        "new executable rule.",
    ),
    "recall": (
        "Working 2 - Recall",
        "A stored predicate matched this anomaly, so CEE reused it instead of learning "
        "again: the rule's action was dispatched, and the telemetry is re-measured so "
        "the rule's confidence can be reinforced or demoted.",
    ),
}


class CEEEngine:
    """Stateful, single-threaded-per-frame cognitive loop."""

    def __init__(
        self,
        knowledge: Optional[KnowledgeBase] = None,
        actuators: Optional[ActuatorBank] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.kb = knowledge or KnowledgeBase()
        self.actuators = actuators or ActuatorBank()
        self.event_sink = event_sink

        self.frame_index = 0
        self.prev: Optional[FeatureVector] = None
        self.cooldown = 0
        self.pending: Optional[Dict[str, Any]] = None
        self.last_frame_ts = 0.0

        self.history: Deque[Dict[str, Any]] = deque(maxlen=config.HISTORY_LEN)
        self.events: Deque[Dict[str, Any]] = deque(maxlen=config.EVENT_LOG_LEN)
        self.last_cycle: Dict[str, Any] = self._empty_cycle()
        self.counters = {
            "frames": 0,
            "nominal": 0,
            "anomalies": 0,
            "inductions": 0,
            "recalls": 0,
            "novel": 0,
            "helped": 0,
            "failed": 0,
            "stream_faults": 0,
            "warmup": 0,
            "revisions": 0,
        }
        self.narration = "Engine cold. Waiting for the first telemetry frame."
        self.last_induction: Optional[Dict[str, Any]] = None
        self.last_recall: Optional[Dict[str, Any]] = None
        self.last_verification: Optional[Dict[str, Any]] = None

    # -- events ------------------------------------------------------------
    def _emit(self, level: str, phase: str, title: str, detail: str = "") -> Dict[str, Any]:
        event = {
            "seq": len(self.events) + 1,
            "t": round(time.time(), 3),
            "clock": self.frame_index,
            "level": level,
            "phase": phase,
            "title": title,
            "detail": detail,
        }
        self.events.append(event)
        if self.event_sink:
            try:
                self.event_sink(event)
            except Exception:
                pass
        return event

    @staticmethod
    def _empty_cycle() -> Dict[str, Any]:
        return {
            "stage": "idle",
            "anomaly": None,
            "induction": None,
            "recall": None,
            "action": None,
            "verification": None,
            "narration": "",
        }

    # -- main entry point --------------------------------------------------
    def ingest(self, frame: Mapping[str, Any]) -> Dict[str, Any]:
        """Run one telemetry frame through the full cognitive pipeline."""
        self.frame_index += 1
        self.counters["frames"] += 1

        features = FeatureVector.from_frame(frame, self.prev)
        self.last_frame_ts = time.time()
        self.actuators.step()

        cycle = self._empty_cycle()
        anomaly = None
        verify_blocked = False

        # --- SENSE + DETECT ------------------------------------------------
        # A drivetrain pulling away from rest reads as stalled on every channel,
        # so the spin-up window excuses that signature.  It must not excuse a
        # sensor, thermal or radar fault arriving in the same breath, so the
        # excuse only applies while the deviation lives entirely in the
        # drivetrain's own channels.
        spinning_up = self.frame_index <= WARMUP_FRAMES
        warming_up = False
        if spinning_up:
            self.counters["warmup"] = self.counters.get("warmup", 0) + 1
            deviant = {a for a, d in features.deviation.items() if abs(d) > 0.12}
            drivetrain_only = bool(deviant) and deviant <= set(WARMUP_EXEMPT_AXES)
            # Spin-up is a rover climbing *towards* its bands, so every
            # drivetrain reading must still be short of them.  A wheel already
            # outrunning the ground it is on is wheelspin, and that is a fault
            # however early it arrives.
            climbing = all(
                features.deviation.get(axis, 0.0) < 0.0 for axis in deviant
            )
            warming_up = drivetrain_only and climbing

        if features.severity >= ANOMALY_THRESHOLD and not warming_up:
            axes = features.dominant_axes(limit=3)
            if not features.stream_ok:
                axes = ["stream_ok", *axes]
            anomaly = {
                "axes": axes,
                "severity": round(features.severity, 3),
                "residual": round(features.residual(axes), 3),
                "raw_residual": round(features.raw_residual(axes), 2),
                "signature": self._signature_text(features, axes),
                "kin_residual": round(features.kin_residual, 3),
            }
            cycle["stage"] = "detect"
            cycle["anomaly"] = anomaly
            self.counters["anomalies"] += 1
            if not features.stream_ok:
                self.counters["stream_faults"] += 1
        else:
            self.counters["nominal"] += 1

        # --- VERIFY (previous action, judged now) --------------------------
        verification = None
        if self.pending is not None:
            if not features.stream_ok:
                # The link is still down, so the effect of the action cannot be
                # measured.  Judging a rule on evidence it could not possibly
                # gather would poison its confidence, so the trial is deferred.
                verify_blocked = True
                self.pending["deferred"] = self.pending.get("deferred", 0) + 1
            else:
                age = self.frame_index - self.pending["applied_at"]
                residual_now = features.residual(self.pending["axes"])
                if age >= VERIFY_DELAY:
                    verification = self.kb.verify(
                        self.pending["rule_id"],
                        residual_now,
                        self.pending["residual_before"],
                        self.pending.get("expected_residual"),
                    )
                    self.last_verification = verification
                    self.pending = None
                    self.counters[
                        "helped" if verification["verdict"] == "helped" else "failed"
                    ] += 1
                    level = "ok" if verification["verdict"] == "helped" else "warn"
                    cycle["stage"] = "verify"
                    cycle["verification"] = verification
                    self.narration = (
                        f"Verification - {verification['rule_id']} "
                        f"({verification['title']}) moved the residual from "
                        f"{verification['residual_before']:.2f} to "
                        f"{verification['residual_after']:.2f} against a predicted "
                        f"{verification['expected']:.2f}. {verification['detail'].capitalize()}. "
                        f"Confidence is now {verification['confidence']:.2f} "
                        f"({verification['delta']:+.2f})."
                    )
                    self._emit(
                        level,
                        "verify",
                        f"{verification['rule_id']} {verification['verdict']}",
                        self.narration,
                    )
                else:
                    self.pending["residual_now"] = residual_now

        # --- COGNITION: Working 2 (recall) or Working 1 (induction) --------
        action_applied: Optional[Dict[str, Any]] = None
        if (
            anomaly is not None
            and verification is None
            and not verify_blocked
            and self.cooldown <= 0
        ):
            hits = self.kb.recall(features)
            trusted = [h for h in hits if h.rule.confidence >= config.RECALL_MIN_CONFIDENCE]

            if trusted:
                action_applied = self._run_recall(trusted[0], features, anomaly, hits)
                cycle["stage"] = "recall"
                self.counters["recalls"] += 1
            elif self.actuators.busy or frame.get("intervened"):
                # This frame was produced while one of our own interventions was
                # still in force.  Learning from it would manufacture rules that
                # describe our mitigation rather than the rover's behaviour, so
                # the engine waits for a clean observation before inducing.
                cycle["stage"] = "hold"
                cycle["suppressed"] = "induction suppressed: frame already under intervention"
            elif hits:
                # A stored rule does cover this failure, but its confidence fell
                # below the trust floor -- its remedy was judged not to work.
                # The honest response is to revise that rule in place, not to
                # invent a second rule for a failure the engine already names.
                self.counters["novel"] += 1
                action_applied = self._run_induce(
                    features, anomaly, revise=hits[0].rule.id
                )
                cycle["stage"] = "revise"
                cycle["revised"] = hits[0].rule.id
                self.counters["revisions"] += 1
            else:
                self.counters["novel"] += 1
                action_applied = self._run_induce(features, anomaly)
                cycle["stage"] = "induce"
                self.counters["inductions"] += 1

            if action_applied:
                cycle["action"] = action_applied
                self.pending = {
                    "rule_id": action_applied["rule_id"],
                    "axes": anomaly["axes"],
                    "residual_before": anomaly["residual"],
                    "expected_residual": self._expected_residual(
                        action_applied["action"], features, anomaly["axes"]
                    ),
                    "applied_at": self.frame_index,
                }
                self.cooldown = COOLDOWN_FRAMES

        if self.cooldown > 0:
            self.cooldown -= 1

        # --- NOMINAL narration ---------------------------------------------
        if anomaly is None and verification is None and cycle["stage"] == "idle":
            cycle["stage"] = "warmup" if warming_up else "nominal"
            if warming_up:
                self.narration = (
                    f"Stream open. Frame {self.frame_index} of {WARMUP_FRAMES}: observing "
                    "the drivetrain spin up before judging it, so that acceleration from "
                    "rest is never mistaken for a fault."
                )
            else:
                self.narration = (
                    "All monitored channels are inside their healthy bands. "
                    f"The knowledge graph is idle but retains {len(self.kb.rules)} learned "
                    "rule(s) and will act on them without being asked."
                )

        # --- telemetry history ----------------------------------------------
        record = {
            "i": self.frame_index,
            "t": round(time.time(), 2),
            "v": {k: round(v, 2) for k, v in features.values.items()},
            "sev": round(features.severity, 3),
            "stage": cycle["stage"],
            "anomalous": anomaly is not None,
            "rule": (action_applied or {}).get("rule_id"),
            "stream_ok": features.stream_ok,
        }
        self.history.append(record)

        self.prev = features
        self.last_cycle = cycle
        return cycle

    # -- Working 2 ---------------------------------------------------------
    def _run_recall(
        self,
        hit,
        features: FeatureVector,
        anomaly: Dict[str, Any],
        all_hits: List[Any],
    ) -> Dict[str, Any]:
        rule = hit.rule
        primitive = PRIMITIVES[rule.action]
        before = rule.confidence

        self.kb.register_application(rule.id)
        intervention = self.actuators.apply(rule.action, rule.id, note=rule.text)

        recall_payload = {
            "chosen": hit.to_dict(),
            "considered": [h.to_dict() for h in all_hits[:6]],
            "considered_count": len(all_hits),
            "rule": rule.to_dict(),
            "primitive": primitive.to_dict(),
            "why": (
                f"{rule.id} matched {hit.satisfied}/{len(rule.conditions)} conditions and holds "
                f"confidence {before:.2f}. It already encodes the correct recovery for this "
                f"failure, so CEE reused it rather than inventing a second rule."
            ),
            "reinforcement": self._reinforcement_preview(hit),
        }
        self.last_recall = recall_payload

        self.narration = (
            f"Working 2 - Recall. {rule.id} matched this anomaly ({hit.satisfied}/"
            f"{len(rule.conditions)} conditions, confidence {before:.2f}). Its action "
            f"{primitive.title} was dispatched to the actuators and is held for "
            f"{intervention.holds_for} frames."
        )
        self._emit(
            "recall",
            "recall",
            f"Working 2  {rule.id} recalled",
            f"{rule.text}  ->  {primitive.consequence}",
        )
        return {
            "rule_id": rule.id,
            "rule_title": rule.title,
            "action": rule.action,
            "action_title": primitive.title,
            "consequence": primitive.consequence,
            "working": 2,
            "match": f"{hit.satisfied}/{len(rule.conditions)}",
            "conditions": [c.to_dict() for c in rule.conditions],
            "confidence": round(before, 3),
        }

    def _reinforcement_preview(self, hit) -> Dict[str, Any]:
        rule = hit.rule
        return {
            "rule_id": rule.id,
            "confidence": round(rule.confidence, 3),
            "success_rate": round(rule.success_rate, 3),
            "judged": rule.times_helped + rule.times_failed,
            "applied": rule.times_applied,
            "trace": [round(c, 3) for c in rule.confidence_trace[-16:]],
        }

    # -- Working 1 ---------------------------------------------------------
    def _run_induce(
        self,
        features: FeatureVector,
        anomaly: Dict[str, Any],
        revise: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.kb.induce(features, revise=revise)
        rule = result.rule
        primitive = PRIMITIVES[rule.action]

        self.actuators.apply(rule.action, rule.id, note=rule.text)
        payload = result.to_dict()
        self.last_induction = payload

        self.narration = (
            f"Working 1 - Induction. No stored predicate explained this anomaly, so CEE "
            f"generalised {len(anomaly['axes'])} deviating channel(s) into {rule.id} "
            f"({rule.text}) and bound it to {primitive.title}, predicted to cut the residual "
            f"from {result.candidates[0].current_residual:.2f} to "
            f"{result.candidates[0].projected_residual:.2f}."
        )
        self._emit(
            "induce",
            "induce",
            f"Working 1  {rule.id} induced",
            rule.text,
        )
        return {
            "rule_id": rule.id,
            "rule_title": rule.title,
            "action": rule.action,
            "action_title": primitive.title,
            "consequence": primitive.consequence,
            "working": 1,
            "match": "new",
            "conditions": [c.to_dict() for c in rule.conditions],
            "confidence": round(rule.confidence, 3),
        }

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _expected_residual(
        action: str, features: FeatureVector, axes: Sequence[str]
    ) -> float:
        """What the chosen action's forward model says the residual should be."""
        primitive = PRIMITIVES.get(action)
        if primitive is None:
            return features.residual(axes)
        try:
            return primitive.project(features).residual(axes)
        except Exception:
            return features.residual(axes)

    @staticmethod
    def _signature_text(features: FeatureVector, axes: List[str]) -> str:
        parts = []
        for axis in axes:
            spec = AXIS_BY_KEY.get(axis)
            if not spec:
                if axis == "stream_ok":
                    parts.append("link_severed")
                continue
            lo, hi = spec["nominal"]
            v = features.values.get(axis, 0.0)
            if v > hi:
                parts.append(f"{spec['short']}_HIGH({v:.1f}>{hi:.0f})")
            elif v < lo:
                parts.append(f"{spec['short']}_LOW({v:.1f}<{lo:.0f})")
        if not parts:
            parts.append("link_severed")
        return " + ".join(parts)

    def watchdog(self) -> Optional[Dict[str, Any]]:
        """Raise a synthetic link fault when the pipeline goes silent."""
        if not self.last_frame_ts or self.frame_index == 0:
            return None
        silence = time.time() - self.last_frame_ts
        if silence < config.STREAM_TIMEOUT:
            return None
        if self.prev is not None and not self.prev.stream_ok:
            return None

        frame = {
            "timestamp": round(time.time(), 2),
            "stream_ok": False,
            "wheel_rpm_avg": 0,
            "imu_accel_x": 0,
            "imu_pitch_deg": 0,
            "ultrasonic_cm": 0,
            "system_temp_c": 0,
        }
        self._emit(
            "warn",
            "detect",
            "Telemetry watchdog",
            f"No frame received for {silence:.1f}s - synthesising a link-severed frame.",
        )
        return self.ingest(frame)

    # -- reporting ---------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        features = self.prev
        return {
            "frame_index": self.frame_index,
            "counters": dict(self.counters),
            "actuators": self.actuators.to_dict(),
            "knowledge": self.kb.snapshot(),
            "cycle": self.last_cycle,
            "narration": self.narration,
            "last_induction": self.last_induction,
            "last_recall": self.last_recall,
            "last_verification": self.last_verification,
            "telemetry": features.to_dict() if features else None,
            "display": features.display() if features else None,
            "pipeline": PIPELINE_STAGES,
            "workings": WORKING_DESCRIPTIONS,
            "threshold": ANOMALY_THRESHOLD,
            "recall_min_confidence": config.RECALL_MIN_CONFIDENCE,
        }