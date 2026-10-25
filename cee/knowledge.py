"""Persistent operational knowledge base.

This module owns the two cognitive workings and nothing else:

  WORKING 1 - INDUCTION  (:meth:`KnowledgeBase.induce`)
      Generalise an observation into a symbolic predicate.  Each deviating
      axis contributes one comparison whose threshold is synthesised as the
      midpoint between the healthy operating band and the observed value, so
      the rule generalises to a *region* of telemetry rather than memorising a
      single reading.  A matching primitive is then selected by forward
      simulation (see actions.py) and committed as an executable rule.

  WORKING 2 - RECALL    (:meth:`KnowledgeBase.recall` / :meth:`verify`)
      Score every stored predicate against the current anomaly, execute the
      best-supported match, then observe the telemetry to confirm whether the
      action actually helped.  Rules are reinforced or demoted accordingly,
      which is what makes the knowledge base an *evolving* one.

Rules are persisted to disk so accumulated experience survives restarts.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import config
from .actions import PRIMITIVES, ActionPrimitive, ScoredAction, score_actions
from .features import Condition, FeatureVector, resolve_axis

# Pseudo-axes let a predicate express a *relationship* between channels
# (drivetrain mismatch, link integrity) instead of a single reading.
PSEUDO_AXES: Dict[str, Dict[str, Any]] = {
    "kin_residual": {
        "label": "Traction Mismatch",
        "unit": "m/s2",
        "nominal": (-0.21, 0.21),
        "description": "acceleration actually achieved minus acceleration the wheel speed implies",
    },
    "stream_ok": {
        "label": "Link Integrity",
        "unit": "",
        "nominal": (0.5, 1.0),
        "description": "whether the telemetry pipeline is delivering frames at all",
    },
    "severity": {
        "label": "Anomaly Severity",
        "unit": "",
        "nominal": (0.0, 0.18),
        "description": "how far the worst channel sits outside its healthy band",
    },
}

# Nominal driving can already leave a channel a little outside its band, so only
# deviations past this floor are worth writing a condition for.
CONDITION_DEVIATION_FLOOR = 0.25
# A mismatch past this is a genuine drivetrain fault rather than sensor noise.
KIN_RESIDUAL_FLOOR = 0.32
# Relative slack when deciding two synthesised thresholds describe one region.
NEAR_DUPLICATE_TOLERANCE = 0.35

ROLE_LABELS = {
    "wheel_rpm_avg": "wheel speed",
    "imu_accel_x": "chassis acceleration",
    "imu_pitch_deg": "chassis pitch",
    "ultrasonic_cm": "forward radar return",
    "system_temp_c": "compute temperature",
    "kin_residual": "drivetrain traction mismatch",
    "stream_ok": "telemetry link integrity",
    "severity": "unattributed anomaly",
}


def resolve_axis_for_rule(features: FeatureVector, axis: str) -> float:
    """Backwards-compatible alias for :func:`cee.features.resolve_axis`."""
    return resolve_axis(features, axis)


@dataclass
class Rule:
    """One executable unit of learned behaviour."""

    id: str
    action: str
    conditions: List[Condition]
    text: str
    title: str
    rationale: str
    confidence: float = config.CONFIDENCE_START
    times_induced: int = 1
    times_applied: int = 0
    times_helped: int = 0
    times_failed: int = 0
    created_at: float = field(default_factory=time.time)
    last_used_at: float = 0.0
    last_outcome: str = "not yet verified"
    signature: str = ""
    confidence_trace: List[float] = field(default_factory=list)
    outcome_trace: List[int] = field(default_factory=list)

    # -- predicate ---------------------------------------------------------
    def matches(self, features: FeatureVector) -> bool:
        return all(cond.evaluate(features) for cond in self.conditions)

    def explains(self, features: FeatureVector) -> List[bool]:
        return [cond.evaluate(features) for cond in self.conditions]

    @property
    def specificity(self) -> float:
        return 1.0 + 0.09 * (len(self.conditions) - 1)

    @property
    def success_rate(self) -> float:
        judged = self.times_helped + self.times_failed
        return (self.times_helped / judged) if judged else 0.0

    def to_dict(self) -> Dict[str, Any]:
        primitive = PRIMITIVES.get(self.action)
        return {
            "id": self.id,
            "action": self.action,
            "action_title": primitive.title if primitive else self.action,
            "conditions": [c.to_dict() for c in self.conditions],
            "text": self.text,
            "title": self.title,
            "rationale": self.rationale,
            "confidence": round(self.confidence, 3),
            "times_induced": self.times_induced,
            "times_applied": self.times_applied,
            "times_helped": self.times_helped,
            "times_failed": self.times_failed,
            "success_rate": round(self.success_rate, 3),
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "last_outcome": self.last_outcome,
            "signature": self.signature,
            "specificity": round(self.specificity, 3),
            "confidence_trace": [round(c, 3) for c in self.confidence_trace[-24:]],
            "outcome_trace": self.outcome_trace[-24:],
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "Rule":
        return Rule(
            id=payload["id"],
            action=payload["action"],
            conditions=[Condition.from_dict(c) for c in payload.get("conditions", [])],
            text=payload.get("text", ""),
            title=payload.get("title", ""),
            rationale=payload.get("rationale", ""),
            confidence=float(payload.get("confidence", config.CONFIDENCE_START)),
            times_induced=int(payload.get("times_induced", 1)),
            times_applied=int(payload.get("times_applied", 0)),
            times_helped=int(payload.get("times_helped", 0)),
            times_failed=int(payload.get("times_failed", 0)),
            created_at=float(payload.get("created_at", time.time())),
            last_used_at=float(payload.get("last_used_at", 0.0)),
            last_outcome=payload.get("last_outcome", "not yet verified"),
            signature=payload.get("signature", ""),
            confidence_trace=list(payload.get("confidence_trace", [])),
            outcome_trace=list(payload.get("outcome_trace", [])),
        )


@dataclass
class InductionResult:
    """Full audit trail of one WORKING 1 event, for the dashboard."""

    rule: Rule
    candidates: List[ScoredAction]
    deviation_table: List[Dict[str, Any]]
    observation: Dict[str, float]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "deviation_table": self.deviation_table,
            "observation": self.observation,
            "explanation": self.explanation,
        }


@dataclass
class RecallHit:
    rule: Rule
    score: float
    condition_results: List[bool]
    satisfied: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule.id,
            "action": self.rule.action,
            "title": self.rule.title,
            "confidence": round(self.rule.confidence, 3),
            "score": round(self.score, 3),
            "satisfied": self.satisfied,
            "of": len(self.rule.conditions),
            "condition_results": self.condition_results,
            "confidence_before": round(
                max(0.0, self.rule.confidence - config.CONFIDENCE_DECAY), 3
            ),
        }


class KnowledgeBase:
    """Thread-safe store of induced rules with recall and reinforcement."""

    NEAR_DUPLICATE_TOLERANCE = NEAR_DUPLICATE_TOLERANCE

    def __init__(self, path=config.KNOWLEDGE_FILE, autoload: bool = True) -> None:
        self.path = path
        self.rules: List[Rule] = []
        self._counter = 0
        self._lock = threading.RLock()
        self.stats = {
            "anomalies_seen": 0,
            "inductions": 0,
            "recalls": 0,
            "verified_success": 0,
            "verified_failure": 0,
            "novel_detections": 0,
        }
        if autoload:
            self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> int:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        with self._lock:
            self.rules = [Rule.from_dict(r) for r in raw.get("rules", [])]
            self._counter = int(raw.get("counter", len(self.rules)))
            self.stats.update(raw.get("stats", {}))
        return len(self.rules)

    def save(self) -> None:
        with self._lock:
            payload = {
                "version": config.__dict__.get("CONFIDENCE_START") and 2,
                "counter": self._counter,
                "saved_at": time.time(),
                "stats": dict(self.stats),
                "rules": [r.to_dict() for r in self.rules],
            }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def reset(self, wipe_file: bool = True) -> None:
        with self._lock:
            self.rules = []
            self._counter = 0
            for key in self.stats:
                self.stats[key] = 0
        if wipe_file:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass

    # -- WORKING 1: induction ---------------------------------------------
    def _synthesise_conditions(
        self, features: FeatureVector, max_conditions: int = 3
    ) -> List[Condition]:
        """Generalise an observation into a predicate over the failing region.

        Each deviating channel contributes one comparison whose threshold sits
        *outside* the healthy band, part-way between the band edge and the
        observed value.  The rule therefore covers a region of telemetry rather
        than a single reading, while a nominal frame can never satisfy it --
        which is exactly what makes recall trustworthy.
        """
        conditions: List[Condition] = []

        if not features.stream_ok:
            conditions.append(
                Condition(
                    axis="stream_ok",
                    op="<",
                    value=0.5,
                    role=ROLE_LABELS["stream_ok"],
                )
            )

        # A large drivetrain mismatch is the physically meaningful statement of
        # "wheels are spinning but the chassis is not moving", so it earns a
        # slot of its own rather than being implied by two raw channels.
        if features.stream_ok and abs(features.kin_residual) > KIN_RESIDUAL_FLOOR:
            observed = features.kin_residual
            lo, hi = PSEUDO_AXES["kin_residual"]["nominal"]
            if observed < lo:
                threshold = lo - 0.5 * abs(observed - lo)
            else:
                threshold = hi + 0.5 * abs(observed - hi)
            conditions.append(
                Condition(
                    axis="kin_residual",
                    op="<" if observed < lo else ">",
                    value=round(threshold, 3),
                    role=ROLE_LABELS["kin_residual"],
                )
            )

        ranked = sorted(
            features.deviation.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        for axis, deviation in ranked:
            if len(conditions) >= max_conditions:
                break
            if abs(deviation) < CONDITION_DEVIATION_FLOOR or axis in features.masked:
                continue
            spec = config.AXIS_BY_KEY.get(axis)
            if not spec:
                continue
            observed = features.values.get(axis, 0.0)
            lo, hi = spec["nominal"]
            if observed > hi:
                threshold = hi + 0.5 * (observed - hi)
                op = ">"
            elif observed < lo:
                threshold = lo - 0.5 * (lo - observed)
                op = "<"
            else:
                continue
            conditions.append(
                Condition(
                    axis=axis,
                    op=op,
                    value=round(threshold, 3),
                    role=ROLE_LABELS.get(axis, axis),
                )
            )

        if not conditions:
            # Nothing cleared the deviation floor, so there is no dramatic
            # failure to name -- but the frame *is* anomalous, and it must still
            # become a predicate that localises.  Falling back to a channel-wide
            # statement such as "severity > 0" would produce a rule that fires on
            # every future anomaly and therefore explains nothing, so the
            # worst-offender is named at its own midpoint threshold instead.
            ranked_any = sorted(
                (axis for axis in features.deviation if axis in config.AXIS_BY_KEY),
                key=lambda a: abs(features.deviation[a]),
                reverse=True,
            )
            axis = ranked_any[0] if ranked_any else None
            if axis is not None:
                spec = config.AXIS_BY_KEY[axis]
                observed = features.values.get(axis, 0.0)
                lo, hi = spec["nominal"]
                if observed > hi:
                    op, value = ">", round(hi + 0.5 * (observed - hi), 3)
                else:
                    op, value = "<", round(lo - 0.5 * (lo - observed), 3)
                conditions.append(
                    Condition(
                        axis=axis,
                        op=op,
                        value=value,
                        role=ROLE_LABELS.get(axis, axis),
                    )
                )
            else:
                # No sensor channel deviated at all -- only the derived link or
                # coupling diagnostic did, and neither is available here.
                conditions.append(
                    Condition(
                        axis="severity",
                        op=">",
                        value=0.0,
                        role="unattributed anomaly",
                    )
                )
        return self._harden(conditions, features)

    @staticmethod
    def _nominal_features() -> FeatureVector:
        return FeatureVector.from_frame(
            {spec["key"]: (spec["nominal"][0] + spec["nominal"][1]) / 2.0 for spec in config.AXES}
        )

    def _harden(
        self, conditions: List[Condition], features: FeatureVector
    ) -> List[Condition]:
        """Guarantee the induced predicate rejects healthy telemetry.

        If a nominal frame would satisfy the predicate, push that threshold
        further out past the operating band until it cannot.  A rule that fires
        during normal driving is worse than no rule at all, because it trains
        the engine to distrust itself.
        """
        nominal = self._nominal_features()
        for cond in conditions:
            guard = 0
            while cond.evaluate(nominal) and guard < 6:
                if cond.axis in PSEUDO_AXES:
                    lo, hi = PSEUDO_AXES[cond.axis]["nominal"]
                    step = (hi - lo) or 1.0
                    if cond.axis == "severity":
                        # Severity is only ever "off nominal" upwards, so a
                        # stricter predicate must demand a *higher* severity.
                        cond.value = hi + step * (guard + 1)
                        guard += 1
                        continue
                else:
                    spec = config.AXIS_BY_KEY.get(cond.axis)
                    if not spec:
                        break
                    lo, hi = spec["nominal"]
                    step = (hi - lo) or max(1.0, abs(cond.value) * 0.25)
                if cond.op in ("<", "<="):
                    cond.value = lo - step * (guard + 1)
                else:
                    cond.value = hi + step * (guard + 1)
                guard += 1
        return conditions

    @staticmethod
    def _signature(conditions: Sequence[Condition]) -> str:
        payload = "|".join(sorted(f"{c.axis}{c.op}{c.value:.2f}" for c in conditions))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def _is_near_duplicate(cls, rule: "Rule", conditions: Sequence[Condition], action: str) -> bool:
        """True when a stored rule already covers this failure region.

        Two things are compared.  The remedy must be the same, because a rule is
        a predicate *and* a remedy: changing one without the other is a new
        idea, not a new observation.  The predicate must describe the same
        *axes* with the same *directions*, but thresholds are compared loosely,
        because successive samples of one physical fault land slightly apart and
        a stuck rover re-observes the same fault every frame forever.

        Requiring the action to match is what keeps a graph from filling with
        cosmetic variants of a single fault.
        """
        if rule.action != action:
            return False
        existing = {c.axis: c for c in rule.conditions}
        incoming = {c.axis: c for c in conditions}
        if set(existing) != set(incoming):
            return False
        for axis, cond in incoming.items():
            other = existing[axis]
            if other.op != cond.op:
                return False
            span = max(abs(cond.value), 1e-6)
            if abs(other.value - cond.value) / span > cls.NEAR_DUPLICATE_TOLERANCE:
                return False
        return True

    @classmethod
    def _covers(cls, rule: "Rule", conditions: Sequence[Condition]) -> bool:
        """True when an existing rule's region already spans the new observation.

        This is the weaker test used when the remedy changed: the predicate may
        already cover this telemetry, in which case the engine rebinds the known
        region to the newly chosen remedy instead of storing a second rule that
        would fire on identical frames.
        """
        existing = {c.axis: c for c in rule.conditions}
        incoming = {c.axis: c for c in conditions}
        if not incoming:
            return False
        for axis, cond in incoming.items():
            other = existing.get(axis)
            if other is None or other.op != cond.op:
                return False
            # The stored rule must remain satisfied by the new observation, so
            # that rebinding it can only widen coverage, never contradict it.
            observed = cond.value
            if cond.op in ("<", "<="):
                if not (other.value < observed):
                    return False
            elif not (other.value > observed):
                return False
        return True

    @staticmethod
    def _name_for(conditions: Sequence[Condition], primitive: ActionPrimitive) -> str:
        primary = conditions[0].role if conditions else "telemetry"
        verb = {
            "cap_motor_torque": "Traction Slip",
            "emergency_brake": "Obstacle Proximity",
            "reverse_and_recalculate_path": "Kinetic Reversal",
            "engage_max_torque_burst": "Traction Stall",
            "enable_4x4_crawl_mode": "Incline Overrun",
            "navigate_by_imu_only": "Perception Loss",
            "throttle_cognitive_hz": "Thermal Overrun",
            "trigger_hardware_safing_mode": "Link Severance",
            "recalibrate_radar_sweep": "Radar Latch Fault",
            "hold_and_reacquire": "Unexplained Drift",
        }.get(primitive.key, "Anomaly")
        return f"{verb} @ {primary}"

    def _revise(
        self,
        rule: "Rule",
        primitive: ActionPrimitive,
        conditions: Sequence[Condition],
        features: FeatureVector,
    ) -> "Rule":
        """Re-point a known rule at a better remedy, in place.

        The predicate is widened slightly towards the fresh observation and the
        remedy is swapped.  Confidence starts over from a neutral value because
        the new remedy is a fresh hypothesis that has to earn its trust through
        verification, and the previous ledger is kept so the dashboard can still
        show what happened.
        """
        rule.conditions = list(conditions)
        rule.action = primitive.key
        rule.title = self._name_for(rule.conditions, primitive)
        rule.text = primitive.render(rule.conditions, features.values)
        rule.rationale = (
            f"{rule.id} was re-pointed at {primitive.title.lower()} after its original "
            f"remedy failed verification on the same region "
            f"({', '.join(c.to_string() for c in rule.conditions)})."
        )
        rule.signature = self._signature(rule.conditions)
        rule.times_induced += 1
        rule.confidence = config.CONFIDENCE_START
        rule.confidence_trace.append(rule.confidence)
        rule.outcome_trace.append("revised")
        rule.last_outcome = "revised after failure"
        return rule

    def induce(
        self,
        features: FeatureVector,
        forced_action: Optional[str] = None,
        revise: Optional[str] = None,
    ) -> InductionResult:
        """WORKING 1 - synthesise, score and commit a rule.

        ``revise`` names an existing rule whose predicate already covers this
        failure but whose remedy has lost confidence.  In that case the predicate
        is kept and the remedy is re-chosen, because the engine has already
        learned *what* is wrong and only needs a better response.
        """
        with self._lock:
            self.stats["anomalies_seen"] += 1
            self.stats["inductions"] += 1

            axes = features.dominant_axes(limit=3)
            if not axes:
                axes = sorted(
                    features.deviation,
                    key=lambda k: abs(features.deviation[k]),
                    reverse=True,
                )[:2]

            candidates = score_actions(features, axes)
            winner = next((c for c in candidates if c.feasible), None)
            if winner is None:
                winner = candidates[-1]
            if forced_action and forced_action in PRIMITIVES:
                for cand in candidates:
                    if cand.key == forced_action:
                        winner = cand
                        break

            primitive = PRIMITIVES[winner.key]

            # A revision keeps the predicate the engine already trusts and only
            # re-chooses the remedy, so the rule keeps naming the same failure.
            revising = next((r for r in self.rules if r.id == revise), None)
            if revising is not None:
                if revising.action == primitive.key:
                    # Same remedy, so nothing has actually been learned yet.
                    revising.times_induced += 1
                    self.stats["inductions"] -= 1
                    return self._induction_result(revising, candidates, features, axes)
                rule = self._revise(revising, primitive, list(revising.conditions), features)
                self.stats["inductions"] -= 1
                return self._induction_result(rule, candidates, features, axes)

            conditions = self._synthesise_conditions(features)
            signature = self._signature(conditions)

            # Novelty detection can fire late, so induction re-checks the graph
            # three ways before committing anything new:
            #   1. the exact same predicate and remedy      -> reinforce in place
            #   2. the same region and remedy, thresholds
            #      a little apart                           -> reinforce in place
            #   3. the same region, but a different remedy -> rebind the region
            duplicate = next(
                (r for r in self.rules if r.signature == signature and r.action == primitive.key),
                None,
            ) or next(
                (r for r in self.rules if self._is_near_duplicate(r, conditions, primitive.key)),
                None,
            )
            rebound = None
            if duplicate is None:
                rebound = next(
                    (r for r in self.rules if self._covers(r, conditions)),
                    None,
                )

            if duplicate is not None:
                duplicate.times_induced += 1
                duplicate.confidence = min(
                    config.CONFIDENCE_CEIL, duplicate.confidence + 0.03
                )
                duplicate.confidence_trace.append(duplicate.confidence)
                self.stats["inductions"] -= 1
                rule = duplicate
            elif rebound is not None:
                # The predicate is already understood but its remedy has been
                # shown to fail, so the honest update is to re-point the known
                # region at a remedy the forward models prefer -- not to add a
                # second rule that would fire on the very same frames.
                rule = self._revise(rebound, primitive, conditions, features)
                self.stats["inductions"] -= 1
            else:
                self._counter += 1
                rule = Rule(
                    id=f"RULE_{self._counter:03d}",
                    action=primitive.key,
                    conditions=conditions,
                    text=primitive.render(conditions, features.values),
                    title=self._name_for(conditions, primitive),
                    rationale=self._rationale(features, axes, primitive, winner),
                    signature=signature,
                )
                rule.confidence_trace = [rule.confidence]
                self.rules.insert(0, rule)

            return self._induction_result(rule, candidates, features, axes)

    def _induction_result(
        self,
        rule: "Rule",
        candidates: List[Any],
        features: FeatureVector,
        axes: Sequence[str],
    ) -> InductionResult:
        """Package an induced or revised rule for the dashboard."""
        primitive = PRIMITIVES[rule.action]

        deviation_table = [
            {
                "axis": axis,
                "label": PSEUDO_AXES[axis]["label"]
                if axis in PSEUDO_AXES
                else config.AXIS_BY_KEY[axis]["short"],
                "value": round(resolve_axis(features, axis), 2),
                "nominal": list(PSEUDO_AXES[axis]["nominal"])
                if axis in PSEUDO_AXES
                else list(config.AXIS_BY_KEY[axis]["nominal"]),
                "deviation": round(
                    4.0
                    if axis == "stream_ok" and not features.stream_ok
                    else features.deviation.get(axis, 0.0),
                    3,
                ),
            }
            for axis in axes
            if axis in config.AXIS_BY_KEY or axis in PSEUDO_AXES
        ]

        explanation = (
            f"{rule.id} learned that "
            f"{' AND '.join(c.to_string() for c in rule.conditions)} predicts "
            f"{primitive.title.lower()}, so it will apply `{primitive.key}` "
            f"({primitive.consequence.lower()}) on every future frame that "
            f"satisfies the predicate."
        )

        return InductionResult(
            rule=rule,
            candidates=candidates,
            deviation_table=deviation_table,
            observation=dict(features.values),
            explanation=explanation,
        )

    @staticmethod
    def _rationale(
        features: FeatureVector,
        axes: Sequence[str],
        primitive: ActionPrimitive,
        scored: ScoredAction,
    ) -> str:
        drop = 0.0
        if scored.current_residual > 0:
            drop = (scored.current_residual - scored.projected_residual) / scored.current_residual
        axes_txt = ", ".join(config.AXIS_BY_KEY[a]["short"] for a in axes if a in config.AXIS_BY_KEY)
        return (
            f"{primitive.summary}. Deviation on {axes_txt} totals "
            f"{scored.current_residual:.2f}; {primitive.title} is predicted to leave "
            f"{scored.projected_residual:.2f} ({drop * 100:.0f}% reduction) at a cost of "
            f"{primitive.cost:.2f}, the best trade-off among {len(PRIMITIVES)} primitives."
        )

    # -- WORKING 2: recall and reinforcement -------------------------------
    def recall(self, features: FeatureVector) -> List[RecallHit]:
        """WORKING 2 - rank stored rules against the live anomaly."""
        hits: List[RecallHit] = []
        with self._lock:
            for rule in self.rules:
                results = rule.explains(features)
                if not all(results):
                    continue
                score = rule.confidence * rule.specificity
                hits.append(
                    RecallHit(
                        rule=rule,
                        score=score,
                        condition_results=results,
                        satisfied=sum(results),
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def register_application(self, rule_id: str) -> Optional[Rule]:
        with self._lock:
            rule = self.get(rule_id)
            if rule is None:
                return None
            rule.times_applied += 1
            rule.last_used_at = time.time()
            rule.confidence = max(
                config.CONFIDENCE_FLOOR, rule.confidence - config.CONFIDENCE_DECAY
            )
            rule.confidence_trace.append(rule.confidence)
            self.stats["recalls"] += 1
            return rule

    def verify(
        self,
        rule_id: str,
        residual_now: float,
        residual_before: float,
        expected_residual: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Judge the last application and reinforce or penalise accordingly.

        The verdict is measured against the rule's *own forward model* rather
        than against an arbitrary fraction: a rule is credited when the
        telemetry it predicted is the telemetry that arrived.  A rule whose
        prediction was never achievable is a bad model, and demoting it is the
        only honest response.
        """
        with self._lock:
            rule = self.get(rule_id)
            if rule is None:
                return {"verdict": "unknown", "rule_id": rule_id}

            if expected_residual is None:
                expected_residual = residual_before * 0.65
            budget = expected_residual * 1.25 + 0.15

            helped = residual_now <= budget
            delta = 0.0
            if helped:
                rule.times_helped += 1
                rule.last_outcome = (
                    f"confirmed: deviation fell to {residual_now:.2f}, "
                    f"inside its predicted {budget:.2f}"
                )
                rule.confidence = min(
                    config.CONFIDENCE_CEIL, rule.confidence + config.CONFIDENCE_GAIN
                )
                delta = config.CONFIDENCE_GAIN
                self.stats["verified_success"] += 1
            else:
                rule.times_failed += 1
                rule.last_outcome = (
                    f"contested: deviation still {residual_now:.2f}, "
                    f"its model only predicted {budget:.2f}"
                )
                rule.confidence = max(
                    config.CONFIDENCE_FLOOR, rule.confidence - config.CONFIDENCE_PENALTY
                )
                delta = -config.CONFIDENCE_PENALTY
                self.stats["verified_failure"] += 1

            rule.confidence_trace.append(rule.confidence)
            rule.outcome_trace.append(1 if helped else 0)
            return {
                "verdict": "helped" if helped else "failed",
                "rule_id": rule.id,
                "title": rule.title,
                "action": rule.action,
                "confidence": round(rule.confidence, 3),
                "delta": round(delta, 3),
                "residual_before": round(residual_before, 3),
                "residual_after": round(residual_now, 3),
                "expected": round(expected_residual, 3),
                "budget": round(budget, 3),
                "detail": rule.last_outcome,
            }

    # -- accessors ---------------------------------------------------------
    def get(self, rule_id: str) -> Optional[Rule]:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def ranked(self) -> List[Rule]:
        with self._lock:
            return sorted(self.rules, key=lambda r: (r.confidence, r.times_applied), reverse=True)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            applied = sum(r.times_applied for r in self.rules)
            judged = sum(r.times_helped + r.times_failed for r in self.rules)
            helped = sum(r.times_helped for r in self.rules)
            return {
                "rules": [r.to_dict() for r in self.ranked()],
                "count": len(self.rules),
                "applied": applied,
                "judged": judged,
                "helped": helped,
                "mean_confidence": round(
                    sum(r.confidence for r in self.rules) / len(self.rules), 3
                )
                if self.rules
                else 0.0,
                "reinforcement_rate": round(helped / judged, 3) if judged else 0.0,
                "recall_rate": round(self.stats["recalls"] / self.stats["anomalies_seen"], 3)
                if self.stats["anomalies_seen"]
                else 0.0,
                "stats": dict(self.stats),
            }