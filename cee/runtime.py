"""Process wiring: scenario source -> bus -> engine -> dashboard.

``CEERuntime`` owns every long-lived object in the system and is the single
place a launcher needs to touch.  In local mode everything runs in one process
over the in-process bus, so ``python program_3_dashboard.py`` is the whole demo.
With ``use_mqtt`` the same runtime mirrors frames to a session-scoped broker
topic so the rover, the brain and the dashboard can be split across processes.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from . import config
from .actions import PRIMITIVES
from .actuators import ActuatorBank
from .bus import TelemetryBus
from .engine import CEEEngine
from .knowledge import KnowledgeBase


@dataclass(frozen=True)
class Hazard:
    """A physical failure the scenario source can inject into the rover."""

    key: str
    label: str
    blurb: str
    domain: str
    burst: int
    targets: Dict[str, Any]


def _r(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)


HAZARDS: Dict[str, Hazard] = {
    h.key: h
    for h in (
        Hazard(
            key="slip",
            label="Traction slip",
            blurb="Wheels spin far faster than the acceleration they produce.",
            domain="navigation",
            burst=10,
            targets={"wheel_rpm_avg": (148.0, 178.0), "imu_accel_x": (0.18, 0.52)},
        ),
        Hazard(
            key="wall_collision",
            label="Wall impact",
            blurb="Chassis is being pushed backwards at point-blank radar range.",
            domain="navigation",
            burst=8,
            targets={
                "wheel_rpm_avg": (0.0, 8.0),
                "imu_accel_x": (-1.25, -0.55),
                "ultrasonic_cm": (3.0, 9.0),
            },
        ),
        Hazard(
            key="mud_stuck",
            label="Mud / sand stall",
            blurb="High resistance: wheels and acceleration collapse with clear path.",
            domain="navigation",
            burst=12,
            targets={"wheel_rpm_avg": (22.0, 44.0), "imu_accel_x": (0.02, 0.28)},
        ),
        Hazard(
            key="offroad",
            label="Severe incline",
            blurb="Chassis pitch leaves its level envelope while climbing.",
            domain="navigation",
            burst=12,
            targets={"imu_pitch_deg": (18.0, 26.0), "wheel_rpm_avg": (58.0, 80.0)},
        ),
        Hazard(
            key="item_crossing",
            label="Obstacle crossing",
            blurb="A dynamic object crosses the forward radar envelope.",
            domain="navigation",
            burst=7,
            targets={"ultrasonic_cm": (11.0, 21.0)},
        ),
        Hazard(
            key="thermal_overload",
            label="Thermal overload",
            blurb="Compute thermal envelope exceeded under sustained load.",
            domain="system",
            burst=12,
            targets={"system_temp_c": (79.0, 86.0)},
        ),
        Hazard(
            key="sensor_blindness",
            label="Radar blinded",
            blurb="Ultrasonic head is covered and reports the invalid sentinel.",
            domain="system",
            burst=10,
            targets={"ultrasonic_cm": (999.0, 999.0)},
        ),
        Hazard(
            key="telemetry_loss",
            label="Telemetry severed",
            blurb="The Wi-Fi link to the rover drops and the frame is lost.",
            domain="system",
            burst=7,
            targets={"__sever_link__": True},
        ),
        Hazard(
            key="battery_depletion",
            label="Pack depletion",
            blurb="Cell voltage sags under sustained drive load and will not hold charge.",
            domain="power",
            burst=14,
            targets={"battery_v": (9.6, 10.6)},
        ),
        Hazard(
            key="steering_actuator_jam",
            label="Steering jam",
            blurb="Steer actuator cannot reach its commanded angle; heading error grows.",
            domain="actuator",
            burst=12,
            targets={"steer_err_deg": (19.0, 31.0), "imu_yaw_rate": (9.0, 16.0)},
        ),
        Hazard(
            key="heading_drift",
            label="Heading drift",
            blurb="Rover yaws steadily off the commanded straight line with no obstacle.",
            domain="navigation",
            burst=13,
            targets={"imu_yaw_rate": (12.0, 21.0), "odom_drift_m": (1.5, 2.8)},
        ),
        Hazard(
            key="payload_shift",
            label="Payload shift",
            blurb="Cargo has shifted and the chassis is leaning outside its level envelope.",
            domain="mechanical",
            burst=12,
            targets={"tilt_roll_deg": (14.0, 23.0), "imu_pitch_deg": (4.0, 11.0)},
        ),
        Hazard(
            key="encoder_desync",
            label="Encoder desync",
            blurb="Wheel odometry diverges from what the chassis is actually doing.",
            domain="localisation",
            burst=12,
            targets={"odom_drift_m": (2.9, 4.6), "gps_fix_quality": (1.8, 2.9)},
        ),
        Hazard(
            key="motor_overcurrent",
            label="Motor overcurrent",
            blurb="A winding is drawing well past its design current and heating itself.",
            domain="mechanical",
            burst=11,
            targets={"motor_current_a": (19.0, 26.0), "system_temp_c": (56.0, 68.0)},
        ),
        Hazard(
            key="gps_degraded",
            label="GPS degraded",
            blurb="Satellite geometry collapses and the fix becomes too loose to steer by.",
            domain="environment",
            burst=13,
            targets={"gps_fix_quality": (6.5, 10.5), "odom_drift_m": (1.2, 2.4)},
        ),
        Hazard(
            key="dust_blinding",
            label="Dust blinding",
            blurb="Airborne particulate is obscuring the optical head.",
            domain="environment",
            burst=12,
            targets={"dust_opacity": (0.72, 0.96), "ultrasonic_cm": (30.0, 62.0)},
        ),
    )
}


def nominal_sample() -> Dict[str, float]:
    """A healthy telemetry frame, drawn from inside every operating band."""
    return {
        "wheel_rpm_avg": _r(85.5, 94.5),
        "imu_accel_x": _r(1.22, 1.48),
        "imu_pitch_deg": _r(-1.8, 1.8),
        "ultrasonic_cm": _r(55.0, 145.0),
        "system_temp_c": _r(35.2, 37.8),
        "battery_v": _r(12.3, 12.85),
        "steer_err_deg": _r(-1.3, 1.3),
        "imu_yaw_rate": _r(-1.5, 1.5),
        "tilt_roll_deg": _r(-1.2, 1.2),
        "odom_drift_m": _r(0.04, 0.30),
        "gps_fix_quality": _r(0.7, 1.3),
        "motor_current_a": _r(4.2, 5.6),
        "dust_opacity": _r(0.06, 0.20),
    }


class ScenarioSource:
    """Generates the rover's telemetry stream and injects hazard bursts."""

    def __init__(self, bus: TelemetryBus, actuators: ActuatorBank) -> None:
        self.bus = bus
        self.actuators = actuators
        self.active: Optional[Hazard] = None
        self.burst_left = 0
        self.last_key: Optional[str] = None
        self.paused = False
        self.speed = 1.0
        self.auto = False
        self.auto_interval = 14.0
        self._auto_at = 0.0
        self._lock = threading.RLock()
        self.emitted = 0

    # -- control -----------------------------------------------------------
    def inject(self, key: str, burst: Optional[int] = None) -> Optional[Hazard]:
        hazard = HAZARDS.get(key)
        if hazard is None:
            return None
        with self._lock:
            self.active = hazard
            self.burst_left = burst if burst is not None else hazard.burst
            self.last_key = key
        return hazard

    def clear(self) -> None:
        with self._lock:
            self.active = None
            self.burst_left = 0

    def repeat_last(self) -> Optional[Hazard]:
        if not self.last_key:
            return None
        return self.inject(self.last_key)

    def maybe_auto(self) -> None:
        if not self.auto:
            return
        now = time.monotonic()
        if now < self._auto_at:
            return
        self._auto_at = now + self.auto_interval / max(0.25, self.speed)
        with self._lock:
            if self.burst_left > 0:
                return
        key = random.choice(list(HAZARDS))
        self.inject(key)

    # -- frame synthesis ---------------------------------------------------
    def build_frame(self) -> Dict[str, Any]:
        nominal = nominal_sample()
        target = dict(nominal)
        sever_link = False

        with self._lock:
            hazard = self.active
            burst = self.burst_left
            if hazard is not None:
                if burst > 0:
                    for axis, span in hazard.targets.items():
                        if axis == "__sever_link__":
                            sever_link = True
                            continue
                        target[axis] = _r(*span) if isinstance(span, tuple) else span

        # Hardware safing deliberately holds the outage open long enough for the
        # brain to latch a safe pose, then releases the link.
        if sever_link and self.actuators.safing:
            with self._lock:
                self.burst_left = max(self.burst_left, 3)

        published = self.actuators.fold(nominal, target)

        frame: Dict[str, Any] = {
            "timestamp": round(time.time(), 3),
            "frame_seq": self.emitted,
            "stream_ok": not sever_link,
            # Channels the engine itself had already been folded into on this
            # frame.  Learning from such a frame would describe our own
            # mitigation instead of the rover's behaviour.
            "intervened": sorted({i.key for i in self.actuators.active}),
        }
        frame.update(published)
        frame["masked"] = sorted(self.actuators.masked_axes())
        if hazard is not None:
            frame["scenario"] = hazard.key
        return frame

    def advance(self) -> Dict[str, Any]:
        frame = self.build_frame()
        with self._lock:
            if self.burst_left > 0:
                self.burst_left -= 1
                if self.burst_left == 0:
                    self.active = None
            self.emitted += 1
        return frame

    # -- reporting ---------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "paused": self.paused,
                "speed": self.speed,
                "auto": self.auto,
                "auto_interval": self.auto_interval,
                "emitted": self.emitted,
                "active_hazard": self.active.key if self.active else None,
                "active_label": self.active.label if self.active else None,
                "burst_left": self.burst_left,
                "last_hazard": self.last_key,
                "burst_total": self.active.burst if self.active else 0,
            }


class CEERuntime:
    """Everything the project needs, wired together."""

    def __init__(
        self,
        use_mqtt: bool = False,
        autoload_memory: bool = True,
        topic: Optional[str] = None,
    ) -> None:
        config.ensure_state_dir()
        self.kb = KnowledgeBase(autoload=autoload_memory)
        self.actuators = ActuatorBank()
        self.bus = TelemetryBus(use_mqtt=use_mqtt, topic=topic)
        self.engine = CEEEngine(
            knowledge=self.kb,
            actuators=self.actuators,
            event_sink=self._on_event,
        )
        self.scenario = ScenarioSource(self.bus, self.actuators)
        self.bus.subscribe(self.engine.ingest)
        self.started_at = time.time()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._dirty = threading.Event()
        self._pending_save = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self, produce: bool = True) -> None:
        if produce and self.bus.use_mqtt:
            self.bus.connect_mqtt()
        if produce and not self.bus.use_mqtt:
            # Nothing to connect: the in-process bus is already live.
            pass
        if produce:
            self._thread = threading.Thread(target=self._loop, name="cee-sim", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.kb.save()
        self.bus.disconnect_mqtt()

    def _loop(self) -> None:
        interval = config.FRAME_INTERVAL
        while not self._stop.is_set():
            started = time.monotonic()
            with self._lock:
                paused = self.scenario.paused
                speed = self.scenario.speed
            if not paused:
                try:
                    self.scenario.maybe_auto()
                    frame = self.scenario.advance()
                    self.bus.publish(frame)
                except Exception:
                    pass
                self._dirty.set()
            sleep = max(0.05, interval / max(0.25, speed))
            remaining = sleep - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)
            else:
                self._stop.wait(0.01)

    def publish_external(self, frame: Mapping[str, Any]) -> Dict[str, Any]:
        """Feed a frame from an outside source (MQTT, rover game, tests)."""
        self.bus.publish(dict(frame), mirror=False)
        self._dirty.set()
        return self.engine.last_cycle

    # -- persistence -------------------------------------------------------
    def _on_event(self, _event: Dict[str, Any]) -> None:
        self._dirty.set()

    def maybe_save(self, interval: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._pending_save < interval:
            return
        self._pending_save = now
        self.kb.save()

    # -- controls ----------------------------------------------------------
    def control(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        action = str(payload.get("action", "")).strip()
        scenario = self.scenario

        if action == "inject":
            hazard = scenario.inject(str(payload.get("hazard", "")))
            return {"ok": hazard is not None, "hazard": hazard.key if hazard else None}
        if action == "repeat":
            hazard = scenario.repeat_last()
            return {"ok": hazard is not None, "hazard": hazard.key if hazard else None}
        if action == "clear_hazard":
            scenario.clear()
            return {"ok": True}
        if action == "pause":
            scenario.paused = True
            return {"ok": True, "paused": True}
        if action == "resume":
            scenario.paused = False
            return {"ok": True, "paused": False}
        if action == "toggle_pause":
            scenario.paused = not scenario.paused
            return {"ok": True, "paused": scenario.paused}
        if action == "speed":
            try:
                scenario.speed = max(0.25, min(8.0, float(payload.get("value", 1.0))))
            except (TypeError, ValueError):
                pass
            return {"ok": True, "speed": scenario.speed}
        if action == "auto":
            scenario.auto = bool(payload.get("value", not scenario.auto))
            return {"ok": True, "auto": scenario.auto}
        if action == "reset_memory":
            self.kb.reset()
            self.actuators.clear()
            self.engine.last_induction = None
            self.engine.last_recall = None
            self.engine.last_verification = None
            return {"ok": True}
        if action == "reset_interventions":
            self.actuators.clear()
            return {"ok": True}
        if action == "save":
            self.kb.save()
            return {"ok": True}
        return {"ok": False, "error": f"unknown action: {action}"}

    # -- reporting ---------------------------------------------------------
    def hazard_catalog(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": h.key,
                "label": h.label,
                "blurb": h.blurb,
                "domain": h.domain,
                "burst": h.burst,
            }
            for h in HAZARDS.values()
        ]

    def state(self) -> Dict[str, Any]:
        self.maybe_save()
        engine = self.engine
        knowledge = self.kb.snapshot()
        telemetry = engine.snapshot()

        verdict = self._verdict(engine)
        uptime = time.time() - self.started_at

        return {
            "meta": {
                "session": config.SESSION_ID,
                "version": "2.0.0",
                "uptime": round(uptime, 1),
                "memory_file": str(config.KNOWLEDGE_FILE),
                "frame_interval": config.FRAME_INTERVAL,
            },
            "transport": self.bus.status(),
            "axes": list(config.AXES),
            "scenario": self.scenario.status(),
            "hazards": self.hazard_catalog(),
            "status": verdict,
            "telemetry": telemetry["telemetry"],
            "display": telemetry["display"],
            "cycle": engine.last_cycle,
            "pipeline": telemetry["pipeline"],
            "workings": telemetry["workings"],
            "narration": engine.narration,
            "knowledge": knowledge,
            "actuators": telemetry["actuators"],
            "induction": engine.last_induction,
            "recall": engine.last_recall,
            "verification": engine.last_verification,
            "counters": engine.counters,
            "history": list(engine.history)[-140:],
            "events": list(engine.events)[-60:][::-1],
            "primitives": [p.to_dict() for p in PRIMITIVES.values()],
        }

    @staticmethod
    def _verdict(engine: CEEEngine) -> Dict[str, Any]:
        if not engine.prev:
            return {"level": "idle", "label": "STANDBY", "detail": "No telemetry received yet"}
        if not engine.prev.stream_ok:
            return {
                "level": "critical",
                "label": "LINK SEVERED",
                "detail": "Telemetry pipeline has stopped delivering frames",
            }
        if engine.cooldown > 0 or engine.pending is not None:
            return {
                "level": "acting",
                "label": "MITIGATING",
                "detail": "A recalled rule is active and awaiting verification",
            }
        if engine.prev.severity >= 0.18:
            return {
                "level": "warn",
                "label": "ANOMALY",
                "detail": "Deviation detected outside the healthy envelope",
            }
        return {
            "level": "nominal",
            "label": "NOMINAL",
            "detail": "All channels inside their healthy operating bands",
        }