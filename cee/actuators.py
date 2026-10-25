"""Actuator model driven by recalled and induced rules.

A rule that is never executed is a comment, not a rule.  Every primitive in
:mod:`cee.actions` carries a command payload; applying it writes a *bounded
intervention* into this model, and the vehicle simulator asks the model to fold
those interventions into the telemetry it is about to publish.

An intervention is built from the same primitives as the forward model used at
induction time: ``MITIGATION`` mirrors ``actions.attenuate``, ``BOOSTS`` mirrors
``actions.boost`` and ``ENFORCERS`` mirrors ``actions.reseat_drivetrain``.  That
is deliberate.  When the engine later checks a rule's prediction against
reality, the two sides have to be speaking about the same intervention.

The model is a mitigation on the *fault term* rather than a hard normaliser,
which mirrors real behaviour: a control action reduces a fault but does not
abolish it.  The engine can therefore still observe the anomaly persisting at
lower magnitude, judge whether its rule helped, and recall the same rule again
once the intervention decays.  That loop -- induce, apply, verify, decay,
recall -- is the whole point of the system.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Mapping

from .actions import band_base
from .config import AXES, DRIVETRAIN_ACCEL_PER_RPM

DEFAULT_HOLD_FRAMES = 4

AXIS_RANGES: Dict[str, Any] = {spec["key"]: spec["range"] for spec in AXES}

# How strongly each primitive attenuates the fault on each channel.
# 1.0 = no effect. 0.0 = the channel is fully restored.
MITIGATION: Dict[str, Dict[str, float]] = {
    "cap_motor_torque": {"wheel_rpm_avg": 0.15},
    "emergency_brake": {
        "wheel_rpm_avg": 0.55,
        "imu_accel_x": 0.65,
        "ultrasonic_cm": 0.70,
    },
    "reverse_and_recalculate_path": {"wheel_rpm_avg": 0.55},
    "engage_max_torque_burst": {"wheel_rpm_avg": 0.45, "imu_accel_x": 0.50},
    "enable_4x4_crawl_mode": {"imu_pitch_deg": 0.30, "wheel_rpm_avg": 0.55},
    "navigate_by_imu_only": {},
    "throttle_cognitive_hz": {"system_temp_c": 0.42, "wheel_rpm_avg": 0.90},
    "trigger_hardware_safing_mode": {
        "wheel_rpm_avg": 0.10,
        "imu_accel_x": 0.10,
        "imu_pitch_deg": 0.20,
        "ultrasonic_cm": 0.35,
    },
    "recalibrate_radar_sweep": {},
    "hold_and_reacquire": {
        "wheel_rpm_avg": 0.45,
        "imu_accel_x": 0.45,
        "imu_pitch_deg": 0.45,
    },
    "enter_energy_saver_mode": {
        "battery_v": 0.30,
        "motor_current_a": 0.25,
        "wheel_rpm_avg": 0.60,
        "imu_accel_x": 0.60,
    },
    "reanchor_localisation_fusion": {},
    "power_down_for_thermal_soak": {
        "dust_opacity": 0.35,
        "motor_current_a": 0.20,
        "wheel_rpm_avg": 0.25,
        "imu_accel_x": 0.25,
        "imu_yaw_rate": 0.30,
    },
    "level_payload_and_hold": {
        "tilt_roll_deg": 0.20,
        "wheel_rpm_avg": 0.15,
        "imu_accel_x": 0.15,
        "imu_pitch_deg": 0.60,
    },
    "cycle_steer_actuator_to_stops": {"imu_yaw_rate": 0.35, "wheel_rpm_avg": 0.70},
}

# Absolute offsets applied after attenuation (torque burst, throttle shed).
BOOSTS: Dict[str, Dict[str, float]] = {
    "engage_max_torque_burst": {"wheel_rpm_avg": 55.0, "imu_accel_x": 0.70},
}

# Post-conditions that re-establish a physical coupling the motor cannot set
# directly.
def _reseat(v: Dict[str, float]) -> Dict[str, float]:
    """Re-couple chassis acceleration to the (possibly changed) wheel speed."""
    return {**v, "imu_accel_x": v.get("wheel_rpm_avg", 90.0) * DRIVETRAIN_ACCEL_PER_RPM}


ENFORCERS: Dict[str, Callable[[Dict[str, float]], Dict[str, float]]] = {
    "cap_motor_torque": _reseat,
    "reverse_and_recalculate_path": _reseat,
}

INTERVENTION_LABEL = {
    "cap_motor_torque": "Motor torque limited to 50%",
    "emergency_brake": "Emergency brake asserted",
    "reverse_and_recalculate_path": "Reversing, path recalculated",
    "engage_max_torque_burst": "Maximum torque burst engaged",
    "enable_4x4_crawl_mode": "4x4 crawl gearing engaged",
    "navigate_by_imu_only": "Radar demoted, IMU-only navigation",
    "throttle_cognitive_hz": "Cognitive clock throttled",
    "trigger_hardware_safing_mode": "Hardware safing mode latched",
    "recalibrate_radar_sweep": "Radar sweep recalibrating",
    "hold_and_reacquire": "Coasting to reacquire telemetry",
    "enter_energy_saver_mode": "Energy saver, mission task aborted",
    "reanchor_localisation_fusion": "Pose re-anchored, holding position",
    "power_down_for_thermal_soak": "Powered down for thermal soak",
    "level_payload_and_hold": "Payload re-levelling, holding position",
    "cycle_steer_actuator_to_stops": "Steering actuator cycled and re-zeroed",
}

# Interventions that change what the engine is allowed to *believe* rather than
# what the vehicle does.  Demoting an untrustworthy channel is a genuine
# perception decision: its reading stops counting as evidence of a fault.
INTERVENTION_MASKS: Dict[str, tuple] = {
    "navigate_by_imu_only": ("ultrasonic_cm",),
    # A degraded satellite fix is not evidence of a fault, it is a loss of
    # precision, so once the rover has re-anchored onto a trusted reference the
    # quality channel stops counting against it.
    "reanchor_localisation_fusion": ("gps_fix_quality", "odom_drift_m"),
    # Dust on an optical head degrades perception rather than proving a fault,
    # so once the rover has powered down the dust reading stops counting
    # against it.
    "power_down_for_thermal_soak": ("dust_opacity",),
}

# Interventions that write an absolute value the attenuator cannot reach.
INTERVENTION_OVERRIDES: Dict[str, Dict[str, float]] = {
    "recalibrate_radar_sweep": {"ultrasonic_cm": 90.0},
    "reanchor_localisation_fusion": {
        "odom_drift_m": 0.18,
        "gps_fix_quality": 1.0,
        "imu_yaw_rate": 1.0,
        "steer_err_deg": 0.0,
    },
    "cycle_steer_actuator_to_stops": {"steer_err_deg": 0.0},
    "level_payload_and_hold": {"tilt_roll_deg": 0.2},
}


@dataclass
class Intervention:
    key: str
    label: str
    rule_id: str
    factors: Dict[str, float]
    frames_left: int
    started_at: float
    holds_for: int
    boosts: Dict[str, float] = field(default_factory=dict)
    masks: frozenset = field(default_factory=frozenset)

    def factor(self, axis: str) -> float:
        return self.factors.get(axis, 1.0)

    def mitigation(self) -> float:
        """Fraction of the fault this intervention removes (0..1)."""
        if not self.factors:
            return 0.0
        return 1.0 - min(self.factors.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "rule_id": self.rule_id,
            "frames_left": self.frames_left,
            "hold_total": self.holds_for,
            "strength": round(self.mitigation(), 3),
            "masks": sorted(self.masks),
            "boosts": dict(self.boosts),
            "started_at": self.started_at,
        }


class ActuatorBank:
    """Bounded control interventions currently held against the vehicle."""

    def __init__(self, hold_frames: int = DEFAULT_HOLD_FRAMES) -> None:
        self.hold_frames = hold_frames
        self.active: List[Intervention] = []
        self.history: Deque[Dict[str, Any]] = deque(maxlen=40)
        self.applied_total = 0

    # -- actuation ---------------------------------------------------------
    def apply(
        self,
        key: str,
        rule_id: str,
        note: str = "",
        hold: int | None = None,
    ) -> Intervention:
        frames = hold if hold is not None else self.hold_frames
        intervention = Intervention(
            key=key,
            label=INTERVENTION_LABEL.get(key, key.replace("_", " ")),
            rule_id=rule_id,
            factors=dict(MITIGATION.get(key, {})),
            boosts=dict(BOOSTS.get(key, {})),
            frames_left=frames,
            started_at=time.time(),
            holds_for=frames,
            masks=frozenset(INTERVENTION_MASKS.get(key, ())),
        )
        # Repeating the same action refreshes rather than stacks, so a recalled
        # rule keeps its mitigation alive without stacking effects.
        self.active = [i for i in self.active if i.key != key]
        self.active.append(intervention)
        self.applied_total += 1

        entry = intervention.to_dict()
        entry["note"] = note
        self.history.appendleft(entry)
        return intervention

    def step(self) -> None:
        """Age every intervention by one telemetry frame."""
        for intervention in list(self.active):
            intervention.frames_left -= 1
            if intervention.frames_left <= 0:
                self.active.remove(intervention)

    def clear(self) -> None:
        self.active = []

    # -- queries -----------------------------------------------------------
    def factor(self, axis: str) -> float:
        f = 1.0
        for intervention in self.active:
            f *= intervention.factor(axis)
        return max(0.0, min(1.0, f))

    def masked_axes(self) -> frozenset:
        out: set = set()
        for intervention in self.active:
            out |= intervention.masks
        return frozenset(out)

    def overridden_axes(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for intervention in self.active:
            for axis, value in INTERVENTION_OVERRIDES.get(intervention.key, {}).items():
                out[axis] = value
        return out

    @property
    def radar_trust(self) -> bool:
        return "ultrasonic_cm" not in self.masked_axes()

    @property
    def safing(self) -> bool:
        return any(i.key == "trigger_hardware_safing_mode" for i in self.active)

    @property
    def busy(self) -> bool:
        return bool(self.active)

    def fold(
        self, nominal: Mapping[str, float], target: Mapping[str, float]
    ) -> Dict[str, float]:
        """Fold active interventions into a telemetry frame.

        ``nominal`` is the healthy reading for this frame and ``target`` is what
        the scenario wanted to publish.  The published value is the healthy
        baseline plus the *attenuated* fault, then any boosts and post-conditions
        the active interventions carry.
        """
        values: Dict[str, float] = {
            axis: float(target.get(axis, nominal.get(axis, 0.0)))
            for axis in nominal
        }

        for axis in list(values):
            factor = self.factor(axis)
            if factor >= 1.0:
                continue
            base = float(nominal.get(axis, band_base(axis)))
            values[axis] = base + (values[axis] - base) * factor

        for intervention in self.active:
            for axis, delta in intervention.boosts.items():
                values[axis] = values.get(axis, 0.0) + delta
            enforcer = ENFORCERS.get(intervention.key)
            if enforcer is not None:
                values = enforcer(values)

        values.update(self.overridden_axes())

        out: Dict[str, float] = {}
        for axis, value in values.items():
            spec = AXIS_RANGES.get(axis)
            if axis == "ultrasonic_cm":
                out[axis] = round(max(0.0, value), 2)
            elif spec is not None:
                lo, hi = spec
                out[axis] = round(max(lo - 25.0, min(hi + 25.0, value)), 2)
            else:
                out[axis] = round(value, 2)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": [
                i.to_dict() for i in sorted(self.active, key=lambda i: -i.mitigation())
            ],
            "history": list(self.history)[:12],
            "applied_total": self.applied_total,
            "radar_trust": self.radar_trust,
            "safing": self.safing,
        }