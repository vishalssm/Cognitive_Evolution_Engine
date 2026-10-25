"""Primitive action library.

An induced rule is a predicate plus an *action*.  The engine cannot invent new
motor commands, so it reasons over a fixed library of vehicle/compute
primitives.  What it genuinely does decide is:

  * which predicates to generalise from an observation (see knowledge.py), and
  * which primitive to attach, by forward-simulating each primitive's effect on
    the telemetry and choosing the one that predicts the smallest residual.

This is a neuro-symbolic split: the symbols (conditions, rules) evolve; the
actuator vocabulary is fixed and interpretable.

The forward model and the real actuators are deliberately built from the same
pieces -- ``attenuate``/``boost`` here, ``MITIGATION``/``enforce`` in
actuators.py -- so that when a rule's prediction is later checked against
reality, the comparison is meaningful.  A primitive whose model does not match
its implementation gets demoted, which is the point of verifying at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .config import AXES, AXIS_BY_KEY, DRIVETRAIN_ACCEL_PER_RPM
from .features import FeatureVector, axis_deviation


# -- shared numerical helpers ----------------------------------------------

def band_base(axis: str) -> float:
    """Centre of an axis' healthy operating band -- the no-fault reading."""
    lo, hi = AXIS_BY_KEY[axis]["nominal"]
    return (lo + hi) / 2.0


def attenuate(values: Mapping[str, float], factors: Mapping[str, float]) -> Dict[str, float]:
    """Scale each channel's distance from its healthy band by ``factors``.

    A factor of 1.0 leaves the deviation untouched, 0.0 snaps the channel back
    to nominal.  Anchoring on the band centre rather than on the raw value is
    what makes the model physically sensible: a control intervention reduces a
    *fault*, it does not scale the healthy reading.
    """
    out = dict(values)
    for axis, factor in factors.items():
        base = band_base(axis)
        out[axis] = base + (float(values.get(axis, base)) - base) * factor
    return out


def boost(values: Mapping[str, float], deltas: Mapping[str, float]) -> Dict[str, float]:
    """Add an absolute offset to a channel (torque bursts, throttle sheds)."""
    out = dict(values)
    for axis, delta in deltas.items():
        base = band_base(axis)
        out[axis] = float(values.get(axis, base)) + delta
    return out


def reseat_drivetrain(values: Mapping[str, float]) -> Dict[str, float]:
    """Re-couple chassis acceleration to wheel speed.

    Applying wheel-side torque changes are only useful if the chassis follows;
    this is the relation the IMU and the wheel encoders share.
    """
    out = dict(values)
    out["imu_accel_x"] = out.get("wheel_rpm_avg", 90.0) * DRIVETRAIN_ACCEL_PER_RPM
    return out


@dataclass(frozen=True)
class Projected:
    """Predicted telemetry state after applying a primitive."""

    values: Dict[str, float]
    masked: frozenset = field(default_factory=frozenset)

    def deviation(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for spec in AXES:
            key = spec["key"]
            out[key] = (
                0.0
                if key in self.masked
                else axis_deviation(spec, self.values.get(key, 0.0))
            )
        return out

    def residual(self, axes: Sequence[str]) -> float:
        dev = self.deviation()
        return sum(abs(dev.get(a, 0.0)) for a in axes if a not in self.masked)


@dataclass(frozen=True)
class ActionPrimitive:
    key: str
    title: str
    summary: str
    consequence: str
    cost: float
    influences: Sequence[str]
    applies: Callable[[FeatureVector], bool]
    project: Callable[[FeatureVector], Projected]
    render: Callable[[Sequence[Any], Dict[str, float]], str]
    command: Callable[[FeatureVector], Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "summary": self.summary,
            "consequence": self.consequence,
            "cost": self.cost,
            "influences": list(self.influences),
        }


def _v(features: FeatureVector) -> Dict[str, float]:
    return dict(features.values)


# --- primitives ------------------------------------------------------------

def _applies_cap(features: FeatureVector) -> bool:
    # Wheels outrunning the acceleration they should produce == tyre slip.
    return (
        features.deviation.get("wheel_rpm_avg", 0.0) > 0.2
        and features.kin_residual < -0.25
    )


def _project_cap(features: FeatureVector) -> Projected:
    # Traction control exists to stop wheelspin, so it converges wheel speed
    # onto the value consistent with the grip actually available and re-couples
    # the chassis to it, rather than merely halving the fault.
    values = attenuate(_v(features), {"wheel_rpm_avg": 0.15})
    values = reseat_drivetrain(values)
    return Projected(values)


def _render_cap(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN cap_motor_torque(0.50) AND reseat_traction_model()"


def _cmd_cap(features: FeatureVector) -> Dict[str, Any]:
    return {"torque_cap": 0.5, "traction_reseat": True, "note": "Torque capped to 50%"}


def _applies_brake(features: FeatureVector) -> bool:
    return features.deviation.get("ultrasonic_cm", 0.0) < -0.2


def _project_brake(features: FeatureVector) -> Projected:
    values = attenuate(
        _v(features),
        {"wheel_rpm_avg": 0.55, "imu_accel_x": 0.65, "ultrasonic_cm": 0.70},
    )
    return Projected(values)


def _render_brake(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN emergency_brake() AND clear_path_window(0.8s)"


def _cmd_brake(features: FeatureVector) -> Dict[str, Any]:
    return {"brake": True, "brake_hold_s": 1.4, "note": "Emergency brake engaged"}


def _applies_reverse(features: FeatureVector) -> bool:
    # Negative forward acceleration means the chassis is being pushed backwards.
    return features.deviation.get("imu_accel_x", 0.0) < -0.15


def _project_reverse(features: FeatureVector) -> Projected:
    values = attenuate(_v(features), {"wheel_rpm_avg": 0.55})
    values = reseat_drivetrain(values)
    return Projected(values)


def _render_reverse(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN reverse_and_recalculate_path()"


def _cmd_reverse(features: FeatureVector) -> Dict[str, Any]:
    return {"reverse": True, "path_replan": True, "note": "Reversing and replanning"}


def _applies_burst(features: FeatureVector) -> bool:
    return (
        features.deviation.get("wheel_rpm_avg", 0.0) < -0.1
        and features.deviation.get("imu_accel_x", 0.0) < -0.1
        and features.deviation.get("ultrasonic_cm", 0.0) > -0.5
    )


def _project_burst(features: FeatureVector) -> Projected:
    values = attenuate(_v(features), {"wheel_rpm_avg": 0.45, "imu_accel_x": 0.50})
    values = boost(values, {"wheel_rpm_avg": 55.0, "imu_accel_x": 0.70})
    return Projected(values)


def _render_burst(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN engage_max_torque_burst()"


def _cmd_burst(features: FeatureVector) -> Dict[str, Any]:
    return {"torque_boost": 2.2, "note": "Maximum torque burst"}


def _applies_crawl(features: FeatureVector) -> bool:
    # Crawl gearing is a traction remedy for an *inclined chassis*.  A chassis
    # that is merely leaning because the cargo shifted is a load problem, so
    # roll alone no longer qualifies it.
    return (
        abs(features.deviation.get("imu_pitch_deg", 0.0)) > 0.4
        and features.deviation.get("tilt_roll_deg", 0.0) < 0.15
    )


def _project_crawl(features: FeatureVector) -> Projected:
    values = attenuate(_v(features), {"imu_pitch_deg": 0.30, "wheel_rpm_avg": 0.55})
    return Projected(values)


def _render_crawl(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN enable_4x4_crawl_mode()"


def _cmd_crawl(features: FeatureVector) -> Dict[str, Any]:
    return {"crawl_mode": True, "note": "4x4 low-ratio crawl engaged"}


def _applies_imu_only(features: FeatureVector) -> bool:
    return not features.radar_ok


def _project_imu_only(features: FeatureVector) -> Projected:
    # Perception channel is declared untrustworthy, so its deviation leaves the
    # residual instead of being "fixed".
    return Projected(_v(features), frozenset({"ultrasonic_cm"}))


def _render_imu_only(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds) or "radar_invalid"
    return f"IF {trigger} THEN navigate_by_imu_only() AND demote_radar_channel()"


def _cmd_imu_only(features: FeatureVector) -> Dict[str, Any]:
    return {"radar_trust": False, "note": "Radar demoted, IMU-only navigation"}


def _applies_throttle(features: FeatureVector) -> bool:
    return features.deviation.get("system_temp_c", 0.0) > 0.3


def _project_throttle(features: FeatureVector) -> Projected:
    values = attenuate(_v(features), {"system_temp_c": 0.42, "wheel_rpm_avg": 0.90})
    return Projected(values)


def _render_throttle(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN throttle_cognitive_hz(-1) AND shed_background_tasks()"


def _cmd_throttle(features: FeatureVector) -> Dict[str, Any]:
    return {"cpu_throttle": True, "note": "Cognitive clock throttled"}


def _applies_safing(features: FeatureVector) -> bool:
    return not features.stream_ok


def _project_safing(features: FeatureVector) -> Projected:
    values = attenuate(
        _v(features),
        {
            "wheel_rpm_avg": 0.10,
            "imu_accel_x": 0.10,
            "imu_pitch_deg": 0.20,
            "ultrasonic_cm": 0.35,
        },
    )
    return Projected(values)


def _render_safing(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds) or "data_stream_lost"
    return f"IF {trigger} THEN trigger_hardware_safing_mode() AND hold_last_safe_pose()"


def _cmd_safing(features: FeatureVector) -> Dict[str, Any]:
    return {"safing": True, "hold_pose": True, "note": "Hardware safing mode"}


def _applies_recalibrate(features: FeatureVector) -> bool:
    raw = features.values.get("ultrasonic_cm", 0.0)
    return raw < 3.0 or (900.0 <= raw < 998.0)


def _project_recalibrate(features: FeatureVector) -> Projected:
    values = dict(_v(features))
    values["ultrasonic_cm"] = 90.0
    return Projected(values)


def _render_recalibrate(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN recalibrate_radar_sweep() AND clear_fault_latch()"


def _cmd_recalibrate(features: FeatureVector) -> Dict[str, Any]:
    return {"radar_recal": True, "note": "Radar sweep recalibrating"}


def _applies_hold(features: FeatureVector) -> bool:
    # Coasting is a *perception* remedy: it stops the rover churning while the
    # operator decides whether the fault is real.  It cannot restart a
    # drivetrain, so when wheel speed and chassis acceleration are both clearly
    # low the rover is stalling rather than being misread, and the burst
    # primitive is the one that can actually recover it.
    drivetrain_stalled = (
        features.deviation.get("wheel_rpm_avg", 0.0) < -0.15
        and features.deviation.get("imu_accel_x", 0.0) < -0.15
    )
    return features.stream_ok and not drivetrain_stalled


def _project_hold(features: FeatureVector) -> Projected:
    # Coasting bleeds energy off the moving channels; it does not snap them
    # back into the healthy band, so the projection stays pessimistic.
    values = attenuate(
        _v(features), {"wheel_rpm_avg": 0.45, "imu_accel_x": 0.45, "imu_pitch_deg": 0.45}
    )
    return Projected(values)


def _render_hold(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN hold_and_reacquire()"


def _cmd_hold(features: FeatureVector) -> Dict[str, Any]:
    return {"coast": True, "note": "Holding for reacquisition"}


# --- degraded-mode primitives ------------------------------------------------
#
# Power, localisation and environmental faults cannot be fixed by commanding
# the drivetrain harder.  More torque on a flat pack drains it faster, and
# driving faster on a drifted pose drives further off course.  These primitives
# acknowledge that limit: the remedy is to *reduce* what the rover is doing and
# declare what it no longer trusts.


def _applies_energy(features: FeatureVector) -> bool:
    # Voltage below the healthy pack band is an energy fault, whatever else is
    # also wrong.  The rover sheds load rather than asking the motors for more.
    return features.deviation.get("battery_v", 0.0) < -0.15


def _project_energy(features: FeatureVector) -> Projected:
    # Cutting load lets the pack recover toward its resting voltage, and the
    # rover coasts, so motion channels bleed down with it.
    values = attenuate(
        _v(features),
        {
            "battery_v": 0.30,
            "motor_current_a": 0.25,
            "wheel_rpm_avg": 0.60,
            "imu_accel_x": 0.60,
        },
    )
    return Projected(values)


def _render_energy(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN enter_energy_saver_mode() AND abort_mission_task()"


def _cmd_energy(features: FeatureVector) -> Dict[str, Any]:
    return {"energy_saver": True, "abort_task": True, "note": "Energy saver, mission aborted"}


def _applies_reanchor(features: FeatureVector) -> bool:
    # Either the wheel odometry disagrees with the chassis, or the fix itself has
    # degraded.  Both mean the rover no longer knows where it is, and no amount
    # of wheel torque repairs that.  A steering actuator that will not reach its
    # commanded angle is part of the same fault: the pose is no longer
    # controllable, so the rover holds station and lets the actuator reset.
    # Yaw rate only counts as a *pose* fault when the steering actuator is
    # actually obeying its command.  A jammed actuator makes the rover leave
    # course too, but re-anchoring the pose would not fix the actuator, and
    # treating it as a pose fault would hide the real cause behind a generic
    # remedy.
    steering_fault = abs(features.deviation.get("steer_err_deg", 0.0)) > 0.25
    return (
        features.deviation.get("odom_drift_m", 0.0) > 0.15
        or features.deviation.get("gps_fix_quality", 0.0) > 0.15
        or (features.deviation.get("imu_yaw_rate", 0.0) > 0.25 and not steering_fault)
    )


def _project_reanchor(features: FeatureVector) -> Projected:
    # Re-anchoring drives the pose estimate back onto a trusted reference and
    # stops the rover compounding the error while it sorts itself out.
    values = dict(_v(features))
    values["odom_drift_m"] = 0.18
    values["gps_fix_quality"] = 1.0
    values["imu_yaw_rate"] = 1.0
    values["steer_err_deg"] = 0.0
    return Projected(values)


def _render_reanchor(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN reanchor_localisation_fusion() AND hold_station_keeping()"


def _cmd_reanchor(features: FeatureVector) -> Dict[str, Any]:
    return {"reanchor": True, "station_keep": True, "note": "Re-anchoring pose, holding position"}


def _applies_shelter(features: FeatureVector) -> bool:
    # Environmental degradation and motor overcurrent.  A motor winding drawing
    # far more than its design current is heating itself, so the safe move is to
    # stop driving and let it cool rather than shed the one load that would help.
    return (
        features.deviation.get("dust_opacity", 0.0) > 0.18
        or features.deviation.get("motor_current_a", 0.0) > 0.22
    )


def _project_shelter(features: FeatureVector) -> Projected:
    values = attenuate(
        _v(features),
        {
            "dust_opacity": 0.35,
            "motor_current_a": 0.20,
            "wheel_rpm_avg": 0.25,
            "imu_accel_x": 0.25,
            "imu_yaw_rate": 0.30,
        },
    )
    return Projected(values)


def _render_shelter(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN power_down_for_thermal_soak() AND enter_safe_shelter()"


def _cmd_shelter(features: FeatureVector) -> Dict[str, Any]:
    return {"thermal_soak": True, "shelter": True, "note": "Powered down for thermal soak"}


def _applies_relevel(features: FeatureVector) -> bool:
    # The chassis is leaning.  This is a load problem, not a traction or
    # perception one, so the remedy is to come to a stop and let the cargo
    # settle rather than change gearing or trust a different sensor.
    return abs(features.deviation.get("tilt_roll_deg", 0.0)) > 0.15


def _project_relevel(features: FeatureVector) -> Projected:
    # Stopping transfers the load back onto the suspension, which is what
    # actually brings the chassis back level.
    values = attenuate(
        _v(features),
        {
            "tilt_roll_deg": 0.20,
            "wheel_rpm_avg": 0.15,
            "imu_accel_x": 0.15,
            "imu_pitch_deg": 0.60,
        },
    )
    return Projected(values)


def _render_relevel(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN level_payload_and_hold() AND reset_suspension_load()"


def _cmd_relevel(features: FeatureVector) -> Dict[str, Any]:
    return {"relevel": True, "hold": True, "note": "Payload re-levelling, holding"}


def _applies_steer_reset(features: FeatureVector) -> bool:
    # A steering actuator that will not reach its commanded angle is an
    # actuator fault in its own right.  It is distinguished from a pose fault by
    # odometry and fix quality staying healthy: the rover knows where it is, it
    # simply cannot turn.
    return (
        abs(features.deviation.get("steer_err_deg", 0.0)) > 0.25
        and features.deviation.get("odom_drift_m", 0.0) < 0.15
        and features.deviation.get("gps_fix_quality", 0.0) < 0.15
    )


def _project_steer_reset(features: FeatureVector) -> Projected:
    # Cycling the actuator to its mechanical stops re-seats the encoder offset,
    # which is what makes the commanded angle trustworthy again.
    values = dict(_v(features))
    values["steer_err_deg"] = 0.0
    values = attenuate(values, {"imu_yaw_rate": 0.35, "wheel_rpm_avg": 0.70})
    return Projected(values)


def _render_steer_reset(conds: Sequence[Any], observed: Dict[str, float]) -> str:
    trigger = " AND ".join(c.to_string() for c in conds)
    return f"IF {trigger} THEN cycle_steer_actuator_to_stops() AND clear_steer_encoder_offset()"


def _cmd_steer_reset(features: FeatureVector) -> Dict[str, Any]:
    return {"steer_cycle": True, "steer_recal": True, "note": "Steering actuator cycled and re-zeroed"}


PRIMITIVES: Dict[str, ActionPrimitive] = {}

for _primitive in (
    ActionPrimitive(
        key="cap_motor_torque",
        title="Cap motor torque",
        summary="Wheels spin far faster than the chassis accelerates",
        consequence="Torque limited to 50%, traction model re-seated",
        cost=0.10,
        influences=("wheel_rpm_avg", "imu_accel_x"),
        applies=_applies_cap,
        project=_project_cap,
        render=_render_cap,
        command=_cmd_cap,
    ),
    ActionPrimitive(
        key="emergency_brake",
        title="Emergency brake",
        summary="Forward radar reports a dangerously close return",
        consequence="Motors braked, obstacle window cleared",
        cost=0.14,
        influences=("wheel_rpm_avg", "imu_accel_x", "ultrasonic_cm"),
        applies=_applies_brake,
        project=_project_brake,
        render=_render_brake,
        command=_cmd_brake,
    ),
    ActionPrimitive(
        key="reverse_and_recalculate_path",
        title="Reverse and recalculate",
        summary="Chassis acceleration is negative (impact or reversal)",
        consequence="Rover backs off and re-plans its route",
        cost=0.16,
        influences=("imu_accel_x", "wheel_rpm_avg"),
        applies=_applies_reverse,
        project=_project_reverse,
        render=_render_reverse,
        command=_cmd_reverse,
    ),
    ActionPrimitive(
        key="engage_max_torque_burst",
        title="Maximum torque burst",
        summary="Wheels and acceleration both collapsed with a clear path ahead",
        consequence="Full torque applied to break out of the stall",
        cost=0.22,
        influences=("wheel_rpm_avg", "imu_accel_x"),
        applies=_applies_burst,
        project=_project_burst,
        render=_render_burst,
        command=_cmd_burst,
    ),
    ActionPrimitive(
        key="enable_4x4_crawl_mode",
        title="4x4 crawl mode",
        summary="Chassis pitch has left its level envelope",
        consequence="Low-ratio gearing engaged to hold traction",
        cost=0.13,
        influences=("imu_pitch_deg", "wheel_rpm_avg"),
        applies=_applies_crawl,
        project=_project_crawl,
        render=_render_crawl,
        command=_cmd_crawl,
    ),
    ActionPrimitive(
        key="navigate_by_imu_only",
        title="Navigate by IMU only",
        summary="Radar channel is reporting an impossible value",
        consequence="Radar demoted, dead-reckoning navigation",
        cost=0.09,
        influences=("ultrasonic_cm",),
        applies=_applies_imu_only,
        project=_project_imu_only,
        render=_render_imu_only,
        command=_cmd_imu_only,
    ),
    ActionPrimitive(
        key="throttle_cognitive_hz",
        title="Throttle cognitive clock",
        summary="Compute thermal envelope exceeded",
        consequence="Cognitive frequency reduced, background load shed",
        cost=0.17,
        influences=("system_temp_c", "wheel_rpm_avg"),
        applies=_applies_throttle,
        project=_project_throttle,
        render=_render_throttle,
        command=_cmd_throttle,
    ),
    ActionPrimitive(
        key="trigger_hardware_safing_mode",
        title="Hardware safing mode",
        summary="Telemetry pipeline has been severed",
        consequence="Motion inhibited, safe pose held until the link returns",
        cost=0.05,
        influences=("wheel_rpm_avg", "imu_accel_x", "ultrasonic_cm", "imu_pitch_deg"),
        applies=_applies_safing,
        project=_project_safing,
        render=_render_safing,
        command=_cmd_safing,
    ),
    ActionPrimitive(
        key="recalibrate_radar_sweep",
        title="Recalibrate radar sweep",
        summary="Radar return is stuck at or near the floor",
        consequence="Fault latch cleared, sweep restarted",
        cost=0.11,
        influences=("ultrasonic_cm",),
        applies=_applies_recalibrate,
        project=_project_recalibrate,
        render=_render_recalibrate,
        command=_cmd_recalibrate,
    ),
    ActionPrimitive(
        key="hold_and_reacquire",
        title="Hold and reacquire",
        summary="Residual deviation is not explained by a specific fault",
        consequence="Coasted, awaiting stable telemetry",
        cost=0.20,
        influences=("wheel_rpm_avg", "imu_accel_x", "imu_pitch_deg"),
        applies=_applies_hold,
        project=_project_hold,
        render=_render_hold,
        command=_cmd_hold,
    ),
    ActionPrimitive(
        key="enter_energy_saver_mode",
        title="Energy saver mode",
        summary="Pack voltage has sagged below its operating band",
        consequence="Load shed, drive torque released, mission task aborted",
        cost=0.04,
        influences=("battery_v", "motor_current_a", "wheel_rpm_avg", "imu_accel_x"),
        applies=_applies_energy,
        project=_project_energy,
        render=_render_energy,
        command=_cmd_energy,
    ),
    ActionPrimitive(
        key="reanchor_localisation_fusion",
        title="Re-anchor pose fusion",
        summary="Wheel odometry has drifted or the satellite fix has degraded",
        consequence="Pose reset to last trusted reference, rover holds position",
        cost=0.12,
        influences=("odom_drift_m", "gps_fix_quality", "imu_yaw_rate", "steer_err_deg"),
        applies=_applies_reanchor,
        project=_project_reanchor,
        render=_render_reanchor,
        command=_cmd_reanchor,
    ),
    ActionPrimitive(
        key="power_down_for_thermal_soak",
        title="Power down for thermal soak",
        summary="Motor winding current or airborne dust exceeds its envelope",
        consequence="Drive electronics powered down, rover shelters in place",
        cost=0.02,
        influences=("motor_current_a", "dust_opacity", "wheel_rpm_avg", "imu_accel_x"),
        applies=_applies_shelter,
        project=_project_shelter,
        render=_render_shelter,
        command=_cmd_shelter,
    ),
    ActionPrimitive(
        key="level_payload_and_hold",
        title="Level payload and hold",
        summary="Chassis is leaning: the load has shifted off level",
        consequence="Rover stopped, suspension load reset to level the cargo",
        cost=0.03,
        influences=("tilt_roll_deg", "imu_pitch_deg", "wheel_rpm_avg", "imu_accel_x"),
        applies=_applies_relevel,
        project=_project_relevel,
        render=_render_relevel,
        command=_cmd_relevel,
    ),
    ActionPrimitive(
        key="cycle_steer_actuator_to_stops",
        title="Cycle steering actuator",
        summary="Steer actuator is not reaching its commanded angle",
        consequence="Actuator driven to its stops, encoder offset cleared",
        cost=0.08,
        influences=("steer_err_deg", "imu_yaw_rate", "wheel_rpm_avg"),
        applies=_applies_steer_reset,
        project=_project_steer_reset,
        render=_render_steer_reset,
        command=_cmd_steer_reset,
    ),
):
    PRIMITIVES[_primitive.key] = _primitive


@dataclass
class ScoredAction:
    """One candidate primitive, scored against the current anomaly."""

    primitive: ActionPrimitive
    score: float
    projected_residual: float
    current_residual: float
    feasible: bool
    rejected_reason: str = ""

    @property
    def key(self) -> str:
        return self.primitive.key

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.primitive.key,
            "title": self.primitive.title,
            "summary": self.primitive.summary,
            "consequence": self.primitive.consequence,
            "cost": round(self.primitive.cost, 3),
            "score": round(self.score, 3),
            "projected_residual": round(self.projected_residual, 3),
            "current_residual": round(self.current_residual, 3),
            "feasible": self.feasible,
            "rejected_reason": self.rejected_reason,
        }


INFEASIBLE_PENALTY = 10.0


def score_actions(
    features: FeatureVector,
    axes: Sequence[str],
    feasibility_margin: float = 0.35,
) -> List[ScoredAction]:
    """Rank every primitive for the current anomaly.

    The score is the residual the action is predicted to leave behind, plus a
    cost penalty.  Infeasible primitives are pushed to the bottom rather than
    removed, so the dashboard can show *why* they were rejected.
    """
    scored: List[ScoredAction] = []
    current_residual = features.residual(axes)

    for primitive in PRIMITIVES.values():
        try:
            feasible = bool(primitive.applies(features))
        except Exception:
            feasible = False

        if not feasible:
            scored.append(
                ScoredAction(
                    primitive=primitive,
                    score=current_residual + feasibility_margin + INFEASIBLE_PENALTY,
                    projected_residual=current_residual,
                    current_residual=current_residual,
                    feasible=False,
                    rejected_reason="does not fit this failure signature",
                )
            )
            continue

        try:
            projected = primitive.project(features)
            residual = projected.residual(axes)
        except Exception:
            scored.append(
                ScoredAction(
                    primitive=primitive,
                    score=current_residual + feasibility_margin + INFEASIBLE_PENALTY,
                    projected_residual=current_residual,
                    current_residual=current_residual,
                    feasible=False,
                    rejected_reason="forward model unavailable",
                )
            )
            continue

        scored.append(
            ScoredAction(
                primitive=primitive,
                score=residual + primitive.cost,
                projected_residual=residual,
                current_residual=current_residual,
                feasible=True,
            )
        )

    scored.sort(key=lambda s: (not s.feasible, s.score))
    return scored


def render_rule_text(
    primitive: ActionPrimitive,
    conditions: Sequence[Any],
    observed: Mapping[str, float],
) -> str:
    return primitive.render(conditions, dict(observed))