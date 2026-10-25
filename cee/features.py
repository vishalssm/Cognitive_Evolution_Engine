"""Telemetry frames and the feature layer that makes them comparable.

A raw telemetry frame is five loosely-scaled numbers.  Two frames can look
identical while meaning opposite things (0 rpm from a severed link vs. 0 rpm
from a stationary rover), so the engine never reasons over raw values.  It
reasons over a :class:`FeatureVector`: each axis re-expressed as a signed,
scale-normalised distance from its healthy operating band, plus a handful of
derived physical quantities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .config import (
    AXES,
    AXIS_BY_KEY,
    AXIS_KEYS,
    DRIVETRAIN_ACCEL_PER_RPM,
    SENSOR_INVALID_FLOOR,
)

# Signed deviation is clipped so one wild axis cannot silently dominate the
# whole reasoning step; the raw magnitude is kept alongside for display.
DEVIATION_CLIP = 4.0


def _round(value: float, digits: int = 3) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return round(value, digits)


def axis_deviation(spec: Mapping[str, Any], value: float) -> float:
    """Signed, scale-normalised distance of ``value`` from an axis' healthy band.

    Shared by the feature layer and by the action layer's forward model so a
    prediction of "what the telemetry would look like after this action" is
    scored on exactly the same scale as an observation.
    """
    invalid_band = spec.get("invalid")
    if invalid_band and value >= invalid_band[0]:
        return DEVIATION_CLIP
    lo, hi = spec["nominal"]
    if lo <= value <= hi:
        return 0.0
    anchor = (lo + hi) / 2.0
    scale = spec["scale"] or 1.0
    return max(-DEVIATION_CLIP, min(DEVIATION_CLIP, (value - anchor) / scale))


PSEUDO_AXIS_KEYS = ("kin_residual", "stream_ok", "severity")


def resolve_axis(features: "FeatureVector", axis: str) -> float:
    """Read a channel for predicate evaluation, including derived channels."""
    if axis == "kin_residual":
        return features.kin_residual
    if axis == "stream_ok":
        return 1.0 if features.stream_ok else 0.0
    if axis == "severity":
        return features.severity
    return features.values.get(axis, 0.0)


@dataclass
class Condition:
    """One atomic comparison inside an induced rule predicate.

    ``axis``   telemetry channel the comparison reads; may be a derived
               channel (``kin_residual``, ``stream_ok``)
    ``op``     one of ``<``, ``<=``, ``>``, ``>=``
    ``value``  threshold, synthesised as a midpoint between the healthy band
               and the observation that triggered induction
    ``role``   plain-language name for the physical failure this condition
               represents, used to humanise the generated rule
    """

    axis: str
    op: str
    value: float
    role: str

    def evaluate(self, features: "FeatureVector") -> bool:
        v = resolve_axis(features, self.axis)
        if self.op == "<":
            return v < self.value
        if self.op == "<=":
            return v <= self.value
        if self.op == ">":
            return v > self.value
        if self.op == ">=":
            return v >= self.value
        return True

    def holds_on(self, raw: Mapping[str, float]) -> bool:
        v = float(raw.get(self.axis, 0.0))
        if self.op == "<":
            return v < self.value
        if self.op == "<=":
            return v <= self.value
        if self.op == ">=":
            return v >= self.value
        if self.op == ">":
            return v > self.value
        return True

    def to_string(self) -> str:
        pretty = {
            "wheel_rpm_avg": "rpm",
            "imu_accel_x": "accel_x",
            "imu_pitch_deg": "pitch",
            "ultrasonic_cm": "radar_cm",
            "system_temp_c": "cpu_temp",
        }.get(self.axis, self.axis)
        symbol = {"<": "<", "<=": "<=", ">": ">", ">=": ">="}[self.op]
        return f"{pretty} {symbol} {round(self.value, 2)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis,
            "op": self.op,
            "value": _round(self.value, 2),
            "role": self.role,
            "text": self.to_string(),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "Condition":
        return Condition(
            axis=payload["axis"],
            op=payload["op"],
            value=float(payload["value"]),
            role=payload.get("role", ""),
        )


@dataclass
class FeatureVector:
    """Normalised view of one telemetry frame."""

    values: Dict[str, float] = field(default_factory=dict)
    deviation: Dict[str, float] = field(default_factory=dict)
    valid: Dict[str, bool] = field(default_factory=dict)
    masked: frozenset = field(default_factory=frozenset)
    stream_ok: bool = True
    radar_ok: bool = True
    severity: float = 0.0
    kin_residual: float = 0.0
    timestamp: float = 0.0
    dt: float = 0.0
    rank: int = 0

    # -- construction ------------------------------------------------------
    @staticmethod
    def from_frame(
        frame: Mapping[str, Any],
        prev: Optional["FeatureVector"] = None,
    ) -> "FeatureVector":
        values: Dict[str, float] = {}
        valid: Dict[str, bool] = {}
        deviation: Dict[str, float] = {}
        masked = frozenset(frame.get("masked") or ())

        for spec in AXES:
            key = spec["key"]
            raw = frame.get(key, None)
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = 0.0
            if not math.isfinite(v):
                v = 0.0
            values[key] = v

            invalid_band = spec.get("invalid")
            is_invalid = bool(invalid_band and v >= invalid_band[0])
            valid[key] = not is_invalid
            deviation[key] = axis_deviation(spec, v)

        stream_ok = frame.get("stream_ok")
        if stream_ok is None:
            # The legacy convention: a severed link emits an all-zero frame.
            stream_ok = not all(values[k] == 0.0 for k in AXIS_KEYS)
        stream_ok = bool(stream_ok)

        radar_ok = valid.get("ultrasonic_cm", True)

        if stream_ok:
            kin_residual = values["imu_accel_x"] - (
                values["wheel_rpm_avg"] * DRIVETRAIN_ACCEL_PER_RPM
            )
        else:
            kin_residual = 0.0

        fv = FeatureVector(
            values=values,
            deviation=deviation,
            valid=valid,
            masked=masked,
            stream_ok=stream_ok,
            radar_ok=radar_ok,
            kin_residual=kin_residual,
            timestamp=float(frame.get("timestamp", 0.0) or 0.0),
        )
        fv.severity = fv.compute_severity()
        if prev is not None:
            fv.dt = max(0.0, fv.timestamp - prev.timestamp) if fv.timestamp and prev.timestamp else 0.0
        return fv

    # -- reasoning helpers -------------------------------------------------
    def compute_severity(self) -> float:
        if not self.stream_ok:
            return DEVIATION_CLIP
        return max((abs(d) for k, d in self.deviation.items() if k not in self.masked),
                   default=0.0)

    def dominant_axes(self, limit: int = 3, floor: float = 0.12) -> List[str]:
        """Axes that have left their healthy band, worst first.

        Channels the engine has been told not to trust are excluded: an
        untrusted sensor is not evidence of a fault.
        """
        ranked = sorted(
            (
                (k, d)
                for k, d in self.deviation.items()
                if abs(d) >= floor and k not in self.masked
            ),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        return [k for k, _ in ranked[:limit]]

    def residual(self, axes: Iterable[str]) -> float:
        return sum(
            abs(self.deviation.get(a, 0.0))
            for a in axes
            if a not in self.masked
        )

    def raw_residual(self, axes: Iterable[str]) -> float:
        """Un-normalised distance from the healthy band, for human display."""
        total = 0.0
        for axis in axes:
            spec = AXIS_BY_KEY.get(axis)
            if not spec:
                continue
            lo, hi = spec["nominal"]
            v = self.values.get(axis, 0.0)
            if v < lo:
                total += lo - v
            elif v > hi:
                total += v - hi
        return total

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": {k: _round(v, 2) for k, v in self.values.items()},
            "deviation": {k: _round(v, 3) for k, v in self.deviation.items()},
            "valid": dict(self.valid),
            "masked": sorted(self.masked),
            "stream_ok": self.stream_ok,
            "radar_ok": self.radar_ok,
            "severity": _round(self.severity, 3),
            "kin_residual": _round(self.kin_residual, 3),
            "timestamp": self.timestamp,
        }

    def display(self) -> Dict[str, str]:
        out = {}
        for spec in AXES:
            key = spec["key"]
            out[key] = {
                "value": _round(self.values.get(key, 0.0), 2),
                "nominal": list(spec["nominal"]),
                "warn": list(spec["warn"]),
                "danger": list(spec["danger"]),
                "range": list(spec["range"]),
                "invalid": self.valid.get(key, True) or key in self.masked,
                "masked": key in self.masked,
                "deviation": _round(self.deviation.get(key, 0.0), 3),
            }
        return out


def band_status(spec: Mapping[str, Any], value: float, valid: bool = True) -> str:
    """Classify a value against an axis spec: ok / warn / danger / invalid."""
    if not valid:
        return "invalid"
    danger_lo, danger_hi = spec["danger"]
    if (danger_hi > danger_lo and (value < danger_lo or value > danger_hi)) or (
        danger_hi == danger_lo and value > danger_hi
    ):
        return "danger"
    warn_lo, warn_hi = spec["warn"]
    if (warn_hi > warn_lo and (value < warn_lo or value > warn_hi)) or (
        warn_hi == warn_lo and value > warn_hi
    ):
        return "warn"
    return "ok"