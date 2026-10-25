"""Central configuration for the Cognitive Evolution Engine.

Every tunable in the project lives here so the engine, the simulator, the
rover and the dashboard can never drift apart.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
WEB_DIR = PKG_DIR / "web"

STATE_DIR = Path(os.environ.get("CEE_STATE_DIR", str(PROJECT_DIR / ".cee_state")))
KNOWLEDGE_FILE = STATE_DIR / "knowledge.json"

# --- Networking -------------------------------------------------------------
HOST = os.environ.get("CEE_HOST", "127.0.0.1")
PORT = int(os.environ.get("CEE_PORT", "8801"))

BROKER = os.environ.get("CEE_BROKER", "broker.hivemq.com")
BROKER_PORT = int(os.environ.get("CEE_BROKER_PORT", "1883"))
SESSION_ID = os.environ.get("CEE_SESSION_ID", uuid.uuid4().hex[:8])
TOPIC_ROOT = os.environ.get("CEE_TOPIC_ROOT", "cee")
TOPIC_TELEMETRY = f"{TOPIC_ROOT}/{SESSION_ID}/telemetry"
TOPIC_EVENTS = f"{TOPIC_ROOT}/{SESSION_ID}/events"

# --- Cadence ----------------------------------------------------------------
FRAME_INTERVAL = float(os.environ.get("CEE_FRAME_INTERVAL", "0.5"))
RULE_VERIFY_WINDOW = 5          # frames to wait before judging an action
STREAM_TIMEOUT = 3.0            # seconds of silence before a stream fault
HISTORY_LEN = 240                # frames retained for dashboard charts
EVENT_LOG_LEN = 300

# --- Cognition --------------------------------------------------------------
RECALL_MIN_CONFIDENCE = 0.34    # below this a match is treated as "novel"
CONFIDENCE_START = 0.55
CONFIDENCE_FLOOR = 0.05
CONFIDENCE_CEIL = 0.98
CONFIDENCE_GAIN = 0.085         # added when an action measurably helps
CONFIDENCE_PENALTY = 0.115      # subtracted when it does not
CONFIDENCE_DECAY = 0.012        # slow forgetting between verifications
PREDICATE_MATCH_TOLERANCE = 0.06  # relative slack when re-matching predicates

# --- Telemetry axes ---------------------------------------------------------
# nominal: healthy operating band, used as the anchor point for deviation
# scale:   divisor that maps a value onto a dimensionless deviation
# range:   absolute bounds used by dashboard gauges
# warn/danger: presentation bands, also reported to the UI
AXES = (
    {
        "key": "wheel_rpm_avg",
        "label": "Wheel RPM",
        "short": "RPM",
        "unit": "rpm",
        "nominal": (85.0, 95.0),
        "scale": 90.0,
        "range": (0.0, 200.0),
        "warn": (0.0, 120.0),
        "danger": (0.0, 160.0),
    },
    {
        "key": "imu_accel_x",
        "label": "IMU Accel X",
        "short": "ACC",
        "unit": "m/s2",
        "nominal": (1.2, 1.5),
        "scale": 1.4,
        "range": (-3.0, 3.0),
        "warn": (0.4, 2.3),
        "danger": (0.05, 2.9),
    },
    {
        "key": "imu_pitch_deg",
        "label": "IMU Pitch",
        "short": "PITCH",
        "unit": "deg",
        "nominal": (-2.0, 2.0),
        "scale": 15.0,
        "range": (-40.0, 40.0),
        "warn": (15.0, 15.0),
        "danger": (25.0, 25.0),
    },
    {
        "key": "ultrasonic_cm",
        "label": "Front Radar",
        "short": "RADAR",
        "unit": "cm",
        "nominal": (50.0, 150.0),
        "scale": 100.0,
        "range": (0.0, 320.0),
        "warn": (25.0, 900.0),
        "danger": (10.0, 900.0),
        "invalid": (900.0, 1000.0),
    },
    {
        "key": "system_temp_c",
        "label": "CPU Thermal",
        "short": "TEMP",
        "unit": "C",
        "nominal": (35.0, 38.0),
        "scale": 45.0,
        "range": (25.0, 90.0),
        "warn": (75.0, 75.0),
        "danger": (82.0, 82.0),
    },
    # --- power ------------------------------------------------------------
    {
        "key": "battery_v",
        "label": "Pack Voltage",
        "short": "BATT",
        "unit": "V",
        "nominal": (12.2, 12.9),
        "scale": 1.2,
        "range": (9.0, 14.5),
        "warn": (11.0, 14.4),
        "danger": (10.0, 14.5),
    },
    # --- actuator fault ---------------------------------------------------
    {
        "key": "steer_err_deg",
        "label": "Steer Error",
        "short": "STEER",
        "unit": "deg",
        "nominal": (-1.6, 1.6),
        "scale": 4.0,
        "range": (-45.0, 45.0),
        "warn": (6.0, 6.0),
        "danger": (13.0, 13.0),
    },
    # --- attitude ---------------------------------------------------------
    {
        "key": "imu_yaw_rate",
        "label": "Yaw Rate",
        "short": "YAW",
        "unit": "dps",
        "nominal": (-1.8, 1.8),
        "scale": 5.0,
        "range": (-70.0, 70.0),
        "warn": (6.0, 6.0),
        "danger": (15.0, 15.0),
    },
    {
        "key": "tilt_roll_deg",
        "label": "Roll Tilt",
        "short": "ROLL",
        "unit": "deg",
        "nominal": (-1.5, 1.5),
        "scale": 4.0,
        "range": (-35.0, 35.0),
        "warn": (8.0, 8.0),
        "danger": (17.0, 17.0),
    },
    # --- localisation -----------------------------------------------------
    {
        "key": "odom_drift_m",
        "label": "Odo Drift",
        "short": "DRIFT",
        "unit": "m",
        "nominal": (0.0, 0.35),
        "scale": 0.6,
        "range": (0.0, 6.0),
        "warn": (1.0, 1.0),
        "danger": (2.6, 2.6),
    },
    {
        "key": "gps_fix_quality",
        "label": "GPS Quality",
        "short": "GPS",
        "unit": "hdop",
        "nominal": (0.6, 1.4),
        "scale": 1.6,
        "range": (0.2, 12.0),
        "warn": (3.0, 3.0),
        "danger": (6.0, 6.0),
    },
    # --- drive train ------------------------------------------------------
    {
        "key": "motor_current_a",
        "label": "Motor Current",
        "short": "CURR",
        "unit": "A",
        "nominal": (4.0, 5.8),
        "scale": 6.0,
        "range": (0.0, 30.0),
        "warn": (11.0, 11.0),
        "danger": (18.0, 18.0),
    },
    # --- environment ------------------------------------------------------
    {
        "key": "dust_opacity",
        "label": "Dust Opacity",
        "short": "DUST",
        "unit": "pct",
        "nominal": (0.04, 0.22),
        "scale": 0.5,
        "range": (0.0, 1.0),
        "warn": (0.55, 0.55),
        "danger": (0.82, 0.82),
    },
)

AXIS_KEYS = tuple(a["key"] for a in AXES)
AXIS_BY_KEY = {a["key"]: a for a in AXES}

SENSOR_INVALID_FLOOR = 900.0    # radar reports >= this when blinded

# The rover's drivetrain couples wheel speed to chassis acceleration.  A wheel
# spinning far faster than the acceleration it should produce is the physical
# signature of tyre slip.  The constant is chosen so that the centre of the
# healthy wheel-speed band maps onto the centre of the healthy acceleration
# band, which keeps the derived mismatch channel at zero during nominal driving.
DRIVETRAIN_ACCEL_PER_RPM = 0.015


def ensure_state_dir() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR