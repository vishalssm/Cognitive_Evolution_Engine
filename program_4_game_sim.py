"""Program 4 — Rover teleoperation game, instrumented by the CEE.

Drive a rover into zero-traction ice, high-resistance mud and a pitch crater and
watch the engine work: when the telemetry leaves its expected envelope the
brain induces a rule, and the moment you steer back into the same zone it
recalls that rule and intervenes on the drivetrain.  The engine's commands are
applied to this simulation, so its mitigations are physically felt here.

    py -3 program_4_game_sim.py
    py -3 program_4_game_sim.py --mqtt       # also publish frames to a broker

Controls: W throttle · S reverse/brake · A/D skid-steer · R reset position
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cee import config  # noqa: E402
from cee.runtime import CEERuntime  # noqa: E402

BROKER = "broker.hivemq.com"

# --- world ---------------------------------------------------------------
WORLD_W, WORLD_H = 800, 660
MAZE_TOP = 30

# --- physics -------------------------------------------------------------
# Telemetry calibration: the sensor scaling below is chosen so that cruising at
# the speed limit lands inside the engine's nominal operating bands
# (85-95 rpm, 1.2-1.5 m/s^2).  If the game's sensors disagree with the engine's
# model of a healthy rover, every frame reads as a fault and the engine never
# sees the difference between driving well and driving badly.
MAX_SPEED = 11.0
ACCELERATION = 0.5
BRAKING = 0.8
FRICTION_NORMAL = 0.95
TURN_SPEED = 4.0
START_X, START_Y = 350.0, 520.0
START_ANGLE = 270.0

TEMP_IDLE = 36.5         # compute temperature with the throttle closed
TEMP_UNDER_LOAD = 0.04   # warming per frame while the driver is on the throttle
TEMP_COOL_RATE = 0.35    # cooling per frame once the load is lifted
# The chassis sheds heat towards the idle temperature rather than running away,
# so holding the throttle settles at a steady operating temperature instead of
# climbing without bound until it is declared a thermal fault.
TEMP_COOL_RATIO = 0.22

MOTOR_RPM = 175.0        # wheel rpm the motor is driven to, grip permitting or not
RPM_PER_SPEED = 9.5      # wheel rpm per m/s of travel
ACCEL_PER_SPEED = 0.142  # chassis m/s^2 per m/s of travel
BURST_FORCE = 3.2        # torque the engine injects on the rover's behalf

# --- zones (x0, y0, x1, y1, key, label, colour) --------------------------
ZONES = (
    (50, 60, 250, 250, "ice", "HAZARD: ZERO TRACTION", "#38bdf8"),
    (450, 60, 650, 250, "mud", "HAZARD: HIGH RESISTANCE", "#fbbf24"),
    (60, 380, 240, 520, "crater", "HAZARD: EXTREME PITCH", "#a78bfa"),
)

BG = "#05070b"
PANEL = "#0b1017"
EDGE = "#1a2430"
TEXT = "#e6edf3"
MUTED = "#8b9aab"
DIM = "#5a6b7d"
CYAN = "#38bdf8"
GREEN = "#34d399"
AMBER = "#fbbf24"
RED = "#f87171"
VIOLET = "#a78bfa"

MONO = ("Consolas", 10)
MONO_B = ("Consolas", 10, "bold")


class RoverGame:
    def __init__(self, use_mqtt: bool, fresh: bool = False) -> None:
        self.runtime = CEERuntime(
            use_mqtt=use_mqtt, autoload_memory=not fresh, topic=None
        )
        if fresh:
            self.runtime.kb.reset()
        self.runtime.start(produce=False)  # this process owns the frame stream

        # world state
        self.x, self.y = START_X, START_Y
        self.velocity = 0.0
        self.angle = START_ANGLE
        self.cpu_temp = 36.5
        # Channels the CEE engine monitors beyond the original five.  The rover
        # has to publish all of them: a channel it never reports is read as a
        # silent zero, which the engine would rightly treat as a hard fault.
        self.pack_v = 12.7
        self.steer_err = 0.0
        self.yaw_rate = 0.0
        self.roll_tilt = 0.0
        self.odom_drift = 0.05
        self.gps_hdop = 0.95
        self.motor_amps = 4.8
        self.dust = 0.10
        self.keys = {"w": False, "a": False, "s": False, "d": False}

        # engine feedback (plain strings, painted to widgets by _paint_verdicts)
        self.last_rule: str | None = None
        self.last_phase = ""
        self.verdict_w1 = "no rule written yet"
        self.verdict_w2 = "no rule matched yet"
        self.event_cursor = 0
        self.zone = "nominal"

        self._build_ui()
        self._bind_keys()
        self.game_loop()

    # -- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("CEE · Rover Teleoperation · Instrumented by the Cognitive Engine")
        self.root.geometry(f"1360x760")
        self.root.configure(bg=BG)

        header = tk.Frame(self.root, bg=PANEL, highlightbackground=EDGE, highlightthickness=1)
        header.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(
            header,
            text="ROVER TELEOPERATION  //  COGNITIVE EVOLUTION ENGINE",
            font=("Consolas", 15, "bold"),
            fg=CYAN,
            bg=PANEL,
        ).pack(side="left", padx=14, pady=9)
        self.lbl_rec = tk.Label(header, text="● LIVE", font=("Consolas", 12, "bold"), fg=RED, bg=PANEL)
        self.lbl_rec.pack(side="right", padx=14, pady=9)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        cam = tk.Frame(body, bg=PANEL, highlightbackground=EDGE, highlightthickness=2)
        cam.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.canvas = tk.Canvas(cam, width=WORLD_W, height=WORLD_H, bg="#080b11", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._draw_static_world()
        self._create_dynamic()

        self.dash = tk.Frame(body, bg=PANEL, width=430, highlightbackground=EDGE, highlightthickness=2)
        self.dash.pack(side="right", fill="y", padx=(5, 0))
        self.dash.pack_propagate(False)
        self._build_dashboard()

        self.footer = tk.Label(
            self.root,
            text="W throttle   S reverse/brake   A/D skid-steer   R reset position",
            font=MONO,
            fg=DIM,
            bg=BG,
        )
        self.footer.pack(pady=(4, 8))

    def _draw_static_world(self) -> None:
        c = self.canvas
        for i in range(0, WORLD_W, 40):
            c.create_line(i, 0, i, WORLD_H, fill="#111a23", dash=(2, 5))
        for i in range(0, WORLD_H, 40):
            c.create_line(0, i, WORLD_W, i, fill="#111a23", dash=(2, 5))

        c.create_rectangle(0, 0, WORLD_W, MAZE_TOP, fill="#151515", outline=RED, dash=(4, 4))
        c.create_text(WORLD_W / 2, MAZE_TOP / 2, text="[ STRUCTURAL BARRIER ]", fill=RED, font=MONO)

        for x0, y0, x1, y1, _key, label, colour in ZONES:
            c.create_rectangle(x0, y0, x1, y1, outline=colour, dash=(3, 3), width=2)
            c.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text=label,
                fill=colour,
                font=MONO_B,
            )
        # zone keys mirror the dashboard hazard catalogue
        self.zone_items = {}
        for x0, y0, x1, y1, key, label, colour in ZONES:
            tag = c.create_rectangle(x0, y0, x1, y1, outline="", fill=colour)
            self.zone_items[tag] = key

    def _create_dynamic(self) -> None:
        c = self.canvas
        self.c_ring = c.create_oval(0, 0, 0, 0, outline="#1e2a36", width=1)
        self.c_t = c.create_line(0, 0, 0, 0, fill=CYAN, width=2)
        self.c_b = c.create_line(0, 0, 0, 0, fill=CYAN, width=2)
        self.c_l = c.create_line(0, 0, 0, 0, fill=CYAN, width=2)
        self.c_r = c.create_line(0, 0, 0, 0, fill=CYAN, width=2)
        self.c_radar = c.create_line(0, 0, 0, 0, fill=RED, dash=(3, 3))
        self.c_rover = c.create_polygon(0, 0, 0, 0, 0, 0, fill=GREEN, outline="#ffffff", width=2)
        self.c_intervene = c.create_oval(0, 0, 0, 0, outline="", fill="")
        self.c_trail = c.create_line(0, 0, 0, 0, fill="#1d3a4d", width=2)
        self.trail: list[float] = []

    def _bar(self, parent, label: str, colour: str):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=18, pady=(9, 0))
        head = tk.Frame(row, bg=PANEL)
        head.pack(fill="x")
        tk.Label(head, text=label, font=MONO, fg=DIM, bg=PANEL).pack(side="left")
        val = tk.Label(head, text="0.0", font=MONO_B, fg=TEXT, bg=PANEL)
        val.pack(side="right")
        strip = tk.Canvas(row, height=9, bg="#05070b", highlightthickness=0, bd=0)
        strip.pack(fill="x", pady=(4, 0))
        bar = strip.create_rectangle(0, 0, 0, 9, fill=colour)
        return strip, bar, val

    def _section(self, parent, title: str) -> tk.Frame:
        wrap = tk.Frame(parent, bg=PANEL)
        wrap.pack(fill="x", padx=18, pady=(16, 0))
        tk.Label(wrap, text=title, font=MONO_B, fg=DIM, bg=PANEL).pack(anchor="w")
        line = tk.Frame(wrap, bg=EDGE, height=1)
        line.pack(fill="x", pady=(5, 0))
        inner = tk.Frame(wrap, bg=PANEL)
        inner.pack(fill="both", expand=True, pady=(7, 0))
        return inner

    def _build_dashboard(self) -> None:
        tk.Label(
            self.dash,
            text="ENGINE · LIVE FEED",
            font=("Consolas", 13, "bold"),
            fg=TEXT,
            bg=PANEL,
        ).pack(pady=(16, 2))

        self.lbl_phase = tk.Label(self.dash, text="STANDBY", font=("Consolas", 16, "bold"), fg=MUTED, bg=PANEL)
        self.lbl_phase.pack(pady=(2, 10))

        self.b_rpm = self._bar(self.dash, "THROTTLE (RPM)", CYAN)
        self.b_acc = self._bar(self.dash, "CHASSIS ACCEL (M/S²)", VIOLET)
        self.b_pitch = self._bar(self.dash, "INCLINE (PITCH °)", AMBER)
        self.b_radar = self._bar(self.dash, "FORWARD RADAR (CM)", GREEN)
        self.b_temp = self._bar(self.dash, "CPU THERMALS (°C)", RED)

        # Working 1
        self.w1 = self._section(self.dash, "WORKING 1 · INDUCTION")
        self.lbl_w1 = tk.Label(
            self.w1, text="no rule written yet", font=MONO, fg=MUTED, bg=PANEL, wraplength=390, justify="left"
        )
        self.lbl_w1.pack(anchor="w")

        # Working 2
        self.w2 = self._section(self.dash, "WORKING 2 · RECALL")
        self.lbl_w2 = tk.Label(
            self.w2, text="no rule matched yet", font=MONO, fg=MUTED, bg=PANEL, wraplength=390, justify="left"
        )
        self.lbl_w2.pack(anchor="w")

        # Knowledge ledger
        self.kb = self._section(self.dash, "OPERATIONAL KNOWLEDGE GRAPH")
        self.lbl_kb = tk.Label(
            self.kb, text="empty", font=MONO, fg=DIM, bg=PANEL, wraplength=390, justify="left"
        )
        self.lbl_kb.pack(anchor="w")

        self.lbl_sys = tk.Label(
            self.dash,
            text=f"SYS: TRANSMITTING · {config.TOPIC_TELEMETRY}",
            font=MONO,
            fg=GREEN,
            bg=PANEL,
            wraplength=390,
            justify="left",
        )
        self.lbl_sys.pack(side="bottom", pady=16)

    # -- input -------------------------------------------------------------
    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress>", self._press)
        self.root.bind("<KeyRelease>", self._release)
        self.root.focus_force()

    def _press(self, event) -> None:
        key = event.keysym.lower()
        if key in self.keys:
            self.keys[key] = True
        elif key == "r":
            self.x, self.y, self.velocity, self.angle = START_X, START_Y, 0.0, START_ANGLE
            self.trail.clear()
            # The onboard channels return with the rover, so a restart is not
            # silently reporting stale power and pose state.
            self.pack_v, self.steer_err, self.yaw_rate = 12.7, 0.0, 0.0
            self.roll_tilt, self.odom_drift = 0.0, 0.05
            self.gps_hdop, self.motor_amps, self.dust = 0.95, 4.8, 0.10

    def _release(self, event) -> None:
        key = event.keysym.lower()
        if key in self.keys:
            self.keys[key] = False

    # -- thermal -----------------------------------------------------------
    def _heat(self, amount: float) -> None:
        """Warm the compute toward the temperature this load actually sustains.

        A real chassis reaches an equilibrium rather than climbing forever: the
        radiator sheds heat as fast as the load adds it.  Without that ceiling a
        sustained climb would eventually be *correctly* flagged as a thermal
        fault, but it would be an artefact of the game rather than a hazard the
        operator caused.
        """
        ceiling = TEMP_IDLE + amount / TEMP_COOL_RATIO
        self.cpu_temp = min(ceiling, self.cpu_temp + amount)

    def _cool(self, rate: float = 1.0) -> None:
        self.cpu_temp = max(TEMP_IDLE, self.cpu_temp - TEMP_COOL_RATE * rate)

    # -- physics -----------------------------------------------------------
    def _zone_effects(self) -> tuple[float, float, float, str]:
        """Return (friction, traction, pitch, zone key) for the current position.

        Engine interventions are folded in here, which is what makes the brain's
        remedies physically felt: a torque cap really does curtail wheelspin in
        this simulation, a crawl mode really does trade speed for grip, and a
        brake really does arrest the chassis.
        """
        friction, traction, pitch, zone = FRICTION_NORMAL, 1.0, 0.0, "nominal"
        for x0, y0, x1, y1, key, _label, _colour in ZONES:
            if x0 < self.x < x1 and y0 < self.y < y1:
                if key == "ice":
                    friction, traction = 0.99, 0.1
                elif key == "mud":
                    friction, traction = 0.80, 0.4
                elif key == "crater":
                    traction, pitch = 0.5, 22.0
                zone = key
                break

        active = {i.key: i for i in self.runtime.actuators.active}

        # Cap motor torque: the driver still commands throttle, but the wheels are
        # held to the speed the ground can actually take, so the chassis stops
        # outrunning them and grip returns.
        cap = active.get("cap_motor_torque")
        if cap is not None:
            grip = max(0.15, cap.factor("wheel_rpm_avg"))
            traction = min(1.0, traction + (1.0 - traction) * grip)

        # Max torque burst: deliberately over-torque to break out of a stall.
        burst = active.get("engage_max_torque_burst")
        if burst is not None:
            traction = max(traction, 0.85)
            friction = min(0.95, friction + 0.15)

        # 4x4 crawl: trade top speed for sure traction.
        crawl = active.get("enable_4x4_crawl_mode")
        if crawl is not None:
            traction = max(traction, 0.7)
            pitch = pitch * 0.4

        return friction, traction, pitch, zone

    def _physics(self) -> dict:
        friction, traction, pitch, zone = self._zone_effects()
        active = {i.key: i for i in self.runtime.actuators.active}

        force = 0.0
        if self.keys["w"]:
            force = ACCELERATION
            self._heat(TEMP_UNDER_LOAD)
        elif self.keys["s"]:
            force = -BRAKING

        # Backing away is the rover's own gear, not an intervention, so it needs
        # no entry here -- but the engine's path replan command has to be felt or
        # a recalled rule can only ever be demoted for a reason that is really
        # the simulator's.
        if "reverse_and_recalculate_path" in active and force >= 0.0:
            force = -BRAKING * 0.9
        elif "hold_and_reacquire" in active and not self.keys["w"]:
            # Coasting: the rover is held rather than driven, so the throttle
            # does nothing and only rolling drag applies.
            force = 0.0
            friction = 0.90

        # A torque burst is the engine supplying the torque the driver is not,
        # so it drives a stalled rover even when nobody is on the throttle --
        # otherwise the rule that requests it would be demoted forever for a
        # reason that is really the simulator's.  It is a *bounded* intervention,
        # so it raises the force the drivetrain can deliver but cannot exceed
        # the same ceiling the driver gets: without the cap a burst compounds
        # every frame until wheel speed runs away into the hundreds.
        burst = active.get("engage_max_torque_burst")
        if burst is not None:
            force = min(
                force + BURST_FORCE * burst.factor("wheel_rpm_avg"), BURST_FORCE
            )

        # An emergency brake or hardware safing command overrides the driver.
        if "emergency_brake" in active or "trigger_hardware_safing_mode" in active:
            force = -BRAKING * 1.6
            self._cool(0.6)

        # Traction control and a 4x4 gear split both cut how much torque reaches
        # the wheels, so they throttle the rover back as well as lifting the
        # slip itself.
        for limiter in ("cap_motor_torque", "enable_4x4_crawl_mode"):
            limited = active.get(limiter)
            if limited is not None:
                force *= max(0.15, limited.factor("wheel_rpm_avg"))

        # Backing off the throttle gives the onboard computer headroom, which is
        # why it cools only when the driver has stopped asking for power.
        if not self.keys["w"] and "throttle_cognitive_hz" in active:
            self._cool(1.4)

        if not self.keys["w"]:
            self._cool()

        if self.keys["a"]:
            self.angle -= TURN_SPEED
        if self.keys["d"]:
            self.angle += TURN_SPEED

        self.velocity = max(
            -MAX_SPEED, min(MAX_SPEED, (self.velocity + force * traction) * friction)
        )

        rad = math.radians(self.angle)
        self.x += self.velocity * math.cos(rad)
        self.y += self.velocity * math.sin(rad)

        # Wheels turn at whatever speed the motor is driving them, which is not
        # the speed the ground allows when traction is poor -- that difference
        # *is* the wheelspin the engine detects as a traction mismatch.
        ground_rpm = abs(self.velocity * RPM_PER_SPEED)
        rpm = ground_rpm
        if force > 0 and traction < 0.5:
            rpm = max(ground_rpm, MOTOR_RPM)

        # Traction control can only remove the excess spin, never add speed the
        # ground was already permitting.  Cancelling it out against the ground
        # value keeps the wheel channel and the chassis channel consistent, so
        # the remedy shows up as reduced slip rather than as a stall.
        cap = active.get("cap_motor_torque")
        if cap is not None and rpm > ground_rpm:
            kept = cap.factor("wheel_rpm_avg")
            rpm = ground_rpm + (rpm - ground_rpm) * kept

        radar = 150.0
        if math.sin(rad) < 0:
            radar = min(150.0, abs(self.y - MAZE_TOP) / max(0.001, abs(math.sin(rad))))

        # Radar blinded / recalibrated: the head is covered and reports the sentinel.
        if "recalibrate_radar_sweep" in active or "navigate_by_imu_only" in active:
            radar = 999.0

        if self.y < MAZE_TOP + 10:
            self.y, self.velocity, rpm = MAZE_TOP + 10, -self.velocity * 0.5, 5.0

        self.x = max(20, min(WORLD_W - 20, self.x))
        self.y = max(MAZE_TOP + 10, min(WORLD_H - 20, self.y))
        self.zone = zone

        # --- the remaining monitored channels -----------------------------
        # Each is a real consequence of how the rover is being driven, so the
        # engine's rules are reacting to the driving rather than to noise.
        turning = TURN_SPEED if (self.keys["a"] or self.keys["d"]) else 0.0
        # Steering lag and yaw rate are modelled as small tracking errors against
        # the commanded turn.  They stay inside their healthy bands while the
        # actuator is working; it is only when something is actually wrong with
        # the steering gear that the error opens out.
        self.yaw_rate = self.yaw_rate * 0.6 + turning * 0.16 + (
            self.velocity * math.sin(rad) * 0.010
        )
        if abs(self.yaw_rate) < 0.05:
            self.yaw_rate = 0.0

        # Steering lag: proportional to how hard the rover is turning.
        self.steer_err = self.steer_err * 0.7 + turning * 0.05

        # A heavily loaded chassis leans, and the worse the traction the more it
        # does -- so driving into sand both bogs the drive and tips the body.
        self.roll_tilt = self.roll_tilt * 0.75 + (1.0 - traction) * 1.0 * max(
            0.0, abs(self.velocity) / max(0.001, MAX_SPEED)
        )

        # Odometer error accumulates with distance travelled, faster off-road.
        # Real wheel odometry does not drift without bound: it wanders within a
        # small band set by wheel slip and uneven ground.  It only leaves that
        # band when something is genuinely wrong with an encoder, which is
        # exactly the distinction the engine's drift rule exists to catch.
        self.odom_drift += abs(self.velocity) * 0.0022 * (2.0 - traction)
        self.odom_drift -= (self.odom_drift - 0.15) * 0.14
        self.odom_drift = max(0.0, self.odom_drift)

        # Pack drains while driving and recovers slowly at rest.
        if abs(self.velocity) > 0.2:
            self.pack_v = max(10.6, self.pack_v - 0.0022)
        else:
            self.pack_v = min(13.0, self.pack_v + 0.0015)

        # Motor current follows the torque actually being asked for.  A rover at
        # a steady cruise sits mid-band; only wheelspin or a hard stall pushes
        # the winding past its design figure.
        self.motor_amps = 4.4 + abs(force) * 0.55 + max(0.0, rpm - ground_rpm) * 0.02

        # Dust hangs in the air once the rover has been working, and settles
        # back towards the ambient level rather than to nothing.
        self.dust += (0.12 - self.dust) * 0.06
        self.dust += 0.004 if abs(force) > 1.2 else 0.0
        self.dust = min(1.0, max(0.02, self.dust))

        # A degraded fix or a re-anchored pose is reported by the intervention
        # itself, exactly as a real rover would mark the channel unreliable.
        if "reanchor_localisation_fusion" in active:
            self.odom_drift, self.gps_hdop, self.steer_err = 0.18, 1.0, 0.0
        if "cycle_steer_actuator_to_stops" in active:
            self.steer_err = 0.0

        return {
            "wheel_rpm_avg": round(rpm, 2),
            "imu_accel_x": round(self.velocity * ACCEL_PER_SPEED, 2),
            "imu_pitch_deg": round(pitch, 2),
            "ultrasonic_cm": round(radar, 2),
            "system_temp_c": round(self.cpu_temp, 1),
            "battery_v": round(self.pack_v, 2),
            "steer_err_deg": round(self.steer_err, 2),
            "imu_yaw_rate": round(self.yaw_rate, 2),
            "tilt_roll_deg": round(self.roll_tilt, 2),
            "odom_drift_m": round(self.odom_drift, 2),
            "gps_fix_quality": round(self.gps_hdop, 2),
            "motor_current_a": round(self.motor_amps, 2),
            "dust_opacity": round(self.dust, 3),
        }

    # -- engine bridge -----------------------------------------------------
    def _feed_engine(self, payload: dict) -> None:
        frame = {
            "timestamp": round(time.time(), 3),
            "frame_seq": self._frame_seq,
            "stream_ok": True,
        }
        frame.update(payload)
        self._frame_seq += 1
        self.runtime.publish_external(frame)
        self._drain_events()

    _frame_seq = 0

    def _drain_events(self) -> None:
        for event in self.runtime.state()["events"][::-1]:
            if event["seq"] <= self.event_cursor:
                continue
            self.event_cursor = event["seq"]
            self.last_phase = event["phase"]
            self.last_rule = event["title"]

            if event["phase"] == "induce" and self.runtime.engine.last_induction:
                rule = self.runtime.engine.last_induction["rule"]
                self.verdict_w1 = f"{rule['id']}  conf {rule['confidence']:.2f}\n{rule['text']}"
            elif event["phase"] == "recall" and self.runtime.engine.last_recall:
                rec = self.runtime.engine.last_recall
                self.verdict_w2 = (
                    f"{rec['chosen']['rule_id']} matched "
                    f"{rec['chosen']['satisfied']}/{rec['chosen']['of']} "
                    f"(conf {rec['chosen']['confidence']:.2f})\n"
                    f"{rec['primitive']['title']}"
                )

    def _sync_dashboard(self) -> None:
        state = self.runtime.state()
        telemetry = state["telemetry"]
        values = telemetry["values"]

        for (strip, bar, val), key, scale, ceiling in (
            (self.b_rpm, "wheel_rpm_avg", 1.0, 180.0),
            (self.b_acc, "imu_accel_x", 4.0, 2.0),
            (self.b_pitch, "imu_pitch_deg", 1.0, 30.0),
            (self.b_radar, "ultrasonic_cm", 1.0, 150.0),
            (self.b_temp, "system_temp_c", 1.0, 90.0),
        ):
            raw = values[key]
            fraction = min(1.0, abs(raw) / ceiling)
            width = max(0.0, min(400.0, fraction * 400.0))
            strip.coords(bar, 0, 0, width, 9)
            decimals = 2 if ceiling <= 10 else 1
            val.configure(text=f"{raw:.{decimals}f}")

        status = state["status"]
        colour = {
            "nominal": GREEN,
            "acting": AMBER,
            "anomalous": AMBER,
            "critical": RED,
            "idle": MUTED,
        }.get(status["level"], MUTED)
        self.lbl_phase.configure(text=status["label"], fg=colour)

        knowledge = state["knowledge"]
        if knowledge["count"]:
            top = knowledge["rules"][0]
            self.lbl_kb.configure(
                text=(
                    f"{knowledge['count']} rule(s) · mean conf "
                    f"{knowledge['mean_confidence']:.2f}\n"
                    f"{top['id']}  {top['title']}  [{top['confidence']:.2f}]"
                ),
                fg=CYAN,
            )
        else:
            self.lbl_kb.configure(text="empty — steer into a hazard zone", fg=DIM)

        self.lbl_w1.configure(text=self.verdict_w1, fg=AMBER if self.last_phase == "induce" else MUTED)
        self.lbl_w2.configure(text=self.verdict_w2, fg=GREEN if self.last_phase == "recall" else MUTED)

        active = state["actuators"]["active"]
        if active:
            names = ", ".join(i["label"] for i in active)
            self.lbl_sys.configure(text=f"SYS: INTERVENING · {names}", fg=AMBER)
        else:
            self.lbl_sys.configure(
                text=f"SYS: TRANSMITTING · {state['transport']['mode']}", fg=GREEN
            )

    # -- render ------------------------------------------------------------
    def _draw_rover(self) -> None:
        rad = math.radians(self.angle)
        nose = (self.x + 20 * math.cos(rad), self.y + 20 * math.sin(rad))
        left = (self.x + 15 * math.cos(rad + 2.5), self.y + 15 * math.sin(rad + 2.5))
        right = (self.x + 15 * math.cos(rad - 2.5), self.y + 15 * math.sin(rad - 2.5))

        self.canvas.coords(self.c_rover, *nose, *left, *right)
        self.canvas.coords(self.c_radar, *nose, nose[0] + math.cos(rad) * 150, nose[1] + math.sin(rad) * 150)
        self.canvas.coords(self.c_ring, self.x - 40, self.y - 40, self.x + 40, self.y + 40)
        self.canvas.coords(self.c_t, self.x, self.y - 50, self.x, self.y - 35)
        self.canvas.coords(self.c_b, self.x, self.y + 35, self.x, self.y + 50)
        self.canvas.coords(self.c_l, self.x - 50, self.y, self.x - 35, self.y)
        self.canvas.coords(self.c_r, self.x + 35, self.y, self.x + 50, self.y)

        # Pulse ring while an engine intervention is in force.
        active = self.runtime.actuators.active
        if active:
            phase = int(time.time() * 4) % 2
            r = 46 + phase * 8
            self.canvas.coords(self.c_intervene, self.x - r, self.y - r, self.x + r, self.y + r)
            self.canvas.itemconfigure(self.c_intervene, outline=AMBER, fill="")
        else:
            self.canvas.itemconfigure(self.c_intervene, outline="")

    def _highlight_zone(self) -> None:
        for tag, key in self.zone_items.items():
            self.canvas.itemconfigure(tag, outline="" if key == self.zone else "", fill="")

    # -- main loop ---------------------------------------------------------
    def game_loop(self) -> None:
        payload = self._physics()
        self._feed_engine(payload)
        self._draw_rover()
        self._sync_dashboard()

        self.trail.append(self.x)
        self.trail.append(self.y)
        if len(self.trail) > 120:
            del self.trail[:2]
        if len(self.trail) >= 4:
            self.canvas.coords(self.c_trail, *self.trail)
            self.canvas.itemconfigure(self.c_trail, state="normal")
        else:
            self.canvas.itemconfigure(self.c_trail, state="hidden")

        self.root.after(40, self.game_loop)

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self.runtime.control({"action": "save"})
            self.runtime.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="CEE instrumented rover game")
    parser.add_argument("--mqtt", action="store_true", help=f"mirror frames to {BROKER}")
    args = parser.parse_args()

    try:
        for stream in (sys.stdout, sys.stderr):
            stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    RoverGame(use_mqtt=args.mqtt).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
