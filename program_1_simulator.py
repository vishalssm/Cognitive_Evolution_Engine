"""Program 1 — Full-spectrum hazard simulator.

A thin console launcher over the shared :mod:`cee.runtime` scenario generator.
It streams the same telemetry the dashboard renders, so you can watch the
engine learn without opening a browser.

    py -3 program_1_simulator.py                 # live, auto-hazards
    py -3 program_1_simulator.py --hazard slip   # repeat one fault continuously
    py -3 program_1_simulator.py --mqtt          # mirror frames to a broker
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cee import config  # noqa: E402
from cee.runtime import HAZARDS, CEERuntime  # noqa: E402


def _utf8_console() -> None:
    """Windows consoles still default to cp1252, which chokes on the box drawing."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _banner() -> None:
    transport = f"local bus (broker available: {config.BROKER})"
    print("=" * 78)
    print("  COGNITIVE EVOLUTION ENGINE  ·  PROGRAM 1  ·  HAZARD SIMULATOR")
    print("=" * 78)
    print(f"  scenarios  {len(HAZARDS)} fault modes")
    print(f"  rate       {1.0 / config.FRAME_INTERVAL:.0f} Hz")
    print(f"  transport  {transport}")
    print(f"  memory     {config.KNOWLEDGE_FILE}")
    print("-" * 78)
    for key, hazard in HAZARDS.items():
        print(f"  {key:<18} {hazard.label:<22} {hazard.blurb}")
    print("=" * 78)
    print("  Watching for Working 1 (induction) and Working 2 (recall).\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="CEE hazard simulator")
    parser.add_argument("--hazard", choices=sorted(HAZARDS), help="repeat this fault continuously")
    parser.add_argument("--mqtt", action="store_true", help="mirror frames to an MQTT broker")
    parser.add_argument("--quiet", action="store_true", help="print only cognitive events")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard persisted rules so Working 1 induction is visible from scratch",
    )
    args = parser.parse_args()

    _utf8_console()
    _banner()

    # Autoloading learned rules is the correct default for a running system, but
    # it hides Working 1 entirely on a demo: the graph is never empty, so
    # nothing looks like it was learned rather than remembered.
    runtime = CEERuntime(use_mqtt=args.mqtt, autoload_memory=not args.fresh)
    if args.fresh:
        runtime.kb.reset()

    try:
        runtime.start(produce=True)
        if args.hazard:
            runtime.control({"action": "inject", "hazard": args.hazard})
        else:
            runtime.control({"action": "auto", "value": True})

        shown_events = 0
        shown_frames = 0
        while True:
            time.sleep(0.4)
            state = runtime.state()
            counters = state["counters"]

            if not args.quiet and counters["frames"] > shown_frames:
                shown_frames = counters["frames"]
                action = (state["cycle"] or {}).get("action") or {}
                tail = ""
                if action:
                    tail = f"   [W{action['working']} {action['rule_id']} → {action['action_title']}]"
                print(
                    f"frame {counters['frames']:>5}  {state['status']['label']:<16}"
                    f"sev {state['telemetry']['severity']:.2f}"
                    f"  rpm {state['telemetry']['values']['wheel_rpm_avg']:>6.1f}"
                    f"  acc {state['telemetry']['values']['imu_accel_x']:>5.2f}"
                    f"  radar {state['telemetry']['values']['ultrasonic_cm']:>6.1f}"
                    f"  temp {state['telemetry']['values']['system_temp_c']:>5.1f}"
                    f"  rules {state['knowledge']['count']:>2}"
                    f"  recalls {counters['recalls']:>3}{tail}"
                )

            for event in state["events"][::-1]:
                if event["seq"] > shown_events:
                    shown_events = event["seq"]
                    print(f"    · [{event['phase']:<7}] {event['title']}")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Simulator shutdown.")
    finally:
        knowledge = runtime.kb.snapshot()
        runtime.control({"action": "save"})
        runtime.stop()
        print(
            f"[SYSTEM] {knowledge['count']} rule(s) retained "
            f"(mean confidence {knowledge['mean_confidence']:.2f}, "
            f"{(knowledge['stats'].get('recalls') or 0)} recall applications)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
