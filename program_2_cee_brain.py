"""Program 2 — Headless cognitive brain.

Runs the reflective reasoning engine with no simulator attached.  It subscribes
to the telemetry bus and narrates its own reasoning, which is the clearest way to
read what the two workings are doing.

    py -3 program_2_cee_brain.py                 # synthesise its own frames
    py -3 program_2_cee_brain.py --mqtt          # consume a broker topic
    py -3 program_2_cee_brain.py --hazard slip   # one fault, repeatedly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cee import config  # noqa: E402
from cee.runtime import HAZARDS, CEERuntime  # noqa: E402

BANNER = r"""
==============================================================================
  COGNITIVE EVOLUTION ENGINE  ·  PROGRAM 2  ·  HEADLESS BRAIN
==============================================================================
  The reasoning engine with no dashboard attached.  Watch it narrate its own
  induction, recall and verification decisions.

  Working 1  induction   nothing in memory explains this -> write a rule
  Working 2  recall      a stored predicate matches      -> reuse the rule
  verify                 reality vs the rule's own prediction -> reinforce
==============================================================================
"""


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="CEE headless brain")
    parser.add_argument("--mqtt", action="store_true", help="consume frames from a broker")
    parser.add_argument("--hazard", choices=sorted(HAZARDS), help="repeat this fault continuously")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard persisted rules so Working 1 induction is visible from scratch",
    )
    args = parser.parse_args()

    _utf8_console()
    print(BANNER)
    print(f"  memory      {config.KNOWLEDGE_FILE}")
    print(f"  transport   {'MQTT ' + config.BROKER if args.mqtt else 'local bus'}")
    print(f"  topic       {config.TOPIC_TELEMETRY}\n")

    runtime = CEERuntime(use_mqtt=args.mqtt, autoload_memory=not args.fresh)
    if args.fresh:
        runtime.kb.reset()

    try:
        if args.mqtt:
            # Consume only: an external publisher owns the frame stream.
            runtime.start(produce=False)
            runtime.bus.connect_mqtt()
            print("[SYSTEM] Listening. Waiting for telemetry on the broker…\n")
            _idle_while_mqtt(runtime)
        else:
            runtime.start(produce=True)
            if args.hazard:
                runtime.control({"action": "inject", "hazard": args.hazard})
            else:
                runtime.control({"action": "auto", "value": True})
            _narrate(runtime)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Brain shutdown.")
    finally:
        runtime.control({"action": "save"})
        knowledge = runtime.kb.snapshot()
        runtime.stop()
        stats = knowledge.get("stats", {})
        print(
            f"\n[SYSTEM] {knowledge['count']} rule(s) · mean confidence "
            f"{knowledge['mean_confidence']:.2f} · {stats.get('inductions', 0)} induced · "
            f"{stats.get('recalls', 0)} recalled · {knowledge['helped']} helped"
        )
        for rule in knowledge["rules"]:
            judged = rule["times_helped"] + rule["times_failed"]
            print(f"    {rule['id']:<9} conf {rule['confidence']:.2f}  applied {rule['times_applied']:<3}"
                  f" {rule['times_helped']}/{judged if judged else 0} helped  {rule['title']}")
    return 0


def _narrate(runtime: CEERuntime) -> None:
    """Print the engine's reasoning as it happens."""
    seen = 0
    import time

    while True:
        time.sleep(0.35)
        for event in runtime.state()["events"][::-1]:
            if event["seq"] <= seen:
                continue
            seen = event["seq"]
            bar = {"induce": "-", "recall": "-", "verify": "-", "warn": "!"}.get(event["phase"], " ")
            print(f"  {bar} [{event['phase']:<7}] {event['title']}")
            if event.get("detail"):
                print(f"           {event['detail']}\n")


def _idle_while_mqtt(runtime: CEERuntime) -> None:
    import time

    seen = 0
    while True:
        time.sleep(0.5)
        for event in runtime.state()["events"][::-1]:
            if event["seq"] <= seen:
                continue
            seen = event["seq"]
            print(f"  [{event['phase']:<7}] {event['title']}")
            if event.get("detail"):
                print(f"           {event['detail']}\n")


if __name__ == "__main__":
    raise SystemExit(main())
