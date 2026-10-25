"""Program 3 — Live operations console.

Serves the web dashboard and runs the engine behind it.  This is the whole
demo: open the printed URL and inject a fault twice.

    py -3 program_3_dashboard.py                 # http://127.0.0.1:8801/
    py -3 program_3_dashboard.py --port 9000
    py -3 program_3_dashboard.py --hazard slip   # auto-repeat one fault
    py -3 program_3_dashboard.py --mqtt --external   # dashboard only
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cee import config  # noqa: E402
from cee.runtime import HAZARDS, CEERuntime  # noqa: E402
from cee.server import DashboardServer  # noqa: E402

BANNER = r"""
==============================================================================
  COGNITIVE EVOLUTION ENGINE  ·  PROGRAM 3  ·  OPERATIONS CONSOLE
==============================================================================
  Working 1 (induction)  a fault with no known explanation becomes a new rule
  Working 2 (recall)     the same fault later reuses that rule without relearning

  Try this: click one hazard twice.  The first click fills the Working 1 panel
  with the symbol it invented; the second fills the Working 2 panel with the
  rule it recalled.
==============================================================================
"""


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="CEE dashboard")
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--hazard", choices=sorted(HAZARDS), help="auto-repeat this fault")
    parser.add_argument("--mqtt", action="store_true", help="use a broker topic")
    parser.add_argument("--external", action="store_true", help="do not synthesise frames")
    parser.add_argument("--open", action="store_true", help="launch the default browser")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard persisted rules so Working 1 induction is visible from scratch",
    )
    args = parser.parse_args()

    _utf8_console()
    print(BANNER)

    # Loading saved rules is what a deployed system should do, but for a demo it
    # hides Working 1: the graph starts non-empty, so nothing on screen looks
    # invented rather than remembered.
    runtime = CEERuntime(use_mqtt=args.mqtt, autoload_memory=not args.fresh)
    if args.fresh:
        runtime.kb.reset()
        print("  knowledge graph reset; induction starts from an empty graph")
    server = DashboardServer(runtime, host=args.host, port=args.port)

    try:
        server.start()
        produce = not args.external
        runtime.start(produce=produce)
        if args.mqtt and produce:
            runtime.bus.connect_mqtt()

        if args.hazard and produce:
            runtime.control({"action": "inject", "hazard": args.hazard})
        elif produce:
            runtime.control({"action": "auto", "value": False})

        print(f"  dashboard   {server.url}")
        print(f"  telemetry   {'broker ' + config.BROKER if args.mqtt else 'in-process bus'}")
        print(f"  topic       {config.TOPIC_TELEMETRY}")
        print(f"  memory      {config.KNOWLEDGE_FILE}")
        print(f"  scenarios   {', '.join(sorted(HAZARDS))}")
        print("\n  Ctrl+C to stop.\n")

        if args.open:
            threading.Timer(0.8, lambda: webbrowser.open(server.url)).start()

        _status_line(runtime)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Dashboard shutdown.")
    except OSError as exc:
        print(f"\n[SYSTEM] Cannot bind {args.host}:{args.port} — {exc}")
        print("         Another instance is probably running. Try --port 8802.")
        return 1
    finally:
        server.stop()
        runtime.control({"action": "save"})
        runtime.stop()
    return 0


def _status_line(runtime: CEERuntime) -> None:
    """A single line that keeps rewriting itself with the running totals."""
    import shutil

    seen = 0
    while True:
        time.sleep(0.6)
        state = runtime.state()
        counters = state["counters"]
        knowledge = state["knowledge"]
        stats = knowledge.get("stats", {})
        text = (
            f"  frames {counters['frames']:<6} anomalies {counters['anomalies']:<5} "
            f"induced {stats.get('inductions', 0):<3} recalled {stats.get('recalls', 0):<4} "
            f"rules {knowledge['count']:<3} mean conf {knowledge['mean_confidence']:.2f}"
        )
        width = max(0, (shutil.get_terminal_size((100, 24)).columns - 1) - len(text))
        sys.stdout.write("\r" + text + " " * width)
        sys.stdout.flush()

        for event in state["events"][::-1]:
            if event["seq"] <= seen:
                continue
            seen = event["seq"]
            if event["phase"] in ("induce", "recall", "verify"):
                sys.stdout.write("\r" + " " * (len(text) + width) + "\r")
                print(f"  > [{event['phase']:<7}] {event['title']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
