"""
Cognitive Evolution Engine (CEE)
================================

An online rule-induction and rule-application agent for autonomous rover
telemetry.  The package is split so that every layer has exactly one owner:

    config.py     single source of truth for axes, thresholds, wiring
    features.py   raw telemetry frame -> comparable feature vector
    actions.py    primitive action library + applicability scoring
    knowledge.py  persistent knowledge base: induce / recall / reinforce
    engine.py     reflective reasoning engine (the only brain in the system)
    actuators.py  vehicle + compute actuators that rules actually drive
    bus.py        local-first transport with an optional MQTT bridge
    runtime.py    wires bus + engine + simulator + controls together
    server.py     stdlib HTTP + Server-Sent-Events dashboard host

The engine implements two cognitive workings, both visible in the dashboard:

    WORKING 1 - INDUCTION
        When a telemetry anomaly is detected and no stored rule matches it,
        the engine generalises the observation into a symbolic predicate
        (`axis <op> threshold AND ...`), scores every primitive action
        against the predicted post-action residual, and commits the winning
        action to the knowledge base as a new, executable rule.

    WORKING 2 - RECALL
        When an anomaly is detected and a stored rule matches, the engine
        recalls it by confidence, executes its action against the actuators,
        observes the effect on the telemetry, and reinforces or penalises the
        rule's confidence.  Rules that repeatedly fail to help are demoted.
"""

__all__ = ["__version__"]

__version__ = "2.0.0"