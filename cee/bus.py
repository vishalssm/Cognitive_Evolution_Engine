"""Telemetry transport.

Local-first by default.  The in-process bus delivers frames synchronously so
that the engine, the simulator and the dashboard all observe a frame in the
same instant, which keeps the UI honest and makes the demo work with no
network at all.

MQTT is opt-in.  When enabled the same frames are mirrored to a broker on a
session-scoped topic (``cee/<session>/telemetry``) so separate processes --
and separate machines -- can run the rover, the brain and the dashboard
independently without colliding with unrelated traffic.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from . import config


FrameHandler = Callable[[Dict[str, Any]], None]


class TelemetryBus:
    """In-process pub/sub with an optional MQTT bridge."""

    def __init__(
        self,
        use_mqtt: bool = False,
        topic: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> None:
        self.handlers: List[FrameHandler] = []
        self.use_mqtt = use_mqtt
        self.topic = topic or config.TOPIC_TELEMETRY
        self.client_id = client_id or f"cee_{config.SESSION_ID}"
        self.client = None
        self.connected = False
        self.last_error = ""
        self.published = 0
        self.received = 0
        self._lock = threading.RLock()
        self._ready = threading.Event()

    # -- local path --------------------------------------------------------
    def subscribe(self, handler: FrameHandler) -> None:
        with self._lock:
            self.handlers.append(handler)

    def publish(self, frame: Dict[str, Any], *, mirror: bool = True) -> None:
        """Deliver a frame to local subscribers, then mirror it if MQTT is on."""
        with self._lock:
            handlers = list(self.handlers)
        for handler in handlers:
            try:
                handler(frame)
            except Exception as exc:  # a bad subscriber must not kill the bus
                self.last_error = f"handler: {exc}"
        self.published += 1
        if mirror and self.use_mqtt and self.client is not None:
            try:
                self.client.publish(self.topic, json.dumps(frame), qos=0)
            except Exception as exc:
                self.last_error = f"publish: {exc}"

    # -- optional MQTT bridge ---------------------------------------------
    def connect_mqtt(self) -> bool:
        if not self.use_mqtt:
            return False
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self.last_error = "paho-mqtt is not installed"
            self.use_mqtt = False
            return False

        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, self.client_id)
        except TypeError:  # paho 1.x
            client = mqtt.Client(client_id=self.client_id)

        client.on_connect = self._on_connect
        client.on_message = self._on_mqtt_message
        client.on_disconnect = self._on_disconnect

        try:
            client.connect(config.BROKER, config.BROKER_PORT, keepalive=30)
            client.loop_start()
        except Exception as exc:
            self.last_error = f"{config.BROKER}: {exc}"
            self.connected = False
            # Local operation continues regardless.
            self._ready.set()
            return False

        self.client = client
        self._ready.wait(timeout=3.0)
        return self.connected

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        try:
            code = int(reason_code)
        except (TypeError, ValueError):
            code = 0
        if code == 0:
            self.connected = True
            self.last_error = ""
            try:
                client.subscribe(self.topic)
            except Exception:
                pass
        else:
            self.connected = False
            self.last_error = f"broker refused connection: {reason_code}"
        self._ready.set()

    def _on_disconnect(self, client, userdata, flags=None, reason_code=None, properties=None) -> None:
        self.connected = False

    def _on_mqtt_message(self, client, userdata, message) -> None:
        try:
            frame = json.loads(message.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        self.received += 1
        with self._lock:
            handlers = list(self.handlers)
        for handler in handlers:
            try:
                handler(frame)
            except Exception as exc:
                self.last_error = f"handler: {exc}"

    def disconnect_mqtt(self) -> None:
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        self.client = None
        self.connected = False

    def status(self) -> Dict[str, Any]:
        return {
            "mode": "mqtt" if self.use_mqtt else "local",
            "connected": self.connected if self.use_mqtt else True,
            "topic": self.topic if self.use_mqtt else "in-process",
            "broker": config.BROKER if self.use_mqtt else None,
            "session": config.SESSION_ID,
            "published": self.published,
            "received": self.received,
            "error": self.last_error,
            "ts": round(time.time(), 2),
        }