"""Dashboard host: stdlib HTTP with a Server-Sent-Events stream.

No web framework and no client-side dependencies.  The dashboard is served from
``cee/web`` and updated by SSE, which means the whole project still installs
with nothing beyond ``paho-mqtt``.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from . import config
from .runtime import CEERuntime

PUSH_INTERVAL = 0.35
MAX_BODY = 64 * 1024


class _Handler(BaseHTTPRequestHandler):
    server_version = "CEEDashboard/2.0"
    protocol_version = "HTTP/1.1"

    runtime: CEERuntime = None  # type: ignore[assignment]
    web_dir: Path = config.WEB_DIR
    verbose: bool = False
    _shutdown = threading.Event()

    # -- plumbing ----------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if self.verbose:
            print(f"[http] {self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def _json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _file(self, relative: str) -> None:
        target = (self.web_dir / relative).resolve()
        try:
            target.relative_to(self.web_dir.resolve())
        except ValueError:
            self._json({"error": "forbidden"}, status=403)
            return
        if not target.is_file():
            self._json({"error": "not found", "path": relative}, status=404)
            return
        ctype, _ = mimetypes.guess_type(str(target))
        self._send(200, target.read_bytes(), ctype or "application/octet-stream")

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._file("index.html")
        elif route.startswith("/static/"):
            self._file(route[len("/static/") :])
        elif route == "/api/state":
            self._json(self.runtime.state())
        elif route == "/api/stream":
            self._stream()
        elif route == "/health":
            self._json({"ok": True, "session": config.SESSION_ID})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route != "/api/control":
            self._json({"error": "not found"}, status=404)
            return
        try:
            length = min(int(self.headers.get("Content-Length") or 0), MAX_BODY)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            self._json({"ok": False, "error": "invalid json"}, status=400)
            return
        if not isinstance(payload, dict):
            self._json({"ok": False, "error": "object expected"}, status=400)
            return
        self._json(self.runtime.control(payload))

    # -- server-sent events ------------------------------------------------
    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last = 0.0
        try:
            while not self._shutdown.is_set():
                now = time.monotonic()
                if now - last >= PUSH_INTERVAL:
                    last = now
                    body = json.dumps(self.runtime.state(), separators=(",", ":"))
                    frame = f"data: {body}\n\n"
                    self.wfile.write(frame.encode("utf-8"))
                    self.wfile.flush()
                self._shutdown.wait(0.05)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, OSError):
            pass
        except Exception:
            pass


class DashboardServer:
    """Threaded HTTP server wrapping a :class:`CEERuntime`."""

    def __init__(
        self,
        runtime: CEERuntime,
        host: str = config.HOST,
        port: int = config.PORT,
        verbose: bool = False,
    ) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self.httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        handler = type(
            "BoundHandler",
            (_Handler,),
            {
                "runtime": runtime,
                "web_dir": config.WEB_DIR,
                "verbose": verbose,
            },
        )
        self._handler = handler

    @property
    def url(self) -> str:
        host, port = self.address
        return f"http://{host}:{port}/"

    @property
    def address(self):
        if self.httpd is None:
            return (self.host, self.port)
        return self.httpd.server_address[0], self.httpd.server_address[1]

    def start(self) -> None:
        # The caller's host/port is honoured, otherwise ``--port 8802`` would
        # silently bind the default and two dashboards would collide.
        self.httpd = ThreadingHTTPServer((self.host, self.port), self._handler)
        self.httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, name="cee-http", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        _Handler._shutdown.set()
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        _Handler._shutdown.clear()