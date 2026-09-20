"""Small dependency-free live dashboard for closed-loop trials.

The controller publishes snapshots and JPEG frames here. A local browser polls
them over HTTP; no cloud service or extra Python package is involved.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class DashboardState:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            "status": "IDLE",
            "mode": "TELEOP",
            "task": "",
            "condition": "",
            "checkpoint": "",
            "trial": None,
            "elapsed_s": 0.0,
            "max_trial_s": 0.0,
            "frame": 0,
            "camera_fps": 0.0,
            "control_hz": 0.0,
            "frame_age_ms": None,
            "serial_age_ms": None,
            "latency_ms": None,
            "replan_steps": 0,
            "queue_remaining": 0,
            "inferences": 0,
            "watchdog_events": 0,
            "firmware_deg": None,
            "commanded_deg": None,
            "target_deg": None,
            "action_chunk_deg": [],
            "paused": False,
            "outcome": None,
            "events": [],
            "updated_at": time.time(),
        }
        self._jpeg = None
        self.controls = queue.Queue()

    def update(self, **values):
        with self._lock:
            self._state.update(values)
            self._state["updated_at"] = time.time()

    def add_event(self, label, elapsed_s, color="blue"):
        with self._lock:
            events = list(self._state.get("events", []))
            events.append({"label": label, "t": round(float(elapsed_s), 3), "color": color})
            self._state["events"] = events[-24:]

    def set_frame(self, image, quality=80):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self._lock:
                self._jpeg = encoded.tobytes()

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def jpeg(self):
        with self._lock:
            return self._jpeg


class _DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, state, html):
        self.dashboard_state = state
        self.dashboard_html = html
        super().__init__(address, _DashboardHandler)


class _DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, status, content_type, body, cache="no-store"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, "text/html; charset=utf-8", self.server.dashboard_html)
        elif path == "/api/state":
            body = json.dumps(self.server.dashboard_state.snapshot(), allow_nan=False).encode()
            self._send(200, "application/json", body)
        elif path == "/api/frame.jpg":
            body = self.server.dashboard_state.jpeg()
            if body is None:
                self._send(204, "image/jpeg", b"")
            else:
                self._send(200, "image/jpeg", body)
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        if self.path != "/api/control":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            payload = json.loads(self.rfile.read(length) or b"{}")
            action = payload.get("action")
            if action not in ("toggle_pause", "e_stop", "success", "failure"):
                raise ValueError("unknown action")
            self.server.dashboard_state.controls.put(action)
            self._send(202, "application/json", b'{"accepted":true}')
        except (ValueError, json.JSONDecodeError):
            self._send(400, "application/json", b'{"accepted":false}')


class LiveDashboard:
    def __init__(self, host="127.0.0.1", port=8765, open_browser=True):
        html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
        with open(html_path, "rb") as f:
            html = f.read()
        self.state = DashboardState()
        self.server = _DashboardServer((host, port), self.state, html)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        self.url = f"http://{display_host}:{self.server.server_port}/"
        self.open_browser = open_browser

    def start(self):
        self.thread.start()
        print(f"[OK] live dashboard: {self.url}")
        if self.open_browser:
            threading.Timer(0.4, lambda: webbrowser.open(self.url)).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def update(self, **values):
        self.state.update(**values)

    def add_event(self, *args, **kwargs):
        self.state.add_event(*args, **kwargs)

    def set_frame(self, image):
        self.state.set_frame(image)

    def controls(self):
        actions = []
        while True:
            try:
                actions.append(self.state.controls.get_nowait())
            except queue.Empty:
                return actions
