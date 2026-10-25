"""In-memory stand-in for the room API (see docs/voice/room-api.md).

Stdlib only. Run with `jarvis-room-stub [--port 8790]`.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8790


def _is_int(v: object, lo: int, hi: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


# field -> (validator, description used in error messages)
_SCHEMA = {
    "fan": {
        "on": (lambda v: isinstance(v, bool), "a boolean"),
        "speed": (lambda v: _is_int(v, 1, 3), "an integer 1-3"),
    },
    "window": {"open_pct": (lambda v: _is_int(v, 0, 100), "an integer 0-100")},
    "thermostat": {
        "target_f": (lambda v: _is_int(v, 60, 85), "an integer 60-85"),
        "mode": (lambda v: v in ("cool", "heat", "off"), "one of cool, heat, off"),
    },
    "speakers": {
        "playing": (lambda v: isinstance(v, bool), "a boolean"),
        "query": (lambda v: isinstance(v, str), "a string"),
        "volume": (lambda v: _is_int(v, 0, 100), "an integer 0-100"),
    },
}

_INITIAL = {
    "fan": ("Fan", {"on": False, "speed": 1}),
    "window": ("Window", {"open_pct": 0}),
    "thermostat": ("Thermostat", {"target_f": 72, "mode": "off"}),
    "speakers": ("Speakers", {"playing": False, "query": "", "volume": 30}),
}


class RoomState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._devices = {
            did: {"id": did, "name": name, "kind": did, "state": copy.deepcopy(state)}
            for did, (name, state) in _INITIAL.items()
        }

    def list(self) -> list[dict]:
        with self._lock:
            return copy.deepcopy(list(self._devices.values()))

    def get(self, did: str) -> dict | None:
        with self._lock:
            dev = self._devices.get(did)
            return copy.deepcopy(dev) if dev else None

    def patch(self, did: str, patch: object) -> tuple[int, dict]:
        with self._lock:
            dev = self._devices.get(did)
            if dev is None:
                return 404, {"error": "unknown device"}
            if not isinstance(patch, dict) or not patch:
                return 400, {"error": "patch must be a non-empty JSON object"}
            schema = _SCHEMA[did]
            for key, value in patch.items():
                if key not in schema:
                    return 400, {"error": f"unknown field '{key}' for {did}"}
                check, desc = schema[key]
                if not check(value):
                    return 400, {"error": f"{key} must be {desc}"}
            dev["state"].update(patch)
            return 200, copy.deepcopy(dev)


def _make_handler(room: RoomState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: object) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _device_id(self) -> str | None:
            parts = self.path.split("?")[0].strip("/").split("/")
            return parts[1] if len(parts) == 2 and parts[0] == "devices" else None

        def do_GET(self) -> None:
            if self.path.split("?")[0].rstrip("/") == "/devices":
                self._send(200, room.list())
                return
            did = self._device_id()
            dev = room.get(did) if did else None
            if dev is None:
                self._send(404, {"error": "unknown device"})
            else:
                self._send(200, dev)

        def do_POST(self) -> None:
            did = self._device_id()
            if did is None:
                self._send(404, {"error": "unknown device"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                patch = json.loads(self.rfile.read(length) or b"null")
            except ValueError:
                self._send(400, {"error": "body must be valid JSON"})
                return
            status, body = room.patch(did, patch)
            self._send(status, body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"room-stub: {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}",
                  file=sys.stderr)

    return Handler


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _make_handler(RoomState()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stub room API for the Jarvis voice agent")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    server = make_server("127.0.0.1", args.port)
    print(f"room-stub listening on http://127.0.0.1:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
