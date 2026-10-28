"""A small HTTP server on asyncio streams for the gesture studio: JSON
routes, static files, and server-sent events for the live view. One request
per connection; no dependencies.

The studio can bind and test-run shell commands, so it must not be drivable
by other websites open in the same browser:
- it listens on 127.0.0.1 only;
- the Host header must name localhost (defeats DNS rebinding);
- a request carrying a foreign Origin is refused;
- anything but GET must carry X-Jarvis-Studio, a custom header, which makes
  a cross-site browser request need a CORS preflight this server never grants.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Pattern, Tuple
from urllib.parse import parse_qsl, urlsplit

logger = logging.getLogger(__name__)

_STATUS = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found",
           405: "Method Not Allowed", 413: "Payload Too Large", 500: "Internal Server Error"}
_MAX_BODY = 4 * 1024 * 1024
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")


class StudioError(Exception):
    """Raised by a handler: becomes a 400 with {"error": message}."""


@dataclass
class Request:
    method: str
    path: str
    query: Dict[str, str]
    headers: Dict[str, str]
    body: bytes = b""
    json: Any = None


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"
    headers: Dict[str, str] = field(default_factory=dict)


Handler = Callable[..., Awaitable[Any]]


def _json_response(obj: Any, status: int = 200) -> Response:
    return Response(status, json.dumps(obj).encode())


class HttpServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host, self.port = host, port
        self._routes: List[Tuple[str, Pattern, Handler]] = []
        self._server: Optional[asyncio.base_events.Server] = None
        self._streams: set = set()

    def route(self, method: str, pattern: str):
        def deco(fn: Handler) -> Handler:
            self._routes.append((method, re.compile(pattern + r"$"), fn))
            return fn
        return deco

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        for w in list(self._streams):
            w.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # --- request handling ---

    async def _read(self, reader: asyncio.StreamReader) -> Optional[Request]:
        line = await reader.readline()
        if not line:
            return None
        method, target, _ = line.decode("latin-1").split(" ", 2)
        headers: Dict[str, str] = {}
        while True:
            h = await reader.readline()
            if h in (b"\r\n", b"\n", b""):
                break
            k, _, v = h.decode("latin-1").partition(":")
            headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length", "0") or 0)
        if n > _MAX_BODY:
            raise ValueError("too large")
        body = await reader.readexactly(n) if n else b""
        url = urlsplit(target)
        return Request(method.upper(), url.path, dict(parse_qsl(url.query)), headers, body)

    def _allowed(self, req: Request) -> bool:
        host = req.headers.get("host", "")
        if host.rsplit(":", 1)[0] not in _LOCAL_HOSTS:
            return False
        origin = req.headers.get("origin")
        if origin and urlsplit(origin).hostname not in ("127.0.0.1", "localhost", "::1"):
            return False
        if origin and urlsplit(origin).port != self.port:
            return False
        if req.method != "GET" and req.headers.get("x-jarvis-studio") != "1":
            return False
        return True

    async def _dispatch(self, req: Request) -> Any:
        allowed_methods = []
        for method, pattern, fn in self._routes:
            m = pattern.match(req.path)
            if not m:
                continue
            if method != req.method:
                allowed_methods.append(method)
                continue
            return await fn(req, **m.groupdict())
        if allowed_methods:
            return _json_response({"error": "method not allowed"}, 405)
        return _json_response({"error": "not found"}, 404)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            try:
                req = await self._read(reader)
            except (ValueError, asyncio.IncompleteReadError):
                await self._send(writer, _json_response({"error": "bad request"}, 400))
                return
            if req is None:
                return
            if not self._allowed(req):
                await self._send(writer, _json_response({"error": "forbidden"}, 403))
                return
            if req.body:
                try:
                    req.json = json.loads(req.body)
                except ValueError:
                    await self._send(writer, _json_response({"error": "body isn't JSON"}, 400))
                    return
            try:
                result = await self._dispatch(req)
            except StudioError as e:
                result = _json_response({"error": str(e)}, 400)
            except Exception:
                logger.exception("studio handler failed: %s %s", req.method, req.path)
                result = _json_response({"error": "internal error (see the Jarvis log)"}, 500)
            if hasattr(result, "__aiter__"):
                await self._stream(writer, result)
            elif isinstance(result, Response):
                await self._send(writer, result)
            else:
                await self._send(writer, _json_response(result))
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()

    async def _send(self, writer: asyncio.StreamWriter, resp: Response) -> None:
        head = [f"HTTP/1.1 {resp.status} {_STATUS.get(resp.status, '')}",
                f"Content-Type: {resp.content_type}",
                f"Content-Length: {len(resp.body)}",
                "Cache-Control: no-store",
                "Connection: close"]
        head += [f"{k}: {v}" for k, v in resp.headers.items()]
        writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + resp.body)
        await writer.drain()

    async def _stream(self, writer: asyncio.StreamWriter, events) -> None:
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                     b"Cache-Control: no-store\r\nConnection: keep-alive\r\n\r\n")
        self._streams.add(writer)
        try:
            async for ev in events:
                writer.write(f"data: {json.dumps(ev)}\n\n".encode())
                await writer.drain()
        finally:
            self._streams.discard(writer)
