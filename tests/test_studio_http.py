import asyncio
import json

import pytest

from jarvis.studio.http import HttpServer, Response, StudioError


async def _request(port, method, path, body=None, headers=None, read_lines=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    data = json.dumps(body).encode() if body is not None else b""
    hdrs = {"Host": f"127.0.0.1:{port}", "Content-Length": str(len(data)),
            "X-Jarvis-Studio": "1", **(headers or {})}
    hdrs = {k: v for k, v in hdrs.items() if v is not None}
    head = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in hdrs.items()) + "\r\n"
    writer.write(head.encode() + data)
    await writer.drain()
    if read_lines:
        lines = []
        while len(lines) < read_lines:
            line = await asyncio.wait_for(reader.readline(), 2)
            if not line:
                break
            lines.append(line.decode())
        writer.close()
        return lines
    raw = await asyncio.wait_for(reader.read(), 2)
    writer.close()
    head, _, payload = raw.partition(b"\r\n\r\n")
    status = int(head.split()[1])
    return status, payload


@pytest.fixture
async def server():
    srv = HttpServer("127.0.0.1", 0)

    @srv.route("GET", r"/api/hello")
    async def hello(req):
        return {"hi": req.query.get("name", "there")}

    @srv.route("POST", r"/api/echo/(?P<what>[a-z]+)")
    async def echo(req, what):
        return {"what": what, "body": req.json}

    @srv.route("POST", r"/api/boom")
    async def boom(req):
        raise StudioError("that can't be bound")

    @srv.route("GET", r"/api/stream")
    async def stream(req):
        async def events():
            for i in range(3):
                yield {"i": i}
        return events()

    @srv.route("GET", r"/")
    async def index(req):
        return Response(200, b"<html>ok</html>", "text/html")

    await srv.start()
    yield srv
    await srv.stop()


async def test_get_json_with_query(server):
    status, body = await _request(server.port, "GET", "/api/hello?name=ada")
    assert status == 200 and json.loads(body) == {"hi": "ada"}


async def test_post_json_with_path_params(server):
    status, body = await _request(server.port, "POST", "/api/echo/peace", {"a": 1})
    assert status == 200 and json.loads(body) == {"what": "peace", "body": {"a": 1}}


async def test_handler_errors_come_back_as_400_with_message(server):
    status, body = await _request(server.port, "POST", "/api/boom", {})
    assert status == 400 and json.loads(body) == {"error": "that can't be bound"}


async def test_unknown_route_is_404(server):
    status, _ = await _request(server.port, "GET", "/api/nope")
    assert status == 404


async def test_html(server):
    status, body = await _request(server.port, "GET", "/")
    assert status == 200 and b"ok" in body


async def test_server_sent_events(server):
    lines = await _request(server.port, "GET", "/api/stream", read_lines=12)
    data = [json.loads(l[6:]) for l in lines if l.startswith("data: ")]
    assert data == [{"i": 0}, {"i": 1}, {"i": 2}]


# --- protections: another website in the browser must not drive the studio ---

async def test_post_without_studio_header_is_refused(server):
    status, _ = await _request(server.port, "POST", "/api/echo/x", {}, headers={"X-Jarvis-Studio": None})
    assert status == 403


async def test_cross_origin_request_is_refused(server):
    status, _ = await _request(server.port, "POST", "/api/echo/x", {},
                               headers={"Origin": "https://evil.example"})
    assert status == 403


async def test_same_origin_request_is_allowed(server):
    status, _ = await _request(server.port, "POST", "/api/echo/x", {},
                               headers={"Origin": f"http://127.0.0.1:{server.port}"})
    assert status == 200


async def test_foreign_host_header_is_refused(server):
    # DNS rebinding: a hostile name resolving to 127.0.0.1.
    status, _ = await _request(server.port, "GET", "/api/hello", headers={"Host": "evil.example:80"})
    assert status == 403


async def test_localhost_host_header_is_fine(server):
    status, _ = await _request(server.port, "GET", "/api/hello", headers={"Host": f"localhost:{server.port}"})
    assert status == 200


async def test_malformed_json_is_400(server):
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    body = b"{nope"
    writer.write((f"POST /api/echo/x HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n"
                  f"X-Jarvis-Studio: 1\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body)
    raw = await asyncio.wait_for(reader.read(), 2)
    writer.close()
    assert raw.split()[1] == b"400"
