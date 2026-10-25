"""Room API client and the MCP tools that expose it to Claude.

Claude sees these as mcp__room__list_devices, mcp__room__get_state and
mcp__room__set_state. HTTP errors become tool results with is_error=True so a
failing room never raises into the SDK.
"""
from __future__ import annotations

import json
from urllib.parse import quote
from typing import Any

import httpx
from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server, tool


class RoomError(Exception):
    """The room API rejected a request or could not be reached."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class RoomClient:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    async def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as http:
                resp = await http.request(method, self._base + path, json=body)
        except httpx.HTTPError as exc:
            raise RoomError(f"room API unreachable: {exc}") from exc
        try:
            data = resp.json()
        except ValueError:
            data = None
        if resp.status_code >= 400:
            message = data.get("error") if isinstance(data, dict) else None
            raise RoomError(message or f"HTTP {resp.status_code}", resp.status_code)
        return data

    async def list_devices(self) -> list[dict]:
        return await self._request("GET", "/devices")

    @staticmethod
    def _device_path(device_id: str) -> str:
        if not isinstance(device_id, str) or not device_id.strip():
            raise RoomError("device id is required")
        return f"/devices/{quote(device_id.strip(), safe='')}"

    async def get_state(self, device_id: str) -> dict:
        return await self._request("GET", self._device_path(device_id))

    async def search_music(self, query: str, kind: str = "track") -> list[dict]:
        from urllib.parse import urlencode
        return await self._request("GET", "/search?" + urlencode({"q": query, "kind": kind}))

    async def set_state(self, device_id: str, patch: dict) -> dict:
        return await self._request("POST", self._device_path(device_id), patch)


def _ok(data: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(data)}]}


def _err(exc: RoomError) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {exc}"}], "is_error": True}


async def _guarded(call) -> dict[str, Any]:
    """Run a room call; any failure becomes an is_error result, never an exception."""
    try:
        return _ok(await call())
    except RoomError as exc:
        return _err(exc)
    except Exception as exc:  # nothing may raise into the SDK
        return _err(RoomError(f"{type(exc).__name__}: {exc}"))


def make_room_tools(client: RoomClient) -> list[SdkMcpTool]:
    @tool("list_devices", "List every controllable device in the room with its current state.", {})
    async def list_devices(args: dict[str, Any]) -> dict[str, Any]:
        return await _guarded(client.list_devices)

    @tool("get_state", "Get the current state of one room device by id (e.g. fan, window).",
          {"device": str})
    async def get_state(args: dict[str, Any]) -> dict[str, Any]:
        device = (args or {}).get("device")
        return await _guarded(lambda: client.get_state(device))

    @tool(
        "set_state",
        "Change a room device by sending a partial state patch, e.g. device='fan', "
        "patch={'on': true, 'speed': 2}. Returns the updated device.",
        {"device": str, "patch": dict},
    )
    async def set_state(args: dict[str, Any]) -> dict[str, Any]:
        device = (args or {}).get("device")
        patch = (args or {}).get("patch")
        if not isinstance(patch, dict) or not patch:
            return _err(RoomError("patch must be a non-empty object"))
        return await _guarded(lambda: client.set_state(device, patch))

    @tool(
        "search_music",
        "Search Spotify. kind is track, album, playlist or artist. Returns the top "
        "matches as {uri, name, by}. Play one with set_state(device='speakers', "
        "patch={'query': <uri>}); or skip searching and send patch={'query': 'lofi "
        "beats', 'kind': 'playlist'} to play the top match directly. Speakers also "
        "take {'skip': 'next'|'previous'}, {'shuffle': bool}, {'playing': bool}, "
        "{'volume': 0-100}.",
        {"query": str, "kind": str},
    )
    async def search_music(args: dict[str, Any]) -> dict[str, Any]:
        query = (args or {}).get("query") or ""
        kind = (args or {}).get("kind") or "track"
        return await _guarded(lambda: client.search_music(query, kind))

    return [list_devices, get_state, set_state, search_music]


def build_room_server(base_url: str):
    """Return the in-process MCP server config for ClaudeAgentOptions.mcp_servers."""
    return create_sdk_mcp_server(name="room", tools=make_room_tools(RoomClient(base_url)))
