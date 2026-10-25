"""Room API client and the MCP tools that expose it to Claude.

Claude sees these as mcp__room__list_devices, mcp__room__get_state and
mcp__room__set_state. HTTP errors become tool results with is_error=True so a
failing room never raises into the SDK.
"""
from __future__ import annotations

import json
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

    async def get_state(self, device_id: str) -> dict:
        return await self._request("GET", f"/devices/{device_id}")

    async def set_state(self, device_id: str, patch: dict) -> dict:
        return await self._request("POST", f"/devices/{device_id}", patch)


def _ok(data: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(data)}]}


def _err(exc: RoomError) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {exc}"}], "is_error": True}


def make_room_tools(client: RoomClient) -> list[SdkMcpTool]:
    @tool("list_devices", "List every controllable device in the room with its current state.", {})
    async def list_devices(args: dict[str, Any]) -> dict[str, Any]:
        try:
            return _ok(await client.list_devices())
        except RoomError as exc:
            return _err(exc)

    @tool("get_state", "Get the current state of one room device by id (e.g. fan, window).",
          {"device": str})
    async def get_state(args: dict[str, Any]) -> dict[str, Any]:
        try:
            return _ok(await client.get_state(args["device"]))
        except RoomError as exc:
            return _err(exc)

    @tool(
        "set_state",
        "Change a room device by sending a partial state patch, e.g. device='fan', "
        "patch={'on': true, 'speed': 2}. Returns the updated device.",
        {"device": str, "patch": dict},
    )
    async def set_state(args: dict[str, Any]) -> dict[str, Any]:
        try:
            return _ok(await client.set_state(args["device"], args["patch"]))
        except RoomError as exc:
            return _err(exc)

    return [list_devices, get_state, set_state]


def build_room_server(base_url: str):
    """Return the in-process MCP server config for ClaudeAgentOptions.mcp_servers."""
    return create_sdk_mcp_server(name="room", tools=make_room_tools(RoomClient(base_url)))
