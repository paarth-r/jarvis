# Room API contract

The voice agent controls the room through a small HTTP+JSON API. Any
implementation (the real hardware service, or `jarvis-room-stub`) that follows
this contract works. No auth; LAN only. The agent reads the base URL from the
voice config; the stub listens on `http://127.0.0.1:8790`.

## Endpoints

| Method | Path | Body | Success |
|---|---|---|---|
| GET | `/devices` | none | 200, list of devices |
| GET | `/devices/{id}` | none | 200, one device |
| POST | `/devices/{id}` | JSON state patch | 200, the updated device |

A device is `{"id": str, "name": str, "kind": str, "state": {...}}`.

A patch is a partial `state` object: only the listed fields change.
Example: `POST /devices/fan` with `{"on": true, "speed": 2}`.

## Errors

Errors are JSON `{"error": "<message>"}`; the message is read aloud to Claude,
so keep it short and specific.

- 404 `{"error": "unknown device"}` for an id that does not exist.
- 400 `{"error": "..."}` for a bad patch (unknown field, wrong type, out of range).

## Devices (stub)

| id | state fields |
|---|---|
| `fan` | `on`: bool, `speed`: int 1-3 |
| `window` | `open_pct`: int 0-100 |
| `thermostat` | `target_f`: int 60-85, `mode`: `"cool"` / `"heat"` / `"off"` |
| `speakers` | `playing`: bool, `query`: str, `volume`: int 0-100 |

A real implementation may add devices or fields; the agent discovers them via
`GET /devices`.

## Claude tools

The brain mounts these as the in-process MCP server `room`:
`mcp__room__list_devices`, `mcp__room__get_state(device)`,
`mcp__room__set_state(device, patch)`.

## Running the stub

```
jarvis-room-stub --port 8790
```
