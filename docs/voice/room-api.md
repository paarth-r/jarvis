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

Errors are JSON `{"error": "<message>"}`; the message is returned to Claude as
tool-result text, so keep it short and specific.

- 404 `{"error": "unknown device"}` for an id that does not exist.
- 400 `{"error": "..."}` for a bad patch (unknown field, wrong type, out of range).

## Validation rules

- Requests with a body send `Content-Type: application/json`; every response
  (success and error) is `application/json`.
- A successful POST returns the full updated device object, not just the patch.
- An empty or non-object patch (`{}`, `[]`, `null`, a string) is 400.
- A body that is not valid JSON is 400.
- Unknown fields are rejected with 400.
- Any bad field rejects the whole patch with 400; nothing is partially applied.
- `POST /devices` with no id (and any other unrecognised path) is 404.

## Devices (stub)

| id | state fields |
|---|---|
| `fan` | `on`: bool, `speed`: int 1-3 |
| `window` | `open_pct`: int 0-100 |
| `thermostat` | `target_f`: int 60-85, `mode`: `"cool"` / `"heat"` / `"off"` |
| `speakers` | `playing`: bool, `query`: str, `volume`: int 0-100 |

`jarvis-room` (the real server) implements only `speakers` today, backed by the
Mac's system output volume and the Spotify desktop app; its state also carries a
read-only `now_playing` ("Track by Artist"). Its `speakers` also accept `kind`
(`track` / `album` / `playlist` / `artist`, default `track`: what a free-text `query`
should find), `skip` (`"next"` / `"previous"`) and `shuffle` (bool). Albums,
playlists and artists play as a whole. `GET /search?q=<text>&kind=<kind>&limit=<1-10>`
returns the top matches as `[{"uri", "name", "by", "kind"}]`; play one by sending its
`uri` as the `query`. Search covers public Spotify content; your own playlists and
Liked Songs need a user login, which isn't wired up yet. If the Mac or Spotify fails a command
it answers `502` with `{"error": "..."}`. `jarvis-room-stub` implements all four.

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
