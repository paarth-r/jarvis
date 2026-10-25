# Voice Agent ("hey Claude") — Design

Date: 2026-10-25
Status: approved 2026-10-25 (after prior-art pass)

## Intent

An always-on voice front-end for Claude that lives in the user's room. They say
"hey Claude", talks, and gets the same assistant they get typing into Claude Code
today: same skills, same memory, same notes (wiki + Claude
Memory), same connectors (Google Calendar, Gmail). Plus control
of the physical room (fan, window, temperature, music on the speakers) through
a room API that the user builds separately.

It is not a new, dumber assistant. It is Claude Code without the keyboard. The
test: anything they'd type to Claude Code ("what's due tomorrow", "add tennis
Sunday 7:30", "what did I decide about widgetPose re-id", talking through an
idea) works out loud.

The user's stance: move fast, break things. Intelligence over latency: silence
while Claude works is acceptable; tuning response time is later work.

## Goals

1. Wake on "hey Claude", spoken locally; no audio leaves the box before the wake word.
2. Accurate transcription of the user's speech, including their proper nouns
   (names of people, places and projects).
3. Full Claude Code brain: skills, memory, MCP connectors, wiki read/write.
4. Spoken answers that sound like speech: short, conversational, no markdown.
5. Multi-turn conversation: context carries across turns in one conversation.
6. Follow-up window: after Claude finishes speaking, keep listening ~8 s for a
   follow-up without the wake word (Tesla/Grok-style semi-natural flow).
7. Room control through a room API, stubbed for now.
8. Runs on the Mac now; documented, scripted migration to a Mac mini or the
   Jetson Orin NX later.

## Non-goals (v1)

- The real device layer (smart plugs, window actuator, speaker integration).
  The user builds the room API; this project only calls it.
- Barge-in (interrupting Claude mid-sentence). Needs echo cancellation; later.
- Latency optimization beyond the cheap wins listed below.
- Speaker identification, multi-room, phone app.
- Integration with the vision/gesture runtime in this repo.
- The vault git mirror (needed only at migration; see Migration).

## Anti-goals

- **Bad speech-to-text.** The user named this as the one thing that would make them
  stop using it. STT accuracy on their vocabulary is a first-class requirement and
  gets measured, not assumed.
- Streaming room audio to the cloud before the wake word fires.
- A "voice mode" that is a lobotomized assistant (no skills, no memory, no wiki).

## Decided

### The user's decisions
- Brain is Claude ("powered by you"), with second-brain access.
- Room devices sit behind a room API that the user will build; the agent just calls it.
- Develop on the Mac; later move to a Mac mini or the Orin NX with a
  well-documented, easy transfer.
- The vault should eventually be mirrored to GitHub (private) to remove the
  iCloud dependency. Deferred to migration time.
- Silence while thinking is fine.
- Follow-up without the wake word is wanted if it's cheap; always-wake-word is
  an acceptable fallback.
- **Voice power level:** reading anything, editing the wiki/vault, and room
  control run without asking. Shell commands, and any file edit or delete
  outside the vault, are spoken back and need a "yes" first (same confirmation
  path as outbound actions).
- **Wake word:** v1 ships with the prebuilt "hey Jarvis". "Hey Claude" comes
  later and must be drag-and-drop: drop a trained model file into
  `models/wakeword/` (or point `config.voice.yaml` at it) and it's live, with no
  code change. The assistant itself is always Claude; "Jarvis" is only the
  placeholder trigger phrase, never a rename.

### Claude's calls (the user agreed to the set in conversation, any can be revisited)
- **Brain: Claude Agent SDK (Python, `claude-agent-sdk`) driving headless Claude
  Code**, loading user + project settings so their skills, CLAUDE.md, memory and
  hooks apply. Not the raw Messages API: rebuilding skills/memory/connectors
  would defeat the point.
- **Wake word: openWakeWord**, local. "hey claude" has no prebuilt model, so v1
  ships with the prebuilt "hey jarvis" model as a placeholder and a custom
  "hey claude" model is trained (openWakeWord's synthetic-data pipeline) as a
  separate task. Swappable behind a `WakeWord` interface; any openWakeWord
  model file in `models/wakeword/` is loaded automatically. A community
  "hey claude" openWakeWord model exists (huggingface.co/gdiamos/hey-claude,
  Apache-2.0) and is the first drop-in candidate. Known weaknesses: fires on
  "hey Clyde"/"clawed", ~90% recall, untested against music/TV. It replaces
  "hey Jarvis" only after an overnight false-wake test in their room with music
  playing (measure in hours, fresh `Model` per clip; `reset()` does not clear
  the buffer).
- **End of speech: Silero VAD + smart-turn.** Silero VAD finds speech; the
  open-source smart-turn model (Pipecat's) decides whether they're actually done,
  so a mid-thought pause doesn't cut them off. A hard max (e.g. 3 s of silence)
  still ends the turn.
- **Whisper hallucination guard:** STT only runs on segments the VAD marked as
  speech; transcripts with no real speech or known phantom phrases
  ("thank you for watching", ...) are dropped.
- **STT: Whisper large-v3-turbo, local** (mlx-whisper on the Mac; faster-whisper
  or whisper.cpp with CUDA on the Orin). Behind an `STT` interface so a cloud
  recognizer can be swapped in if local measures badly.
  - **Vocabulary priming:** an initial prompt built from a vocab file of their
    names/terms (seeded by hand from the wiki/memory; regenerable by a script).
  - **Transcript log:** every post-wake utterance's audio + transcript saved
    locally (gitignored) so word error rate on their real speech can be measured.
    A small script computes WER against hand-corrected transcripts.
- **TTS: Kokoro-82M, local** (runs on Mac and Orin, free, decent quality).
  Behind a `TTS` interface; a cloud voice (ElevenLabs/OpenAI) can be swapped in.
- **Speech-shaped output:** a system-prompt append tells Claude it's speaking
  aloud: brief, no markdown/code/lists, numbers spoken naturally; anything long
  (a plan, a diff, a table) goes to the wiki or to their phone via iMessage, and
  Claude says where it put it.
- **Streaming speech:** Claude's text streams; complete sentences go to TTS as
  they arrive so speaking starts before the full answer exists.
- **Audio cues:** a short chime on wake, a soft "working" tone if a turn runs
  past ~3 s, so silence never reads as "it didn't hear me".
- **Confirmation rule:** room actions (fan, music, window) just happen.
  Anything that leaves the room (send email, send iMessage, reply, share a file,
  delete anything) is read back and confirmed out loud ("yes"/"no") before it
  runs. Implemented with the SDK's tool-permission callback: listed tools
  trigger a spoken confirmation; everything else is allowed.
- **Room API as a tool:** an in-process SDK MCP server named `room` exposing
  generic tools: `list_devices`, `get_state(device)`, `set_state(device, ...)`.
  It calls the room API over HTTP at a configured URL. v1 ships a stub room
  server (fake fan/window/thermostat/speakers, state in memory, logs every
  call) so the whole loop works end-to-end today. Generic tools mean new devices
  need no agent changes.
- **Conversation lifetime:** one SDK session per conversation; a new session
  starts after 10 minutes idle, or when they say something like "new
  conversation". Bounds context growth.
- **Session-end hook:** The user has a SessionEnd hook that synthesizes each
  session into the wiki. Voice sessions are many and short; they should still
  be logged, but batched (e.g. one daily voice-log entry) rather than one wiki
  synthesis per conversation. v1: set the skip flag for voice sessions and
  append each turn to a daily `wiki/voice-log/YYYY-MM-DD.md`. Builder may
  adjust the mechanism.
- **Code location:** a new `jarvis/voice/` package in this repo with its own
  entry point `jarvis-voice`, independent of the vision runtime (no camera,
  MediaPipe or Quartz dependency on the voice path). Voice deps go in a
  `[voice]` optional-dependency group. The existing `llm_agent.py` stub is left
  alone; wiring voice into the vision runtime is future work.
- **Config:** `config.voice.yaml` (devices, models, room API URL, follow-up
  window, idle timeout, confirm-tool list).

### Claude's calls from the builder review (the user to check)
- **Confirmation recognition:** after reading an outbound action back, only an
  explicit affirmative ("yes", "send it", "do it") proceeds; anything else,
  including silence or an unclear transcript, cancels and says so. No retries.
- **Messages to the user themself** (iMessage to their own number, "send it to my
  phone") are not outbound and need no confirmation. Uses the installed
  iMessage plugin's reply tool.
- **Working directory:** the vault root (configurable). User-level settings,
  skills, hooks and auto-memory load from `~/.claude` regardless of cwd; the
  Jarvis repo's project settings are deliberately not loaded.
- **Model:** the top model (Opus) at default effort, matching "intelligence over
  latency". Runs on their Claude Code login (subscription limits, not per-token
  billing). Per-turn timeout 3 minutes; on timeout or error it says a short
  apology plus what failed, and returns to listening.
- **Sessions:** default Claude Code session storage, so voice conversations are
  visible to `claude --resume`. A daemon restart always starts fresh.
- **Voice log:** the voice code (not Claude) appends each turn to
  `<vault>/wiki/voice-log/YYYY-MM-DD.md`. The builder may read the user's
  SessionEnd hook to find how it skips a session, and works around it from
  the voice side rather than editing the hook.
- **Audio:** built-in Mac mic/speakers must work for development; the USB puck
  is an upgrade, not a prerequisite. The mic reopens ~300 ms after TTS ends.
  The follow-up window starts when TTS ends; the wake word also works during it.
- **Room API contract:** this repo defines it (`docs/voice/room-api.md`, small
  REST: `GET /devices`, `GET /devices/{id}`, `POST /devices/{id}` with a JSON
  state patch, no auth on the LAN). The stub implements it; the user builds the
  real one against it.

## Pipeline

```
mic ─▶ WakeWord ──fired──▶ chime ─▶ record until VAD silence ─▶ STT ─▶ transcript
                                                                         │
             ┌───────────────────────────────────────────────────────────┘
             ▼
     Agent (Claude Agent SDK session) ──tools──▶ skills / MCP / wiki / room API
             │  streamed text
             ▼
     sentence splitter ─▶ TTS ─▶ speakers
             │
             ▼
     follow-up window (~8 s VAD listen, no wake word) ─▶ STT ─▶ Agent ...
             │ nothing heard
             ▼
     back to WakeWord
```

While TTS plays, the mic is ignored (no barge-in), so Claude can't hear and
answer itself.

## Migration (Mac -> Mac mini / Orin NX)

- `docs/voice/DEPLOY.md` + `scripts/install_voice.sh`: one script installs
  deps, models, and a service (launchd on macOS, systemd on Linux) that starts
  on boot and restarts on crash.
- Claude Code must be installed and logged in on the box (same account) so
  connectors and settings carry over; skills/settings copied or synced.
- **Mac mini:** vault keeps syncing over iCloud; nothing else changes.
- **Orin NX:** iCloud doesn't sync to Linux. Prerequisite: mirror the vault to
  a **private** GitHub repo with auto-commit on the Mac and pull-before-write /
  push-after-write on the Orin. Its own small project when the move happens.
- Mac-only features (iMessage plugin) won't exist on the Orin; "send to phone"
  needs another channel there. Noted, not solved in v1.

## Success

- They say the wake word (v1: "hey Jarvis") + "what's due tomorrow" from across the room and gets a
  correct spoken answer, using the same skills as typed Claude Code.
- "hey Claude, turn on the fan" hits the stub room API and confirms.
- An outbound action (draft-and-send email) asks for spoken confirmation first.
- A follow-up within ~8 s works without the wake word.
- WER is measured on at least 50 of their real utterances; proper nouns
  specifically checked.

## Prior art (2026-10-25 research)

- adewaskar/jarvis: Agent SDK bridge + browser wake word/speech, subscription
  login, read-only tool gate by default. Found claude.ai connectors do NOT
  reach the SDK session.
- Home Assistant forum build: Claude CLI as house brain, `--continue` for
  context, sentence-streamed TTS, 10 s continuous-conversation window (needs a
  state-transition trigger to avoid self-loops), spoken "one moment" ack.
- Pipecat: local Mac pipeline (MLX Whisper, Kokoro, smart-turn) <800 ms. We
  borrow its components, not the framework: Claude Code turns are 10-60 s of
  tool use, not real-time chat. Revisit Pipecat if barge-in is ever built.

## Open risks (verify first, before building around them)

1. **claude.ai connectors (Gmail, Calendar, Drive) are expected NOT to load**
   in a headless Agent SDK session (prior art found this). Spike #1 confirms by
   listing the session's tools. If confirmed, Gmail/Calendar get local MCP
   servers configured on the box; wiki and memory are local and
   unaffected.
2. **Wake-word quality for "hey claude"** once trained; false accepts from
   music/YouTube in the room.
3. **macOS mic permission for a background service.** The service must run as
   a LaunchAgent in their logged-in session with mic access granted, or it gets
   silent audio with no error.
4. **Subscription auth:** personal local use of the Agent SDK on their Claude
   Code login appears tolerated; re-check Anthropic's current terms before
   relying on it.
5. **Mic quality across the room.** Laptop mic may be too weak; a USB
   conference puck is the expected fix.

## Privacy (public repo)

The repo is public. Nothing personal is committed: vocab list, vault path,
transcripts, audio, voice logs and `config.voice.local.yaml` are gitignored;
the repo ships an example config only.

## Future

- dum-e: stereo vision mounted on the SO-101 arm with Claude in control of it,
  a real movie-style Dum-E. The room API's generic device tools are the seam:
  the arm becomes another device/tool set. Separate project.

## Left to the builder

- Library versions, audio I/O library (sounddevice is the obvious pick), thread
  vs asyncio layout, exact chime sounds, sentence splitting, the stub room
  server's framework, test strategy, how vocab priming text is formatted.
