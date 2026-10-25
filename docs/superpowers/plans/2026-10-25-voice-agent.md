# Voice Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An always-on "hey Jarvis" voice front-end for headless Claude Code (same skills, memory, wiki, connectors) with room control through a stubbed REST room API.

**Architecture:** One asyncio process (`jarvis-voice`) running a state machine: IDLE (wake word) -> LISTEN (VAD + smart-turn) -> THINK (warm Claude Agent SDK session) -> SPEAK (sentence-streamed Kokoro TTS) -> FOLLOWUP (8 s listen without wake word) -> IDLE. Every stage sits behind a small interface so tests inject fakes and models can be swapped (wake word, STT, TTS). Room devices are reached through an in-process SDK MCP server that calls a REST room API; a stdlib stub server implements that API.

**Tech Stack:** Python 3.12 (separate `.venv-voice`, created with `uv venv -p 3.12 .venv-voice`), claude-agent-sdk 0.2.174, openwakeword (onnx), sounddevice, mlx-whisper (large-v3-turbo), onnxruntime + transformers `WhisperFeatureExtractor` (smart-turn v3.1), kokoro-onnx, httpx, pyyaml, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-10-25-voice-agent-design.md`

## Global Constraints

- Repo is PUBLIC. Never commit: vocab list, vault path, transcripts, audio, voice logs, `config.voice.local.yaml`, model weights. Ship `config.voice.example.yaml` only.
- Voice code imports nothing from the vision runtime (`jarvis/modules`, `jarvis/runtime`, cv2, mediapipe, Quartz).
- Voice deps live in `[project.optional-dependencies] voice = [...]`; existing `dev` extras unchanged.
- Audio in: 16 kHz mono int16, frames of 1280 samples (80 ms) — openWakeWord's native chunk.
- Wake word v1: prebuilt `hey_jarvis`; plus every `*.onnx` in `models/wakeword/` loaded automatically. Default threshold 0.5. The assistant is never renamed; "Jarvis" is only the trigger phrase.
- STT model: `mlx-community/whisper-large-v3-turbo`, `language="en"`, `initial_prompt` from the vocab file.
- Hard end-of-turn: 3.0 s of silence regardless of smart-turn. Smart-turn threshold 0.5, evaluated on the last <= 8 s.
- Follow-up window: 8.0 s, starts when TTS playback ends; mic reopens 0.3 s after TTS ends; wake word also fires during follow-up.
- Working tone if a turn runs past 3.0 s. Per-turn timeout 180 s. Idle session reset 600 s (10 min).
- Brain: `ClaudeAgentOptions(cwd=<vault>, setting_sources=["user","project","local"], model=<config, default "claude-opus-5-5">)`. Client is started at boot and kept warm (claude.ai connectors finish connecting a few seconds after start; measured 2026-10-25).
- Confirmation: only an explicit affirmative proceeds; anything else (silence, unclear, "no") cancels and says so. No retries.
- Every voice SDK session drops `/tmp/claude-skip-synthesis/<session_id>` (the user's SessionEnd synthesis hook skips flagged sessions); the voice code logs turns itself to `<vault>/wiki/voice-log/YYYY-MM-DD.md`.
- No emojis anywhere (code, logs, docs, commits).

## Review Focus

1. Whisper returning phantom text on noise/silence ("Thank you for watching.", "you") -> dropped, no Claude turn. Pinned in Task 4.
2. Claude's reply containing markdown/code fences/URLs -> spoken text stripped of markup, never "asterisk asterisk". Pinned in Task 5.
3. Claude session dies or hangs mid-turn (CLI crash, 180 s timeout) -> spoken short error, loop returns to IDLE, next wake works with a fresh client. Pinned in Task 7.
4. Confirmation transcripts like "yes no wait", "yeah", "", "send it" -> only clean affirmatives proceed. Pinned in Task 6.
5. Wake word firing on Claude's own TTS ("Jarvis" said in a reply) -> mic ignored during playback + 0.3 s tail. Pinned in Task 9.

---

## File Structure

```
jarvis/voice/
  __init__.py
  config.py        VoiceConfig dataclass, load_config(example, local override)
  audio.py         MicStream (frames async iterator, mute()), Player (play/stop), tones
  wakeword.py      WakeWord: prebuilt + drop-in models, process(frame) -> name|None
  turn.py          SmartTurn (onnx) + UtteranceRecorder (Silero VAD + smart-turn + hard stop)
  stt.py           STT protocol, WhisperSTT, clean_transcript(), load_vocab_prompt()
  speech.py        to_speakable(), SentenceSplitter, TTS protocol, KokoroTTS
  permissions.py   classify_tool(), is_affirmative()
  room.py          RoomClient (httpx) + build_room_server() SDK MCP server
  room_stub.py     stdlib REST stub; entry point jarvis-room-stub
  brain.py         Brain: warm ClaudeSDKClient, ask() -> async text stream, idle reset, timeout
  voicelog.py      VoiceLog: wiki daily log + utterance archive (wav + jsonl)
  loop.py          VoiceLoop state machine (all deps injected)
  main.py          jarvis-voice entry point: build real deps, run loop
scripts/
  voice_wer.py     WER over the utterance archive vs hand-corrected transcripts
  install_voice.sh one-shot install (venv, deps, models, LaunchAgent/systemd)
  com.jarvis.voice.plist.template
models/wakeword/.gitkeep         drop-in wake word models (gitignored *.onnx)
config.voice.example.yaml
docs/voice/room-api.md  docs/voice/DEPLOY.md
tests/voice/ test_*.py per module
```

---

### Task 1: Package skeleton, config, deps

**Files:**
- Create: `jarvis/voice/__init__.py`, `jarvis/voice/config.py`, `config.voice.example.yaml`, `models/wakeword/.gitkeep`, `tests/voice/__init__.py`, `tests/voice/test_config.py`
- Modify: `pyproject.toml` (add `voice` extra + `jarvis-voice`, `jarvis-room-stub` scripts), `.gitignore`

**Interfaces:**
- Produces: `VoiceConfig` (frozen dataclass) with fields: `vault: Path`, `model: str = "claude-opus-5-5"`, `wake_threshold: float = 0.5`, `wake_models_dir: Path = Path("models/wakeword")`, `stt_model: str = "mlx-community/whisper-large-v3-turbo"`, `vocab_file: Path | None = Path("vocab.local.txt")`, `self_chat_id: str | None = None`, `tts_voice: str = "af_heart"`, `kokoro_model: Path`, `kokoro_voices: Path`, `smart_turn_model: Path`, `room_api_url: str = "http://127.0.0.1:8790"`, `followup_s: float = 8.0`, `hard_silence_s: float = 3.0`, `working_tone_after_s: float = 3.0`, `turn_timeout_s: float = 180.0`, `idle_reset_s: float = 600.0`, `data_dir: Path = ~/.jarvis/voice`, `input_device: str | int | None = None`, `output_device: str | int | None = None`.
- Produces: `load_config(example: Path, local: Path | None) -> VoiceConfig` — local YAML keys override example keys; `~` expanded; unknown key raises `ValueError` naming it.

- [ ] **Step 1: Write failing tests** in `tests/voice/test_config.py`:
  - `test_local_overrides_example`: example sets `followup_s: 8.0`, local sets `followup_s: 5.0` -> `cfg.followup_s == 5.0`, other fields from example.
  - `test_missing_local_uses_example`: `load_config(ex, None)` works.
  - `test_unknown_key_raises`: local has `folowup_s` -> `ValueError` containing `"folowup_s"`.
  - `test_paths_expand_user`: `vault: ~/x` -> `cfg.vault == Path.home() / "x"`.
- [ ] **Step 2: Run** `.venv-voice/bin/pytest tests/voice/test_config.py -v` -> FAIL (module missing).
- [ ] **Step 3: Implement** `config.py`; write `config.voice.example.yaml` with every field and a placeholder vault (`~/path/to/your/vault`). Add to pyproject: `voice = ["claude-agent-sdk==0.2.174", "openwakeword", "onnxruntime", "sounddevice", "numpy", "mlx-whisper; sys_platform == 'darwin'", "transformers", "kokoro-onnx", "httpx", "pyyaml", "pytest", "pytest-asyncio"]`. `.gitignore` add: `.venv-voice/`, `config.voice.local.yaml`, `models/wakeword/*.onnx`, `models/**/*.onnx`, `models/**/*.bin`, `vocab.local.txt`.
- [ ] **Step 4: Install + run** `uv pip install -p .venv-voice -e ".[voice]"` then `.venv-voice/bin/pytest tests/voice -v` -> PASS. Also run the existing suite with the existing interpreter to confirm nothing broke.
- [ ] **Step 5: Commit** `feat(voice): package skeleton, config, voice extras`

### Task 2: Audio I/O and wake word

**Files:**
- Create: `jarvis/voice/audio.py`, `jarvis/voice/wakeword.py`, `tests/voice/test_wakeword.py`

**Interfaces:**
- Produces: `class MicStream` — `async def frames() -> AsyncIterator[np.ndarray]` (int16, shape (1280,)); `mute(seconds_tail: float = 0.3)` context manager: frames arriving while muted and for the tail are dropped. Built on `sounddevice.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1280)` with `loop.call_soon_threadsafe` into an `asyncio.Queue`.
- Produces: `class Player` — `async def play(samples: np.ndarray, sr: int)`; `stop()`. `def tone(kind: Literal["wake","working","error"]) -> tuple[np.ndarray, int]` (synthesized sine blips, no asset files).
- Produces: `class WakeWord(models_dir: Path, threshold: float, prebuilt: list[str] = ["hey_jarvis"])` — `process(frame: np.ndarray) -> str | None` (name of the model that crossed threshold); `reset()` rebuilds the underlying `openwakeword.model.Model` (its `reset()` does not clear the buffer); `names: list[str]`. Uses `inference_framework="onnx"`; prebuilt paths via `openwakeword.get_pretrained_model_paths("onnx")` after `openwakeword.utils.download_models()` if missing.

- [ ] **Step 1: Write failing tests** (`test_wakeword.py`):
  - `test_dropin_models_discovered`: copy the prebuilt hey_jarvis onnx into a tmp dir as `hey_claude.onnx` -> `WakeWord(tmp, 0.5).names` contains `"hey_jarvis"` and `"hey_claude"`.
  - `test_silence_never_fires`: 200 frames of zeros -> `process` always `None`.
  - `test_fires_on_synthesized_phrase`: generate "hey jarvis" with macOS `say -o x.aiff --data-format=LEI16@16000` (skip test if `say` absent), feed frames -> some frame returns `"hey_jarvis"`.
  - `test_reset_clears_state`: after firing, `reset()` then 50 zero frames -> no fire.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** `audio.py`, `wakeword.py`.
- [ ] **Step 4: Run** `.venv-voice/bin/pytest tests/voice/test_wakeword.py -v` -> PASS.
- [ ] **Step 5: Commit** `feat(voice): mic/player and drop-in wake word`

### Task 3: End-of-utterance (VAD + smart-turn)

**Files:**
- Create: `jarvis/voice/turn.py`, `tests/voice/test_turn.py`

**Interfaces:**
- Produces: `class SmartTurn(model_path: Path)` — `predict(audio: np.ndarray) -> float` (completion probability). Keeps last 8 s, `WhisperFeatureExtractor(chunk_length=8)` with `sampling_rate=16000, max_length=128000, padding="max_length", truncation=True, do_normalize=True, return_tensors="np"`, ONNX input `"input_features"` float32, output `outputs[0][0]` (already sigmoid). Model file: `smart-turn-v3.1.onnx` from `pipecat-ai/smart-turn-v3` (downloaded by install script).
- Produces: `class UtteranceRecorder(vad, smart_turn, hard_silence_s=3.0, start_timeout_s: float, pause_check_s=0.4)` — `async def record(frames: AsyncIterator[np.ndarray]) -> np.ndarray | None`. Returns None if no speech starts within `start_timeout_s`. After speech starts: on each pause >= `pause_check_s` run smart-turn on audio so far; end if prob >= 0.5; always end after `hard_silence_s` of silence. `vad` is any object with `is_speech(frame) -> bool`; real one = `SileroVAD` class in the same file wrapping openWakeWord's bundled silero onnx (`openwakeword.VAD_MODELS`) with threshold 0.5.

- [ ] **Step 1: Write failing tests** with a `FakeVAD` (speech flags from a list) and `FakeTurn` (returns scripted probs):
  - `test_no_speech_returns_none`: all-silence frames, `start_timeout_s=1.0` -> `None`.
  - `test_pause_with_incomplete_turn_keeps_listening`: speech, 0.5 s pause (turn prob 0.1), more speech, pause (prob 0.9) -> returned audio length covers both speech spans.
  - `test_hard_silence_ends_turn`: speech then 3.0 s silence with turn prob always 0.1 -> returns at ~3.0 s of silence.
  - `test_smartturn_real_model_smoke` (skip if model file absent): `predict(np.zeros(16000, int16))` returns float in [0,1].
- [ ] **Step 2: Run** -> FAIL. **Step 3: Implement.** **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(voice): VAD + smart-turn utterance recorder`

### Task 4: STT with vocab priming and hallucination guard

**Files:**
- Create: `jarvis/voice/stt.py`, `tests/voice/test_stt.py`

**Interfaces:**
- Produces: `class STT(Protocol): def transcribe(self, audio: np.ndarray) -> str | None`.
- Produces: `class WhisperSTT(model: str, initial_prompt: str | None)` — runs `mlx_whisper.transcribe(audio_float32, path_or_hf_repo=model, language="en", initial_prompt=..., condition_on_previous_text=False)`, returns `clean_transcript(result)`.
- Produces: `clean_transcript(result: dict) -> str | None` — None when stripped text is empty, or every segment has `no_speech_prob > 0.6`, or the normalized text (lowercase, no punctuation) is in `PHANTOMS = {"thank you for watching", "thanks for watching", "you", "thank you", "bye", "subtitles by the amaraorg community"}`.
- Produces: `load_vocab_prompt(path: Path | None) -> str | None` — one term per line, `#` comments and blanks skipped, joined as `"Vocabulary: a, b, c."`; None if path missing.

- [ ] **Step 1: Write failing tests**: `test_phantom_dropped` ("Thank you for watching." -> None), `test_high_no_speech_dropped`, `test_real_text_kept` ("Turn on the fan." -> same), `test_vocab_prompt_format` (file with comment + 2 terms -> `"Vocabulary: widgetPose, Smith."`), `test_vocab_missing_none`, and `test_whisper_smoke` (skip if not darwin/arm64): `say "add tennis to my calendar on Sunday"` -> transcript contains "tennis" and "Sunday".
- [ ] **Step 2: Run** -> FAIL. **Step 3: Implement.** **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(voice): whisper STT with vocab priming and phantom guard`

### Task 5: Speakable text, sentence streaming, TTS

**Files:**
- Create: `jarvis/voice/speech.py`, `tests/voice/test_speech.py`

**Interfaces:**
- Produces: `to_speakable(text: str) -> str` — removes code fences (replaced with "I put the code in the log."), inline backticks, markdown emphasis/headers/list bullets, converts `[label](url)` to `label`, bare URLs to "a link".
- Produces: `class SentenceSplitter` — `feed(delta: str) -> list[str]` returns completed sentences (split on `. ! ?` followed by whitespace, or newline; does not split "e.g.", "3.5", "Dr."); `flush() -> list[str]`.
- Produces: `class TTS(Protocol): def synth(self, text: str) -> tuple[np.ndarray, int]`; `KokoroTTS(model: Path, voices: Path, voice="af_heart")` via `kokoro_onnx.Kokoro(...).create(text, voice=voice, speed=1.0, lang="en-us")`.

- [ ] **Step 1: Write failing tests**: `test_strips_markdown` ("**Due** tomorrow:\n- Report" -> no `*`, no `-` bullet), `test_code_fence_replaced`, `test_link_label_kept`, `test_splitter_streams` (feed "Hi the" , "re. Report is du", "e. Ok" -> yields "Hi there." then "Report is due."; flush -> "Ok"), `test_splitter_keeps_decimals` ("It's 3.5 hours." one sentence), `test_kokoro_smoke` (skip if model absent): returns non-empty float array, sr 24000.
- [ ] **Step 2: Run** -> FAIL. **Step 3: Implement.** **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(voice): speakable text, sentence streaming, kokoro TTS`

### Task 6: Permission policy and confirmations

**Files:**
- Create: `jarvis/voice/permissions.py`, `tests/voice/test_permissions.py`

**Interfaces:**
- Produces: `classify_tool(name: str, tool_input: dict, vault: Path, self_chat_id: str | None = None) -> Literal["allow", "confirm"]`:
  - `confirm`: `Bash`; `Write`/`Edit`/`NotebookEdit` whose `file_path` resolves outside `vault`; any MCP tool whose name contains (case-insensitive) one of `send_message`, `reply`, `forward`, `share_file`, `delete`, `trash`, `respond_to_event`, `cancel`. (`create_draft` stays `allow`: a draft never leaves the account.)
  - Exception: iMessage `reply` with `tool_input.get("chat_id") == self_chat_id` (the user's own chat) is `allow`.
  - Everything else, including `mcp__room__*`, Read/Grep/Glob, wiki writes: `allow`.
- Produces: `is_affirmative(transcript: str | None) -> bool` — normalized text must equal one of `{"yes", "yeah", "yep", "yes please", "do it", "send it", "go ahead", "confirm", "yes send it", "yes do it"}`. Anything else -> False.
- Produces: `describe_action(name: str, tool_input: dict) -> str` — one-sentence read-back, e.g. Gmail send -> "Send an email to {to} with subject {subject}?"; Bash -> "Run the shell command: {command}?"; fallback "Use {tool short name}?".

- [ ] **Step 1: Write failing tests**: table-driven `test_classify` covering Bash->confirm, Edit inside vault->allow, Edit `/Users/x/Code/a.py`->confirm, `mcp__claude_ai_Gmail__send_message`->confirm, `mcp__claude_ai_Gmail__create_draft`->allow, `mcp__room__set_state`->allow, iMessage reply to self->allow, to other->confirm; `test_affirmative` table: "Yes." True, "yeah" True, "yes no wait" False, "" False, None False, "no" False, "send it" True; `test_describe_gmail`.
- [ ] **Step 2: Run** -> FAIL. **Step 3: Implement.** **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(voice): voice permission policy and confirmation parsing`

### Task 7: Brain (warm Agent SDK session)

**Files:**
- Create: `jarvis/voice/brain.py`, `tests/voice/test_brain.py`

**Interfaces:**
- Consumes: `classify_tool`, `describe_action` (Task 6); `build_room_server` (Task 8 — inject as `extra_mcp: dict` so Task 7 doesn't depend on it).
- Produces: `class Brain(cfg: VoiceConfig, confirm: Callable[[str], Awaitable[bool]], extra_mcp: dict | None = None, client_factory=ClaudeSDKClient)`:
  - `async start()` — connect a client now (warm); remembers `session_id` from the first `ResultMessage`/init and touches `/tmp/claude-skip-synthesis/<session_id>`.
  - `async ask(text: str) -> AsyncIterator[str]` — text deltas (uses `include_partial_messages=True`; yield `TextBlock` text from `StreamEvent` deltas, fall back to `AssistantMessage` TextBlocks if no partials). Raises `BrainTimeout` after `cfg.turn_timeout_s`; on timeout/crash the client is disconnected and a fresh one started in the background.
  - `async maybe_reset()` — if idle >= `cfg.idle_reset_s`, swap in a fresh warm client. `async reset()` forces it ("new conversation").
  - `can_use_tool` callback: `classify_tool(...) == "allow"` -> `PermissionResultAllow()`; else `await confirm(describe_action(...))` -> allow / `PermissionResultDeny(message="Declined by voice.")`.
  - Options: `cwd=cfg.vault`, `setting_sources=["user","project","local"]`, `model=cfg.model`, `system_prompt={"type":"preset","preset":"claude_code","append": VOICE_PROMPT}`, `mcp_servers=extra_mcp`.
  - `VOICE_PROMPT` (exact copy): "You are being spoken to through a voice interface in the user's room and your reply will be read aloud. Answer in short, natural spoken sentences. No markdown, lists, tables, code or URLs. Say numbers and times the way a person would. If the answer is long (a plan, a diff, a table, more than about four sentences), write it to the wiki or send it to the user's phone over iMessage, then say in one sentence where you put it. Room devices are controlled with the room tools."
- Produces: `class BrainTimeout(Exception)`.

- [ ] **Step 1: Write failing tests** with a `FakeClient` (scripted messages, records `query` calls, can hang or raise):
  - `test_streams_text_deltas`: fake yields two partial deltas -> `ask` yields both strings in order.
  - `test_timeout_raises_and_recycles`: fake hangs; `turn_timeout_s=0.1` -> `BrainTimeout`; next `ask` uses a new fake client instance.
  - `test_idle_reset_swaps_client`: advance an injected clock past `idle_reset_s` -> `maybe_reset()` creates a new client.
  - `test_permission_confirm_path`: invoke the options' `can_use_tool("Bash", {"command":"ls"}, ctx)` with confirm returning False -> `PermissionResultDeny`; with `("mcp__room__set_state", {...})` -> Allow without calling confirm.
  - `test_skip_flag_written`: after start with session id "abc" -> file `<tmp>/claude-skip-synthesis/abc` exists (flag dir injectable for tests).
  - `test_live_sdk_smoke` (marked `live`, skipped unless `JARVIS_LIVE=1`): real client, ask "reply with just: ok" -> joined text contains "ok".
- [ ] **Step 2: Run** -> FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests/voice/test_brain.py -v` -> PASS; then `JARVIS_LIVE=1 pytest -m live tests/voice/test_brain.py` -> PASS.
- [ ] **Step 5: Commit** `feat(voice): warm Agent SDK brain with voice permissions`

### Task 8: Room API contract, client tools, stub server

**Files:**
- Create: `docs/voice/room-api.md`, `jarvis/voice/room.py`, `jarvis/voice/room_stub.py`, `tests/voice/test_room.py`

**Interfaces:**
- Contract (write into `room-api.md`): `GET /devices` -> `[{"id","name","kind","state"}]`; `GET /devices/{id}` -> device; `POST /devices/{id}` body = JSON state patch -> updated device; 404 `{"error":"unknown device"}`; 400 `{"error": "..."}` on a bad patch. No auth, LAN only. Stub devices: `fan` {on: bool, speed: 1-3}, `window` {open_pct: 0-100}, `thermostat` {target_f: 60-85, mode: "cool"|"heat"|"off"}, `speakers` {playing: bool, query: str, volume: 0-100}.
- Produces: `RoomClient(base_url)` with async `list_devices()`, `get_state(id)`, `set_state(id, patch: dict)` (httpx); `build_room_server(base_url) -> McpSdkServerConfig` via `create_sdk_mcp_server(name="room", tools=[...])` with tools `list_devices`, `get_state(device)`, `set_state(device, patch)`; HTTP errors become tool results with `is_error=True` and the server's message (never raise into the SDK).
- Produces: `room_stub.main()` -> `ThreadingHTTPServer` on `127.0.0.1:8790` (`--port`), in-memory state, validates ranges, logs each call to stderr; console script `jarvis-room-stub`.

- [ ] **Step 1: Write failing tests**: start stub on an ephemeral port in a thread; `test_list_devices` (4 devices), `test_set_fan_on`, `test_bad_patch_400` (speed 9), `test_unknown_404`, `test_tool_error_is_result` (call `set_state` tool handler for "toaster" -> result `is_error` True, text contains "unknown device").
- [ ] **Step 2: Run** -> FAIL. **Step 3: Implement.** **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(voice): room API contract, MCP room tools, stub server`

### Task 9: Voice log and the state machine

**Files:**
- Create: `jarvis/voice/voicelog.py`, `jarvis/voice/loop.py`, `tests/voice/test_voicelog.py`, `tests/voice/test_loop.py`

**Interfaces:**
- Produces: `VoiceLog(vault: Path, data_dir: Path, clock=datetime.now)` — `log_turn(heard: str, said: str)` appends `- HH:MM **heard:** ... / **said:** ...` to `<vault>/wiki/voice-log/YYYY-MM-DD.md` (creates with `# Voice log YYYY-MM-DD` header); `archive_utterance(audio: np.ndarray, transcript: str | None) -> Path` writes `<data_dir>/utterances/<ts>.wav` and appends `{"ts","wav","transcript","corrected": null}` to `<data_dir>/utterances.jsonl`.
- Produces: `VoiceLoop(mic, player, wake, recorder, stt, brain, tts, log, cfg)` — `async run()` forever; `async handle_turn(audio) -> None` exposed for tests. States: IDLE -> on wake: `wake.reset()`, play wake tone -> LISTEN (`recorder.record`, `start_timeout_s=6`) -> STT (archive every utterance; None -> back to IDLE silently) -> THINK (`brain.maybe_reset()`; if transcript normalized is "new conversation" -> `brain.reset()`, say "Okay, fresh start.") -> stream `brain.ask` through `SentenceSplitter` -> `to_speakable` -> `tts.synth` -> `player.play` under `mic.mute(0.3)`; working tone once if no first sentence by `working_tone_after_s` -> after speaking: FOLLOWUP = `recorder.record(start_timeout_s=cfg.followup_s)`; wake word also accepted in follow-up -> None -> IDLE.
  - `BrainTimeout`/exception -> play error tone + say "Sorry, that took too long." / "Sorry, something broke: {short type name}." -> IDLE.
  - The `confirm(question)` callable given to Brain is `loop.confirm`: speak question under mute, `recorder.record(start_timeout_s=6)`, STT, `is_affirmative`; speaks "Cancelled." when False.

- [ ] **Step 1: Write failing tests** with fakes for every dep (fake mic yields scripted frames and records mute windows; fake player records played items; fake brain yields scripted deltas or raises):
  - `test_wake_to_answer`: wake fires -> STT "what's due" -> brain yields "Report is due. Bye." -> player got wake tone then two synthesized sentences; log has the turn.
  - `test_phantom_transcript_no_brain_call`: STT None -> brain.ask never called.
  - `test_followup_without_wake`: after an answer, the recorder returns follow-up audio -> second brain.ask call without a wake event.
  - `test_timeout_speaks_error_and_idles`: brain raises `BrainTimeout` -> error tone + "took too long" spoken; loop back to IDLE.
  - `test_mic_muted_while_speaking`: every `player.play` happened inside a mute window (Review Focus 5).
  - `test_confirm_declines_on_unclear`: confirm's STT returns "yes no wait" -> returns False, "Cancelled." spoken.
  - `test_new_conversation_resets`.
  - voicelog: `test_daily_file_created_with_header`, `test_archive_appends_jsonl`.
- [ ] **Step 2: Run** -> FAIL. **Step 3: Implement.** **Step 4: Run** `.venv-voice/bin/pytest tests/voice -v` -> all PASS.
- [ ] **Step 5: Commit** `feat(voice): voice loop state machine and voice log`

### Task 10: Entry point, live run on the Mac

**Files:**
- Create: `jarvis/voice/main.py`, `scripts/fetch_voice_models.sh`
- Modify: `README.md` (Voice section: install, run, drop-in wake word, room stub, privacy note)

**Interfaces:**
- Produces: `main()` — args `--config config.voice.example.yaml --local config.voice.local.yaml --list-devices --no-room`; builds real deps, starts `brain.start()` before listening, logs state transitions to stderr as `[voice] IDLE -> LISTEN`.
- `fetch_voice_models.sh` downloads into `models/` (gitignored): `kokoro-v1.0.onnx` + `voices-v1.0.bin` (kokoro-onnx GitHub release), `smart-turn-v3.1.onnx` (HF `pipecat-ai/smart-turn-v3`), openWakeWord prebuilt models; optionally `--hey-claude` fetches `hey_claude.onnx` from HF `gdiamos/hey-claude` into `models/wakeword/`.

- [ ] **Step 1:** Implement `main.py` and the fetch script; run `./scripts/fetch_voice_models.sh`.
- [ ] **Step 2: Live check (the user present):** `jarvis-room-stub` in one terminal, `.venv-voice/bin/jarvis-voice --local config.voice.local.yaml` in another. Verify by voice: "hey Jarvis, turn on the fan" (stub logs POST /devices/fan), "hey Jarvis, what's due tomorrow" (canvas skill answer), a follow-up without the wake word, "hey Jarvis, send an email to myself saying test" (spoken confirmation; answer "no" -> cancelled). Record what failed.
- [ ] **Step 3:** Fix what Step 2 found (each fix gets its test first).
- [ ] **Step 4: Commit** `feat(voice): jarvis-voice entry point, model fetch, README`

### Task 11: WER measurement and deploy kit

**Files:**
- Create: `scripts/voice_wer.py`, `tests/voice/test_wer.py`, `scripts/install_voice.sh`, `scripts/com.jarvis.voice.plist.template`, `docs/voice/DEPLOY.md`

**Interfaces:**
- Produces: `wer(ref: str, hyp: str) -> float` (word-level Levenshtein / ref words, case/punct-insensitive); CLI `voice_wer.py <data_dir>` prints overall WER over rows with `corrected` set, count, and the 10 most-missed reference words (that's where vocab priming goes next).
- `install_voice.sh` (macOS + Linux branches, single-line commands): create `.venv-voice`, install `.[voice]` (Linux: `faster-whisper` instead of mlx-whisper — documented, STT class swap noted as follow-up), fetch models, install LaunchAgent (`~/Library/LaunchAgents`, `KeepAlive`, runs in the user session so mic permission applies) or a systemd user unit.
- `DEPLOY.md`: Mac mini path; Orin path incl. prerequisite private vault git mirror, `claude` login on the box, mic permission note (grant to the venv's python in System Settings > Privacy > Microphone, or audio is silent with no error), iMessage unavailable on Linux.

- [ ] **Step 1: Write failing tests**: `test_wer_identical_zero`, `test_wer_one_sub` ("turn on the fan" vs "turn on the van" -> 0.25), `test_wer_ignores_case_punct`.
- [ ] **Step 2: Run** -> FAIL. **Step 3: Implement** script, installer, template, docs. **Step 4: Run** -> PASS; run `bash -n scripts/install_voice.sh`.
- [ ] **Step 5: Commit** `feat(voice): WER tool and deploy kit`; push to `origin main`.
