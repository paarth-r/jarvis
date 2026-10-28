# Plan: duck music while Jarvis is engaged, and stop-word barge-in

Spec: agreed in conversation on 2026-10-27 (no separate spec file). Summary below is binding.

## Context

`jarvis-voice` (package `jarvis/voice/`, tests `tests/voice/`, run with
`.venv-voice/bin/python -m pytest tests/voice -q`) is a wake-word voice front-end for
headless Claude Code. State machine in `jarvis/voice/loop.py` (`VoiceLoop`):
IDLE -> LISTEN -> STT -> THINK/SPEAK -> FOLLOWUP -> IDLE. Every dependency is
injected; tests use fakes (`tests/voice/test_loop.py`). Playback: `Player` in
`jarvis/voice/audio.py`. Today all playback happens under `mic.mute(0.3)` so the
assistant never hears itself. Claude session: `Brain` in `jarvis/voice/brain.py`.
macOS app control is via `osascript` (`run_osascript` in `jarvis/voice/mac_speakers.py`).
Config: frozen dataclass `VoiceConfig` in `jarvis/voice/config.py`, defaults mirrored
in `config.voice.example.yaml` (unknown keys are rejected).

## Global Constraints

- Jarvis speaks at the Mac's current system output volume. NEVER change the system
  volume for ducking. Duck other streams by lowering the app's own volume
  (`sound volume of application "Spotify"` / `"Music"`, 0-100).
- Ducking covers the whole engaged conversation: from wake-word detection until the
  conversation returns to IDLE (listening, thinking, speaking, follow-up). Duck to
  `duck_ratio` (default 0.2) of the app's previous level, minimum 1 if it was > 0.
- Restore exactly the pre-duck level. If the user changed that app's volume while
  ducked (current level != the level we set), do not overwrite their choice. Restore
  must run on every exit path (exceptions, Ctrl-C), and a crashed process must not
  leave music stuck low: persist pre-duck levels to `<data_dir>/duck.json` while
  ducked, and restore + delete it at startup if present.
- Only touch apps that are already running (`application "X" is running` does not
  launch them). Ducking failures are logged and never fail a turn.
- Restore fades back up (a few steps over about 0.5 s); ducking is immediate.
- Barge-in is stop-word based (no echo cancellation in this plan). Default phrases:
  "wait", "no", "stop", "hold up", "hang on", "hold on", "nevermind", "never mind",
  "jarvis". Configurable as `stop_phrases`. `barge_in: true` by default; false keeps
  today's muted playback exactly.
- Echo rule: a stop phrase heard during playback counts only if it does not appear
  (as whole words, normalized: lowercase, punctuation stripped) in the sentence
  currently being spoken.
- On barge-in: stop playback immediately, stop the rest of Claude's reply WITHOUT
  losing the Claude session (today abandoning `Brain.ask` recycles the client into a
  brand-new session id, which wipes context: that must not happen on barge-in), and
  tell Claude on the next turn roughly where it was cut off, based on what was
  actually played (fraction of the clip's samples played), not what was generated.
- Every task: tests first, full voice suite green before commit, README updated in
  the same commit as user-visible behaviour, conventional commit message ending with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Single-line shell
  commands only. No emojis anywhere.

## Task 1: `MusicDucker` module

Create `jarvis/voice/duck.py` and `tests/voice/test_duck.py`.

- `class MusicDucker(osascript=run_osascript, apps=("Spotify", "Music"), ratio=0.2,
  state_path: Path | None = None, sleep=time.sleep)`. All methods synchronous (callers
  use `asyncio.to_thread`).
- `duck()`: idempotent. For each app not already ducked: if running, read
  `sound volume of application "<app>"`; if > 0, set it to `max(1, round(v*ratio))`
  and remember `(previous, set_to)`. After any change, write the remembered map to
  `state_path` as JSON (if given).
- `restore()`: for each ducked app, read its current volume; if it equals `set_to`,
  fade back to `previous` in 4 steps with `sleep(0.12)` between; otherwise leave it.
  Forget the app, and delete `state_path` once nothing is ducked. An app that quit
  meanwhile is skipped silently.
- `recover()`: if `state_path` exists from a previous crashed run, load it and run the
  same restore logic, then delete the file. Corrupt file: delete and log.
- Every osascript failure (`SpeakerError` or any `Exception`) is logged via
  `logging.getLogger(__name__)` and swallowed per app.
- Tests use a fake osascript that models app running state and volumes, and a fake
  sleep. Cover: duck/restore round trip, idempotent duck, not-running app untouched,
  zero volume untouched, user-changed volume not overwritten, fade steps end at the
  exact previous value, state file written/deleted, recover() after a simulated
  crash, osascript failure swallowed.

## Task 2: Wire ducking into the loop, main and config

- `VoiceConfig`: add `duck_music: bool = True`, `duck_ratio: float = 0.2`; mirror in
  `config.voice.example.yaml` with a comment.
- `VoiceLoop.__init__` gets an optional `ducker: Any = None` (None = no ducking, so
  existing tests are unaffected). In `_conversation`: duck as soon as the wake word
  fired, concurrently with the wake tone (do not delay the tone on osascript), and
  restore in a `finally` around the whole conversation. Also call `duck()` again
  before each spoken sentence, concurrently with TTS synthesis, so music Claude
  started mid-turn is ducked too (duck is idempotent). All via `asyncio.to_thread`,
  exceptions swallowed.
- `main.run`: build `MusicDucker(state_path=cfg.data_dir / "duck.json",
  ratio=cfg.duck_ratio)` when `cfg.duck_music`, call `recover()` at startup, and
  `restore()` in the shutdown `finally`.
- README voice section: describe ducking, the config keys, and the limit that only
  Spotify and Apple Music can be ducked (macOS has no per-app volume API).
- Loop tests with a fake ducker: ducked during the turn, restored after, restored
  when the turn raises, re-ducked before sentences.

## Task 3: `Brain.interrupt()` keeps the session

- Add `async def interrupt(self) -> None` to `Brain`: marks the in-flight `ask` as
  interrupted. When an `ask` generator that was interrupted is closed early, it calls
  the SDK client's `interrupt()` and drains `receive_response()` until the
  `ResultMessage` (bounded by a short timeout, e.g. 5 s), and does NOT recycle the
  client, so `session_id` and context survive. If the interrupt or drain fails or
  times out, fall back to today's recycle.
- An early close without `interrupt()` keeps today's behaviour (recycle).
- Tests in `tests/voice/test_brain.py` with the existing fake client patterns: an
  interrupted ask keeps the same client and session id and calls the client's
  interrupt; a drain timeout recycles; an un-interrupted abandon still recycles.

## Task 4: Player stop reporting and stop-phrase listener

- `Player.play()` returns the fraction of the clip that was played (1.0 when it ran
  to the end, less when `stop()` cut it). Existing callers ignore the return value.
  Test with the existing fake stream factory in `tests/voice/test_wakeword.py` style.
- Create `jarvis/voice/bargein.py` and `tests/voice/test_bargein.py`:
  - `find_stop_phrase(transcript, speaking, phrases) -> str | None`: whole-word,
    normalized match of any phrase in the transcript that is not also in `speaking`.
  - `class BargeInListener(vad, stt, phrases, min_speech_s=0.3, window_s=2.0,
    step_s=0.5)` with `async def listen(frames, speaking: Callable[[], str],
    done: asyncio.Event) -> BargeIn | None`. It pulls frames from the shared async
    iterator until `done` is set (checked after every frame; it must NEVER cancel or
    `aclose` the iterator, which would stop the mic). While the VAD sees speech for
    at least `min_speech_s`, it transcribes the last `window_s` of audio every
    `step_s` (via `asyncio.to_thread`) and returns `BargeIn(phrase, audio)` as soon
    as `find_stop_phrase` hits, where `audio` is the speech audio captured since the
    speech began (so the rest of the user's sentence can be prepended later).
    Returns None when `done` is set or the iterator ends.
  - Tests with fake VAD/STT/frames: stop word fires, a stop word inside the current
    sentence is ignored (echo), short blips under `min_speech_s` never call STT,
    `done` ends listening without closing the iterator, iterator end returns None.

## Task 5: Wire barge-in into the loop

- `VoiceConfig`: `barge_in: bool = True`, `stop_phrases: tuple[str, ...]` with the
  defaults above (YAML list -> tuple); mirror in `config.voice.example.yaml`.
- `VoiceLoop` gets an optional `barge: Any = None` (None = today's muted behaviour;
  existing tests unaffected). When set, `_say_unlocked` plays without `mic.mute`,
  runs `barge.listen(self._stream(), lambda: <current sentence>, done)` concurrently
  with `player.play`, sets `done` when playback ends, and still applies a 0.3 s mute
  tail afterwards. On a hit: `player.stop()`, trace `[interrupt] <phrase>`, append the
  played portion of the sentence (by played fraction, word-rounded) to `self._said`,
  and raise an internal `Interrupted` carrying the captured audio.
- `_answer`/`_process`: on `Interrupted`, call `brain.interrupt()` before the ask
  generator is closed, skip any remaining sentences, log the turn with what was
  actually said, and return a new outcome that makes `_conversation` record the
  user's continuation (`_record` with `cfg.followup_s`), prepend the captured audio,
  and process it as the next turn. That next turn's text sent to Claude is prefixed
  with a short note: `(You were interrupted after saying: "<played portion>". The
  rest was not heard.) ` If the continuation transcript is only stop phrases
  ("wait", "hold up"), say nothing and open the follow-up window; existing cancel
  handling ("stop", "nevermind") still applies.
- `main.build_deps`/`run`: build `BargeInListener(SileroVAD(), stt, cfg.stop_phrases)`
  when `cfg.barge_in` and pass it to `VoiceLoop`.
- README: barge-in section (phrases, echo rule, that it is word-based so expect about
  half a second, config keys, interrupting while Claude is still thinking is not
  covered yet). Loop tests with fake barge listener and player: interrupt stops
  playback and remaining sentences, brain.interrupt called, prefix sent next turn,
  stop-word-only continuation opens follow-up, barge None keeps mute behaviour.
