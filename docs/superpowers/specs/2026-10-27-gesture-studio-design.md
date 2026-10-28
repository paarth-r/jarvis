# Gesture Studio — Design

Date: 2026-10-27
Status: approved 2026-10-27 ("take your best decision and go for it"; calls below are Claude's unless marked)

## Intent

Gesture control today is a set of independent per-frame detectors (pinch,
fist, point). Each one fires on its own, so passing *through* a pose fires it:
closing a fist brushes thumb to index (pinch -> click, held -> drag), and the
index finger lagging the others reads as pointing for a frame (Cmd+Space,
Spotlight). Adding a gesture means editing code; the legacy `gesture_control/`
TCN needs a full retrain and has no "not a gesture" class.

The rework: one place decides which gesture is active; new gestures are
added from a UI with a few takes and no training; the UI grades a candidate
gesture (recognizable, repeatable, distinct) before it can be saved; any
gesture binds to a media key, keyboard shortcut, macro or shell command.

## Goals

1. **Fist then tilt scrolls** (requested). Fist = scroll mode: cursor frozen,
   pinch locked out. Tilting the fist away from its starting angle scrolls.
2. **Arbitration.** At most one gesture active at a time. A gesture becomes
   active only after it has been held for a short dwell; transition poses
   never fire.
3. **Add gestures from a UI** (requested) with a live quality check: how
   recognizable, distinct and repeatable the gesture is (requested).
4. **Similarity check** (requested): a new gesture that is confusable with an
   existing one, a built-in, or everyday transition poses is rejected.
5. **Binding flow** (requested): bind any gesture to media actions, key combos,
   multi-step macros; all from the UI.

## Non-goals (v1)

- Motion gestures (swipe, circle). v1 gestures are static hand shapes; fist
  scroll is the one continuous gesture. The template format leaves room for
  sequences later.
- Gaze mode (`gaze_hci`) keeps its own pinch handling.
- The legacy `gesture_control/` TCN stays as a reference, unused.

## Anti-goals

- A cursor that feels worse. Pinch-click latency must not grow; the dwell
  applies to discrete gestures, not to the pinch FSM.
- Gestures firing while you are recording one in the UI.
- Scores that look precise but mean nothing: every score has a plain
  explanation and a fix hint.

## Prior art used

- $1 recognizer family (Wobbrock et al.): template matching gets ~97% with
  one template, 99% with 3+. -> few-shot nearest-neighbour, no training.
- MediaPipe Model Maker gesture customization: requires a `none` class. ->
  every custom gesture has an acceptance radius; outside all radii = none.
- Xu et al. 2022 (wrist-worn customization, arXiv 2203.15239): enrollment
  rejects gestures similar to existing ones or to daily activity; asks for
  more shots when borderline. -> the distinct score includes a background clip.
- US 9092062 / US 6256033: similarity threshold at enrollment, suggest
  alternatives. -> the scorecard names the nearest gesture and which fingers
  separate them.

## Design (Claude's calls)

### Hand shape signal
`HandPoseModule` also emits MediaPipe's `hand_world_landmarks` (metric,
hand-centred, model-estimated depth) as `PoseEvent.world_landmarks`;
`StereoFusionModule` passes it through as `WorldPoseEvent.hand_shape`. Mono
projection flattens z, so shape features and tilt use `hand_shape`, never
`landmarks_3d`. Cursor code is untouched.

### Features (`jarvis/gestures/features.py`)
Rotation- and scale-invariant vector from `hand_shape`: 15 joint flexion
angles, 4 adjacent-finger spread angles, 4 thumb-tip-to-fingertip distances
normalized by palm width. `palm_frame()` gives the palm's orientation for tilt.

### Built-ins and custom gestures (`jarvis/gestures/`)
- Built-in rules: `pinch` (thumb-index distance; only when the hand is not
  fist-shaped), `fist`, `point`. Reserved: `pinch` = click/drag, `fist` =
  scroll. `point` is bindable (default Cmd+Space, as today).
- Custom gestures: a set of feature samples per gesture; nearest-neighbour
  distance; each gesture's acceptance radius comes from its own spread.
  Match requires distance < radius and a margin over the runner-up.
- `GestureArbiter`: candidate must lead for `dwell_s` (0.15 s) to activate;
  stays active until gone for `release_s` (0.1 s). While a gesture is
  active nothing else activates. Emits `GestureEvent(name, phase=start|hold|end)`.
  A fist activating cancels a pending pinch (no click).

### Fist-tilt scroll (`jarvis/gestures/tilt_scroll.py`)
On fist start, store the palm frame. Pitch delta (knuckles tipping up/down)
-> vertical scroll; the further the tilt the faster (rate control, like a
joystick), with a dead zone (8 deg) and a speed cap. Roll -> horizontal,
off by default to avoid crosstalk. Open hand ends it; the cursor re-anchors
where it was.

### Actions and bindings (`jarvis/gestures/actions.py`)
Action specs: `media` (play_pause, next, prev, volume_up/down, mute),
`keys` (`"cmd+shift+4"`), `text`, `shell`, `macro` (ordered steps of the
above plus `wait`). Optional `repeat_s` re-fires while held (volume).
Legacy dispatcher names (`raycast`, `media_next`, ...) map onto these.

### Library (`~/.jarvis/gestures.json`)
`{version, gestures: {name: {samples, radius, quality, created}},
bindings: {gesture: action}}`. The runtime reloads it when its mtime
changes, so UI edits apply live.

### Quality check (`jarvis/gestures/quality.py`)
Input: 5 takes (~1.5 s held each) + optional ~10 s background clip of
normal hand movement between existing gestures. Three scores, each
pass / warn / fail with an explanation and a hint:
- **Recognizable**: hand tracked in most frames; low frame-to-frame jitter.
- **Repeatable**: takes agree with each other (spread vs. scale).
- **Distinct**: separation from the nearest custom gesture relative to both
  radii; collision rate with the built-in rules (fist/point/pinch) over the
  takes; false-fire rate on the background clip. Names the nearest gesture
  and the fingers that differ most.
Fail blocks saving; warn allows it. Thresholds live in config and are
first guesses until measured on a real hand.

### Studio UI (`jarvis/studio/`)
An aiohttp server inside the runtime (127.0.0.1:8765; a module, so it
shares the camera). Single page, vanilla JS + canvas, no build step.
Sections: Live (skeleton, active gesture, per-gesture match bars, tilt
gauge), Gestures (list, scores, bindings, delete/retrain), Add (name ->
record takes -> background clip -> scorecard -> bind -> save), Binding
editor (media picker, key combo capture, macro steps, test run). While
recording, cursor control and actions are paused.

## Build order (each step committed and checkable)

1. `hand_shape` plumbing.
2. Features + synthetic hand helper for tests.
3. Arbiter + built-ins wired into `GestureHCIModule`; point via dwell
   (fixes fist -> click/drag/Spotlight).
4. Fist-tilt scroll.
5. Actions, bindings, library, dispatcher on `GestureEvent`.
6. Custom gesture matching + hot reload.
7. Quality scoring.
8. Studio server + API.
9. Studio UI.
10. README.
