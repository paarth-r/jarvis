# Hand Gesture Control

This repo has two ways to control your computer with your hands:

1. **Gesture control** – Train on specific gestures (swipe, flick, etc.) and trigger actions (Raycast, media keys).
2. **Mouse control** – Use your hand as a virtual mouse: index finger moves the cursor, thumb+pinky pinch to click (macOS only).

Both use MediaPipe for hand tracking. Gesture control adds a TCN neural network for recognizing learned gestures.

---

## 1. Gesture control

### What it does

- Captures hand landmarks from your webcam
- Recognizes gestures (e.g. initialize, swipe_L, flick_up) from recorded training data
- Triggers actions (Raycast, media keys, volume) when you make a gesture

### Setup

0. Go into the gesture control directory:
```bash
cd gesture_control
```

1. Install dependencies:
```bash
pip install opencv-python mediapipe torch numpy scikit-learn pyautogui pynput
```

2. Create a folder for each gesture inside `gesture_data`:
```bash
cd gesture_data && mkdir your_gesture_name
```
(Use a short name with no spaces, e.g. `flick_up`, `swipe_L`.)

3. Set the `GESTURE` variable in `data_collection.py` to the folder name you created.

4. Collect training data:
```bash
python data_collection.py
```
- Press **r** to start recording a gesture
- Make your gesture
- Press **s** to save it
- Repeat for many examples (do this for each gesture, changing `GESTURE` and folder as needed)

5. Normalize the data:
```bash
python normalize_recordings.py
```

6. Train the model:
```bash
python train.py
```

7. Add an action for your gesture in `utils.py`: add the gesture name to the `ACTIONS` dict and point it to a handler (e.g. for Raycast, media, volume).

8. Run the system (from `gesture_control`):
```bash
python run.py
```
Gesture classes are read automatically from the subfolders in `gesture_data/`; you only need to add the action in `utils.py` for new gestures.

### Files

- `gesture_control/data_collection.py` – Records gesture sequences from webcam
- `gesture_control/train.py` – Trains the TCN model on collected data
- `gesture_control/run.py` – Main gesture recognition loop and inference
- `gesture_control/normalize_recordings.py` – Prepares data for training
- `gesture_control/utils.py` – Maps gesture names to actions (keyboard/media)
- `gesture_control/gesture_data/` – Recorded gesture sequences (one folder per gesture)
- `gesture_control/models/` – Saved trained models

### Adding new gestures

1. Create a new folder under `gesture_data/` (e.g. `gesture_data/my_gesture/`).
2. Set `GESTURE` in `data_collection.py` to that folder name and record examples.
3. Run `normalize_recordings.py` and `train.py` again.
4. In `utils.py`, add your gesture to the `ACTIONS` dict and implement or reuse an action handler.

No need to edit `run.py`; it discovers gesture classes from the `gesture_data/` folders.

---

## 2. Mouse control

**macOS only.** Uses the camera to track your **left hand** and inject mouse events via Quartz (CoreGraphics). No ML: cursor follows your index finger; a thumb–pinky pinch triggers a click.

### What it does

- **Move** – Cursor follows your index finger tip (with gain and deadzone).
- **Click** – Pinch thumb and pinky together to click. Optional **click halo**: if a button or link is near the click point, the app uses the Accessibility API to activate it so the click “snaps” to the control.
- **Overlay** – Hand skeleton, gesture state, FPS, and latency on the camera view.

### Setup

0. Go into the mouse control directory:
```bash
cd mouse_control
```

1. Install dependencies:
```bash
pip install opencv-python mediapipe pyobjc-framework-Quartz
```
(Quartz is required for mouse injection on macOS.)

2. Run (allow camera and, for click halo, Accessibility access if prompted):
```bash
python main.py
```
- Show your **left** hand to the camera. Cursor follows your index finger.
- Pinch thumb and pinky to click.
- Press **Esc** to quit.

If you see “Quartz unavailable – no input”, the Quartz framework could not be loaded; ensure you have PyObjC and a macOS environment.

### Files

- `mouse_control/main.py` – Entry point: camera → hand pose → gesture FSM → mouse control; debug overlay
- `mouse_control/camera.py` – Frame capture (1280×800), FPS and timestamps
- `mouse_control/hand_pose.py` – MediaPipe hand tracking, landmark smoothing, thumb–pinky distance for pinch
- `mouse_control/gestures.py` – Pinch FSM (pinch = one click, no drag)
- `mouse_control/control.py` – Quartz mouse move/click/scroll; optional click halo
- `mouse_control/click_halo.py` – macOS Accessibility API: find clickable element near point and perform AXPress

Pretty basic but it works for simple gestures.
