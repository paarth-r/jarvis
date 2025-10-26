# Hand Gesture Control

A simple hand gesture recognition system that lets you control your computer with hand movements. Uses MediaPipe for hand tracking and a neural network to recognize gestures.

## What it does

- Captures hand landmarks from your webcam
- Recognizes specific gestures (currently just "initialize")
- Triggers actions like opening Raycast when you make the gesture

## Setup

0. Make sure directory is correct:
```bash
cd main
```

1. Install dependencies:
```bash
pip install opencv-python mediapipe torch numpy scikit-learn pyautogui
```
2. Create a training folder inside of main/gesture_data:
```bash
cd gesture_data && mkdir GESTURE NAME
```

3. modify GESTURE variable in data_collection.py to match created folder

4. Collect training data:
```bash
python data_collection.py
```
- Press 'r' to start recording a gesture
- Make your gesture
- Press 's' to save it
- Repeat for different gestures

5. Normalize the data:
```bash
python normalize_recordings.py
```

6. Train the model:
```bash
python train.py
```

7. Add the name of any new gesture to 
```bash
GESTURE_CLASSES
```
list in
```bash
run.py
```

5. Run the model:
```bash
python run.py
```

## Files

- `data_collection.py` - Records gesture data from webcam
- `train.py` - Trains the TCN model on collected data
- `run.py` - Main gesture recognition and control loop
- `normalize_recordings.py` - Prepares data for training
- `gesture_data/` - Contains recorded gesture sequences
- `models/` - Saved trained models

## Adding new gestures

1. Change `GESTURE` in `data_collection.py` to your new gesture name
2. Record some examples of the gesture
3. Update `GESTURE_CLASSES` in `run.py`
4. Add the action in the `perform_action()` function
5. Retrain the model

Pretty basic but it works for simple gestures.
