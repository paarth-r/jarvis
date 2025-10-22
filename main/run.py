import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import pyautogui
import time

# === CONFIG ===
MODEL_PATH = "/Users/paarth/Desktop/everything/tonystark/main/models/gesture_tcn.pt"
SEQ_LEN = 30
CONF_THRESH = 0.75
GESTURE_CLASSES = ["initialize"]  # update with your labels

# === MODEL DEFINITION ===
class TCN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, dilation=4, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = x.transpose(1, 2)  # [B, F, T]
        return self.net(x)

# === LOAD MODEL ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TCN(n_features=63, n_classes=len(GESTURE_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === MEDIAPIPE SETUP ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# === BUFFER & STATE ===
seq_buffer = deque(maxlen=SEQ_LEN)
last_action_time = 0
cooldown = 1.0  # seconds between actions
current_pred = "..."

# === ACTION MAP ===
def perform_action(gesture):
    global last_action_time
    if time.time() - last_action_time < cooldown:
        return
    print(f"Action: {gesture}")

    if gesture == "initialize":
        pyautogui.hotkey("command", "space")  # raycast
    # elif gesture == "swipe_right":
    #     pyautogui.hotkey("ctrl", "left")  # previous app

    last_action_time = time.time()

# === MAIN LOOP ===
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        landmarks = np.array([[p.x, p.y, p.z] for p in lm.landmark])
        landmarks -= landmarks[0]  # wrist normalization
        seq_buffer.append(landmarks.flatten())

        if len(seq_buffer) == SEQ_LEN:
            with torch.no_grad():
                inp = torch.tensor(np.expand_dims(seq_buffer, axis=0), dtype=torch.float32).to(device)
                out = model(inp)
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()
                pred_idx = np.argmax(probs)
                conf = probs[pred_idx]

                if conf > CONF_THRESH:
                    gesture = GESTURE_CLASSES[pred_idx]
                    current_pred = f"{gesture} ({conf:.2f})"
                    perform_action(gesture)
    else:
        seq_buffer.clear()

    # === DISPLAY ===
    cv2.putText(frame, current_pred, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()