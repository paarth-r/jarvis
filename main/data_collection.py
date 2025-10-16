import cv2, mediapipe as mp, numpy as np, os, re

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

GESTURE = "swipe_left"
SAVE_DIR = f"main/gesture_data/{GESTURE}"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Find largest existing file number ---
existing_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.npy')]
if existing_files:
    numbers = [int(re.findall(r'\d+', f)[0]) for f in existing_files if re.findall(r'\d+', f)]
    counter = max(numbers) + 1
else:
    counter = 0
# -----------------------------------------

cap = cv2.VideoCapture(1)
recording, seq = False, []

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
        # Normalize relative to wrist (point 0)
        landmarks -= landmarks[0]
        if recording:
            seq.append(landmarks.flatten())

    cv2.putText(frame, f"{GESTURE} | {'REC' if recording else '---'}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if recording else (255, 255, 255), 2)
    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recording, seq = True, []
    elif key == ord('s') and seq:
        filename = f"{SAVE_DIR}/{counter:04d}.npy"
        np.save(filename, np.array(seq))
        print(f"Saved {filename}")
        counter += 1
        recording = False
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()