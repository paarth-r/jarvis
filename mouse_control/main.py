"""
Orchestration + debug overlay.
Pipeline: camera -> hand_pose -> gesture FSM -> control.
Overlay: hand skeleton, gesture state, FPS, latency.
"""
import cv2
import time

from camera import Camera, Frame
from hand_pose import HandPoseEstimator, INDEX_TIP
from gestures import GestureFSM, GestureIntent
from control import MouseController, is_available as control_available

import mediapipe as mp

MP_HANDS = mp.solutions.hands
MP_DRAW = mp.solutions.drawing_utils


def draw_skeleton(frame, pose, mirror_x: bool = False) -> None:
    """Draw hand landmarks and connections on frame (in-place)."""
    if pose is None:
        return
    h, w = frame.shape[:2]
    # Landmarks are normalized 0-1; convert to pixel coords (mirror x if display is flipped)
    lm_px = []
    for i in range(pose.landmarks.shape[0]):
        x = pose.landmarks[i, 0]
        if mirror_x:
            x = 1.0 - x
        x = int(x * w)
        y = int(pose.landmarks[i, 1] * h)
        lm_px.append((x, y))
    # Draw connections (MediaPipe HAND_CONNECTIONS)
    for start_idx, end_idx in MP_HANDS.HAND_CONNECTIONS:
        if 0 <= start_idx < len(lm_px) and 0 <= end_idx < len(lm_px):
            cv2.line(
                frame,
                lm_px[start_idx],
                lm_px[end_idx],
                (0, 255, 0),
                2,
            )
    for pt in lm_px:
        cv2.circle(frame, pt, 4, (0, 255, 255), -1)


def run():
    camera = Camera(width=1280, height=800)
    if not camera.open():
        print("Failed to open camera")
        return
    hand_pose = HandPoseEstimator()
    gesture_fsm = GestureFSM()
    controller = MouseController()

    if not control_available():
        print("Quartz not available; mouse/scroll will not be injected.")

    fps_alpha = 0.1
    fps_est = 30.0
    latency_est_ms = 0.0
    t_start = time.perf_counter()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            t_after_capture = time.perf_counter()
            rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
            pose = hand_pose.process(rgb, frame.timestamp)
            intent = gesture_fsm.update(pose, frame.dt)
            t_after_gesture = time.perf_counter()

            # Control layer: mirror x so hand motion matches cursor (camera is front-facing)
            if pose is not None:
                ix, iy = float(pose.landmarks[INDEX_TIP, 0]), float(pose.landmarks[INDEX_TIP, 1])
                mx, my = 1.0 - ix, iy
                controller.move(mx, my)
                if intent.type == "click":
                    controller.click(mx, my)
            else:
                if controller.is_mouse_down:
                    controller.mouse_up(0.5, 0.5)

            # FPS and latency
            dt = frame.dt if frame.dt > 0 else (t_after_gesture - t_start)
            fps_est = fps_alpha * (1.0 / dt) + (1 - fps_alpha) * fps_est
            latency_est_ms = (t_after_gesture - t_after_capture) * 1000

            # Mirror image so display matches mirrored cursor (hand left = cursor left)
            display = cv2.flip(frame.image, 1)
            draw_skeleton(display, pose, mirror_x=True)
            state_str = gesture_fsm.pinch_state
            cv2.putText(
                display,
                f"Gesture: {intent.type}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                state_str,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                display,
                f"FPS: {fps_est:.1f}  Latency: {latency_est_ms:.1f} ms",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 255, 180),
                1,
            )
            if not control_available():
                cv2.putText(
                    display,
                    "Quartz unavailable - no input",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1,
                )

            cv2.imshow("Hand HCI", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
