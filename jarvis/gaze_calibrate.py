"""Gaze calibration: look at a sequence of dots on the main display, fit the
feature -> screen mapping, report its honest (leave-one-dot-out) error, then
show a live dot that follows your gaze so you can judge it.

    jarvis-gaze-calibrate [--device 0] [--out ~/.jarvis/gaze_calibration.json]
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .modules.gaze import FaceTracker, main_display_size
from .modules.gaze_model import DEFAULT_CALIBRATION_PATH, GazeMapper
from .modules.one_euro import OneEuroFilter

WINDOW = "jarvis gaze calibration"
# 3x3 grid near the edges + 4 inner points: the edges anchor the extremes,
# the inner ring constrains the quadratic terms where people look most.
TARGETS = [
    (0.06, 0.08), (0.50, 0.08), (0.94, 0.08),
    (0.06, 0.50), (0.50, 0.50), (0.94, 0.50),
    (0.06, 0.92), (0.50, 0.92), (0.94, 0.92),
    (0.30, 0.30), (0.70, 0.30), (0.30, 0.70), (0.70, 0.70),
]
SETTLE_S = 0.9    # eyes saccade + settle; discard
COLLECT_S = 1.2


def _canvas(size, text=None):
    w, h = size
    img = np.full((h, w, 3), 24, np.uint8)
    if text:
        for i, line in enumerate(text.split("\n")):
            cv2.putText(img, line, (40, 60 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (220, 220, 220), 2, cv2.LINE_AA)
    return img


def _to_px(pt, size):
    return int(pt[0] * size[0]), int(pt[1] * size[1])


def _robust(samples: np.ndarray) -> np.ndarray:
    """Drop blink/saccade outliers: keep samples within 2.5 MAD of the median."""
    med = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - med), axis=0) + 1e-9
    keep = np.all(np.abs(samples - med) <= 2.5 * 1.4826 * mad, axis=1)
    return samples[keep] if keep.sum() >= 5 else samples


def _collect(cap, tracker, size):
    feats, targets = [], []
    for i, target in enumerate(TARGETS):
        start = time.perf_counter()
        got = []
        while True:
            t = time.perf_counter() - start
            if t > SETTLE_S + COLLECT_S:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            f = tracker.features(frame, time.perf_counter())
            if f is not None and t > SETTLE_S:
                got.append(f)
            img = _canvas(size, None if f is not None else "face not found")
            # shrinking ring draws the eye in; solid dot while sampling
            r = int(8 + 40 * max(0.0, 1.0 - t / SETTLE_S))
            cv2.circle(img, _to_px(target, size), r, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.circle(img, _to_px(target, size), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(img, f"{i + 1}/{len(TARGETS)}", (size[0] - 140, size[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, img)
            if cv2.waitKey(1) & 0xFF == 27:
                return None
        if len(got) < 5:
            print(f"  dot {i + 1}: only {len(got)} face samples, skipping", flush=True)
            continue
        got = _robust(np.array(got))
        feats.append(got)
        targets.append(np.repeat([target], len(got), axis=0))
    return feats, targets


def _holdout_error_px(feats, targets, size, ridge):
    """Leave-one-dot-out: the error you'd see looking at a spot the fit never saw."""
    errs = []
    scale = np.array(size, dtype=float)
    for k in range(len(feats)):
        tr_f = np.vstack([f for j, f in enumerate(feats) if j != k])
        tr_t = np.vstack([t for j, t in enumerate(targets) if j != k])
        m = GazeMapper.fit(tr_f, tr_t, ridge=ridge)
        pred = np.median(m.predict(feats[k]), axis=0)
        errs.append(float(np.linalg.norm((pred - targets[k][0]) * scale)))
    return np.array(errs)


def _preview(cap, tracker, mapper, size, halo_px):
    filt = OneEuroFilter(min_cutoff=0.4, beta=2.0)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.perf_counter()
        f = tracker.features(frame, now)
        img = _canvas(size, "Live gaze preview - ESC to quit")
        for t in TARGETS:
            cv2.circle(img, _to_px(t, size), 4, (90, 90, 90), -1)
        if f is not None:
            p = filt(np.clip(mapper.predict(f), 0, 1), now)
            c = _to_px(p, size)
            cv2.circle(img, c, int(halo_px), (80, 200, 255), 2, cv2.LINE_AA)
            cv2.circle(img, c, 7, (80, 200, 255), -1, cv2.LINE_AA)
        cv2.imshow(WINDOW, img)
        if cv2.waitKey(1) & 0xFF == 27:
            return


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate webcam gaze tracking")
    ap.add_argument("--device", type=int, default=0, help="camera index (default 0, built-in)")
    ap.add_argument("--out", default=str(DEFAULT_CALIBRATION_PATH))
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    size = main_display_size()
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit(f"camera {args.device} failed to open")
    tracker = FaceTracker()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        while True:
            cv2.imshow(WINDOW, _canvas(size, (
                "Gaze calibration\n\nSit how you normally sit. Keep your head still-ish.\n"
                "Follow each dot with your EYES until it moves on.\n\nSPACE to start, ESC to quit")))
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return
            if key == 32:
                break

        collected = _collect(cap, tracker, size)
        if collected is None:
            return
        feats, targets = collected
        if len(feats) < 8:
            sys.exit(f"only {len(feats)} usable dots - need 8+ (is your face in frame?)")

        errs = _holdout_error_px(feats, targets, size, args.ridge)
        mapper = GazeMapper.fit(np.vstack(feats), np.vstack(targets), ridge=args.ridge)
        out = Path(os.path.expanduser(args.out))
        mapper.save(out, extra={
            "screen_points": list(size),
            "holdout_error_px": {"median": float(np.median(errs)), "max": float(errs.max())},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"saved {out}", flush=True)
        print(f"held-out error: median {np.median(errs):.0f}px, max {errs.max():.0f}px "
              f"(display {size[0]}x{size[1]} pt)", flush=True)

        if not args.no_preview:
            _preview(cap, tracker, mapper, size, halo_px=float(np.median(errs)))
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
