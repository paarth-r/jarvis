"""Grade a candidate gesture before it is saved.

Three checks, each pass / warn / fail with a 0-100 score, a one-line summary
and a hint for fixing it:

- recognizable: the tracker sees the hand (few dropouts) and steadily
  (low frame-to-frame jitter; hidden fingers make MediaPipe guess).
- repeatable: the takes agree with each other. Compared on per-take
  averages, so frame jitter cancels out; an odd take out is named so it can
  be redone, and is left out of the saved templates.
- distinct: far enough from every saved gesture (relative to both
  acceptance radii), not read as a built-in (fist / pinch are reserved,
  point is overridden), and not matching everyday hand movement recorded
  as a background clip (Xu et al. 2022's enrollment check).

A fail blocks saving; a warn doesn't. Thresholds are first guesses on
synthetic hands and should be revisited with real takes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .builtins import is_fist, is_pointing
from .features import FEATURE_NAMES, features
from .matcher import fit_radius, take_samples

_ORDER = {"pass": 0, "warn": 1, "fail": 2}
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# Flexion of a finger plus the thumb's reach to it; spreads are left out (one
# finger curling changes its neighbours' spread too).
_FINGER_DIMS = {
    f: [i for i, n in enumerate(FEATURE_NAMES)
        if n.startswith(f + "_") and not n.startswith("thumb_to") or n == f"thumb_to_{f}"]
    for f in _FINGERS
}


@dataclass
class Thresholds:
    tracked_pass: float = 0.9      # fraction of frames with a hand
    tracked_warn: float = 0.75
    jitter_pass: float = 1.0       # mean per-frame feature change while held
    jitter_warn: float = 2.0
    min_frames: int = 5            # tracked frames for a take to count
    min_takes: int = 3
    take_pass: float = 0.5         # take-average distance from the typical take
    take_odd: float = 0.9          # beyond this a take is "odd"
    sep_fail: float = 1.0          # nearest gesture distance / (sum of radii)
    sep_warn: float = 1.25
    builtin_fail: float = 0.2      # fraction of frames read as fist / pinch
    point_warn: float = 0.5
    pinch_m: float = 0.03          # thumb tip to index tip, in hand_shape metres
    background_fail: float = 0.05  # fraction of everyday frames it would match
    background_warn: float = 0.01


def _check(verdict: str, score: float, summary: str, hint: str = "", **details) -> Dict[str, Any]:
    return {"verdict": verdict, "score": int(round(float(np.clip(score, 0, 100)))),
            "summary": summary, "hint": hint, "details": details}


def _worst(*verdicts: str) -> str:
    return max(verdicts, key=_ORDER.__getitem__)


def _grade(value: float, good: float, ok: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "pass" if value >= good else "warn" if value >= ok else "fail"
    return "pass" if value <= good else "warn" if value <= ok else "fail"


def _recognizable(takes, feats, th: Thresholds) -> Dict[str, Any]:
    total = sum(len(t) for t in takes)
    tracked = sum(s is not None for t in takes for s in t) / max(total, 1)
    moving = [np.linalg.norm(np.diff(f, axis=0), axis=1).mean() for f in feats if len(f) > 1]
    jitter = float(np.median(moving)) if moving else float("inf")
    v_track = _grade(tracked, th.tracked_pass, th.tracked_warn)
    v_jit = _grade(jitter, th.jitter_pass, th.jitter_warn, higher_is_better=False)
    verdict = _worst(v_track, v_jit)
    score = min(100 * (tracked - 0.5) / 0.5, 100 * (1 - jitter / (1.5 * th.jitter_warn)))
    if v_track != "pass":
        summary = f"The camera lost your hand in {round(100 * (1 - tracked))}% of frames."
        hint = "Keep your whole hand in view, a bit farther from the camera, in even light."
    elif v_jit != "pass":
        summary = "Tracking was shaky while you held it."
        hint = ("That usually means fingers hidden behind each other or the palm edge-on to "
                "the camera. Try a version where every finger is visible.")
    else:
        summary, hint = "Tracked clearly and steadily.", ""
    return _check(verdict, score, summary, hint, tracked=round(tracked, 3), jitter=round(jitter, 3))


def _repeatable(feats, th: Thresholds):
    """Returns (check, indices of the takes to keep)."""
    usable = [i for i, f in enumerate(feats) if len(f) >= th.min_frames]
    if len(usable) < th.min_takes:
        return _check("fail", 0, f"Only {len(usable)} usable take(s); need {th.min_takes}.",
                      "Record more takes, holding the gesture until the bar fills.",
                      odd_takes=[], spread=None), usable
    centroids = np.stack([feats[i].mean(axis=0) for i in usable])
    typical = np.median(centroids, axis=0)
    dist = np.linalg.norm(centroids - typical, axis=1)
    odd = [usable[k] for k in np.flatnonzero(dist > th.take_odd)]
    keep = [i for i in usable if i not in odd]
    spread = float(dist.max())
    details = dict(odd_takes=[i + 1 for i in odd], spread=round(spread, 3),
                   per_take=[round(float(d), 3) for d in dist])
    score = 100 * (1 - spread / (2 * th.take_odd))
    if len(keep) < th.min_takes:
        return _check("fail", score, "Your takes don't agree with each other.",
                      "Hold the same shape every take, or pick a shape that's easier to repeat.",
                      **details), usable
    if odd:
        names = ", ".join(f"take {i + 1}" for i in odd)
        return _check("warn", max(score, 50), f"{names.capitalize()} looked different from the others.",
                      f"Redo {names}; it's left out for now.", **details), keep
    verdict = "pass" if spread <= th.take_pass else "warn"
    return _check(verdict, score,
                  "Your takes agree." if verdict == "pass" else "Your takes vary a little.",
                  "" if verdict == "pass" else "One more careful take would tighten it up.",
                  **details), keep


def _differing_fingers(a: np.ndarray, b: np.ndarray, at_least: float = 0.5) -> List[str]:
    """Fingers that set two gesture averages apart, most different first.
    [] when no finger differs by `at_least` (the shapes are the same)."""
    diff = {f: float(np.linalg.norm(a[d] - b[d])) for f, d in _FINGER_DIMS.items()}
    top = max(diff.values())
    if top < at_least:
        return []
    return [f for f, v in sorted(diff.items(), key=lambda kv: -kv[1]) if v >= 0.4 * top]


def _distinct(shapes, samples, radius, library, background, th: Thresholds) -> Dict[str, Any]:
    problems = []                          # (verdict, summary, hint)
    details: Dict[str, Any] = {"nearest": None, "separation": None, "differing_fingers": []}

    # Built-ins: the engine reads a fist before any custom gesture, and a
    # thumb on the index starts a pinch, so those shapes can never fire.
    n = max(len(shapes), 1)
    fist = float(sum(is_fist(s) for s in shapes) / n)
    pinch = float(sum(bool(np.linalg.norm(s[4] - s[8]) < th.pinch_m) for s in shapes) / n)
    point = float(sum(is_pointing(s) for s in shapes) / n)
    details["builtins"] = {"fist": round(fist, 3), "pinch": round(pinch, 3), "point": round(point, 3)}
    if fist > th.builtin_fail:
        problems.append(("fail", "It reads as a fist, which is reserved for scrolling.",
                         "Leave at least one finger out."))
    if pinch > th.builtin_fail:
        problems.append(("fail", "Your thumb touches your index: that's a pinch (click).",
                         "Keep the thumb off the index fingertip."))
    if point > th.point_warn:
        problems.append(("warn", "It looks like point; while you hold it, it replaces point.",
                         "Fine if you don't use point; otherwise change another finger too."))

    centroid = samples.mean(axis=0)
    best = None
    for name, g in (library.gestures.items() if library else []):
        other = np.asarray(g["samples"], dtype=np.float32)
        dmin = float(np.linalg.norm(samples[:, None, :] - other[None, :, :], axis=-1).min())
        sep = dmin / (radius + float(g["radius"]))
        if best is None or sep < best[1]:
            best = (name, sep, other.mean(axis=0))
    if best:
        name, sep, other_c = best
        fingers = _differing_fingers(centroid, other_c)
        details.update(nearest=name, separation=round(sep, 3), differing_fingers=fingers)
        v = _grade(sep, th.sep_warn, th.sep_fail)
        if v != "pass":
            if fingers:
                hint = (f"They differ mostly in your {' and '.join(fingers[:2])}; make that "
                        "difference bigger or pick another shape.")
            else:
                hint = "They're practically the same shape; change at least one finger clearly."
            problems.append((v, f"Too close to '{name}'.", hint))

    if background:
        bg = np.stack([features(s) for s in background])
        rel = np.linalg.norm(bg[:, None, :] - samples[None, :, :], axis=-1).min(axis=1) / radius
        hits = float((rel < 1.0).mean())
        details["background_hits"] = round(hits, 3)
        v = _grade(hits, th.background_warn, th.background_fail, higher_is_better=False)
        if v != "pass":
            problems.append((v, f"It matched {round(100 * hits)}% of your everyday hand movement.",
                             "It would fire by accident; pick a shape you don't make without meaning to."))

    score = 100.0
    if best:
        score = min(score, 100 * (best[1] - 0.75) / (1.5 - 0.75))
    score = min(score, 100 * (1 - max(fist, pinch) / 0.4), 100 * (1 - 0.5 * point))
    if "background_hits" in details:
        score = min(score, 100 * (1 - details["background_hits"] / (2 * th.background_fail)))
    if not problems:
        near = f" Nearest is '{best[0]}'." if best else ""
        return _check("pass", score, "Clearly different from everything else." + near, "", **details)
    problems.sort(key=lambda p: -_ORDER[p[0]])
    verdict = problems[0][0]
    return _check(verdict, score, " ".join(p[1] for p in problems),
                  " ".join(p[2] for p in problems if p[2]), **details)


def assess(takes: Sequence[Sequence[Optional[np.ndarray]]], library=None,
           background: Optional[Sequence[Optional[np.ndarray]]] = None,
           thresholds: Optional[Thresholds] = None) -> Dict[str, Any]:
    """takes: per take, the (21, 3) hand_shape of each frame (None = no hand).
    background: optional everyday-movement frames. library: saved gestures.
    Returns the report the studio shows, plus the templates to save."""
    th = thresholds or Thresholds()
    feats = [np.stack([features(s) for s in t if s is not None])
             if any(s is not None for s in t) else np.zeros((0, len(FEATURE_NAMES)))
             for t in takes]
    recognizable = _recognizable(takes, feats, th)
    repeatable, keep = _repeatable(feats, th)
    keep_feats = [feats[i] for i in keep if len(feats[i])]
    if not keep_feats:
        distinct = _check("fail", 0, "Nothing to compare: no usable takes.", "Record the takes again.")
        return {"verdict": "fail", "radius": 0.0, "samples": [],
                "checks": {"recognizable": recognizable, "repeatable": repeatable, "distinct": distinct}}
    samples = take_samples(keep_feats)
    radius = fit_radius(keep_feats)
    shapes = [s for i in keep for s in takes[i] if s is not None]
    bg = [s for s in (background or []) if s is not None]
    distinct = _distinct(shapes, samples, radius, library, bg, th)
    checks = {"recognizable": recognizable, "repeatable": repeatable, "distinct": distinct}
    return {
        "verdict": _worst(*(c["verdict"] for c in checks.values())),
        "radius": radius,
        "samples": samples.tolist(),
        "takes_used": [i + 1 for i in keep],
        "checks": checks,
    }
