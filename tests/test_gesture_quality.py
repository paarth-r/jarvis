import numpy as np
import pytest

from jarvis.gestures.library import GestureLibrary
from jarvis.gestures.matcher import fit_radius, take_samples
from jarvis.gestures.quality import assess
from tests.hand_synth import FIST, OPEN, PEACE, POINT, ROCK, THREE, hand, pinch, rotation
from tests.test_gesture_matcher import takes_of


def shape_takes(pose, n_takes=5, frames=30, noise=0.002, seed=0, wobble=0.15, drop=0.0):
    """Raw (21, 3) shapes per take, with None where tracking dropped out."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_takes):
        R = rotation(*rng.uniform(-wobble, wobble, 3))
        take = []
        for _ in range(frames):
            take.append(None if rng.random() < drop else hand(pose, R=R, noise=noise, rng=rng))
        out.append(take)
    return out


def _lib_with(tmp_path, **poses):
    lib = GestureLibrary(tmp_path / "g.json")
    for name, pose in poses.items():
        takes = takes_of(pose, seed=7)
        lib.add_gesture(name, take_samples(takes), fit_radius(takes))
    return lib


def _check(report, name):
    return report["checks"][name]


def test_a_clean_distinct_gesture_passes_everything(tmp_path):
    lib = _lib_with(tmp_path, peace=PEACE)
    r = assess(shape_takes(THREE), lib)
    assert r["verdict"] == "pass", r
    for c in ("recognizable", "repeatable", "distinct"):
        assert _check(r, c)["verdict"] == "pass"
        assert 0 <= _check(r, c)["score"] <= 100
    assert r["radius"] > 0 and len(r["samples"]) > 0


def test_every_check_explains_itself():
    r = assess(shape_takes(THREE), None)
    for c in r["checks"].values():
        assert c["summary"] and isinstance(c["hint"], str)


def test_tracking_dropouts_hurt_recognizable():
    r = assess(shape_takes(THREE, drop=0.4), None)
    assert _check(r, "recognizable")["verdict"] == "fail"
    assert _check(r, "recognizable")["details"]["tracked"] == pytest.approx(0.6, abs=0.1)


def test_shaky_tracking_hurts_recognizable():
    r = assess(shape_takes(THREE, noise=0.012), None)
    assert _check(r, "recognizable")["verdict"] in ("warn", "fail")


def test_one_odd_take_is_named():
    takes = shape_takes(THREE, n_takes=4) + shape_takes(ROCK, n_takes=1, seed=3)
    r = assess(takes, None)
    rep = _check(r, "repeatable")
    assert rep["verdict"] in ("warn", "fail")
    assert rep["details"]["odd_takes"] == [5]
    assert "take 5" in rep["hint"]


def test_inconsistent_takes_fail_repeatable():
    takes = (shape_takes(THREE, 2) + shape_takes(PEACE, 2, seed=1) + shape_takes(ROCK, 1, seed=2))
    assert _check(assess(takes, None), "repeatable")["verdict"] == "fail"


def test_too_close_to_an_existing_gesture_fails_and_names_it(tmp_path):
    lib = _lib_with(tmp_path, peace=PEACE)
    r = assess(shape_takes(PEACE, seed=11), lib)
    d = _check(r, "distinct")
    assert d["verdict"] == "fail" and r["verdict"] == "fail"
    assert d["details"]["nearest"] == "peace"
    assert "peace" in d["summary"]
    assert d["details"]["differing_fingers"] == []          # it's the same shape
    assert "practically the same" in d["hint"]


def test_report_is_plain_json(tmp_path):
    import json
    lib = _lib_with(tmp_path, peace=PEACE)
    r = assess(shape_takes(ROCK), lib, background=[hand(OPEN)] * 5)
    assert json.loads(json.dumps(r))["verdict"] == r["verdict"]


def test_differing_fingers_point_at_what_separates_two_gestures(tmp_path):
    lib = _lib_with(tmp_path, peace=PEACE)
    d = _check(assess(shape_takes(ROCK), lib), "distinct")
    assert d["details"]["nearest"] == "peace"
    assert set(d["details"]["differing_fingers"]) == {"middle", "pinky"}


def test_fist_like_gesture_fails_because_fist_is_reserved():
    d = _check(assess(shape_takes(FIST), None), "distinct")
    assert d["verdict"] == "fail"
    assert d["details"]["builtins"]["fist"] > 0.9
    assert "fist" in d["summary"]


def test_pinch_like_gesture_fails_because_it_would_click():
    takes = [[pinch(R=rotation(0.05 * i, 0, 0)) for i in range(20)] for _ in range(5)]
    d = _check(assess(takes, None), "distinct")
    assert d["verdict"] == "fail" and "pinch" in d["summary"]


def test_point_like_gesture_warns_that_it_overrides_point():
    d = _check(assess(shape_takes(POINT), None), "distinct")
    assert d["verdict"] == "warn" and "point" in d["summary"]


def test_matching_everyday_movement_fails():
    background = [hand(OPEN)] * 40 + [hand(THREE)] * 20 + [None] * 5
    d = _check(assess(shape_takes(THREE), None, background=background), "distinct")
    assert d["verdict"] == "fail"
    assert d["details"]["background_hits"] == pytest.approx(20 / 60, abs=0.02)


def test_background_without_the_gesture_is_fine():
    background = [hand(OPEN)] * 40 + [hand(FIST)] * 20
    d = _check(assess(shape_takes(THREE), None, background=background), "distinct")
    assert d["verdict"] == "pass"
    assert d["details"]["background_hits"] == 0.0


def test_too_few_takes_is_a_repeatable_fail():
    r = assess(shape_takes(THREE, n_takes=1), None)
    assert _check(r, "repeatable")["verdict"] == "fail"
    assert r["verdict"] == "fail"


def test_no_tracked_frames_at_all_does_not_crash():
    r = assess([[None] * 10] * 3, None)
    assert r["verdict"] == "fail"
    assert r["samples"] == []
