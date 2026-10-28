import numpy as np
import pytest

from jarvis.gestures.features import features
from jarvis.gestures.library import GestureLibrary
from jarvis.gestures.matcher import Matcher, fit_radius, take_samples
from tests.hand_synth import FIST, OPEN, PEACE, POINT, ROCK, THREE, hand, rotation


def takes_of(pose, n_takes=5, frames=30, noise=0.002, seed=0, wobble=0.15):
    """n takes of a pose: each take at a slightly different hand angle, with
    per-frame landmark noise (like MediaPipe jitter)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_takes):
        R = rotation(*rng.uniform(-wobble, wobble, 3))
        out.append(np.stack([features(hand(pose, R=R, noise=noise, rng=rng)) for _ in range(frames)]))
    return out


def _matcher(**poses):
    m = Matcher()
    for name, pose in poses.items():
        takes = takes_of(pose, seed=hash(name) % 1000)
        m.add(name, take_samples(takes), fit_radius(takes))
    return m


def test_recognizes_each_trained_gesture():
    m = _matcher(peace=PEACE, rock=ROCK, three=THREE)
    for name, pose in (("peace", PEACE), ("rock", ROCK), ("three", THREE)):
        for f in takes_of(pose, n_takes=1, seed=99)[0]:
            assert m.match(f)[0] == name


def test_untrained_shapes_are_none():
    m = _matcher(peace=PEACE, rock=ROCK)
    for pose in (OPEN, FIST):
        assert m.match(features(hand(pose)))[0] is None


def test_scores_are_between_0_and_1_and_highest_for_the_match():
    m = _matcher(peace=PEACE, rock=ROCK)
    name, scores = m.match(features(hand(PEACE)))
    assert set(scores) == {"peace", "rock"}
    assert all(0.0 <= s <= 1.0 for s in scores.values())
    assert scores["peace"] > scores["rock"]


def test_empty_matcher_matches_nothing():
    assert Matcher().match(features(hand(PEACE))) == (None, {})


def test_ambiguous_between_two_overlapping_gestures_is_none():
    m = Matcher()
    takes = takes_of(PEACE)
    samples, r = take_samples(takes), fit_radius(takes)
    m.add("a", samples, r)
    m.add("b", samples + 1e-3, r)            # practically the same gesture
    assert m.match(features(hand(PEACE)))[0] is None


def test_shakier_takes_get_a_wider_radius():
    steady = fit_radius(takes_of(PEACE, noise=0.001, wobble=0.05))
    shaky = fit_radius(takes_of(PEACE, noise=0.004, wobble=0.3))
    assert shaky > steady


def test_radius_has_a_floor():
    identical = [np.tile(features(hand(PEACE)), (10, 1))] * 3
    assert fit_radius(identical) >= Matcher.MIN_RADIUS


def test_take_samples_subsamples_long_takes():
    s = take_samples([np.zeros((100, 23))] * 5, per_take=12)
    assert s.shape == (60, 23)


def test_from_library(tmp_path):
    lib = GestureLibrary(tmp_path / "g.json")
    takes = takes_of(PEACE)
    lib.add_gesture("peace", take_samples(takes), fit_radius(takes))
    assert Matcher.from_library(lib).match(features(hand(PEACE)))[0] == "peace"
