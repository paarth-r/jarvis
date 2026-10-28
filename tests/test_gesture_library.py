import json
import os

import numpy as np
import pytest

from jarvis.gestures.library import GestureLibrary, check_name


def _lib(tmp_path):
    return GestureLibrary(tmp_path / "gestures.json")


def test_missing_file_is_an_empty_library(tmp_path):
    lib = _lib(tmp_path)
    assert lib.gestures == {} and lib.bindings == {}


def test_round_trip(tmp_path):
    lib = _lib(tmp_path)
    samples = np.random.default_rng(0).normal(size=(5, 23)).astype(np.float32)
    lib.add_gesture("peace", samples, radius=0.4, quality={"verdict": "pass"})
    lib.set_binding("peace", {"type": "media", "key": "play_pause"})
    again = _lib(tmp_path)
    np.testing.assert_allclose(again.samples("peace"), samples, atol=1e-5)   # stored at 5 dp
    assert again.gestures["peace"]["radius"] == 0.4
    assert again.bindings["peace"] == {"type": "media", "key": "play_pause"}


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    lib = _lib(tmp_path)
    lib.set_binding("point", {"type": "keys", "combo": "cmd+space"})
    assert sorted(os.listdir(tmp_path)) == ["gestures.json"]
    assert json.loads((tmp_path / "gestures.json").read_text())["version"] == 1


def test_remove_gesture_drops_its_binding(tmp_path):
    lib = _lib(tmp_path)
    lib.add_gesture("rock", np.zeros((3, 23)), radius=0.3)
    lib.set_binding("rock", {"type": "media", "key": "next"})
    lib.remove_gesture("rock")
    assert "rock" not in lib.gestures and "rock" not in lib.bindings


def test_bindings_for_builtins_are_allowed_but_not_reserved_ones(tmp_path):
    lib = _lib(tmp_path)
    lib.set_binding("point", {"type": "media", "key": "next"})
    with pytest.raises(ValueError):
        lib.set_binding("fist", {"type": "media", "key": "next"})
    with pytest.raises(ValueError):
        lib.set_binding("nope", {"type": "media", "key": "next"})   # no such gesture


def test_invalid_actions_are_rejected(tmp_path):
    lib = _lib(tmp_path)
    with pytest.raises(ValueError):
        lib.set_binding("point", {"type": "keys", "combo": "cmd+zz"})


def test_clearing_a_binding(tmp_path):
    lib = _lib(tmp_path)
    lib.set_binding("point", {"type": "media", "key": "next"})
    lib.set_binding("point", None)
    assert "point" not in lib.bindings


@pytest.mark.parametrize("bad", ["", "fist", "Pinch", "point", "a" * 33, "no/slash", "  "])
def test_bad_names(bad):
    assert check_name(bad)


@pytest.mark.parametrize("good", ["peace", "thumbs up", "rock-on", "three_2"])
def test_good_names(good):
    assert check_name(good) == []


def test_reload_if_changed_picks_up_other_writers(tmp_path):
    a, b = _lib(tmp_path), _lib(tmp_path)
    assert a.reload_if_changed() is False
    b.add_gesture("ok", np.ones((2, 23)), radius=0.2)
    os.utime(tmp_path / "gestures.json", (1e9 + 5, 1e9 + 5))   # mtime differs for sure
    assert a.reload_if_changed() is True
    assert "ok" in a.gestures
    assert a.reload_if_changed() is False


def test_corrupt_file_loads_empty_and_is_kept_aside(tmp_path):
    (tmp_path / "gestures.json").write_text("{not json")
    lib = _lib(tmp_path)
    assert lib.gestures == {}
    assert (tmp_path / "gestures.json.corrupt").exists()
