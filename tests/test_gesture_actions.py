import pytest

from jarvis.gestures.actions import (
    MEDIA_KEYS, ActionRunner, describe, normalize, parse_combo, validate,
)


class FakeBackend:
    def __init__(self):
        self.calls = []

    def media(self, key):
        self.calls.append(("media", key))

    def hotkey(self, keys):
        self.calls.append(("hotkey", tuple(keys)))

    def text(self, s):
        self.calls.append(("text", s))

    def shell(self, cmd):
        self.calls.append(("shell", cmd))

    def sleep(self, s):
        self.calls.append(("wait", s))


def test_parse_combo_maps_modifiers_and_named_keys():
    assert parse_combo("cmd+shift+4") == ["command", "shift", "4"]
    assert parse_combo("Ctrl + Option + Space") == ["ctrl", "option", "space"]
    assert parse_combo("cmd+return") == ["command", "enter"]
    assert parse_combo("f5") == ["f5"]


@pytest.mark.parametrize("bad", ["", "cmd+", "cmd+shift", "cmd+notakey", "cmd+a+b+"])
def test_parse_combo_rejects_bad_combos(bad):
    with pytest.raises(ValueError):
        parse_combo(bad)


def test_legacy_names_normalize_to_specs():
    assert normalize("raycast") == {"type": "keys", "combo": "cmd+space"}
    assert normalize("media_next") == {"type": "media", "key": "next"}
    assert normalize("volume_up") == {"type": "media", "key": "volume_up"}
    assert normalize({"type": "media", "key": "mute"}) == {"type": "media", "key": "mute"}
    assert normalize("") is None and normalize(None) is None


def test_validate_reports_problems_in_plain_words():
    assert validate({"type": "media", "key": "play_pause"}) == []
    assert validate({"type": "media", "key": "louder"})
    assert validate({"type": "keys", "combo": "cmd+zz"})
    assert validate({"type": "teleport"})
    assert validate({"type": "shell", "command": "  "})
    assert validate({"type": "macro", "steps": []})
    assert validate({"type": "macro", "steps": [{"type": "wait", "seconds": -1}]})
    assert validate({"type": "macro", "steps": [{"type": "macro", "steps": [{"type": "text", "text": "x"}]}]})
    assert validate({"type": "media", "key": "next", "repeat_s": 0.01})
    assert validate("nonsense-legacy-name")


def test_runner_runs_each_kind():
    b = FakeBackend()
    r = ActionRunner(b)
    r.run({"type": "media", "key": "next"})
    r.run({"type": "keys", "combo": "cmd+shift+4"})
    r.run({"type": "text", "text": "hi"})
    r.run({"type": "shell", "command": "jarvis-dj next"})
    assert b.calls == [("media", "next"), ("hotkey", ("command", "shift", "4")),
                       ("text", "hi"), ("shell", "jarvis-dj next")]


def test_macro_runs_steps_in_order_with_waits():
    b = FakeBackend()
    ActionRunner(b).run({"type": "macro", "steps": [
        {"type": "keys", "combo": "cmd+c"},
        {"type": "wait", "seconds": 0.2},
        {"type": "media", "key": "play_pause"},
    ]})
    assert b.calls == [("hotkey", ("command", "c")), ("wait", 0.2), ("media", "play_pause")]


def test_runner_refuses_invalid_specs_without_raising():
    b = FakeBackend()
    assert ActionRunner(b).run({"type": "media", "key": "louder"}) is False
    assert b.calls == []


def test_every_media_key_is_describable():
    for k in MEDIA_KEYS:
        assert describe({"type": "media", "key": k})


def test_describe_is_human_readable():
    assert describe({"type": "keys", "combo": "cmd+shift+4"}) == "Keys: cmd+shift+4"
    assert describe({"type": "media", "key": "play_pause"}) == "Media: play / pause"
    assert describe({"type": "macro", "steps": [{"type": "text", "text": "a"}] * 3}) == "Macro: 3 steps"
    assert describe(None) == "Nothing"
