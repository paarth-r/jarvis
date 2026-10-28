import re

from jarvis.voice.duck import MusicDucker
from jarvis.voice.mac_speakers import SpeakerError


class FakeApps:
    """Models which apps are running and their sound volumes."""

    def __init__(self, **volumes):
        self.volumes = dict(volumes)  # app -> volume; absent means not running
        self.scripts = []
        self.fail = False

    def __call__(self, script):
        self.scripts.append(script)
        if self.fail:
            raise SpeakerError("boom")
        m = re.fullmatch(r'application "(\w+)" is running', script)
        if m:
            return "true" if m.group(1) in self.volumes else "false"
        m = re.fullmatch(r'tell application "(\w+)" to sound volume', script)
        if m:
            if m.group(1) not in self.volumes:
                raise SpeakerError("not running")
            return str(self.volumes[m.group(1)])
        m = re.fullmatch(r'tell application "(\w+)" to set sound volume to (\d+)', script)
        if m:
            if m.group(1) not in self.volumes:
                raise SpeakerError("not running")
            self.volumes[m.group(1)] = int(m.group(2))
            return ""
        raise AssertionError(f"unexpected script: {script}")


def make(apps, tmp_path=None, **kw):
    sleeps = []
    path = tmp_path / "duck.json" if tmp_path else None
    return MusicDucker(apps, state_path=path, sleep=sleeps.append, **kw), sleeps


def test_duck_restore_round_trip():
    apps = FakeApps(Spotify=80, Music=50)
    ducker, _ = make(apps)
    ducker.duck()
    assert apps.volumes == {"Spotify": 16, "Music": 10}
    ducker.restore()
    assert apps.volumes == {"Spotify": 80, "Music": 50}


def test_duck_is_idempotent():
    apps = FakeApps(Spotify=80)
    ducker, _ = make(apps)
    ducker.duck()
    ducker.duck()
    assert apps.volumes["Spotify"] == 16
    ducker.restore()
    assert apps.volumes["Spotify"] == 80


def test_not_running_app_untouched():
    apps = FakeApps(Spotify=60)
    ducker, _ = make(apps)
    ducker.duck()
    assert not any("Music" in s and "sound volume" in s for s in apps.scripts)
    ducker.restore()
    assert apps.volumes == {"Spotify": 60}


def test_zero_volume_untouched():
    apps = FakeApps(Spotify=0)
    ducker, _ = make(apps)
    ducker.duck()
    assert not any("set sound volume" in s for s in apps.scripts)
    ducker.restore()
    assert apps.volumes["Spotify"] == 0


def test_minimum_duck_level_is_one():
    apps = FakeApps(Spotify=2)
    ducker, _ = make(apps)
    ducker.duck()
    assert apps.volumes["Spotify"] == 1
    ducker.restore()
    assert apps.volumes["Spotify"] == 2


def test_user_changed_volume_not_overwritten():
    apps = FakeApps(Spotify=80)
    ducker, sleeps = make(apps)
    ducker.duck()
    apps.volumes["Spotify"] = 45  # user turned it up while ducked
    ducker.restore()
    assert apps.volumes["Spotify"] == 45
    assert sleeps == []


def test_fade_steps_end_at_exact_previous():
    apps = FakeApps(Spotify=83)
    ducker, sleeps = make(apps)
    ducker.duck()
    start = apps.volumes["Spotify"]
    apps.scripts.clear()
    ducker.restore()
    sets = [int(s.rsplit(" ", 1)[1]) for s in apps.scripts if "set sound volume" in s]
    assert len(sets) == 4
    assert sets == sorted(sets) and sets[0] > start and sets[-1] == 83
    assert sleeps == [0.12] * 3


def test_state_file_written_and_deleted(tmp_path):
    apps = FakeApps(Spotify=80)
    ducker, _ = make(apps, tmp_path)
    ducker.duck()
    assert (tmp_path / "duck.json").exists()
    ducker.restore()
    assert not (tmp_path / "duck.json").exists()


def test_no_state_file_when_nothing_ducked(tmp_path):
    ducker, _ = make(FakeApps(), tmp_path)
    ducker.duck()
    assert not (tmp_path / "duck.json").exists()


def test_recover_after_crash(tmp_path):
    apps = FakeApps(Spotify=80)
    crashed, _ = make(apps, tmp_path)
    crashed.duck()  # process dies here, music stuck low
    assert apps.volumes["Spotify"] == 16
    fresh, _ = make(apps, tmp_path)
    fresh.recover()
    assert apps.volumes["Spotify"] == 80
    assert not (tmp_path / "duck.json").exists()


def test_recover_respects_user_change(tmp_path):
    apps = FakeApps(Spotify=80)
    crashed, _ = make(apps, tmp_path)
    crashed.duck()
    apps.volumes["Spotify"] = 30
    fresh, _ = make(apps, tmp_path)
    fresh.recover()
    assert apps.volumes["Spotify"] == 30
    assert not (tmp_path / "duck.json").exists()


def test_recover_corrupt_file_deleted(tmp_path):
    (tmp_path / "duck.json").write_text("{not json")
    ducker, _ = make(FakeApps(Spotify=80), tmp_path)
    ducker.recover()
    assert not (tmp_path / "duck.json").exists()


def test_recover_without_file_is_noop(tmp_path):
    apps = FakeApps(Spotify=80)
    ducker, _ = make(apps, tmp_path)
    ducker.recover()
    assert apps.scripts == []


def test_app_quit_while_ducked_is_skipped():
    apps = FakeApps(Spotify=80)
    ducker, _ = make(apps)
    ducker.duck()
    del apps.volumes["Spotify"]
    apps.scripts.clear()
    ducker.restore()  # no raise
    assert not any("set sound volume" in s for s in apps.scripts)
    apps.volumes["Spotify"] = 70
    apps.scripts.clear()
    ducker.restore()  # forgotten: a relaunched app is left alone
    assert apps.volumes["Spotify"] == 70


def test_osascript_failure_swallowed(caplog):
    apps = FakeApps(Spotify=80)
    apps.fail = True
    ducker, _ = make(apps)
    ducker.duck()
    ducker.restore()
    assert "boom" in caplog.text


def test_failure_in_one_app_does_not_block_other():
    class Flaky(FakeApps):
        def __call__(self, script):
            if '"Spotify"' in script:
                raise RuntimeError("spotify broke")
            return super().__call__(script)

    apps = Flaky(Music=50)
    ducker, _ = make(apps)
    ducker.duck()
    assert apps.volumes["Music"] == 10


def test_ratio_is_configurable():
    apps = FakeApps(Spotify=80)
    ducker, _ = make(apps, ratio=0.5)
    ducker.duck()
    assert apps.volumes["Spotify"] == 40


def test_state_written_before_set(tmp_path):
    seen = []

    class Spy(FakeApps):
        def __call__(self, script):
            if "set sound volume" in script:
                seen.append((tmp_path / "duck.json").exists())
            return super().__call__(script)

    ducker, _ = make(Spy(Spotify=80), tmp_path)
    ducker.duck()
    assert seen == [True]


def test_restore_failure_keeps_state_and_retries(tmp_path):
    class Flaky(FakeApps):
        failing = False

        def __call__(self, script):
            if self.failing and "sound volume" in script:
                raise SpeakerError("transient")
            return super().__call__(script)

    apps = Flaky(Spotify=80)
    ducker, _ = make(apps, tmp_path)
    ducker.duck()
    apps.failing = True
    ducker.restore()
    assert (tmp_path / "duck.json").exists() and apps.volumes["Spotify"] == 16
    apps.failing = False
    ducker.restore()
    assert apps.volumes["Spotify"] == 80
    assert not (tmp_path / "duck.json").exists()


def test_restore_retry_after_partial_fade():
    class DiesMidFade(FakeApps):
        sets = 0

        def __call__(self, script):
            if "set sound volume" in script and self.armed:
                self.sets += 1
                if self.sets == 3:
                    self.armed = False
                    raise SpeakerError("transient")
            return super().__call__(script)

    apps = DiesMidFade(Spotify=80)
    apps.armed = False
    ducker, _ = make(apps)
    ducker.duck()
    apps.armed = True
    ducker.restore()
    assert apps.volumes["Spotify"] < 80
    ducker.restore()
    assert apps.volumes["Spotify"] == 80


def test_failed_app_is_skipped_until_the_conversation_ends():
    # A hung Spotify times out after 10 s: re-ducking it before every sentence would
    # stall each one again.
    class Hung(FakeApps):
        hung = True

        def __call__(self, script):
            if self.hung and '"Spotify" to' in script:
                raise SpeakerError("osascript failed: timed out")
            return super().__call__(script)

    apps = Hung(Spotify=80, Music=50)
    ducker, _ = make(apps)
    ducker.duck()
    calls = len(apps.scripts)
    ducker.duck()
    assert not any("Spotify" in s for s in apps.scripts[calls:])
    ducker.restore()  # the conversation is over: try it again next time
    apps.hung = False
    ducker.duck()
    assert apps.volumes == {"Spotify": 16, "Music": 10}


class SetFails(FakeApps):
    """The set raises (osascript timed out), applied or not."""

    applied = True

    def __call__(self, script):
        if "set sound volume to" in script and not getattr(self, "restoring", False):
            if self.applied:
                super().__call__(script)
            raise SpeakerError("osascript failed: timed out")
        return super().__call__(script)


def test_set_that_timed_out_after_applying_is_still_restored(tmp_path):
    apps = SetFails(Spotify=80)
    ducker, _ = make(apps, tmp_path)
    ducker.duck()
    assert apps.volumes["Spotify"] == 16
    assert (tmp_path / "duck.json").exists()  # a crash now must not strand it low
    apps.restoring = True
    ducker.restore()
    assert apps.volumes["Spotify"] == 80
    assert not (tmp_path / "duck.json").exists()


def test_set_that_failed_unapplied_restores_harmlessly(tmp_path):
    apps = SetFails(Spotify=80)
    apps.applied = False
    ducker, _ = make(apps, tmp_path)
    ducker.duck()
    assert apps.volumes["Spotify"] == 80
    apps.restoring = True
    ducker.restore()
    assert apps.volumes["Spotify"] == 80
    assert not (tmp_path / "duck.json").exists()
