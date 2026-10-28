import pytest


@pytest.fixture(autouse=True)
def _private_gesture_library(tmp_path, monkeypatch):
    """Never read or write the real ~/.jarvis/gestures.json from tests."""
    monkeypatch.setenv("JARVIS_GESTURES_PATH", str(tmp_path / "gestures.json"))
