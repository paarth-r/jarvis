import numpy as np

from jarvis.studio.recorder import to_frames

S = np.zeros((21, 3), dtype=np.float32)


def test_steady_stream_has_no_gaps():
    samples = [(1.0 + i / 30, S) for i in range(30)]
    frames = to_frames(samples, 1.0, 2.0)
    assert len(frames) == 30 and all(f is not None for f in frames)


def test_dropout_in_the_middle_becomes_none_frames():
    samples = [(1.0 + i / 30, S) for i in range(10)] + [(1.0 + i / 30, S) for i in range(20, 30)]
    frames = to_frames(samples, 1.0, 2.0)
    assert sum(f is None for f in frames) == 10
    assert frames[10] is None and frames[9] is not None


def test_hand_missing_at_start_and_end_counts():
    samples = [(1.0 + i / 30, S) for i in range(10, 20)]
    frames = to_frames(samples, 1.0, 1.0 + 30 / 30)
    assert sum(f is None for f in frames) == 20


def test_backpressure_halving_the_rate_is_not_a_dropout():
    # The pose model skipping every other camera frame is just a lower rate.
    samples = [(1.0 + i / 15, S) for i in range(15)]
    frames = to_frames(samples, 1.0, 2.0)
    assert all(f is not None for f in frames)


def test_a_one_frame_hiccup_is_not_a_dropout():
    # The pose model occasionally skipping one frame under load.
    ts = [1.0 + i / 30 for i in range(30) if i not in (7, 19)]
    frames = to_frames([(t, S) for t in ts], 1.0, 2.0)
    assert all(f is not None for f in frames)


def test_no_hand_at_all_is_all_none():
    frames = to_frames([], 1.0, 2.5)
    assert len(frames) == 45 and all(f is None for f in frames)
