import numpy as np

from jarvis.modules.one_euro import OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out = f(np.array([0.3, -0.2]), 0.0)
    assert np.allclose(out, [0.3, -0.2])


def test_suppresses_jitter_at_rest():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=10.0)
    noise = rng.normal(0.0, 0.001, size=(120, 2))
    outs = np.array([f(n, i / 30.0) for i, n in enumerate(noise)])
    # filtered spread is a fraction of the raw spread
    assert outs[30:].std() < 0.5 * noise[30:].std()


def test_beta_reduces_lag_in_fast_motion():
    def lag(beta):
        f = OneEuroFilter(min_cutoff=1.0, beta=beta)
        out = None
        for i in range(10):
            out = f(np.array([0.5 * i / 30.0]), i / 30.0)   # 0.5 m/s ramp
        return 0.5 * 9 / 30.0 - out[0]

    assert lag(20.0) < 0.5 * lag(0.0)


def test_reset_forgets_history():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f(np.array([0.0]), 0.0)
    f(np.array([1.0]), 0.033)
    f.reset()
    assert np.allclose(f(np.array([5.0]), 0.1), [5.0])


def test_non_increasing_timestamp_holds_value():
    f = OneEuroFilter()
    f(np.array([1.0]), 1.0)
    assert np.allclose(f(np.array([9.0]), 1.0), [1.0])
