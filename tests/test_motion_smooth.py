import numpy as np


def noisy(T=200):
    rng = np.random.default_rng(3)
    truth = np.c_[np.linspace(0, 1, T), np.sin(np.linspace(0, 3, T)), np.zeros(T)]
    return truth, truth + rng.normal(scale=.01, size=truth.shape)


def test_smoothing_lowers_acceleration_and_stays_close():
    from motion_smooth import smooth_object
    truth, values = noisy()
    out = smooth_object(values, np.arange(200), 100.)
    assert np.abs(np.diff(out, 2, axis=0)).mean() < .2 * np.abs(np.diff(values, 2, axis=0)).mean()
    assert np.abs(out - truth).mean() < np.abs(values - truth).mean()


def test_zero_strength_is_identity():
    from motion_smooth import smooth_object
    _, values = noisy()
    np.testing.assert_array_equal(smooth_object(values, np.arange(200), 0.), values)


def test_unscored_bridge_does_not_pull_scored_frames():
    from motion_smooth import smooth_object
    T = 100
    values = np.zeros((T, 3))
    values[:, 0] = np.linspace(0, 1, T)
    frames = np.r_[10:40, 60:90]
    bent = values.copy()
    bent[40:60, 0] += .5        # junk in the unscored gap
    out = smooth_object(bent, frames, 10.)
    assert np.abs(out[frames, 0] - values[frames, 0]).max() < .01
