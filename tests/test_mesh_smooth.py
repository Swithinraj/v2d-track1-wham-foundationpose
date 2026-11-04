import numpy as np
import pytest


def track(frames, noise=.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.asarray(frames, float)
    truth = np.stack([np.c_[np.sin(t / 40), t * .002, np.full(len(t), 3.)],
                      np.c_[np.sin(t / 40) + .3, t * .002 + .1, np.full(len(t), 3.1)]], axis=1)
    return truth, truth + rng.normal(scale=noise, size=truth.shape)


def test_noise_falls_and_shape_is_kept():
    from mesh_smooth import smooth_scored
    frames = np.arange(300)
    truth, values = track(frames)
    out = smooth_scored(values, frames, 300.)
    assert out.shape == values.shape and out.dtype == np.float32
    assert np.linalg.norm(out - truth, axis=-1).mean() < .4 * np.linalg.norm(values - truth, axis=-1).mean()
    assert np.abs(np.diff(out, 2, axis=0)).mean() < .05 * np.abs(np.diff(values, 2, axis=0)).mean()


def test_a_gap_is_not_treated_as_adjacent_frames():
    from mesh_smooth import smooth_scored
    frames = np.r_[0:50, 150:200]
    t = frames.astype(float)
    values = np.stack([np.c_[t * .01, np.zeros(100), np.zeros(100)]], axis=1)
    # A straight line across the gap stays exactly on the line: a smoother that ignored the gap would bend it.
    out = smooth_scored(values, frames, 1000.)
    np.testing.assert_allclose(out, values, atol=1e-4)


def test_zero_strength_is_identity():
    from mesh_smooth import smooth_scored
    frames = np.arange(40)
    _, values = track(frames)
    np.testing.assert_allclose(smooth_scored(values, frames, 0.), values, atol=1e-6)


def test_mismatch_is_rejected():
    from mesh_smooth import smooth_scored
    _, values = track(np.arange(10))
    with pytest.raises(ValueError):
        smooth_scored(values, np.arange(9), 10.)
    with pytest.raises(ValueError):
        smooth_scored(values, np.arange(10)[::-1], 10.)
