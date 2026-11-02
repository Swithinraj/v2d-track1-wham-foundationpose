import numpy as np
import pytest


def body(frames=80):
    rng = np.random.default_rng(1)
    points = rng.normal(size=(50, 3)) * [.2, .5, .15]
    centers = np.c_[np.linspace(0, .4, frames), np.full(frames, .3), np.full(frames, 3.)]
    return points[None] + centers[:, None]


def test_zero_weight_changes_nothing():
    from human_root_mix import mix_shift
    base = body()
    other = base + np.c_[np.zeros(80), np.zeros(80), np.linspace(0, .2, 80)][:, None]
    np.testing.assert_array_equal(mix_shift(base, other, 0.), np.zeros((80, 3)))


def test_half_weight_moves_halfway_and_the_first_frame_stays():
    from human_root_mix import mix_shift
    base = body()
    offset = np.c_[np.zeros(80), np.zeros(80), np.linspace(0, .2, 80)]
    shift = mix_shift(base, base + offset[:, None], .5)
    np.testing.assert_array_equal(shift[0], [0, 0, 0])
    np.testing.assert_allclose(shift, .5 * offset, atol=1e-3)


def test_a_different_global_placement_does_not_count_as_motion():
    from human_root_mix import mix_shift, similarity
    base = body()
    angle = .14
    rotation = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
    other = 1.1 * (base @ rotation) + [.5, -.2, .3]
    # The scorer's similarity removes this entirely.
    s, r, t = similarity(other[0], base[0])
    np.testing.assert_allclose(s * (other[0] @ r) + t, base[0], atol=1e-9)
    np.testing.assert_allclose(mix_shift(base, other, 1.), 0, atol=1e-9)


def test_jitter_in_the_other_estimate_is_smoothed_away():
    from human_root_mix import mix_shift
    base = body(200)
    jitter = np.random.default_rng(2).normal(scale=.02, size=(200, 3))
    shift = mix_shift(base, base + jitter[:, None], 1.)
    assert np.abs(np.diff(shift, n=2, axis=0)).max() < .002


def test_mismatched_inputs_are_rejected():
    from human_root_mix import mix_shift
    with pytest.raises(ValueError):
        mix_shift(body(10), body(11), .5)


def test_full_shift_fills_unscored_frames():
    from human_root_mix import full_shift
    frames = np.array([2, 3, 7])
    shift = np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0]], float)
    out = full_shift(shift, frames, 10)
    np.testing.assert_allclose(out[:, 0], [0, 0, 0, 1, 2, 3, 4, 5, 5, 5])


def test_root_mix_smooths_on_video_time_across_missing_scored_frames():
    from human_root_mix import mix_shift
    frames=np.r_[np.arange(20),np.arange(200,220)]
    base=body(len(frames))
    offset=np.c_[np.zeros(len(frames)),np.zeros(len(frames)),frames*.001]
    shift=mix_shift(base,base+offset[:,None],.5,frames=frames)
    np.testing.assert_allclose(shift,.5*offset,atol=1e-8)
    np.testing.assert_array_equal(shift[0],[0,0,0])


@pytest.mark.parametrize('frames',[np.array([0,1,1]),np.array([0,2,-1]),np.array([0,1]),np.array([0,.5,2])])
def test_root_mix_rejects_invalid_video_frame_axis(frames):
    from human_root_mix import mix_shift
    with pytest.raises(ValueError,match='frames'):
        mix_shift(body(3),body(3),.5,frames=frames)


def test_zero_smoothing_with_scoring_gaps_keeps_observed_motion():
    from human_root_mix import mix_shift
    base=body(3)
    offset=np.array([[0,0,0],[0,0,.01],[0,0,.04]])
    shift=mix_shift(base,base+offset[:,None],.5,lam=0,frames=np.array([0,1,4]))
    np.testing.assert_allclose(shift,.5*offset,atol=1e-10)
