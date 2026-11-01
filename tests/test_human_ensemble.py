import numpy as np
import pytest


def sequence(frames=30, seed=0, noise=.01):
    rng = np.random.default_rng(seed)
    body = np.random.default_rng(99).normal(size=(40, 3)) * [.2, .5, .15]
    centers = np.c_[np.linspace(0, .5, frames), np.zeros(frames), np.full(frames, 3.)]
    return body[None] + centers[:, None] + rng.normal(scale=noise, size=(frames, 40, 3))


def test_mean_of_a_member_with_itself_is_itself():
    from human_ensemble import aligned_mean
    a = sequence()
    np.testing.assert_allclose(aligned_mean([a, a.copy()], [.5, .5]), a, atol=1e-10)


def test_overall_placement_of_a_member_does_not_matter():
    from human_ensemble import aligned_mean
    a = sequence(seed=1)
    angle = .3
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    moved = 1.2 * (a @ rotation) + [1., 2., 3.]
    np.testing.assert_allclose(aligned_mean([a, moved], [.5, .5]), a, atol=1e-9)


def test_independent_errors_shrink_and_the_bound_holds():
    from human_ensemble import aligned_mean
    truth = sequence(noise=0.)
    members = [truth + np.random.default_rng(k).normal(scale=.02, size=truth.shape) for k in range(4)]
    mean = aligned_mean(members, [.25] * 4)
    error = lambda x: np.linalg.norm(x - truth, axis=-1).mean()
    assert error(mean) <= np.mean([error(m) for m in members])
    assert error(mean) < .7 * min(error(m) for m in members)
    # Acceleration error is linear in position: the same bound.
    acc = lambda x: np.linalg.norm(np.diff(x, 2, axis=0) - np.diff(truth, 2, axis=0), axis=-1).mean()
    assert acc(mean) <= np.mean([acc(m) for m in members]) * 1.001


def test_weights_are_checked():
    from human_ensemble import aligned_mean
    a = sequence()
    with pytest.raises(ValueError):
        aligned_mean([a, a], [.5, .6])
    with pytest.raises(ValueError):
        aligned_mean([a, a[:-1]], [.5, .5])
    with pytest.raises(ValueError):
        aligned_mean([a, a], [1.5, -.5])
