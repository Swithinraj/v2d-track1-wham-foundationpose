import numpy as np
from feet_hold import hold_targets, smooth, stance_runs


def foot(T=80, seed=5):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(12, 3)) * 30 + [100., -800., 2500.]             # mm, a dozen foot vertices
    path = np.zeros((T, 3))
    path[40:50] = np.linspace(0, 300, 10)[:, None] * [1, 0, 0]               # one step of 30 cm over ten frames
    path[50:] = [300, 0, 0]
    return base[None] + path[:, None] + rng.normal(scale=1.5, size=(T, 12, 3))


def test_smooth_keeps_lines_and_detects_the_step():
    line = np.arange(30)[:, None] * [2., -1., .5]
    np.testing.assert_allclose(smooth(line, 100.), line, atol=1e-8)
    runs = stance_runs(foot().mean(1), 4., 10, 100.)
    assert len(runs) == 2 and runs[0][0] == 0 and runs[1][1] == 80
    assert runs[0][1] <= 40 and runs[1][0] >= 50                                 # the step itself is not held


def test_hold_pins_stances_with_ramps_and_leaves_the_step():
    F = foot()
    targets, held = hold_targets(F, 4., 10, 4, 100.)
    (a, b), (c, d) = stance_runs(F.mean(1), 4., 10, 100.)
    assert held == (b - a) + (d - c)
    inner = targets[a + 4:b - 4]
    np.testing.assert_allclose(inner, np.broadcast_to(F[a:b].mean(0), inner.shape), atol=1e-9)
    np.testing.assert_allclose(targets[a], F[a] + (F[a:b].mean(0) - F[a]) * (0.5 - 0.5 * np.cos(np.pi / 4)), atol=1e-9)
    np.testing.assert_array_equal(targets[b:c], F[b:c])
    acc = lambda x: np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1).mean()
    assert acc(targets[a + 4:b - 4]) < 1e-9 < acc(F[a + 4:b - 4])


def test_moving_foot_is_never_held():
    T = 60
    F = np.zeros((T, 4, 3)) + np.arange(T)[:, None, None] * [10., 0, 0]        # 1 cm per frame, always moving
    targets, held = hold_targets(F, 4., 10, 4, 100.)
    assert held == 0
    np.testing.assert_array_equal(targets, F)
