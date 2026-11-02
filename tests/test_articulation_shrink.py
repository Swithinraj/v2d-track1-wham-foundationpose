import numpy as np
from articulation_shrink import targets


def lin_smoother(lam):
    def smooth(x):
        T = len(x); D = np.diff(np.eye(T), 2, axis=0)
        return np.linalg.solve(np.eye(T) + lam * D.T @ D, np.asarray(x, float).reshape(T, -1)).reshape(np.shape(x))
    return smooth


def second(x):
    return x[2:] - 2 * x[1:-1] + x[:-2]


def test_root_motion_is_kept_and_articulation_is_scaled():
    rng = np.random.default_rng(4)
    T, N = 70, 6
    root = np.cumsum(rng.normal(size=(T, 3)), axis=0) * 5          # mm, a wandering root
    body = rng.normal(size=(T, N, 3)) * 2 + rng.normal(size=(1, N, 3)) * 100
    V = body + root[:, None]
    smooth = lin_smoother(1e3)
    X = targets(V, root, .6, smooth)
    # a rigid translation of the whole body is pure root motion and passes through untouched
    np.testing.assert_allclose(targets(root[:, None] + np.zeros((T, N, 3)) + 7., root, .6, smooth),
                               root[:, None] + np.zeros((T, N, 3)) + 7., atol=1e-8)
    rel, rel_x = V - root[:, None], X - root[:, None]
    rel_s = smooth(rel.reshape(T, -1)).reshape(rel.shape)
    np.testing.assert_allclose(second(rel_x), second(rel_s) + .6 * (second(rel) - second(rel_s)), atol=1e-8)
    np.testing.assert_allclose(targets(V, root, 1., smooth), V, atol=1e-12)
