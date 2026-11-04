import numpy as np
from scipy.spatial.transform import Rotation


def rotation(angle):
    return Rotation.from_rotvec([0, angle, .1 * angle]).as_matrix()


def scorer(points, fit):
    s, R, t = fit
    return s * (points @ R) + t


def test_object_moves_exactly_as_the_similarity_moves_its_points():
    from object_transfer import transfer_object
    fit = (1.03, rotation(.2), np.array([.3, -.1, .2]))
    R_o = Rotation.from_rotvec(np.random.default_rng(0).normal(size=(5, 3))).as_matrix()
    t_o = np.random.default_rng(1).normal(size=(5, 3)) + [0, 0, 3]
    s_o = 1.2
    mesh = np.random.default_rng(2).normal(size=(30, 3))
    old = s_o * np.einsum("tij,pj->tpi", R_o, mesh) + t_o[:, None]
    new_R, new_t, new_s = transfer_object(R_o, t_o, s_o, fit)
    new = new_s * np.einsum("tij,pj->tpi", new_R, mesh) + new_t[:, None]
    np.testing.assert_allclose(new, scorer(old, fit), atol=1e-12)
    np.testing.assert_allclose(new_R @ new_R.swapaxes(1, 2), np.broadcast_to(np.eye(3), (5, 3, 3)), atol=1e-12)


def test_aligned_scene_is_the_same_whichever_person_the_scorer_sees():
    """The point of the transfer: fitting on the new person gives the object the old person's aligned place."""
    from human_root_mix import similarity
    from object_transfer import transfer_object
    rng = np.random.default_rng(3)
    person_old = rng.normal(size=(200, 3)) * [.2, .5, .15] + [0, 0, 3]
    truth = 1.01 * (person_old @ rotation(.05)) + [.1, .1, -.1]
    fit_new = (1.02, rotation(.08), np.array([.2, .0, .1]))
    person_new = scorer(person_old, fit_new)
    R_o, t_o = rotation(.4)[None], np.array([[.3, 1., 3.2]])
    mesh = rng.normal(size=(20, 3))
    aligned_old = scorer(.9 * np.einsum("tij,pj->tpi", R_o, mesh) + t_o[:, None], similarity(person_old, truth))
    new_R, new_t, new_s = transfer_object(R_o, t_o, .9, similarity(person_old, person_new))
    aligned_new = scorer(new_s * np.einsum("tij,pj->tpi", new_R, mesh) + new_t[:, None], similarity(person_new, truth))
    np.testing.assert_allclose(aligned_new, aligned_old, atol=1e-9)
