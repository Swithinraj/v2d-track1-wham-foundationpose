import numpy as np


def test_a_plane_stays_a_plane_through_the_camera_change():
    from floor_planes import camera_map, map_normal
    old = dict(fx=820., cx=768., cy=576.)
    new = dict(fx=820., cx=718., cy=529.)
    normal = np.array([-.02, -.993, -.114]) / np.linalg.norm([-.02, -.993, -.114])
    rng = np.random.default_rng(0)
    # Points on the plane normal . X = -1.3
    points = rng.uniform(-2, 2, (200, 3))
    points[:, 2] = rng.uniform(1, 6, 200)
    points -= ((points @ normal + 1.3)[:, None]) * normal
    moved = points @ camera_map(old, new).T
    mapped = map_normal(normal, camera_map(old, new))
    level = moved @ mapped
    assert np.ptp(level) < 1e-9
    np.testing.assert_allclose(np.linalg.norm(mapped), 1.)


def test_same_pixel_after_the_camera_change():
    from floor_planes import camera_map
    old = dict(fx=820., cx=768., cy=576.)
    new = dict(fx=820., cx=718., cy=529.)
    X = np.array([.4, 1.1, 3.])
    Y = camera_map(old, new) @ X
    np.testing.assert_allclose(old["fx"] * X[0] / X[2] + old["cx"], new["fx"] * Y[0] / Y[2] + new["cx"])
    np.testing.assert_allclose(old["fx"] * X[1] / X[2] + old["cy"], new["fx"] * Y[1] / Y[2] + new["cy"])


def test_normals_average_within_a_mounting_only():
    from floor_planes import session_normals
    tilt = lambda deg: np.array([0., -np.cos(np.radians(deg)), -np.sin(np.radians(deg))])
    normals = [tilt(5.), tilt(5.4), tilt(5.2), tilt(2.), tilt(5.1)]
    cameras = ["a", "a", "a", "a", "b"]
    out = session_normals(normals, cameras)
    assert out[0][1] == 3             # the 2 degree mounting and the other camera stay out
    assert out[3][1] == 1
    assert out[4][1] == 1
    np.testing.assert_allclose(out[3][0], tilt(2.))
    np.testing.assert_allclose(np.degrees(np.arctan2(-out[0][0][2], -out[0][0][1])), 5.2, atol=1e-3)
