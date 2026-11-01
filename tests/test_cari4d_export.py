
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def poses(T=20):
    pose = np.eye(4)[None].repeat(T, 0)
    pose[:, :3, :3] = Rotation.from_rotvec(np.c_[np.zeros(T), np.linspace(0, 1, T), np.zeros(T)]).as_matrix()
    pose[:, :3, 3] = np.c_[np.linspace(0, 1, T), np.ones(T), np.full(T, 3.)]
    return pose


def test_pose_split():
    from cari4d_export import split_pose
    rotation, translation = split_pose(poses())
    np.testing.assert_allclose(translation[10], poses()[10, :3, 3])
    np.testing.assert_allclose(rotation[10], poses()[10, :3, :3])


@pytest.mark.parametrize("bad", ["scaled", "reflected", "nan", "shape"])
def test_pose_split_rejects_non_rigid(bad):
    from cari4d_export import split_pose
    pose = poses()
    if bad == "scaled":
        pose[3, :3, :3] *= 1.1
    if bad == "reflected":
        pose[3, :3, 0] *= -1
    if bad == "nan":
        pose[3, 0, 3] = np.nan
    if bad == "shape":
        pose = pose[:, :3]
    with pytest.raises(ValueError):
        split_pose(pose)


def test_episode_is_indexed_by_video_frame_and_scored_values_are_exact():
    from cari4d_episodes import build_episode
    frames = np.array([2, 3, 4, 9, 10, 11])
    pose = np.arange(6 * 136, dtype=float).reshape(6, 136)
    full = poses(15)
    data = build_episode(frames, pose, np.ones(68), np.zeros(45), full[:, :3, :3], full[:, :3, 3])
    assert data["pose"].shape == (12, 136)
    np.testing.assert_allclose(data["pose"][frames], pose)
    np.testing.assert_allclose(data["object_translation"][frames], full[frames, :3, 3])
    np.testing.assert_allclose(data["object_rotation"][frames], full[frames, :3, :3], atol=1e-9)
    assert data["object_scale"] == 1.
    np.testing.assert_allclose(data["pose"][0], pose[0])


def test_episode_rejects_a_frame_count_mismatch():
    from cari4d_episodes import build_episode
    full = poses(15)
    with pytest.raises(ValueError, match="scored"):
        build_episode(np.array([1, 2, 3]), np.zeros((2, 136)), np.ones(68), np.zeros(45), full[:, :3, :3], full[:, :3, 3])
    with pytest.raises(ValueError, match="cover"):
        build_episode(np.array([1, 2, 30]), np.zeros((3, 136)), np.ones(68), np.zeros(45), full[:, :3, :3], full[:, :3, 3])


def test_mesh_shape_check():
    from cari4d_mesh import shapes_match
    assert shapes_match([.13, .14, .27], [.145, .163, .302])
    # The paint roller's input mesh is only the head.
    assert not shapes_match([.065, .066, .216], [.134, .234, .333])
