import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation


def rows_for(episode, frames, pose, rotation, translation, scales, shape, scale, vertices, faces, commit="c"):
    """Long rows in the official layout (array order, 3 numbers per row, unused tail 0)."""
    from parquet_episodes import ARRAYS, EPISODE_FRAME
    out = []

    def add(name, frame, values):
        index = ARRAYS.index(name)
        for point in range(len(values)):
            out.append((f"t1_{episode:06d}_{frame:06d}_{index}_{point:06d}", *values[point], commit))

    def triples(flat):
        flat = np.asarray(flat, float)
        padded = np.r_[flat, np.zeros(-len(flat) % 3)]
        return padded.reshape(-1, 3)

    add("mhr_scales", EPISODE_FRAME, triples(scales))
    add("mhr_shape", EPISODE_FRAME, triples(shape))
    add("object_scale", EPISODE_FRAME, [[scale, 0, 0]])
    add("object_mesh_vertices", EPISODE_FRAME, vertices)
    add("object_mesh_faces", EPISODE_FRAME, faces)
    for k, frame in enumerate(frames):
        add("mhr_pose", int(frame), triples(pose[k]))
        add("object_rotation", int(frame), rotation[k])
        add("object_translation", int(frame), [translation[k]])
    return out


def small_episode(frames):
    T = len(frames)
    pose = np.zeros((T, 136))
    pose[:, 0] = np.asarray(frames, float) * .1
    pose[:, 5] = np.asarray(frames, float) * .01
    rotation = Rotation.from_rotvec(np.c_[np.zeros(T), np.asarray(frames) * .02, np.zeros(T)]).as_matrix()
    translation = np.c_[np.asarray(frames) * .01, np.ones(T), np.full(T, 3.)]
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 9, 9]], float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [0, 0, 0], [0, 0, 0]], float)
    return dict(pose=pose, rotation=rotation, translation=translation, scales=np.arange(68.) / 68,
                shape=np.arange(45.) / 45, scale=1.25, vertices=vertices, faces=faces)


def frame_of(episode):
    columns = ["row_id", "x", "y", "z", "code_commit_url"]
    return pd.DataFrame(episode, columns=columns)


def test_scored_values_survive_exactly():
    from parquet_episodes import episodes_from_rows
    frames = np.array([3, 4, 5, 6, 10, 11, 12])
    e = small_episode(frames)
    data = episodes_from_rows(frame_of(rows_for(2, frames, e["pose"], e["rotation"], e["translation"], e["scales"],
                                                e["shape"], e["scale"], e["vertices"], e["faces"])))[2]
    np.testing.assert_array_equal(data["frames"], frames)
    np.testing.assert_allclose(data["pose"][frames], e["pose"], atol=1e-12)
    np.testing.assert_allclose(data["object_translation"][frames], e["translation"], atol=1e-12)
    np.testing.assert_allclose(data["object_rotation"][frames], e["rotation"], atol=1e-12)
    np.testing.assert_allclose(data["scales"], e["scales"])
    np.testing.assert_allclose(data["shape"], e["shape"])
    assert data["object_scale"] == 1.25


def test_gaps_are_bridged_and_ends_held():
    from parquet_episodes import episodes_from_rows
    frames = np.array([3, 4, 5, 6, 10, 11, 12])
    e = small_episode(frames)
    data = episodes_from_rows(frame_of(rows_for(0, frames, e["pose"], e["rotation"], e["translation"], e["scales"],
                                                e["shape"], e["scale"], e["vertices"], e["faces"])))[0]
    assert len(data["pose"]) == 13
    np.testing.assert_allclose(data["pose"][:3], np.broadcast_to(e["pose"][0], (3, 136)))
    # Linear across the unscored gap 7, 8, 9.
    np.testing.assert_allclose(data["pose"][8, 0], .8)
    np.testing.assert_allclose(data["object_translation"][8, 0], .08)
    rotation = data["object_rotation"]
    np.testing.assert_allclose(rotation @ rotation.swapaxes(1, 2), np.broadcast_to(np.eye(3), rotation.shape), atol=1e-9)
    np.testing.assert_allclose(Rotation.from_matrix(rotation[8]).as_rotvec(), [0, .16, 0], atol=1e-9)


def test_padding_faces_and_unused_vertices_are_removed():
    from parquet_episodes import episodes_from_rows
    frames = np.array([0, 1, 2])
    e = small_episode(frames)
    data = episodes_from_rows(frame_of(rows_for(0, frames, e["pose"], e["rotation"], e["translation"], e["scales"],
                                                e["shape"], e["scale"], e["vertices"], e["faces"])))[0]
    assert len(data["mesh_faces"]) == 4
    assert data["mesh_faces"].max() < 4
    assert len(data["mesh_vertices"]) == 4


def test_missing_rows_are_an_error():
    from parquet_episodes import episodes_from_rows
    frames = np.array([0, 1, 2])
    e = small_episode(frames)
    rows = frame_of(rows_for(0, frames, e["pose"], e["rotation"], e["translation"], e["scales"], e["shape"], e["scale"],
                             e["vertices"], e["faces"]))
    with pytest.raises(ValueError, match="episode 0"):
        episodes_from_rows(rows.iloc[1:])


def test_directory_writes_npz_mesh_and_frames(tmp_path):
    import json
    import trimesh
    from parquet_episodes import episodes_from_rows, write_episodes
    frames = np.array([0, 1, 2, 3])
    e = small_episode(frames)
    data = episodes_from_rows(frame_of(rows_for(7, frames, e["pose"], e["rotation"], e["translation"], e["scales"],
                                                e["shape"], e["scale"], e["vertices"], e["faces"])))
    write_episodes(data, tmp_path / "out")
    with np.load(tmp_path / "out/episode_000007.npz") as archive:
        assert set(archive.files) == {"pose", "scales", "shape", "object_rotation", "object_translation", "object_scale"}
        assert archive["pose"].shape == (4, 136)
    mesh = trimesh.load(tmp_path / "out/episode_000007_object.glb", force="mesh", process=False)
    assert len(mesh.faces) == 4
    assert json.loads((tmp_path / "out/scored_frames.json").read_text())["7"] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="exists"):
        write_episodes(data, tmp_path / "out")
