import numpy as np


def box_scene(depth_error=1.0, frames=60, lift_from=None):
    """A 40 cm cube resting on the floor at y = 1.3 in a camera-at-origin frame (y down, floor normal up)."""
    half = .2
    vertices = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)], float)
    rotation = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
    true = np.tile([.3, 1.3 - half, 3.], (frames, 1))
    if lift_from is not None:
        true[lift_from:, 1] -= np.linspace(0, .6, frames - lift_from)
    return vertices, rotation, true * depth_error, depth_error


NORMAL = np.array([0., -1., 0.])
FLOOR = -1.3


def test_resting_object_is_scaled_onto_the_floor_about_the_camera():
    from object_floor_scale import floor_scale
    vertices, rotation, translation, _ = box_scene(depth_error=1.05)
    kappa, report = floor_scale(vertices, rotation, translation, 1.05, NORMAL, FLOOR)
    assert report["accepted"], report
    np.testing.assert_allclose(kappa, 1 / 1.05, atol=1e-6)
    assert report["evidence"] == "rest"


def test_correct_object_is_left_alone():
    from object_floor_scale import floor_scale
    vertices, rotation, translation, _ = box_scene()
    kappa, report = floor_scale(vertices, rotation, translation, 1., NORMAL, FLOOR)
    np.testing.assert_allclose(kappa, 1., atol=1e-6)


def test_lifted_object_uses_its_time_on_the_floor():
    from object_floor_scale import floor_scale
    vertices, rotation, translation, _ = box_scene(depth_error=1.04, lift_from=30)
    kappa, report = floor_scale(vertices, rotation, translation, 1.04, NORMAL, FLOOR)
    assert report["accepted"], report
    np.testing.assert_allclose(kappa, 1 / 1.04, atol=1e-3)


def test_still_in_the_hand_above_the_floor_does_not_override_the_lowest_plateau():
    from object_floor_scale import floor_scale
    vertices, rotation, translation, _ = box_scene()
    translation = translation.copy()
    translation[:15, 1] += .03          # 3 cm below the floor for a while
    translation[15:, 1] -= .06          # then still, held 6 cm above it, for longer
    kappa, report = floor_scale(vertices, rotation, translation, 1., NORMAL, FLOOR)
    assert report["accepted"], report
    np.testing.assert_allclose(kappa, 1.3 / 1.33, atol=1e-3)


def test_object_on_a_table_abstains():
    from object_floor_scale import floor_scale
    vertices, rotation, translation, _ = box_scene()
    translation = translation.copy()
    translation[:, 1] -= .8
    kappa, report = floor_scale(vertices, rotation, translation, 1., NORMAL, FLOOR)
    assert not report["accepted"]
    assert kappa == 1.


def test_object_below_the_floor_without_rest_uses_the_violation():
    from object_floor_scale import floor_scale
    vertices, rotation, translation, _ = box_scene(depth_error=1.04)
    translation = translation.copy()
    translation[:, 0] += np.linspace(0, 3., len(translation))
    kappa, report = floor_scale(vertices, rotation, translation, 1.04, NORMAL, FLOOR)
    assert report["accepted"], report
    assert report["evidence"] == "below floor"
    assert kappa < 1


def test_implausible_scale_is_rejected_not_clipped():
    from object_floor_scale import floor_scale
    vertices, rotation, translation, _ = box_scene(depth_error=1.4)
    kappa, report = floor_scale(vertices, rotation, translation, 1.4, NORMAL, FLOOR)
    assert not report["accepted"]
    assert kappa == 1.


def test_directory_scales_translation_and_scale_only(tmp_path):
    import json
    import trimesh
    from object_floor_scale import process_directory
    vertices, rotation, translation, _ = box_scene(depth_error=1.05)
    base = tmp_path / "base"
    base.mkdir()
    faces = trimesh.creation.box(extents=[.4, .4, .4]).faces
    trimesh.Trimesh(vertices, trimesh.creation.box(extents=[.4, .4, .4]).faces, process=False).export(base / "episode_000000_object.glb")
    assert len(faces)
    pose = np.arange(60 * 136, dtype=float).reshape(60, 136)
    np.savez(base / "episode_000000.npz", pose=pose, scales=np.ones(68), shape=np.zeros(45), object_rotation=rotation,
             object_translation=translation, object_scale=np.float64(1.05))
    report = process_directory(base, tmp_path / "out", {"episode_000000": NORMAL.tolist()}, {"episode_000000": FLOOR},
                               {"0": list(range(60))})
    with np.load(tmp_path / "out/episode_000000.npz") as out:
        kappa = report["episode_000000"]["kappa"]
        np.testing.assert_allclose(out["object_translation"], translation * kappa)
        np.testing.assert_allclose(out["object_scale"], 1.05 * kappa)
        np.testing.assert_array_equal(out["pose"], pose)
        np.testing.assert_array_equal(out["object_rotation"], rotation)
    assert json.loads((tmp_path / "out/object_floor_scale_report.json").read_text())["episode_000000"]["accepted"]


def directory(tmp_path, offset=0.):
    import trimesh
    vertices, rotation, translation, _ = box_scene()
    translation = translation.copy()
    translation[:, 1] -= offset
    base = tmp_path / "base"
    base.mkdir()
    trimesh.Trimesh(vertices, trimesh.creation.box(extents=[.4, .4, .4]).faces, process=False).export(base / "episode_000000_object.glb")
    np.savez(base / "episode_000000.npz", pose=np.zeros((60, 136)), scales=np.ones(68), shape=np.zeros(45),
             object_rotation=rotation, object_translation=translation, object_scale=np.float64(1.))
    return base, translation


def test_fixed_scale_ignores_the_evidence(tmp_path):
    from object_floor_scale import process_directory
    base, translation = directory(tmp_path)
    report = process_directory(base, tmp_path / "out", {"episode_000000": NORMAL.tolist()}, {"episode_000000": FLOOR},
                               {"0": list(range(60))}, fixed_kappa=.97)
    assert report["episode_000000"]["kappa"] == .97
    with np.load(tmp_path / "out/episode_000000.npz") as out:
        np.testing.assert_allclose(out["object_translation"], translation * .97)


def test_abstain_scale_only_touches_objects_without_evidence(tmp_path):
    from object_floor_scale import process_directory
    base, translation = directory(tmp_path, offset=.8)
    report = process_directory(base, tmp_path / "out", {"episode_000000": NORMAL.tolist()}, {"episode_000000": FLOOR},
                               {"0": list(range(60))}, abstain_kappa=.97)
    assert report["episode_000000"]["kappa"] == .97
    base2 = tmp_path / "second"
    base2.mkdir()
    base3, _ = directory(base2)
    report = process_directory(base3, base2 / "out", {"episode_000000": NORMAL.tolist()}, {"episode_000000": FLOOR},
                               {"0": list(range(60))}, abstain_kappa=.97)
    np.testing.assert_allclose(report["episode_000000"]["kappa"], 1., atol=1e-6)
