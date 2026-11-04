import numpy as np
import pytest
from object_camera_scale import scale_objects


def test_scaling_keeps_every_pixel_and_changes_only_objects():
    rng = np.random.default_rng(2)
    T = 12
    data = dict(pose=rng.normal(size=(T, 136)).astype(np.float32), scales=np.ones(68, np.float32), shape=np.zeros(45, np.float32),
                object_rotation=np.tile(np.eye(3), (T, 1, 1)), object_translation=rng.normal(size=(T, 3)) + [0, 0, 2.5],
                object_scale=np.array(0.8))
    local = rng.normal(size=(50, 3)) * .05
    out = scale_objects(data, 1.01)
    project = lambda d: (lambda p: p[..., :2] / p[..., 2:])(float(d["object_scale"]) * local[None] @ np.eye(3) + d["object_translation"][:, None])
    np.testing.assert_allclose(project(out), project(data), atol=1e-12)
    np.testing.assert_allclose(out["object_translation"], data["object_translation"] * 1.01)
    assert float(out["object_scale"]) == pytest.approx(0.808)
    for key in ("pose", "scales", "shape", "object_rotation"):
        assert out[key] is data[key]
    with pytest.raises(ValueError):
        scale_objects(data, 0.)


def test_cli_per_episode_factors(tmp_path, monkeypatch):
    import json, sys
    import object_camera_scale
    T = 5
    src = tmp_path / "in"; src.mkdir()
    for e in (0, 1):
        np.savez(src / f"episode_{e:06d}.npz", object_translation=np.ones((T, 3)) * (e + 1), object_scale=np.array(1.),
                 pose=np.zeros((T, 136), np.float32))
    (tmp_path / "f.json").write_text(json.dumps({"1": 1.05}))
    monkeypatch.setattr(sys, "argv", ["x", "--input", str(src), "--out", str(tmp_path / "out"), "--factors", str(tmp_path / "f.json")])
    object_camera_scale.main()
    with np.load(tmp_path / "out/episode_000000.npz") as a, np.load(tmp_path / "out/episode_000001.npz") as b:
        np.testing.assert_array_equal(a["object_translation"], np.ones((T, 3)))
        np.testing.assert_allclose(b["object_translation"], np.ones((T, 3)) * 2 * 1.05)
        assert float(b["object_scale"]) == pytest.approx(1.05)
