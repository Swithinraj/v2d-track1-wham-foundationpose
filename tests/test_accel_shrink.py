import numpy as np
import pytest
from accel_shrink import acceleration, shrink, stretches


def noisy_track(T=90, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    return np.hstack([.01 * t, np.sin(t / 15.), -.002 * t]) + rng.normal(scale=.003, size=(T, 3))


def second(x):
    return x[2:] - 2 * x[1:-1] + x[:-2]


def test_k_one_is_identity_and_linear_motion_is_untouched():
    x = noisy_track()
    frames = np.arange(len(x))
    np.testing.assert_allclose(shrink(x, frames, 1.), x, atol=1e-12)
    line = np.arange(50)[:, None] * np.array([[.02, -.01, .005]]) + 1.
    np.testing.assert_allclose(shrink(line, np.arange(50), .3), line, atol=1e-9)


def test_acceleration_is_exactly_k_times_the_residual_and_anchor_is_pinned():
    x = noisy_track()
    frames = np.arange(10, 80)
    y = shrink(x, frames, .6)
    assert np.array_equal(y[frames[0]], x[frames[0]])
    from floor_root import whittaker
    smooth = whittaker(x[frames], np.ones(len(frames)), 1e4)
    np.testing.assert_allclose(second(y[frames]), second(smooth) + .6 * (second(x[frames]) - second(smooth)), atol=1e-12)
    assert acceleration(y, frames) < .7 * acceleration(x, frames)
    np.testing.assert_array_equal(y[:10], x[:10])
    np.testing.assert_array_equal(y[80:], x[80:])


def test_stretches_are_independent_across_unscored_gaps():
    x = noisy_track(120)
    x[60:] += 5.0                                   # a jump inside the unscored gap
    frames = np.r_[np.arange(5, 40), np.arange(70, 110)]
    assert [len(p) for p in stretches(frames)] == [35, 40]
    y = shrink(x, frames, .5)
    reference = shrink(x - np.where(np.arange(120)[:, None] >= 60, 5.0, 0.), frames, .5)
    np.testing.assert_allclose(y[70:110] - 5.0, reference[70:110], atol=1e-9)
    np.testing.assert_array_equal(y[40:70], x[40:70])


def test_cli_changes_only_requested_columns_and_reports_the_original(tmp_path, monkeypatch):
    import json, sys
    import accel_shrink
    T = 60
    frames = list(range(5, 55))
    pose = np.zeros((T, 136), np.float32); pose[:, :3] = noisy_track(T); pose[:, 3:] = .1
    episode = dict(pose=pose, scales=np.ones(68, np.float32), shape=np.zeros(45, np.float32),
                   object_rotation=np.tile(np.eye(3), (T, 1, 1)), object_translation=noisy_track(T, 7), object_scale=np.array(1.))
    source = tmp_path / "in"; source.mkdir()
    np.savez(source / "episode_000000.npz", **episode)
    (tmp_path / "scored.json").write_text(json.dumps({"0": frames}))
    monkeypatch.setattr(sys, "argv", ["accel_shrink.py", "--input", str(source), "--out", str(tmp_path / "out"), "--k", ".5",
                                      "--root", "--scored", str(tmp_path / "scored.json")])
    accel_shrink.main()
    with np.load(tmp_path / "out/episode_000000.npz") as written:
        np.testing.assert_array_equal(written["pose"][:, 3:], pose[:, 3:])
        np.testing.assert_array_equal(written["object_translation"], episode["object_translation"])
        assert np.array_equal(written["pose"][5, :3], pose[5, :3]) and not np.array_equal(written["pose"][:, :3], pose[:, :3])
    report = json.loads((tmp_path / "out/accel_shrink_report.json").read_text())["episodes"]["episode_000000"]["root_translation_dm"]
    assert report["acceleration_after"] < .75 * report["acceleration_before"]


def test_rejects_invalid_inputs():
    x = noisy_track()
    with pytest.raises(ValueError):
        shrink(x, np.arange(len(x)), 1.5)
    with pytest.raises(ValueError):
        shrink(x, np.array([5, 4, 6]), .5)
    with pytest.raises(ValueError):
        shrink(x, np.arange(len(x) + 1), .5)
    bad = x.copy(); bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        shrink(bad, np.arange(len(x)), .5)
