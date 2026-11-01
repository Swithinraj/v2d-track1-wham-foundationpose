import warnings

import numpy as np
import pytest
from PIL import Image

from moge_lock import floor_from_master, lock


def write_depths(path, maps):
    for frame, depth in enumerate(maps):
        pixels = np.where(np.isfinite(depth) & (depth > 0),
                          np.round(65535 / (depth + 1)), 0).astype(np.uint16)
        Image.fromarray(pixels).save(path / f"{frame:06d}.png")


def test_recovers_global_scale_without_removing_scene_geometry(tmp_path):
    plane = np.linspace(2, 4, 100).reshape(10, 10)
    scales = np.array([0.96, 0.98, 1, 1.02, 1.04])
    write_depths(tmp_path, scales[:, None, None] * plane)
    a, master, static = lock(tmp_path, 5, 1)
    np.testing.assert_allclose(a, scales, atol=0.0002)
    np.testing.assert_allclose(master, plane, atol=0.0004)
    assert static.all()


@pytest.mark.parametrize("valid_pixels", [0, 1, 10])
def test_insufficient_background_abstains_without_nan_scales(tmp_path, valid_pixels):
    maps = np.full((5, 10, 10), np.nan)
    maps.reshape(5, -1)[:, :valid_pixels] = 3
    write_depths(tmp_path, maps)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        a, master, static = lock(tmp_path, 5, 1)
    np.testing.assert_array_equal(a, np.ones(5))
    assert not static.any()


def test_one_empty_frame_does_not_poison_valid_frames(tmp_path):
    scales = np.array([0.98, 1, 1, 1.02, 1])
    maps = np.broadcast_to(scales[:, None, None] * 3, (5, 10, 10)).copy()
    maps[2] = np.nan
    write_depths(tmp_path, maps)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        a, master, static = lock(tmp_path, 5, 1)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, scales, atol=0.0002)
    assert np.isfinite(master).all()


@pytest.mark.parametrize("frames,step", [(0, 1), (2, 0)])
def test_rejects_invalid_sampling_arguments(tmp_path, frames, step):
    with pytest.raises(ValueError, match="positive"):
        lock(tmp_path, frames, step)


def test_floor_rejects_empty_evidence():
    with pytest.raises(ValueError, match="floor"):
        floor_from_master(np.full((10, 10), np.nan), np.zeros((10, 10), bool),
                          1, dict(fx=100, fy=100, cx=5, cy=5), 1,
                          np.array([0., -1., 0.]), np.zeros((5, 22, 3)),
                          np.random.default_rng(0))


def test_floor_rejects_collinear_evidence():
    master = np.full((1, 40), 3.)
    with pytest.raises(ValueError, match="floor"):
        floor_from_master(master, np.ones_like(master, bool), 1,
                          dict(fx=100, fy=100, cx=20, cy=0), 1,
                          np.array([0., -1., 0.]), np.zeros((5, 22, 3)),
                          np.random.default_rng(0))
