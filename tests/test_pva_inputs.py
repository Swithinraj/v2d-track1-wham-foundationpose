import numpy as np
import pytest

from pva_inputs import box_from_mask, expand_boxes, crop_rgb, validate_sequence


def test_mask_box_uses_exclusive_upper_bounds():
    mask = np.zeros((20, 30), np.uint8)
    mask[4:15, 6:18] = 255
    np.testing.assert_array_equal(box_from_mask(mask), [6, 4, 18, 15])
    assert np.isnan(box_from_mask(np.zeros_like(mask))).all()


def test_aspect_crop_contains_full_person_and_interpolates_short_missing_span():
    source = np.array([[10, 20, 110, 70], [np.nan]*4, [30, 20, 130, 70]])
    boxes = expand_boxes(source, padding=1.0)
    np.testing.assert_allclose((boxes[:, 2]-boxes[:, 0])/(boxes[:, 3]-boxes[:, 1]), .75)
    assert np.all(boxes[[0, 2], :2] <= source[[0, 2], :2])
    assert np.all(boxes[[0, 2], 2:] >= source[[0, 2], 2:])
    np.testing.assert_allclose(boxes[1], (boxes[0]+boxes[2])/2)


def test_no_track_or_long_missing_span_fails():
    with pytest.raises(ValueError, match='no valid'):
        expand_boxes(np.full((12,4), np.nan))
    boxes = np.tile([0.,0.,30.,40.], (20,1))
    boxes[2:15] = np.nan
    with pytest.raises(ValueError, match='missing span'):
        expand_boxes(boxes, max_gap=10)


def test_crop_keeps_rgb_channels_and_expected_shape():
    image = np.zeros((256,192,3),np.uint8)
    image[:] = [17,101,203]
    result = crop_rgb(image, np.array([0.,0.,192.,256.]))
    assert result.shape == (256,192,3)
    np.testing.assert_array_equal(result[128,96], [17,101,203])


def test_reject_wrong_fps_and_misaligned_frame_counts():
    validate_sequence(np.arange(5), 30., 5)
    with pytest.raises(ValueError,match='30 FPS'):
        validate_sequence(np.arange(5), 25., 5)
    with pytest.raises(ValueError,match='consecutive'):
        validate_sequence(np.array([0,1,3,4,5]), 30., 5)
    with pytest.raises(ValueError,match='frame count'):
        validate_sequence(np.arange(4), 30., 5)
