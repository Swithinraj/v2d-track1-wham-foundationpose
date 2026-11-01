import numpy as np
import pytest


def test_group_median_is_kept_and_larger_objects_come_closer():
    from object_relative_size import relative_scales
    kappa = relative_scales(["a", "a", "a"], [1., 1.1, 1.2], [False] * 3)
    np.testing.assert_allclose(kappa, [1.1, 1., 1.1 / 1.2])


def test_floor_episodes_are_untouched_and_do_not_enter_the_median():
    from object_relative_size import relative_scales
    kappa = relative_scales(["a", "a", "a"], [1., 1.2, 5.], [False, False, True])
    assert kappa[2] == 1.
    np.testing.assert_allclose(kappa[:2], [1.1, 1.1 / 1.2])


def test_alone_in_its_group_stays():
    from object_relative_size import relative_scales
    np.testing.assert_array_equal(relative_scales(["a", "b", "b"], [1., 2., 3.], [False, False, True]), [1., 1., 1.])


def test_strength_scales_the_correction():
    from object_relative_size import relative_scales
    kappa = relative_scales(["a", "a"], [1., 1.21], [False, False], strength=.5)
    middle = np.median([1., 1.21])
    np.testing.assert_allclose(kappa, [(middle / 1.) ** .5, (middle / 1.21) ** .5])


def test_invalid_input():
    from object_relative_size import relative_scales
    with pytest.raises(ValueError):
        relative_scales(["a"], [1., 2.], [False])
    with pytest.raises(ValueError):
        relative_scales(["a"], [0.], [False])
