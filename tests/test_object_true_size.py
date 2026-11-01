import numpy as np
import pytest


def test_object_names_from_tasks():
    from object_true_size import object_kind
    assert object_kind("Pick up the foam roll on the table. Use your hand") == "pink_foam_roll"
    assert object_kind("Pick up the roller on the table. Slowly roll it") == "paint_roller"
    assert object_kind("Walk over to the iron on the floor and pick it") == "iron"
    assert object_kind("Pick up the pan from the table using one hand") == "black_pan"
    assert object_kind("Walk over to the hula hoop on the floor") == "hula_hoop"
    with pytest.raises(ValueError):
        object_kind("Dance in the environment")


def test_floor_evidence_calibrates_episodes_without_it():
    from object_true_size import calibrated_sizes
    # True size 1.0. Episode 0 is on the floor and 5 % too far (size 1.05, floor kappa 1/1.05).
    # Episode 1 only sits on a table and is 8 % too far: only size constancy can correct it.
    sizes, kappas = calibrated_sizes(["a", "a"], [1.05, 1.08], [1 / 1.05, np.nan], .97)
    np.testing.assert_allclose(sizes["a"], 1.)
    np.testing.assert_allclose(kappas, [1 / 1.05, 1 / 1.08])


def test_object_without_any_floor_evidence_uses_the_default():
    from object_true_size import calibrated_sizes
    sizes, kappas = calibrated_sizes(["a", "a", "a"], [1., 1.1, 1.2], [np.nan] * 3, .97)
    np.testing.assert_allclose(sizes["a"], 1.1 * .97)
    np.testing.assert_allclose(kappas[1], .97)
    np.testing.assert_allclose(kappas[0], 1.1 * .97)


def test_objects_do_not_mix():
    from object_true_size import calibrated_sizes
    sizes, kappas = calibrated_sizes(["a", "b"], [1., 2.], [.9, 1.1], .97)
    np.testing.assert_allclose([sizes["a"], sizes["b"]], [.9, 2.2])
    np.testing.assert_allclose(kappas, [.9, 1.1])


def test_invalid_input_is_rejected():
    from object_true_size import calibrated_sizes
    with pytest.raises(ValueError):
        calibrated_sizes(["a"], [1., 2.], [np.nan], .97)
    with pytest.raises(ValueError):
        calibrated_sizes(["a"], [0.], [np.nan], .97)
