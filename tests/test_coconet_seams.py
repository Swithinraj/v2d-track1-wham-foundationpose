import numpy as np


def test_detects_artificial_offset_at_a_window_join():
    from coconet_seams import seam_statistics
    points=np.c_[np.arange(300)*.001,np.zeros(300),np.ones(300)]
    points[96:,0]+=.1
    result=seam_statistics(points,96)
    assert result['boundary_mean_cm']>1
    assert result['interior_mean_cm']<1e-10


def test_real_acceleration_is_present_in_both_regions():
    from coconet_seams import seam_statistics
    points=np.c_[np.arange(300,dtype=float)**2*.0001,np.zeros(300),np.ones(300)]
    result=seam_statistics(points,96)
    np.testing.assert_allclose(result['boundary_mean_cm'],.02,atol=1e-10)
    np.testing.assert_allclose(result['interior_mean_cm'],.02,atol=1e-10)


def test_scoring_gaps_are_not_treated_as_motion_discontinuities():
    from coconet_seams import seam_statistics
    points=np.c_[np.arange(300)*.001,np.zeros(300),np.ones(300)]
    points[96:,0]+=.1
    frames=np.r_[np.arange(90),np.arange(102,300)]
    result=seam_statistics(points,96,frames)
    assert result['boundary_mean_cm']<1e-10
