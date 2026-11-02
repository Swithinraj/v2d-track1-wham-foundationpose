import numpy as np
import pytest

from pva_root_dynamics import solve_delta, torso_residuals


def test_dynamics_correction_recovers_known_motion_and_keeps_anchor():
    t=np.arange(101)
    truth=np.stack([.01*np.sin(2*np.pi*t/10),np.zeros(101),np.zeros(101)],1)
    result,report=solve_delta(np.diff(truth,axis=0),np.diff(truth,n=2,axis=0),
                              np.ones(100),np.ones(99),anchor=50,
                              velocity_weight=1e6,acceleration_weight=1e6,max_shift=.1)
    assert report['accepted']
    assert np.array_equal(result[50],np.zeros(3))
    assert np.sqrt(np.mean((result-truth)**2))<1e-4


def test_no_motion_evidence_returns_exact_zero():
    d,r=solve_delta(np.ones((9,3)),np.ones((8,3)),np.zeros(9),np.zeros(8),anchor=3)
    np.testing.assert_array_equal(d,np.zeros((10,3)))
    assert not r['accepted']


def test_implausible_shift_is_rejected_as_a_whole():
    d,r=solve_delta(np.ones((19,3)),np.zeros((18,3)),np.ones(19),np.ones(18),
                     anchor=0,velocity_weight=1000,acceleration_weight=0,max_shift=.05)
    assert not r['accepted']
    np.testing.assert_array_equal(d,np.zeros((20,3)))


def test_torso_targets_convert_si_units_and_ignore_unscored_stencils():
    T=8;J=4;fps=30.
    base=np.zeros((T,J,3));v=np.ones((T-1,J,3))*.3;a=np.ones((T-2,J,3))*9.
    confidence=np.ones((T,J));scored=np.array([1,1,1,0,1,1,1,1],bool)
    rv,ra,wv,wa=torso_residuals(base,v,a,confidence,scored,fps=fps)
    np.testing.assert_allclose(rv[wv>0],.01)
    np.testing.assert_allclose(ra[wa>0],.01)
    np.testing.assert_array_equal(rv[wv==0],0.)
    np.testing.assert_array_equal(ra[wa==0],0.)
    np.testing.assert_array_equal(wv>0,[1,1,0,0,1,1,1])
    np.testing.assert_array_equal(wa>0,[1,0,0,0,1,1])


def test_nonfinite_input_and_invalid_anchor_fail():
    v=np.zeros((9,3));a=np.zeros((8,3))
    with pytest.raises(ValueError,match='anchor'):
        solve_delta(v,a,np.ones(9),np.ones(8),anchor=10)
    v[0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        solve_delta(v,a,np.ones(9),np.ones(8),anchor=0)


def test_reject_fractional_anchor_and_nonfinite_hyperparameters():
    v=np.zeros((9,3));a=np.zeros((8,3))
    with pytest.raises(ValueError,match='anchor'):
        solve_delta(v,a,np.ones(9),np.ones(8),anchor=1.5)
    with pytest.raises(ValueError,match='finite'):
        solve_delta(v,a,np.ones(9),np.ones(8),anchor=0,acceleration_weight=np.nan)
