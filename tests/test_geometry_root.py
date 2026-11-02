import numpy as np
import pytest
from geometry_root import fit_translation, project, smooth_anchored


def scene():
    rng=np.random.default_rng(19)
    joints=rng.normal(size=(16,3))*[.3,.7,.12]+[.1,.2,3.5]
    K=np.array([[1100.,0,640],[0,1050.,360],[0,0,1.]])
    return joints,K


def test_recover_metric_translation_with_unequal_focals():
    j,K=scene();truth=np.array([.02,-.01,.08])
    delta,report=fit_translation(j,project(j+truth,K),np.ones(len(j)),K)
    assert report['accepted']
    np.testing.assert_allclose(delta,truth,atol=1e-7)


def test_low_confidence_missing_keypoints_and_outlier_do_not_drive_fit():
    j,K=scene();truth=np.array([.01,.02,-.04]);uv=project(j+truth,K);conf=np.ones(len(j))
    uv[0]+=100;uv[1]=np.nan;conf[0]=0
    delta,report=fit_translation(j,uv,conf,K)
    assert report['accepted']
    np.testing.assert_allclose(delta,truth,atol=1e-7)
    uv=project(j+truth,K);uv[0]+=100
    delta,report=fit_translation(j,uv,conf*0+1,K)
    assert np.linalg.norm(delta-truth)<.02


def test_reject_unobservable_depth_and_insufficient_support():
    j,K=scene();j[:]=[0,0,4]
    _,report=fit_translation(j,project(j,K),np.ones(len(j)),K)
    assert not report['accepted']
    j,K=scene()
    _,report=fit_translation(j,project(j,K),np.zeros(len(j)),K)
    assert not report['accepted']


def test_temporal_fit_uses_real_grid_and_hard_anchor_without_fake_gap_observations():
    T=60;t=np.arange(T);raw=np.stack([.001*t,np.zeros(T),.0005*t],1)
    weights=np.ones_like(raw);weights[20:30]=0;raw[20:30]=999
    fitted,report=smooth_anchored(raw,weights,anchor=0,lam=10,prior=1e-8)
    assert report['accepted']
    np.testing.assert_array_equal(fitted[0],np.zeros(3))
    np.testing.assert_allclose(fitted[:,0],t*.001,atol=1e-6)
    weights[0]=0
    _,report=smooth_anchored(raw,weights,anchor=0)
    assert not report['accepted']


def test_reject_nonfinite_and_fractional_anchor():
    with pytest.raises(ValueError):smooth_anchored(np.zeros((5,3)),np.ones((5,3)),anchor=.5)
    j,K=scene();K[0,0]=np.nan
    with pytest.raises(ValueError):fit_translation(j,project(j,np.eye(3)),np.ones(len(j)),K)
