import numpy as np
import pytest
from validate_pva_candidate import check_preserved, independent_keypoint_errors, MHR_JOINTS, MP_JOINTS


def fixture():
    return dict(pose=np.zeros((5,136)),scales=np.zeros(68),shape=np.zeros(45),
                object_rotation=np.tile(np.eye(3),(5,1,1)),object_translation=np.zeros((5,3)),object_scale=np.array(1.))


def test_validator_rejects_changed_identity_objects_anchor_and_root_only_articulation():
    base=fixture()
    for key in ('shape','object_translation','pose'):
        candidate={k:v.copy() for k,v in base.items()}
        candidate[key].flat[0]=1
        with pytest.raises(ValueError): check_preserved(base,candidate,[0,1,4],root_only=True)
    candidate={k:v.copy() for k,v in base.items()}
    candidate['pose'][2,15]=.1
    with pytest.raises(ValueError): check_preserved(base,candidate,[0,1,4],root_only=True)
    candidate['pose'][2,15]=0;candidate['pose'][2,0]=.1
    check_preserved(base,candidate,[0,1,4],root_only=True)


def test_independent_keypoints_choose_actor_from_baseline_and_keep_same_targets():
    base=np.zeros((5,127,3));base[...,2]=2
    candidate=base.copy();candidate[...,0]=.02
    kp=np.zeros((5,2,33,5));kp[...,3]=1
    kp[:,0,:,:2]=100
    kp[:,1,MP_JOINTS,0]=1 # true actor is second person; x=1px at baseline.
    K=np.diag([100.,100.,1.])
    report=independent_keypoint_errors(base,candidate,kp,K,[0,1,4])
    assert report['observations']==36
    assert report['mean_before_px']==1
    assert report['mean_after_px']==0
    kp[:,:,:,3]=0
    assert independent_keypoint_errors(base,candidate,kp,K,[0,1,4])['mean_before_px'] is None
