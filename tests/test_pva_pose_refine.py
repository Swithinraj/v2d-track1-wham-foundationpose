import numpy as np
import pytest
from pva_pose_refine import temporal_support, apply_pose_updates, displacement_gate


def test_pose_derivatives_exclude_scored_gaps():
    pair,triplet=temporal_support([5,6,7,11,12,13])
    np.testing.assert_array_equal(pair,[1,1,0,1,1])
    np.testing.assert_array_equal(triplet,[1,0,0,1])


def test_reject_reordered_or_duplicate_frame_ids():
    for frames in ([0,1,1],[0,3,2],[0,.5,1],[-1,0,1],[0,1,np.nan]):
        with pytest.raises(ValueError,match='increasing'):
            temporal_support(frames)


def test_float64_export_preserves_locked_columns_anchor_and_interpolates_gaps():
    base=np.full((6,4),1.123456789012345,dtype=np.float64)
    frames=np.array([1,2,5])
    delta=np.array([[.1,.2],[.2,.3],[.5,.6]],dtype=np.float32)
    result=apply_pose_updates(base,frames,delta,[0,2])
    np.testing.assert_array_equal(result[:,[1,3]],base[:,[1,3]])
    np.testing.assert_array_equal(result[1],base[1])
    np.testing.assert_allclose(result[3,[0,2]]-base[3,[0,2]],[.3,.4],atol=1e-7)
    assert result.dtype==np.float64
    np.testing.assert_array_equal(base,np.full((6,4),1.123456789012345))


def test_displacement_gate_checks_articulated_pose_inside_unscored_gap():
    # A rotation delta can move a joint farther at an intervening articulation
    # than at either endpoint: the gate must inspect the exported full sequence.
    base=np.zeros((5,2,3));candidate=base.copy()
    candidate[[0,4],1,0]=.02
    candidate[2,1,0]=.06
    assert displacement_gate(base[[0,4]],candidate[[0,4]])[0]
    accepted,maximum=displacement_gate(base,candidate)
    assert not accepted and maximum==.06
    candidate[2,1,0]=np.nan
    assert not displacement_gate(base,candidate)[0]
