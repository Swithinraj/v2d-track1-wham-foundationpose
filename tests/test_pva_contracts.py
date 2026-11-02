import numpy as np
import pytest

from pva_contracts import motion_fingerprint, require_motion_match


def test_joint_cache_cannot_be_applied_to_another_candidate():
    data={'pose':np.zeros((5,136),np.float32),'shape':np.zeros(45,np.float32),'scales':np.zeros(68,np.float32)}
    original=motion_fingerprint(data)
    require_motion_match({'source_motion_sha256':np.array(original)},data)
    data['pose'][3,0]=.01
    with pytest.raises(ValueError,match='different'):
        require_motion_match({'source_motion_sha256':np.array(original)},data)
    with pytest.raises(ValueError,match='provenance'):
        require_motion_match({},data)


def test_motion_identity_affects_fingerprint_but_objects_do_not():
    data={'pose':np.zeros((5,136)),'shape':np.zeros(45),'scales':np.zeros(68)}
    h=motion_fingerprint(data)
    assert motion_fingerprint(dict(data,object_translation=np.ones((5,3))))==h
    data['shape'][0]=1
    assert motion_fingerprint(data)!=h
