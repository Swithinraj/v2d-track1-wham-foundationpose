import numpy as np
import pytest


def geometry():
    import trimesh
    mesh=trimesh.creation.box(extents=[.3,.3,.3])
    R=np.broadcast_to(np.eye(3),(12,3,3)).copy()
    t=np.tile([0.,0.,2.],(12,1))
    camera=dict(fx=300.,fy=300.,cx=160.,cy=120.,width=320,height=240)
    return mesh,R,t,camera


def test_silhouette_prefers_correct_translation():
    from contact_translation import silhouette_gate, render_mask
    mesh,R,t,K=geometry()
    truth=render_mask(mesh.vertices,mesh.faces,R[0],t[0],1.,K)
    masks={i:truth for i in range(12)}
    biased=t-[.06,0,0]
    report=silhouette_gate(mesh.vertices,mesh.faces,R,biased,1.,np.array([.06,0,0]),K,masks)
    assert report['accepted']
    assert report['iou_after']>.99
    assert report['iou_after']>report['iou_before']+.1


def test_silhouette_rejects_worse_placement():
    from contact_translation import silhouette_gate, render_mask
    mesh,R,t,K=geometry()
    masks={i:render_mask(mesh.vertices,mesh.faces,R[i],t[i],1.,K) for i in range(12)}
    report=silhouette_gate(mesh.vertices,mesh.faces,R,t,1.,np.array([.06,0,0]),K,masks)
    assert not report['accepted']


def test_empty_masks_cannot_approve_a_proposal():
    from contact_translation import silhouette_gate
    mesh,R,t,K=geometry()
    masks={i:np.zeros((60,80),bool) for i in range(12)}
    report=silhouette_gate(mesh.vertices,mesh.faces,R,t,1.,np.array([.06,0,0]),K,masks)
    assert not report['accepted']


def test_behind_camera_mesh_is_rejected():
    from contact_translation import render_mask
    mesh,R,t,K=geometry()
    with pytest.raises(ValueError,match='camera'):
        render_mask(mesh.vertices,mesh.faces,R[0],[0,0,-1],1.,K)
