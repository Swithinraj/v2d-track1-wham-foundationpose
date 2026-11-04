import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def solver():
    assert importlib.util.find_spec("contact_translation") is not None, "contact translation solver is missing"
    from contact_translation import fit_translation
    return fit_translation


def fixture_scene(offset=(.035, -.02, .06), frames=24):
    # Analytic sphere sampled on a Fibonacci lattice, with exact known contacts.
    k = np.arange(3000)
    z = 1 - 2 * (k + .5) / len(k)
    angle = k * np.pi * (3 - np.sqrt(5))
    normals = np.c_[np.sqrt(1-z*z)*np.cos(angle), np.sqrt(1-z*z)*np.sin(angle), z]
    vertices = .12 * normals
    rotations = Rotation.from_rotvec(np.c_[np.zeros(frames), np.linspace(0,.5,frames), np.zeros(frames)]).as_matrix()
    truth = np.c_[np.linspace(-.2,.2,frames), np.zeros(frames), np.full(frames,2.)]
    indices = np.arange(0,3000,100)
    hands = np.einsum('tij,pj->tpi',rotations,vertices[indices])+truth[:,None]
    translation = truth - np.asarray(offset)
    contact = np.ones(hands.shape[:2],bool)
    return vertices,normals,rotations,translation,1.,hands,contact


def test_recovers_known_constant_offset_with_rotating_object():
    fit = solver()(*fixture_scene())
    assert fit['accepted'], fit
    np.testing.assert_allclose(fit['translation_m'],[.035,-.02,.06],atol=.002)
    assert fit['residual_after_m'] < .002


def test_outlier_contacts_do_not_dominate_offset():
    args=list(fixture_scene())
    args[5]=args[5].copy(); args[5][:,::5] += [.25,-.15,.12]
    fit=solver()(*args)
    assert fit['accepted'],fit
    np.testing.assert_allclose(fit['translation_m'],[.035,-.02,.06],atol=.004)


def test_aligned_object_is_left_unchanged():
    fit=solver()(*fixture_scene(offset=(0,0,0)))
    assert not fit['accepted']
    np.testing.assert_array_equal(fit['translation_m'],[0,0,0])


def test_no_contact_abstains():
    args=list(fixture_scene()); args[6][:]=False
    fit=solver()(*args)
    assert not fit['accepted']
    assert 'contact' in fit['reason']
    np.testing.assert_array_equal(fit['translation_m'],[0,0,0])


def test_planar_contact_cannot_determine_three_dimensional_offset():
    args=list(fixture_scene())
    x,y=np.meshgrid(np.linspace(-.2,.2,30),np.linspace(-.2,.2,30))
    args[0]=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    args[1]=np.tile([0.,0.,1.],(x.size,1))
    args[2]=np.broadcast_to(np.eye(3),(24,3,3)).copy()
    args[5]=np.broadcast_to(args[0][::30]+[0,0,2],(24,30,3)).copy()
    fit=solver()(*args)
    assert not fit['accepted']
    assert 'observable' in fit['reason']


def test_inconsistent_offsets_across_time_abstain():
    args=list(fixture_scene())
    args[5]=args[5].copy(); args[5][12:] += [.08,0,0]
    fit=solver()(*args)
    assert not fit['accepted']


def test_constant_offset_preserves_motion_acceleration():
    args=fixture_scene(); fit=solver()(*args)
    assert fit['accepted']
    translated=args[3]+fit['translation_m']
    np.testing.assert_allclose(np.diff(translated,n=2,axis=0),np.diff(args[3],n=2,axis=0),atol=1e-14)


@pytest.mark.parametrize('bad', ['length','nan','scale','rotation'])
def test_rejects_corrupt_inputs(bad):
    args=list(fixture_scene())
    if bad=='length': args[5]=args[5][:-1]
    if bad=='nan': args[3][0,0]=np.nan
    if bad=='scale': args[4]=0
    if bad=='rotation': args[2][0]*=2
    with pytest.raises(ValueError): solver()(*args)


def test_directory_roundtrip_preserves_human_and_rejects_without_image_gate(tmp_path):
    import trimesh
    solver()
    from contact_translation import process_directory
    mesh=trimesh.creation.icosphere(subdivisions=3,radius=.12)
    base=tmp_path/'base'; contacts=tmp_path/'contacts'
    base.mkdir(); contacts.mkdir()
    mesh.export(base/'episode_000000_object.glb')
    T=24; rotation=np.broadcast_to(np.eye(3),(T,3,3)).copy()
    translation=np.tile([0.,0.,1.94],(T,1))
    hands=np.broadcast_to(np.asarray(mesh.vertices)[::20]+[0,0,2.],(T,len(mesh.vertices[::20]),3)).copy()
    payload=dict(pose=np.arange(T*6,dtype=np.float32).reshape(T,6),scales=np.ones(68),shape=np.zeros(45),
                 object_translation=translation,object_rotation=rotation,object_scale=1.)
    np.savez(base/'episode_000000.npz',**payload)
    np.savez(contacts/'episode_000000.npz',hands=hands,contact=np.ones(hands.shape[:2],bool))
    report=process_directory(base,contacts,tmp_path/'out')
    assert not report['episode_000000']['applied']
    out=np.load(tmp_path/'out/episode_000000.npz')
    for key,value in payload.items(): np.testing.assert_array_equal(out[key],value)
    assert (tmp_path/'out/episode_000000_object.glb').read_bytes()==(base/'episode_000000_object.glb').read_bytes()


def test_directory_refuses_in_place_or_existing_output(tmp_path):
    solver()
    from contact_translation import process_directory
    base=tmp_path/'base'; base.mkdir()
    with pytest.raises(ValueError,match='output'):
        process_directory(base,tmp_path/'contacts',base)
