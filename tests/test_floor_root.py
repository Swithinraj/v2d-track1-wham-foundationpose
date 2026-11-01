import importlib.util
import numpy as np
import pytest


def solve(*args,**kwargs):
    assert importlib.util.find_spec('floor_root') is not None,'floor root solver is missing'
    from floor_root import floor_root_shift
    return floor_root_shift(*args,**kwargs)


def example():
    T=180; anchor=30
    ray=np.tile([0.,.2,1.],(T,1)); ray/=np.linalg.norm(ray,axis=1)[:,None]
    drift=.08*(1-np.cos(np.arange(T)*np.pi/120))
    drift-=drift[anchor]
    feet=np.broadcast_to([[-.1,1.,3.],[.1,1.,3.]],(T,2,3)).copy()
    feet+=drift[:,None,None]*ray[:,None]
    confidence=np.ones((T,2))
    return feet,ray,np.array([0.,-1.,0.]),confidence,anchor,drift


def test_known_ray_drift_is_removed_and_anchor_exact():
    feet,ray,n,c,a,drift=example()
    delta,report=solve(feet,ray,n,c,a)
    assert report['accepted'],report
    np.testing.assert_allclose(delta,-drift[:,None]*ray,atol=.002)
    np.testing.assert_array_equal(delta[a],[0,0,0])


def test_missing_contact_abstains():
    feet,ray,n,c,a,drift=example(); c[:]=0
    delta,report=solve(feet,ray,n,c,a)
    assert not report['accepted']
    assert not delta.any()


def test_horizontal_viewing_ray_is_unobservable():
    feet,ray,n,c,a,drift=example(); ray[:]=[0,0,1]
    delta,report=solve(feet,ray,n,c,a)
    assert not report['accepted']
    assert not delta.any()


def test_unobserved_anchor_prevents_arbitrary_floor_height():
    feet,ray,n,c,a,drift=example(); c[a]=0
    delta,report=solve(feet,ray,n,c,a)
    assert not report['accepted']


def test_already_grounded_feet_stay_unchanged():
    feet,ray,n,c,a,drift=example(); feet-=drift[:,None,None]*ray[:,None]
    delta,report=solve(feet,ray,n,c,a)
    np.testing.assert_allclose(delta,0,atol=1e-12)


def test_one_swinging_foot_does_not_lift_whole_body():
    feet,ray,n,c,a,drift=example(); feet[:,0,1]-=.2; c[:,0]=0
    delta,report=solve(feet,ray,n,c,a)
    assert report['accepted']
    np.testing.assert_allclose(delta,-drift[:,None]*ray,atol=.002)


def test_implausible_shift_is_rejected_not_clipped():
    feet,ray,n,c,a,drift=example(); feet[80:,:,1]+=.5
    delta,report=solve(feet,ray,n,c,a)
    assert not report['accepted']
    assert not delta.any()


def test_nonfinite_plane_is_rejected():
    feet,ray,n,c,a,drift=example(); n[:]=np.nan
    with pytest.raises(ValueError): solve(feet,ray,n,c,a)


def test_whittaker_fills_unobserved_frames():
    from floor_root import whittaker
    y=np.arange(50.)
    w=np.ones(50); w[20:30]=0
    out=whittaker(np.where(w>0,y,0.),w,10.)
    np.testing.assert_allclose(out,y,atol=1e-6)


def test_scored_coverage_can_be_set_by_the_caller():
    feet,ray,n,c,a,drift=example(); c[60:]=0
    delta,report=solve(feet,ray,n,c,a,min_coverage=.9)
    assert not report['accepted']
    delta,report=solve(feet,ray,n,c,a,min_coverage=.1)
    assert report['accepted']


def test_anchor_may_borrow_contact_from_nearby_frames_when_allowed():
    feet,ray,n,c,a,drift=example(); c[a]=0
    delta,report=solve(feet,ray,n,c,a,anchor_window=3)
    assert report['accepted']
    np.testing.assert_array_equal(delta[a],[0,0,0])


def test_full_trust_is_the_plain_correction():
    feet,ray,n,c,a,drift=example()
    plain,_=solve(feet,ray,n,c,a)
    same,_=solve(feet,ray,n,c,a,anchor_trust=1.)
    np.testing.assert_array_equal(plain,same)


def test_no_trust_references_the_median_height_and_the_anchor_stays():
    feet,ray,n,c,a,drift=example()
    # The anchor alone sits 5 cm high; every other frame agrees.
    feet=np.broadcast_to([[-.1,1.,3.],[.1,1.,3.]],feet.shape).copy()
    feet[a-20:a+21,:,1]-=.05
    trusting,_=solve(feet,ray,n,c,a,anchor_trust=1.)
    blind,_=solve(feet,ray,n,c,a,anchor_trust=0.)
    np.testing.assert_array_equal(blind[a],[0,0,0])
    far=len(feet)-1
    assert np.linalg.norm(blind[far])<.01                 # far frames already agree with the median: almost no slide
    assert np.linalg.norm(trusting[far])>.2               # trusting that anchor drags them all
    np.testing.assert_array_equal(trusting[a],[0,0,0])
