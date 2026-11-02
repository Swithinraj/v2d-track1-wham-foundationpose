#!/usr/bin/env python3
"""Robust perspective translation and anchored temporal fitting, with no learned 3D targets.

The linear perspective equations initialize a robust pixel reprojection solve.
Reference: https://github.com/nkolot/SPIN/blob/master/utils/geometry.py
The temporal fit uses actual video frames and zero weight for missing evidence.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import spsolve


def project(joints,K):
    q=np.asarray(joints,float)@np.asarray(K,float).T
    return np.divide(q[...,:2],q[...,2:],out=np.full_like(q[...,:2],np.nan),where=q[...,2:]>0)


def fit_translation(joints,keypoints,confidence,K,*,dimensions=3,sigma_px=4.,max_depth_sigma=.1):
    joints,keypoints,confidence,K=map(lambda x:np.asarray(x,float),(joints,keypoints,confidence,K))
    if (joints.ndim!=2 or joints.shape[1]!=3 or keypoints.shape!=(len(joints),2)
            or confidence.shape!=(len(joints),) or K.shape!=(3,3) or not np.isfinite(K).all()
            or K[0,0]<=0 or K[1,1]<=0 or dimensions not in (2,3)):
        raise ValueError('invalid joints/keypoints/camera dimensions or values')
    good=(np.isfinite(joints).all(1)&np.isfinite(keypoints).all(1)&np.isfinite(confidence)
          &(confidence>=.5)&(joints[:,2]>.2))
    if good.sum()<4:return np.zeros(3),{'accepted':False,'reason':'insufficient observations'}
    j,uv,conf=joints[good],keypoints[good],np.clip(confidence[good],0,1)
    normalized=(uv-K[:2,2])/np.diag(K)[:2]
    A=np.zeros((len(j),2,3));A[:,0,0]=1;A[:,1,1]=1;A[:,:,2]=-normalized
    b=normalized*j[:,2,None]-j[:,:2]
    row_scale=np.sqrt(conf[:,None])*np.diag(K)[:2]/j[:,2,None]
    initial=np.linalg.lstsq((A*row_scale[:,:,None]).reshape(-1,3)[:,:dimensions],
                            (b*row_scale).reshape(-1),rcond=None)[0]
    lower=np.full(dimensions,-2.);upper=np.full(dimensions,2.)
    if dimensions==3:lower[2]=max(-2.,.1-j[:,2].min())
    initial=np.clip(initial,lower+1e-6,upper-1e-6)
    def residual(value):
        delta=np.r_[value,0.] if dimensions==2 else value
        return ((project(j+delta,K)-uv)*np.sqrt(conf[:,None])).ravel()
    fit=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',f_scale=sigma_px,
                      max_nfev=60,xtol=1e-9,ftol=1e-9,gtol=1e-9)
    delta=np.r_[fit.x,0.] if dimensions==2 else fit.x
    shifted=j+delta;z=shifted[:,2]
    jac=np.zeros((len(j),2,3))
    jac[:,0,0]=K[0,0]/z;jac[:,1,1]=K[1,1]/z
    jac[:,0,2]=-K[0,0]*shifted[:,0]/z**2;jac[:,1,2]=-K[1,1]*shifted[:,1]/z**2
    pixel_error=np.linalg.norm(project(shifted,K)-uv,axis=1)
    robust=1/np.sqrt(1+(pixel_error/sigma_px)**2)
    weighted=(jac*np.sqrt(conf*robust)[:,None,None]).reshape(-1,3)[:,:dimensions]
    singular=np.linalg.svd(weighted,compute_uv=False)
    observable=bool(singular[-1]>1e-6 and singular[0]/singular[-1]<1e4)
    depth_sigma=None
    if observable and dimensions==3:
        covariance=np.linalg.inv(weighted.T@weighted)*sigma_px**2
        depth_sigma=float(np.sqrt(covariance[2,2]))
        observable=depth_sigma<=max_depth_sigma
    accepted=bool(fit.success and observable and np.isfinite(delta).all() and np.median(pixel_error)<15)
    return delta,{'accepted':accepted,'reason':'usable perspective fit' if accepted else 'weak depth or poor image fit',
                  'observations':int(good.sum()),'median_residual_px':float(np.median(pixel_error)),
                  'depth_sigma_m':depth_sigma,'smallest_singular_value':float(singular[-1])}


def smooth_anchored(raw,weights,*,anchor,lam=1e4,prior=.05):
    raw,weights=np.asarray(raw,float),np.asarray(weights,float)
    if (raw.ndim!=2 or raw.shape[1]!=3 or len(raw)<3 or weights.shape!=raw.shape
            or not np.isfinite(raw).all() or not np.isfinite(weights).all() or (weights<0).any()
            or not isinstance(anchor,(int,np.integer)) or not 0<=anchor<len(raw)
            or not np.isfinite([lam,prior]).all() or lam<0 or prior<=0):
        raise ValueError('invalid temporal fit inputs')
    if not (weights[anchor]>0).all():
        return np.zeros_like(raw),{'accepted':False,'reason':'unsupported anchor'}
    T=len(raw);D=sparse.diags([np.ones(T-2),-2*np.ones(T-2),np.ones(T-2)],
                             [0,1,2],shape=(T-2,T),format='csr')
    keep=np.arange(T)!=anchor;centered=raw-raw[anchor];delta=np.zeros_like(raw)
    for axis in range(3):
        A=sparse.diags(weights[:,axis]+prior)+lam*(D.T@D)
        b=weights[:,axis]*centered[:,axis]
        delta[keep,axis]=spsolve(A.tocsr()[keep][:,keep].tocsc(),b[keep])
    return delta,{'accepted':bool(np.isfinite(delta).all()),'reason':'anchored fit on full video grid',
                  'max_shift_m':float(np.linalg.norm(delta,axis=1).max())}
