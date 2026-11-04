#!/usr/bin/env python3
"""Bounded common-root correction from image-derived torso dynamics.

The measured model targets use SI units. The linear solve uses per-frame changes,
keeps the first scored frame exactly fixed, and leaves identity/articulation/object
arrays untouched. This is an experimental adapter, not a claimed accuracy gain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pva_contracts import require_motion_match

COCO_TORSO=np.array([5,6,11,12])
MHR_TORSO=np.array([75,39,2,18])
FLIP=np.array([1.,-1.,-1.])


def torso_residuals(base,velocity,acceleration,confidence,scored,*,fps=30.):
    base,velocity,acceleration,confidence=map(lambda x:np.asarray(x,float),(base,velocity,acceleration,confidence))
    T,J,_=base.shape
    scored=np.asarray(scored,bool)
    if (velocity.shape!=(T-1,J,3) or acceleration.shape!=(T-2,J,3)
            or confidence.shape!=(T,J) or scored.shape!=(T,)):
        raise ValueError('torso time/joint dimensions do not agree')
    if not np.isfinite(fps) or fps<=0 or not all(np.isfinite(x).all() for x in (base,velocity,acceleration,confidence)):
        raise ValueError('finite torso data and positive FPS required')
    conf=np.where(confidence>=.6,np.clip(confidence,0,1),0.)
    cv=np.minimum(conf[:-1],conf[1:])
    ca=np.minimum(np.minimum(conf[:-2],conf[1:-1]),conf[2:])
    # A scored sample gap is not an observed motion interval.
    cv*= (scored[:-1]&scored[1:])[:,None]
    ca*= (scored[:-2]&scored[1:-1]&scored[2:])[:,None]
    def common(values,weights):
        total=weights.sum(1)
        residual=np.divide((values*weights[...,None]).sum(1),total[:,None],
                           out=np.zeros((len(values),3)),where=total[:,None]>0)
        support=np.where((weights>0).sum(1)>=2,total/J,0.)
        return residual,support
    v,wv=common(velocity/fps-np.diff(base,axis=0),cv)
    a,wa=common(acceleration/fps**2-np.diff(base,n=2,axis=0),ca)
    return v,a,wv,wa


def solve_delta(velocity,acceleration,vweight,aweight,*,anchor,position_weight=1.,
                velocity_weight=0.,acceleration_weight=100.,max_shift=.05):
    v,a,wv,wa=map(lambda x:np.asarray(x,float),(velocity,acceleration,vweight,aweight))
    T=len(v)+1
    if T<3 or v.shape!=(T-1,3) or a.shape!=(T-2,3) or wv.shape!=(T-1,) or wa.shape!=(T-2,):
        raise ValueError('expected velocity [T-1,3], acceleration [T-2,3] and scalar weights')
    if not isinstance(anchor,(int,np.integer)) or not 0<=anchor<T: raise ValueError('anchor must be an integer inside sequence')
    if not all(np.isfinite(x).all() for x in (v,a,wv,wa)):
        raise ValueError('inputs must be finite')
    if not np.isfinite([position_weight,velocity_weight,acceleration_weight,max_shift]).all():
        raise ValueError('hyperparameters must be finite')
    if position_weight<=0 or min(velocity_weight,acceleration_weight,max_shift)<0 or (wv<0).any() or (wa<0).any():
        raise ValueError('positive position weight and nonnegative weights/bound required')
    zero=np.zeros((T,3))
    if not ((wv*velocity_weight).any() or (wa*acceleration_weight).any()):
        return zero,{'accepted':False,'reason':'no usable motion evidence','max_shift_m':0.}
    d1=sparse.diags([-np.ones(T-1),np.ones(T-1)],[0,1],shape=(T-1,T),format='csr')
    d2=sparse.diags([np.ones(T-2),-2*np.ones(T-2),np.ones(T-2)],[0,1,2],shape=(T-2,T),format='csr')
    A=sparse.eye(T,format='csr')*position_weight
    A+=velocity_weight*(d1.T@sparse.diags(wv)@d1)+acceleration_weight*(d2.T@sparse.diags(wa)@d2)
    b=velocity_weight*(d1.T@(wv[:,None]*v))+acceleration_weight*(d2.T@(wa[:,None]*a))
    keep=np.arange(T)!=anchor
    delta=zero.copy();delta[keep]=spsolve(A[keep][:,keep].tocsc(),b[keep])
    maximum=float(np.linalg.norm(delta,axis=1).max())
    report={'accepted':bool(np.isfinite(delta).all() and maximum<=max_shift),
            'max_shift_m':maximum,'rms_shift_m':float(np.sqrt(np.mean(delta**2)))}
    if not report['accepted']:
        report['reason']='correction exceeds displacement bound or is nonfinite'
        return zero,report
    report['reason']='bounded image-derived common torso dynamics'
    return delta,report


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--joints',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--scored',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--velocity-weight',type=float,default=0.)
    p.add_argument('--acceleration-weight',type=float,default=100.)
    p.add_argument('--max-shift',type=float,default=.05)
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    scored=json.loads(args.scored.read_text())
    shutil.copytree(args.base,args.out)
    reports={}
    for path in sorted(args.predictions.glob('episode_??????.npz')):
        tag=path.stem;ep=str(int(tag.split('_')[1]))
        data=dict(np.load(args.base/f'{tag}.npz'));T=len(data['pose'])
        with np.load(path) as pred, np.load(args.joints/f'{tag}.npz') as bodies:
            require_motion_match(bodies,data)
            if len(pred['frames'])<T or not np.array_equal(pred['frames'][:T],np.arange(T)):
                raise ValueError(f'{tag}: predictions do not cover the base frames')
            if not np.array_equal(bodies['frames'],np.arange(T)):
                raise ValueError(f'{tag}: MHR joint frames differ from base')
            mask=np.zeros(T,bool);mask[np.asarray(scored[ep],int)]=True
            conf=pred['confidence'][:T,COCO_TORSO,0]
            v,a,wv,wa=torso_residuals(bodies['joints'][:,MHR_TORSO],pred['velocity'][:T-1,COCO_TORSO],
                    pred['acceleration'][:T-2,COCO_TORSO],conf,mask,fps=float(pred['fps']))
        anchor=int(scored[ep][0])
        delta,report=solve_delta(v,a,wv,wa,anchor=anchor,velocity_weight=args.velocity_weight,
                                acceleration_weight=args.acceleration_weight,max_shift=args.max_shift)
        if report['accepted']:
            changed=data['pose'].copy();changed[:,:3]+= (delta*FLIP*10).astype(changed.dtype)
            assert np.array_equal(changed[anchor],data['pose'][anchor])
            data['pose']=changed
            np.savez(args.out/f'{tag}.npz',**data)
        reports[tag]=report
        print(tag,json.dumps(report),flush=True)
    (args.out/'pva_root_report.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__': main()
