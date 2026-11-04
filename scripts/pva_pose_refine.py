#!/usr/bin/env python3
"""Experimental MHR pose refinement driven by PVA-Net visual dynamics.

Fit only MHR columns that move the twelve observed body joints; preserve identity,
objects, and the first scored pose. Optimize metric joint derivatives, with image
and position priors. Unscored gaps are excluded from derivative losses. This does
not claim the published SMPL optimizer's results transfer automatically to MHR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pva_contracts import motion_fingerprint

MHR_JOINTS=[75,39,76,40,78,42,2,18,3,19,4,20]
COCO_JOINTS=list(range(5,17))


def temporal_support(frames):
    frames=np.asarray(frames)
    if (frames.ndim!=1 or len(frames)<3 or not np.isfinite(frames).all()
            or (frames<0).any() or (frames!=np.floor(frames)).any() or (np.diff(frames)<=0).any()):
        raise ValueError('at least three nonnegative integer strictly increasing video frames required')
    pair=np.diff(frames)==1
    return pair,pair[:-1]&pair[1:]


def apply_pose_updates(base,frames,delta,columns):
    """Interpolate only active-column deltas, retaining original precision elsewhere."""
    base=np.asarray(base);frames=np.asarray(frames);delta=np.asarray(delta)
    temporal_support(frames)
    columns=np.asarray(columns,dtype=int)
    if (base.ndim!=2 or frames[-1]>=len(base) or delta.shape!=(len(frames),len(columns))
            or not np.isfinite(delta).all() or (columns<0).any() or (columns>=base.shape[1]).any()):
        raise ValueError('invalid pose update dimensions, values or indices')
    frames=frames.astype(int)
    delta=delta.copy();delta[0]=0
    result=base.copy()
    for k,column in enumerate(columns):
        result[:,column]+=np.interp(np.arange(len(base)),frames,delta[:,k]).astype(base.dtype)
    result[frames[0]]=base[frames[0]]
    return result


def displacement_gate(base_joints,candidate_joints,maximum=.05):
    """Check every exported frame, including unscored poses inside gaps."""
    base=np.asarray(base_joints);candidate=np.asarray(candidate_joints)
    if base.shape!=candidate.shape or not np.isfinite(candidate).all() or not np.isfinite(base).all():
        return False,float('inf')
    actual=float(np.linalg.norm(candidate-base,axis=-1).max())
    return actual<=maximum,actual


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episode',type=int,required=True)
    p.add_argument('--steps',type=int,default=100)
    args=p.parse_args()
    if args.steps<=0: raise ValueError('positive optimization steps required')
    sys.path.insert(0,str(args.root/'scripts'))
    from human_contact_solver import Body
    tag=f'episode_{args.episode:06d}'
    args.out.mkdir(parents=True,exist_ok=True)
    target=args.out/f'{tag}.npz'
    if target.exists(): raise FileExistsError(target)
    data=dict(np.load(args.base/f'{tag}.npz'))
    pred=dict(np.load(args.predictions/f'{tag}.npz'))
    source_fingerprint=motion_fingerprint(data)
    frames=np.array(json.loads((args.root/'outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json').read_text())[str(args.episode)])
    pair,triplet=temporal_support(frames)
    if pred['frames'][-1]<frames[-1] or not np.array_equal(pred['frames'],np.arange(len(pred['frames']))):
        raise ValueError('PVA predictions must cover the full scored frame IDs')
    fps=float(pred['fps'])
    if not np.isclose(fps,30.,atol=.01): raise ValueError('expected 30 FPS')
    device='cuda';dtype=torch.float32
    tensor=lambda x:torch.as_tensor(x,device=device,dtype=dtype)
    base=tensor(data['pose'][frames]);identity=tensor(np.r_[data['scales'],data['shape']][None])
    body=Body(str(args.root/'weights/mhr/assets/mhr_model.pt'))
    with torch.no_grad():
        _,j0_full=body(tensor(data['pose']),identity)
        j0=j0_full[frames]
        pert=base[0:1].repeat(137,1)
        pert[1+torch.arange(136),torch.arange(136)]+=.01
        _,jp=body(pert,identity)
        moves=(jp[1:,MHR_JOINTS]-jp[:1,MHR_JOINTS]).norm(dim=-1).amax(1)
        columns=torch.nonzero(moves>1e-5).flatten()
    mask=torch.zeros(136,device=device);mask[columns]=1
    limits=torch.full((136,),.15,device=device);limits[:3]=.5
    K=tensor(pred['K'])
    def project(j):
        q=j@K.T
        return q[...,:2]/q[...,2:].clamp_min(.1)
    keypoints=tensor(pred['keypoints'][frames][:,COCO_JOINTS])
    conf=tensor(pred['confidence'][frames][:,COCO_JOINTS,0])
    # Regressors have different anatomical landmark conventions; retain constant
    # offsets and fit image motion instead of forcing incompatible joint centers.
    with torch.no_grad():
        uv0=project(j0[:,MHR_JOINTS])
        valid=conf>=.6
        offsets=torch.zeros((12,2),device=device)
        for k in range(12):
            if valid[:,k].sum()>=3:
                offsets[k]=(keypoints[:,k]-uv0[:,k])[valid[:,k]].median(dim=0).values
        kp_target=keypoints-offsets
        conf=conf*valid*((keypoints-uv0).norm(dim=-1)<40)
    vconf=torch.minimum(conf[:-1],conf[1:])*tensor(pair)[:,None]
    aconf=torch.minimum(torch.minimum(conf[:-2],conf[1:-1]),conf[2:])*tensor(triplet)[:,None]
    # Only valid pairs/triplets consume the targets indexed by their first frame.
    tv=tensor(pred['velocity'][frames[:-1]][:,COCO_JOINTS])/fps
    ta=tensor(pred['acceleration'][frames[:-2]][:,COCO_JOINTS])/fps**2
    latent=torch.zeros_like(base,requires_grad=True)
    anchor_mask=torch.ones((len(base),1),device=device);anchor_mask[0]=0
    optimizer=torch.optim.Adam([latent],lr=.02)
    history=[]
    def weighted_mse(value,weight):
        return (value.square().sum(-1)*weight).sum()/(weight.sum().clamp_min(1)*value.shape[-1])
    def evaluate():
        change=limits*torch.tanh(latent)*mask*anchor_mask
        pose=base+change
        _,joints=body(pose,identity)
        j=joints[:,MHR_JOINTS]
        acc=weighted_mse((j[2:]-2*j[1:-1]+j[:-2]-ta)*1000,aconf)
        vel=weighted_mse((j[1:]-j[:-1]-tv)/.005,vconf)
        img=weighted_mse((project(j)-kp_target)/10,conf)
        position=((joints-j0)/.02).square().mean()
        parameter=(change/(limits/2)).square().mean()
        loss=acc+.1*vel+.1*img+position+.05*parameter
        terms={'acc':float(acc.detach()),'vel':float(vel.detach()),'image':float(img.detach()),
               'position':float(position.detach()),'total':float(loss.detach())}
        return loss,terms,change,joints
    for iteration in range(args.steps):
        optimizer.zero_grad()
        loss,terms,_,_=evaluate()
        if not torch.isfinite(loss): raise ValueError('nonfinite refinement loss')
        loss.backward();optimizer.step()
        if iteration%10==0 or iteration==args.steps-1:
            history.append(dict(step=iteration,**terms));print('PVA_POSE_STEP',tag,iteration,json.dumps(terms),flush=True)
    with torch.no_grad():
        _,terms,change,joints=evaluate()
        selected=columns.cpu().numpy()
        result=apply_pose_updates(data['pose'],frames,change[:,columns].cpu().numpy(),selected)
        _,jfull=body(tensor(result),identity)
        bounded,max_joint=displacement_gate(j0_full.cpu().numpy(),jfull.cpu().numpy())
        accept=(bounded and terms['total']<history[0]['total']
                and terms['image']<=history[0]['image']+.05)
    if not accept: result=data['pose'].copy()
    assert np.array_equal(result[frames[0]],data['pose'][frames[0]])
    data['pose']=result
    np.savez(target,**data)
    report={'accepted':accept,'max_joint_move_m':max_joint,'columns':columns.cpu().tolist(),
            'history':history,'final':terms,'steps':args.steps,'objects_and_identity_unchanged':True,
            'source_motion_sha256':source_fingerprint,'displacement_checked_frames':len(result),
            'prediction_sha256':hashlib.sha256((args.predictions/f'{tag}.npz').read_bytes()).hexdigest()}
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PVA_POSE_DONE',tag,json.dumps({'accepted':accept,'max_joint_move_m':max_joint}),flush=True)


if __name__=='__main__': main()
