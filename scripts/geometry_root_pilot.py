#!/usr/bin/env python3
"""Diagnostic three-clip perspective-root pilot; never packs or uploads a submission."""
import argparse
import json
from pathlib import Path

import numpy as np
from geometry_root import fit_translation, project, smooth_anchored
from pva_contracts import require_motion_match
from validate_pva_candidate import MHR_JOINTS, MP_JOINTS

FIT=np.array([0,1,6,7,10,11]) # shoulders, hips, ankles
TORSO=np.array([0,1,6,7])
HELD=np.array([2,3,4,5,8,9]) # elbows, wrists, knees
GROUPS={'elbows':[2,3],'wrists':[4,5],'knees':[8,9]}


def summarize(before,after,valid):
    if not valid.any():return {'observations':0,'before_px':None,'after_px':None}
    return {'observations':int(valid.sum()),'before_px':float(before[valid].mean()),
            'after_px':float(after[valid].mean())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--episodes',nargs='+',type=int,default=[9,13,24])
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    scored=json.loads((args.root/'outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json').read_text())
    reports={}
    for ep in args.episodes:
        tag=f'episode_{ep:06d}';frames=np.array(scored[str(ep)]);anchor=int(frames[0])
        data=dict(np.load(args.root/f'submissions/episodes_cand_P3/{tag}.npz'))
        with np.load(args.root/f'work/geometry_root/joints_P3_all/{tag}.npz') as cached:
            require_motion_match(cached,data);all_joints=cached['joints']
        joints=all_joints[:,MHR_JOINTS];T=len(joints)
        camera=json.loads((args.root/f'outputs/camera_measured/{tag}/intrinsics.json').read_text())
        K=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1.]])
        keypoints=np.load(args.root/f'outputs/kp2d_vpx/{tag}.npz')['pose']
        uv0=project(joints,K);target=np.zeros((T,12,2));confidence=np.zeros((T,12))
        # Actor assignments and observation masks are frozen from the baseline.
        for f in frames:
            if f>=len(keypoints):continue
            kp=keypoints[f][:,MP_JOINTS];valid=(kp[...,3]>=.6)&np.isfinite(kp[...,:2]).all(-1)
            distance=np.linalg.norm(kp[...,:2]-uv0[f],axis=-1)
            costs=np.array([np.median(d[v]) if v.sum()>=6 else np.inf for d,v in zip(distance,valid)])
            actor=int(costs.argmin())
            if costs[actor]>60:continue
            target[f]=np.where(valid[actor,:,None],kp[actor,:,:2],0)
            confidence[f]=np.where(valid[actor],kp[actor,:,3],0)
        modes={}
        for dims in (2,3):
            raw=np.zeros((T,3));weights=np.zeros((T,3));fits=[]
            for f in frames:
                delta,fit=fit_translation(joints[f,FIT],target[f,FIT],confidence[f,FIT],K,dimensions=dims)
                if dims==3:
                    torso,tfit=fit_translation(joints[f,TORSO],target[f,TORSO],confidence[f,TORSO],K)
                    agreement=bool(tfit['accepted'] and abs(delta[2]-torso[2])<=.05)
                    fit['torso_agrees']=agreement
                    fit['torso_depth_difference_m']=float(abs(delta[2]-torso[2]))
                    fit['accepted']=bool(fit['accepted'] and agreement)
                fits.append(fit)
                if fit['accepted']:
                    raw[f]=delta
                    weight=1. if dims==2 else min(1.,(.04/max(fit['depth_sigma_m'],1e-6))**2)
                    weights[f]=weight
            corrected,temporal=smooth_anchored(raw,weights,anchor=anchor)
            coverage=float((weights[frames,0]>0).mean())
            changed=joints+corrected[:,None]
            before=np.linalg.norm(uv0-target,axis=-1)
            after=np.linalg.norm(project(changed,K)-target,axis=-1)
            valid=confidence>=.6
            holdout=summarize(before[:,HELD],after[:,HELD],valid[:,HELD])
            groups={name:summarize(before[:,ix],after[:,ix],valid[:,ix]) for name,ix in GROUPS.items()}
            maximum=float(np.linalg.norm(corrected,axis=1).max())
            image_gate=bool(holdout['observations']>=30 and holdout['after_px']<=holdout['before_px']-.1
                           and all(x['observations']>=10 and x['after_px']<=x['before_px']+.2 for x in groups.values()))
            gate=bool(temporal['accepted'] and coverage>=.5 and maximum<=.05 and image_gate)
            modes[str(dims)]={'fit_coverage':coverage,'temporal':temporal,'max_shift_m':maximum,
                              'held_out':holdout,'groups':groups,'image_gate':image_gate,
                              'pilot_gate':gate,'reason':'image/size gate only; contact remains unvalidated',
                              'mean_correction_second_difference_cm':float(np.linalg.norm(np.diff(corrected[frames],n=2,axis=0),axis=-1)[(np.diff(frames)[:-1]==1)&(np.diff(frames)[1:]==1)].mean()*100),
                              'fits':fits}
            np.savez(args.out/f'{tag}_{dims}d_diagnostic.npz',delta=corrected,raw=raw,weights=weights,frames=np.arange(T))
            print(tag,dims,json.dumps({k:v for k,v in modes[str(dims)].items() if k!='fits'}),flush=True)
        reports[tag]=modes
    (args.out/'report.json').write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
