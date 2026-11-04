#!/usr/bin/env python3
"""Validate preserved state and report independent 2D evidence, not ground-truth accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np
from pva_contracts import require_motion_match
from pva_pose_refine import temporal_support

MHR_JOINTS=[75,39,76,40,78,42,2,18,3,19,4,20]
MP_JOINTS=[11,12,13,14,15,16,23,24,25,26,27,28]


def check_preserved(base,candidate,frames,root_only=False):
    temporal_support(frames)
    if set(base)!=set(candidate): raise ValueError('episode array keys changed')
    for key in base:
        if base[key].shape!=candidate[key].shape or not np.isfinite(candidate[key]).all():
            raise ValueError(f'invalid shape or nonfinite values: {key}')
        if key!='pose' and not np.array_equal(base[key],candidate[key]):
            raise ValueError(f'identity or object data changed: {key}')
    if not np.array_equal(base['pose'][frames[0]],candidate['pose'][frames[0]]):
        raise ValueError('first scored anchor changed')
    if root_only and not np.array_equal(base['pose'][:,3:],candidate['pose'][:,3:]):
        raise ValueError('root-only candidate changed articulation')
    if frames[-1]>=len(candidate['pose']): raise ValueError('missing scored frames')
    rotation=candidate['object_rotation']
    if not (np.allclose(rotation@rotation.transpose(0,2,1),np.eye(3),atol=1e-5)
            and np.allclose(np.linalg.det(rotation),1,atol=1e-5)):
        raise ValueError('invalid object rotation')


def independent_keypoint_errors(base,candidate,keypoints,K,frames):
    def project(j):
        q=j[:,MHR_JOINTS]@K.T
        return np.divide(q[...,:2],q[...,2:],out=np.full_like(q[...,:2],np.nan),where=q[...,2:]>0)
    uv0,uv1=project(base),project(candidate)
    before=[];after=[]
    for frame in frames:
        if frame>=len(keypoints): continue
        kp=keypoints[frame][:,MP_JOINTS]
        valid=(kp[...,3]>.5)&np.isfinite(kp[...,:2]).all(-1)&np.isfinite(uv0[frame]).all(-1)
        distances=np.linalg.norm(kp[...,:2]-uv0[frame],axis=-1)
        costs=np.array([np.median(d[v]) if v.sum()>=4 else np.inf for d,v in zip(distances,valid)])
        actor=int(costs.argmin())
        if costs[actor]>60: continue
        use=valid[actor]&np.isfinite(uv1[frame]).all(-1)
        before.extend(distances[actor,use])
        after.extend(np.linalg.norm(kp[actor,use,:2]-uv1[frame,use],axis=-1))
    return {'observations':len(before),'mean_before_px':float(np.mean(before)) if before else None,
            'mean_after_px':float(np.mean(after)) if after else None,
            'median_before_px':float(np.median(before)) if before else None,
            'median_after_px':float(np.median(after)) if after else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--base-joints',type=Path,required=True)
    p.add_argument('--candidate-joints',type=Path)
    p.add_argument('--root-only',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if not args.root_only and args.candidate_joints is None:
        p.error('body-pose candidate requires actual candidate MHR joints')
    scored=json.loads((args.root/'outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json').read_text())
    expected={f'episode_{ep:06d}.npz' for ep in range(30)}
    if {x.name for x in args.candidate.glob('episode_??????.npz')}!=expected:
        raise ValueError('exactly 30 complete candidate episodes required')
    reports={}
    for ep in range(30):
        tag=f'episode_{ep:06d}';frames=np.array(scored[str(ep)],dtype=int)
        with np.load(args.base/f'{tag}.npz') as base,np.load(args.candidate/f'{tag}.npz') as candidate:
            check_preserved(base,candidate,frames,root_only=args.root_only)
            with np.load(args.base_joints/f'{tag}.npz') as cache:
                require_motion_match(cache,base);j0=cache['joints']
            if args.root_only:
                delta=(candidate['pose'][:,:3].astype(float)-base['pose'][:,:3])*np.array([1.,-1.,-1.])/10
                j1=j0+delta[:,None]
            else:
                with np.load(args.candidate_joints/f'{tag}.npz') as cache:
                    require_motion_match(cache,candidate);j1=cache['joints']
            if j0.shape!=j1.shape or len(j0)!=len(base['pose']) or not np.isfinite(j1).all():
                raise ValueError(f'{tag}: invalid joint cache dimensions or values')
            maximum=float(np.linalg.norm(j1-j0,axis=-1).max())
            if maximum>.05001: raise ValueError(f'{tag}: exceeds 5 cm displacement limit')
            np.testing.assert_allclose(j0[frames[0]],j1[frames[0]],rtol=0,atol=1e-10)
            camera=json.loads((args.root/f'outputs/camera_measured/{tag}/intrinsics.json').read_text())
            K=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1.]])
            with np.load(args.root/f'outputs/kp2d_vpx/{tag}.npz') as kp:
                evidence=independent_keypoint_errors(j0,j1,kp['pose'],K,frames)
            reports[tag]={'frames':len(j0),'scored_frames':len(frames),'max_joint_move_m':maximum,
                          'independent_2d':evidence}
    before=[];after=[];weights=[]
    for report in reports.values():
        evidence=report['independent_2d']
        if evidence['observations']:
            before.append(evidence['mean_before_px']);after.append(evidence['mean_after_px']);weights.append(evidence['observations'])
    summary={'episodes':30,'preservation_checks':'passed','max_joint_move_m':max(x['max_joint_move_m'] for x in reports.values()),
             'independent_2d_mean_before_px':float(np.average(before,weights=weights)) if weights else None,
             'independent_2d_mean_after_px':float(np.average(after,weights=weights)) if weights else None,
             'independent_2d_improved_episodes':int(np.sum(np.array(after)<before)),
             'evidence_limit':'2D agreement is an implementation check, not proof of 3D accuracy or Kaggle improvement'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'summary':summary,'episodes':reports},indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
