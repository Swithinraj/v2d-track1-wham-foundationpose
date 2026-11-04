#!/usr/bin/env python3
"""Measure CoCoNet window-join acceleration against interior frames and initialization.

Self-acceleration diagnoses discontinuities; it is not acceleration error against GT.
Only consecutive scored triples count when a scored-frame list is provided.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def seam_statistics(points,window_length=96,frames=None):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or window_length<4:
        raise ValueError('expected finite [T,3] points and window length >= 4')
    T=len(points); centers=np.arange(1,T-1)
    valid=np.ones(len(centers),bool)
    if frames is not None:
        use=np.zeros(T,bool); use[np.asarray(frames,int)]=True
        valid=use[:-2]&use[1:-1]&use[2:]
    joins=np.arange(window_length,T,window_length)
    near=np.min(np.abs(centers[:,None]-joins[None]),axis=1) if len(joins) else np.full(len(centers),np.inf)
    boundary=valid&(near<=2); interior=valid&(near>=8)
    acceleration=np.linalg.norm(np.diff(points,n=2,axis=0),axis=1)*100
    out={'boundary_frames':int(boundary.sum()),'interior_frames':int(interior.sum())}
    for name,mask in [('boundary',boundary),('interior',interior)]:
        values=acceleration[mask]
        out[name+'_mean_cm']=float(values.mean()) if len(values) else None
        out[name+'_p90_cm']=float(np.quantile(values,.9)) if len(values) else None
    return out


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--scored',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    scored=json.loads(args.scored.read_text()); report={}
    for path in sorted(args.root.glob('episode_*/inference/coconet.pth')):
        tag=path.parent.parent.name; ep=str(int(tag.split('_')[-1]))
        bundle=torch.load(path,map_location='cpu',weights_only=False)
        window=int(bundle['metadata']['window_length']); result={}
        for label in ['in','pr_initial']:
            block=bundle[label]
            for kind,key in [('human','mhr_trans'),('object','pose_abs')]:
                value=block[key]
                value=value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
                if kind=='object': value=value[:,:3,3]
                result[label+'_'+kind]=seam_statistics(value,window,scored[ep])
        report[tag]=result; print(tag,json.dumps(result),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
