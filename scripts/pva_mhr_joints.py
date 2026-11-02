#!/usr/bin/env python3
"""Export MHR joints for dynamics fitting and verify root units against the model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pva_contracts import motion_fingerprint


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--episodes-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.root/'official/v2d_submission_kit/tools/track1'))
    from mesh_to_mhr_params import MHR
    mhr=MHR(str(args.root/'weights/mhr/assets/mhr_model.pt'),'cuda',chunk=64,precision='float64')
    args.out.mkdir(parents=True,exist_ok=True)
    # COCO shoulders, elbows, wrists, hips, knees, ankles; mapped by anatomy.
    coco=np.arange(5,17); joints=np.array([75,39,76,40,78,42,2,18,3,19,4,20])
    for ep in args.episodes:
        tag=f'episode_{ep:06d}'
        with np.load(args.episodes_dir/f'{tag}.npz') as d:
            signature=motion_fingerprint(d)
            pose=torch.as_tensor(d['pose'],device='cuda',dtype=torch.float64)
            identity=torch.as_tensor(np.r_[d['scales'],d['shape']][None],device='cuda',dtype=torch.float64)
        with torch.no_grad():
            _,base=mhr.run(pose,identity)
            shifted=pose[:3].clone()
            offset=np.array([.002,-.003,.004])
            shifted[:,:3]+=torch.as_tensor(offset*np.array([1.,-1.,-1.])*10,device='cuda')
            _,check=mhr.run(shifted,identity)
        residual=(check[:,joints]-base[:3,joints]).cpu().numpy()-offset
        error=float(np.max(np.abs(residual)))
        if error>1e-8: raise ValueError(f'MHR root units/axes verification failed: {error}')
        values=base.cpu().numpy()
        if not np.isfinite(values).all(): raise ValueError('nonfinite MHR joints')
        np.savez(args.out/f'{tag}.npz',joints=values,frames=np.arange(len(values)),coco_indices=coco,mhr_indices=joints,
                 source_motion_sha256=np.array(signature))
        print('MHR_JOINTS',tag,json.dumps({'frames':len(values),'root_translation_test_max_m':error}),flush=True)


if __name__=='__main__': main()
