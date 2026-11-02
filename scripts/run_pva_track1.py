#!/usr/bin/env python3
"""Run the released PVA-Net on audited Track 1 crops, without a second detector."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    sys.path.insert(0,str(args.upstream))
    from pvanet.model_io import load_full_model
    spec=importlib.util.spec_from_file_location('htd_pva_cli',args.upstream/'scripts/demo_pvanet.py')
    cli=importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    args.out.mkdir(parents=True,exist_ok=True)
    model=load_full_model(args.model,args.device)
    source_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    for ep in args.episodes:
        tag=f'episode_{ep:06d}'; path=args.inputs/f'{tag}.npz'; target=args.out/f'{tag}.npz'
        if target.exists(): raise FileExistsError(target)
        with np.load(path) as z:
            frames=z['frames'].copy(); fps=float(z['fps'])
            if not np.array_equal(frames,np.arange(len(frames))) or not np.isclose(fps,30.,atol=.01):
                raise ValueError('PVA input must contain consecutive 30 FPS frames')
            sample={'imgs':torch.from_numpy(z['images']).permute(0,3,1,2).float()[None].to(args.device)/255.,
                    'bbx_xyxy':torch.from_numpy(z['boxes'])[None].to(args.device),
                    'K_fullimg':torch.from_numpy(z['K'])[None,None].expand(1,len(frames),3,3).contiguous().to(args.device)}
        with torch.inference_mode():
            kp,_,conf,vel,acc,_,_,stats=cli.run_pvanet_inference(model,sample,120,30,False)
        values={k:v[0].detach().cpu().numpy() for k,v in
                [('keypoints',kp),('confidence',conf),('velocity',vel),('acceleration',acc)]}
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError(f'{tag}: nonfinite predictor output')
        if len(values['velocity'])!=len(frames)-1 or len(values['acceleration'])!=len(frames)-2:
            raise ValueError(f'{tag}: unexpected temporal support')
        np.savez(target,**values,frames=frames,fps=fps,K=sample['K_fullimg'][0,0].cpu().numpy())
        report={'source_commit':source_commit,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'model_path':str(args.model),'window':stats,'shapes':{k:list(v.shape) for k,v in values.items()},
                'velocity_units':'metres/second','acceleration_units':'metres/second^2',
                'confidence_median':float(np.median(values['confidence']))}
        target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PVA_DONE',tag,json.dumps(report),flush=True)
        del sample


if __name__=='__main__': main()
