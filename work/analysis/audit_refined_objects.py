"""Read-only trajectory audit of preserved CARI4D stage outputs; no reconstruction."""
import json
from pathlib import Path
import numpy as np
import torch

root=Path('/track1')
scored=json.loads((root/'outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json').read_text())
reports={}
for folder in sorted((root/'work/cari4d/out').glob('episode_*')):
    ep=int(folder.name.split('_')[1]);frames=np.array(scored[str(ep)])
    good=(np.diff(frames)[:-1]==1)&(np.diff(frames)[1:]==1)
    report={}
    for filename,blocks in [('coconet.pth',['in','pr_initial']),('refined.pth',['pr'])]:
        path=folder/'inference'/filename
        if not path.exists():continue
        bundle=torch.load(path,map_location='cpu',weights_only=False)
        for name in blocks:
            block=bundle[name]
            values=block['pose_abs'] if 'pose_abs' in block else block['obj_t']
            if torch.is_tensor(values):values=values.detach().cpu().numpy()
            values=np.asarray(values)
            translation=values[:,:3,3] if values.ndim==3 else values
            acceleration=np.linalg.norm(np.diff(translation[frames],n=2,axis=0),axis=-1)[good]*100
            report[name]={'mean_second_difference_cm':float(acceleration.mean()),
                          'p99_second_difference_cm':float(np.quantile(acceleration,.99)),
                          'finite':bool(np.isfinite(translation).all()),'frames':len(translation)}
        del bundle
    if report:reports[folder.name]=report
    print(folder.name,json.dumps(report),flush=True)
(root/'work/analysis/refined_object_dynamics.json').write_text(json.dumps(reports,indent=2)+'\n')
print('REFINED_OBJECT_AUDIT_COMPLETE',len(reports),flush=True)
