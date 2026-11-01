import json,numpy as np,trimesh,sys
from pathlib import Path
base=Path(sys.argv[1]); bodies=Path(sys.argv[2]); planes=json.load(open('local_recon/kappa_floor.json')); S=json.load(open(base/'scored_frames.json')) if (base/'scored_frames.json').exists() else json.load(open('submissions/episodes_base_BMQ3KGM/scored_frames.json'))
tasks={json.loads(l)['episode_index']:json.loads(l)['tasks'][0] for l in open('data/track_1/meta/episodes.jsonl')}
print('ep  n  task                                  zpel  dz90  hstd h90 h(anch)| obj: omin@anch  omin_med omin_min  otop | moved(cm) ')
res={}
for ep in range(30):
    tag=f'episode_{ep:06d}'; fr=np.array(S[str(ep)]); a=fr[0]
    b=np.load(bodies/f'{tag}.npz'); u=np.array(planes[tag]['normal'],float)
    sL=(b['left_foot'][fr]@u).min(1); sR=(b['right_foot'][fr]@u).min(1); h=np.minimum(sL,sR)
    pel=b['joints'][fr,1]; z=pel[:,2]
    d=np.load(base/f'{tag}.npz'); R=d['object_rotation'][fr]; t=d['object_translation'][fr]; s=float(d['object_scale'])
    m=trimesh.load(base/f'{tag}_object.glb',force='mesh',process=False); V=np.asarray(m.vertices,float)
    O=s*np.einsum('tij,pj->tpi',R,V)+t[:,None]; ou=O@u; omin=ou.min(1); omax=ou.max(1)
    f0=np.median(h)     # human floor level
    h0=h[0]
    res[ep]=dict(h=h,omin=omin,z=z)
    print(f"{ep:2d} {len(fr):3d} {tasks.get(ep,'')[:36]:36s} {np.median(z):4.2f} {100*(np.percentile(z,95)-np.percentile(z,5)):5.1f} {100*h.std():5.1f} {100*(np.percentile(h,95)-np.percentile(h,5)):5.1f} {100*(h0-f0):+5.1f} | {100*(omin[0]-h0):+6.1f} {100*(np.median(omin)-f0):+6.1f} {100*(omin.min()-f0):+6.1f} {100*(np.median(omax)-f0):6.1f} | {100*(np.linalg.norm(t-t.mean(0),axis=1)).max():5.0f}")
