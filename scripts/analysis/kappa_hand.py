# Which object scale about the camera puts the object against the hands? Independent of the floor, no ground truth.
import json,sys,numpy as np,trimesh
from pathlib import Path
from scipy.spatial import cKDTree
objects=sys.argv[1]; hands_dir=sys.argv[2]; out=sys.argv[3]; align_from=sys.argv[4] if len(sys.argv)>4 else None; align_to=sys.argv[5] if len(sys.argv)>5 else None
S=json.load(open('submissions/episodes_base_BMQ3KGM/scored_frames.json'))
def sim3(src,dst):
    a,b=src.mean(0),dst.mean(0); x,y=src-a,dst-b
    U,s,Vt=np.linalg.svd(x.T@y); d=np.ones(3); d[-1]=-1. if np.linalg.det(U@Vt)<0 else 1.
    R=(U*d)@Vt; sc=(s*d).sum()/(x*x).sum(); return sc,R,b-sc*(a@R)
grid=np.arange(.86,1.1401,.01); rows={}
for ep in range(30):
    tag=f'episode_{ep:06d}'; fr=np.array(S[str(ep)]); use=fr[np.unique(np.linspace(0,len(fr)-1,min(60,len(fr))).astype(int))]
    d=np.load(f'{objects}/{tag}.npz'); s=float(d['object_scale']); R=d['object_rotation'][use].astype(float); t=d['object_translation'][use].astype(float)
    m=trimesh.load(f'{objects}/{tag}_object.glb',force='mesh',process=False)
    pts,_=trimesh.sample.sample_surface(m,20000,seed=ep); tree=cKDTree(np.asarray(pts))
    h=np.load(f'{hands_dir}/{tag}.npz')['hands'].astype(float)[use]
    split=h.shape[1]//2
    if align_from:
        a=np.load(f'{align_from}/{tag}.npy')[0].astype(float); b=np.load(f'{align_to}/{tag}.npy')[0].astype(float); sc,Ra,ta=sim3(a,b); h=sc*(h@Ra)+ta
    h=h[:,::6]; sp=split//6
    G=[]
    for k in grid:
        q=np.einsum('tpi,tij->tpj',h-k*t[:,None],R)/(k*s)     # into mesh coordinates (row vectors: X = s R v + t)
        dist,_=tree.query(q.reshape(-1,3)); dist=dist.reshape(q.shape[:2])*k*s
        gap=np.minimum(dist[:,:sp].min(1),dist[:,sp:].min(1)); G.append(gap)
    G=np.array(G)                 # [kappas, frames]
    med=np.median(G,axis=1); low=np.percentile(G,25,axis=1)
    i=int(med.argmin()); kap=grid[i]
    rows[tag]=dict(kappa=float(kap),gap_at_1=float(med[np.argmin(np.abs(grid-1))]),gap_min=float(med[i]),near_share_at_best=float((G[i]<.03).mean()),curve=[float(x) for x in med])
    print(tag[-2:],f"kappa {kap:.2f} gap@1 {100*rows[tag]['gap_at_1']:5.1f} cm  gap@best {100*rows[tag]['gap_min']:5.1f} cm  frames<3cm {rows[tag]['near_share_at_best']:.2f}",flush=True)
json.dump(rows,open(out,'w'),indent=0)
