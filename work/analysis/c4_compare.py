# CARI4D stages against the scored human on one episode: placement, motion, smoothness at window joins. No ground truth.
import json,sys,numpy as np
ep=int(sys.argv[1]) if len(sys.argv)>1 else 28; tag=f'episode_{ep:06d}'
S=json.load(open('submissions/episodes_base_BMQ3KGM/scored_frames.json'))[str(ep)]; f=np.array(S)
b=np.load(f'work/analysis/meshes/BMQ3KGM/{tag}.npy').astype(float)
def sim3(src,dst):
    a,c=src.mean(0),dst.mean(0); x,y=src-a,dst-c
    U,s,Vt=np.linalg.svd(x.T@y); d=np.ones(3); d[-1]=-1. if np.linalg.det(U@Vt)<0 else 1.
    R=(U*d)@Vt; sc=(s*d).sum()/(x*x).sum(); return sc,R,c-sc*(a@R)
print('base (measured camera) anchor pelvis-ish centre',np.round(b[0].mean(0),3),'scored',len(f))
for st in ['init','coconet','refined']:
    v=np.load(f'work/cari4d/export/{st}/{tag}.verts.npy',mmap_mode='r')[:,::8].astype(float)
    sc,R,t=sim3(v[0],b[0]); a=sc*(v@R)+t
    d=np.linalg.norm(a-b,axis=-1).mean(1)
    o=np.load(f'work/cari4d/export/{st}/{tag}.object.npz'); tr=o['object_translation'][f]
    acc=np.linalg.norm(np.diff(v,2,axis=0),axis=-1).mean(1)*100
    acco=np.linalg.norm(np.diff(tr,2,axis=0),axis=-1)*100
    print(f"{st:8s} raw centre {np.round(v[0].mean(0),3).tolist()} align scale {sc:.3f} deg {np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))):.1f} | vs BMQ3KGM after align: mean {100*d.mean():.1f} cm max {100*d.max():.1f} | self-acc body {acc.mean():.4f} p99 {np.percentile(acc,99):.3f} | obj t0 {np.round(tr[0],3).tolist()} self-acc {acco.mean():.4f} p99 {np.percentile(acco,99):.3f}")
K=np.load(f'work/cari4d/export/refined/{tag}.object.npz')['camera']; print('camera',np.round(K,1).tolist())
