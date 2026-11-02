# Do earlier scored files already show how CD-O responds to the object's depth relative to the person? No submissions used.
import numpy as np,trimesh,json
from pathlib import Path
S=json.load(open('submissions/episodes_base_BMQ3KGM/scored_frames.json'))
cdo=dict(BMQ3KGM=18.604,a0=18.020,VPQ3=18.398,BMQ=18.663,k=18.686,nowham=24.257,D=18.604,R2=18.748,BMQO=18.570,Afr=18.743,ktcp=19.351,abl=19.351)
dirs={n:f'submissions/episodes_tp_{n}' for n in cdo}; dirs['BMQ3KGM']='submissions/episodes_cand_Mtp_in'; dirs['ktcp']='submissions/episodes_base_ktcp'; dirs['abl']='submissions/episodes_base_abl'
def sim3(src,dst):
    a,b=src.mean(0),dst.mean(0); x,y=src-a,dst-b
    U,s,Vt=np.linalg.svd(x.T@y); d=np.ones(3); d[-1]=-1. if np.linalg.det(U@Vt)<0 else 1.
    R=(U*d)@Vt; sc=(s*d).sum()/(x*x).sum(); return sc,R,b-sc*(a@R)
base={}
for ep in range(30):
    tag=f'episode_{ep:06d}'; d=np.load(f"{dirs['BMQ3KGM']}/{tag}.npz"); fr=np.array(S[str(ep)]); base[ep]=(np.load(f'work/analysis/meshes/BMQ3KGM/{tag}.npy')[0].astype(float),d['object_translation'][fr].astype(float))
print('candidate CD-O | object distance vs BMQ3KGM after alignment: median ratio, mean abs % | offset cm')
for n in cdo:
    r=[];off=[]
    for ep in range(30):
        tag=f'episode_{ep:06d}'; fr=np.array(S[str(ep)]); d=np.load(f'{dirs[n]}/{tag}.npz'); V=np.load(f'work/analysis/meshes/{n}/{tag}.npy')[0].astype(float)
        sc,R,t=sim3(V,base[ep][0]); o=sc*(d['object_translation'][fr].astype(float)@R)+t; b=base[ep][1]
        r.append(np.median(np.linalg.norm(o,axis=1)/np.linalg.norm(b,axis=1))); off.append(np.median(np.linalg.norm(o-b,axis=1)))
    r=np.array(r); print(f"{n:8s} {cdo[n]:7.3f} | {np.median(r):.4f} {100*np.abs(r-1).mean():5.2f} | {100*np.mean(off):5.1f}")
