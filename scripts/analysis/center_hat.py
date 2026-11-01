# Body-centre trajectories relative to the anchor: how far apart are the backbones, and in which direction?
import numpy as np,json
from pathlib import Path
root=Path('work/analysis/meshes')
def sim3(src,dst):
    a,b=src.mean(0),dst.mean(0); x,y=src-a,dst-b
    U,s,Vt=np.linalg.svd(x.T@y); d=np.ones(3); d[-1]=-1. if np.linalg.det(U@Vt)<0 else 1.
    R=(U*d)@Vt; sc=(s*d).sum()/(x*x).sum(); return sc,R,b-sc*(a@R)
names=['BMQ3KGM','ktcp','lam300','h2t']
print('ep  dist(BMQ,ktcp) dist(BMQ,lam300) dist(BMQ,h2t) | along-ray part lam300 h2t | corr of ray-part lam300 vs h2t | range')
A=[]
for ep in range(30):
    M=[np.load(root/n/f'episode_{ep:06d}.npy').astype(np.float64) for n in names]
    al=[]
    for m in M:
        s,R,t=sim3(m[0],M[0][0]); al.append((s*(m@R)+t).mean(1))
    b=al[0]; ray=b/np.linalg.norm(b,axis=1,keepdims=True)
    d=[np.linalg.norm(a-b,axis=1) for a in al]; r=[((a-b)*ray).sum(1) for a in al]
    c=np.corrcoef(r[2],r[3])[0,1] if r[2].std()>1e-6 else np.nan
    rng=np.ptp(np.linalg.norm(b,axis=1))
    A.append((d[1].mean(),d[2].mean(),d[3].mean(),np.abs(r[2]).mean(),np.abs(r[3]).mean(),c,rng))
    print(f"{ep:2d} {100*d[1].mean():6.1f} {100*d[2].mean():6.1f} {100*d[3].mean():6.1f} | {100*np.abs(r[2]).mean():6.1f} {100*np.abs(r[3]).mean():6.1f} | {c:+5.2f} | {100*rng:5.0f}")
A=np.array(A); print('mean',np.round(100*A[:,:5].mean(0),1),'median corr',round(float(np.nanmedian(A[:,5])),2))
