import numpy as np,json,itertools
from pathlib import Path
names=['BMQ3KGM','ktcp','abl','a0','VPQ3','nowham','BMQ','D','Afr','rdL100','lam300','h2t']
scores=dict(BMQ3KGM=14.354,ktcp=14.310,abl=14.342,a0=14.702,VPQ3=14.506,nowham=14.670,BMQ=14.417,D=14.568,Afr=14.712,rdL100=14.655,lam300=18.99,h2t=21.48)
root=Path('work/analysis/meshes')
def sim3(src,dst):
    a,b=src.mean(0),dst.mean(0); x,y=src-a,dst-b
    U,S,Vt=np.linalg.svd(x.T@y); d=np.ones(3); d[-1]=-1. if np.linalg.det(U@Vt)<0 else 1.
    R=(U*d)@Vt; s=(S*d).sum()/(x*x).sum(); t=b-s*(a@R); return s,R,t
N=len(names); D2=np.zeros((N,N,30)); M1=np.zeros((N,N,30)); DR=np.zeros((N,N,30))
for ep in range(30):
    M=[np.load(root/n/f'episode_{ep:06d}.npy').astype(np.float64) for n in names]
    ref=M[0][0]
    A=[]
    for m in M:
        s,R,t=sim3(m[0],ref); A.append(s*(m@R)+t)
    for i,j in itertools.combinations(range(N),2):
        d=np.linalg.norm(A[i]-A[j],axis=-1)
        D2[i,j,ep]=D2[j,i,ep]=(d**2).mean(); M1[i,j,ep]=M1[j,i,ep]=d.mean()
        dc=np.linalg.norm(A[i].mean(1)-A[j].mean(1),axis=-1); DR[i,j,ep]=DR[j,i,ep]=dc.mean()
np.save('work/analysis/hat_D2.npy',D2)
np.set_printoptions(linewidth=170,precision=1,suppress=True)
print('names',names)
print('mean vertex distance between estimates after anchor alignment (cm), mean over episodes')
print((M1.mean(2)*100))
print('body-centre distance (cm)')
print((DR.mean(2)*100))
# three-cornered hat: D2_ij = v_i + v_j, least squares with v >= 0
from scipy.optimize import nnls
def hat(idx,D):
    rows,y=[],[]
    for a,b in itertools.combinations(range(len(idx)),2):
        r=np.zeros(len(idx)); r[a]=r[b]=1; rows.append(r); y.append(D[idx[a],idx[b]])
    v,_=nnls(np.array(rows),np.array(y)); return v
D=D2.mean(2)
for sub in (['BMQ3KGM','ktcp','VPQ3'],['BMQ3KGM','ktcp','VPQ3','nowham'],['BMQ3KGM','ktcp','VPQ3','nowham','h2t'],['BMQ3KGM','ktcp','VPQ3','nowham','lam300','h2t']):
    idx=[names.index(n) for n in sub]; v=hat(idx,D); w=1/np.maximum(v,1e-6); w/=w.sum()
    print(sub,'independent rms (cm)',np.round(100*np.sqrt(v),1),'inverse-variance w',np.round(w,2),'ensemble rms (cm)',round(100*float(np.sqrt(1/np.sum(1/np.maximum(v,1e-6)))),1))
