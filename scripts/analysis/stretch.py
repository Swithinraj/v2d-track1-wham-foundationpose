# Is depth motion systematically compressed or stretched? Two independent cues, low frequency only, no ground truth.
import json,sys,numpy as np
sys.path.insert(0,'scripts')
from floor_root import stance_feet,whittaker
SCORED=(1,2,18,35,3,19,36,4,20,37,8,24,110,74,38,113,75,39,76,40,78,42)
SC2MP={16:11,17:12,18:13,19:14,20:15,21:16,1:23,2:24,4:25,5:26,7:27,8:28}
BONES=[(16,17),(1,2),(16,18),(17,19),(18,20),(19,21),(1,4),(2,5),(4,7),(5,8),(16,1),(17,2)]
planes=json.load(open('work/analysis/floor_planes_measured.json')); S=json.load(open('submissions/episodes_base_BMQ3KGM/scored_frames.json')); ids=json.load(open('config/camera/episode_cameras.json'))
bodies=sys.argv[1]
rows=[]
print('ep cam   rangeZ  floor_g  size_g   floor_r size_r')
for ep in range(30):
    tag=f'episode_{ep:06d}'; fr=np.array(S[str(ep)]); b=np.load(f'{bodies}/{tag}.npz'); n=np.array(planes[tag]['normal'])
    L,R=b['left_foot'].astype(float),b['right_foot'].astype(float); J=b['joints'].astype(float)[:,list(SCORED)]
    pts,conf=stance_feet(L,R,n,fr)
    h=np.where(conf>0,pts@n,np.inf); st=h.argmin(1); m=h.min(1); ok=np.isfinite(m)
    foot=pts[np.arange(len(pts)),st]; dist=np.linalg.norm(foot,axis=1); ray=foot/np.maximum(dist,1e-6)[:,None]; rn=ray@n
    z=np.linalg.norm(J[:,0],axis=1)
    sel=ok; a=fr[0]
    if sel.sum()<30: continue
    # floor cue: height h = level + rn * e  -> e = (h - level) / rn; our distance error e, in metres along the ray
    # Fit h = c + rn*(g-1)*(dist) : the error grows with distance by factor (g-1), c = level.
    # sliding by -e fixes it. Here e>0 means we are too far by e.
    X=np.c_[np.ones(sel.sum()),rn[sel]*(dist[sel]-dist[sel].mean())]; c,_,_,_=np.linalg.lstsq(X,m[sel],rcond=None); floor_g=1-c[1]     # true = g * ours
    res=m[sel]-X@c; floor_r=1-res.var()/max(m[sel].var(),1e-12)
    # size cue: log(ours/vitpose) = log(z_true/z_ours) ~ (g-1)(dist-mean)/mean
    K=json.load(open(f'outputs/camera_measured/{tag}/intrinsics.json')); kp=np.load(f'outputs/kp2d_vpx/{tag}.npz')['pose'][:,0].astype(float)
    uv=np.stack([K['fx']*J[...,0]/J[...,2]+K['cx'],K['fy']*J[...,1]/J[...,2]+K['cy']],-1)
    c_=np.full(len(J),np.nan)
    for f in fr:
        if f>=len(kp): continue
        r=[]
        for x,y in BONES:
            pa,pb=kp[f,SC2MP[x]],kp[f,SC2MP[y]]
            if min(pa[3],pb[3])>.5:
                lv=np.linalg.norm(pa[:2]-pb[:2]); lo=np.linalg.norm(uv[f,x]-uv[f,y])
                if lv>8 and lo>8: r.append(np.log(lo/lv))
        if len(r)>=3: c_[f]=np.median(r)
    v=np.isfinite(c_)&np.isin(np.arange(len(J)),fr)
    zm=z[v].mean(); X2=np.c_[np.ones(v.sum()),(z[v]-zm)/zm]; d,_,_,_=np.linalg.lstsq(X2,c_[v],rcond=None); size_g=1+d[1]; res2=c_[v]-X2@d; size_r=1-res2.var()/max(c_[v].var(),1e-12)
    rng=np.percentile(z[fr],95)-np.percentile(z[fr],5)
    rows.append((ep,rng,floor_g,size_g,floor_r,size_r))
    print(f"{ep:2d} {ids[tag]['camera'][:5]} {100*rng:6.0f}  {floor_g:6.3f}  {size_g:6.3f}   {floor_r:5.2f} {size_r:5.2f}")
A=np.array(rows); big=A[:,1]>.3
print('episodes with range>30cm:',big.sum(),' floor_g median',round(np.median(A[big,2]),3),'size_g median',round(np.median(A[big,3]),3),' corr(floor_g,size_g)',round(np.corrcoef(A[big,2],A[big,3])[0,1],3))
w=A[big,1]**2
print('range^2-weighted mean: floor_g',round(float((A[big,2]*w).sum()/w.sum()),3),'size_g',round(float((A[big,3]*w).sum()/w.sum()),3))
