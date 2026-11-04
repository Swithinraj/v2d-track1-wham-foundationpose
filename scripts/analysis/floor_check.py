# Does the floor-derived slide agree with apparent size, an independent image cue? No ground truth involved.
import json,sys,numpy as np
sys.path.insert(0,'scripts')
from floor_root import stance_feet,floor_root_shift,whittaker,ENVELOPE_FRAMES
SCORED=(1,2,18,35,3,19,36,4,20,37,8,24,110,74,38,113,75,39,76,40,78,42)
SC2MP={16:11,17:12,18:13,19:14,20:15,21:16,1:23,2:24,4:25,5:26,7:27,8:28}
BONES=[(16,17),(1,2),(16,18),(17,19),(18,20),(19,21),(1,4),(2,5),(4,7),(5,8),(16,1),(17,2)]
planes=json.load(open('work/analysis/floor_planes_measured.json')); S=json.load(open('submissions/episodes_base_BMQ3KGM/scored_frames.json'))
LAM=float(sys.argv[2]) if len(sys.argv)>2 else 1e4
bodies=sys.argv[1]
pool_f,pool_s=[],[]
print('ep  n cover anch-med slideRMS slideMAX  sizeStd  rho  slope | moveZ')
rows=[]
for ep in range(30):
    tag=f'episode_{ep:06d}'; fr=np.array(S[str(ep)]); b=np.load(f'{bodies}/{tag}.npz'); n=np.array(planes[tag]['normal'])
    L,R=b['left_foot'].astype(float),b['right_foot'].astype(float); J=b['joints'].astype(float)[:,list(SCORED)]
    pts,conf=stance_feet(L,R,n,fr)
    st=np.argmin(np.where(conf>0,pts@n,np.inf),axis=1); w=(conf.max(1)>0).astype(float)
    foot=whittaker(np.where(w[:,None]>0,pts[np.arange(len(pts)),st],0.),w+1e-9,100.); ray=foot/np.linalg.norm(foot,axis=1,keepdims=True)
    delta,rep=floor_root_shift(pts,ray,n,conf,int(fr[0]),lam=LAM,anchor_window=ENVELOPE_FRAMES)
    slide=(delta*ray).sum(1)
    zc=np.linalg.norm(J.mean(1),axis=1)
    # apparent size cue
    K=json.load(open(f'outputs/camera_measured/{tag}/intrinsics.json')); kp=np.load(f'outputs/kp2d_vpx/{tag}.npz')['pose'][:,0].astype(float)
    uv=np.stack([K['fx']*J[...,0]/J[...,2]+K['cx'],K['fy']*J[...,1]/J[...,2]+K['cy']],-1)
    T=len(J); c=np.full(T,np.nan)
    for f in fr:
        if f>=len(kp): continue
        r=[]
        for a,bb in BONES:
            pa,pb=kp[f,SC2MP[a]],kp[f,SC2MP[bb]]
            if min(pa[3],pb[3])>.5:
                lv=np.linalg.norm(pa[:2]-pb[:2]); lo=np.linalg.norm(uv[f,a]-uv[f,bb])
                if lv>8 and lo>8: r.append(np.log(lo/lv))
        if len(r)>=3: c[f]=np.median(r)
    ok=np.isfinite(c); wc=ok.astype(float)
    cs=whittaker(np.where(ok,c-np.nanmean(c),0.),wc+1e-9,LAM) if ok.sum()>10 else np.zeros(T)
    a=int(fr[0]); csr=cs-cs[a]
    fl=slide/zc
    sel=fr
    x,y=fl[sel],csr[sel]
    rho=float(np.corrcoef(x,y)[0,1]) if x.std()>1e-6 and y.std()>1e-6 else float('nan')
    slope=float((x*y).sum()/max((x*x).sum(),1e-9))
    pool_f.append(x-x.mean()); pool_s.append(y-y.mean())
    h=(pts[np.arange(len(pts)),st]@n)
    mv=np.percentile(J[fr][:,0,2],95)-np.percentile(J[fr][:,0,2],5)
    print(f"{ep:2d} {len(fr):3d} {rep['coverage']/ (len(fr)/T) if T else 0:4.2f} {100*(rep.get('anchor_height_m',0)-np.median(h[fr][w[fr]>0])):+6.1f} {100*rep.get('translation_rms_m',0):6.1f} {100*rep['translation_max_m']:6.1f} {100*np.nanstd(y):7.1f} {rho:+5.2f} {slope:+5.2f} | {100*mv:5.0f} {'' if rep['accepted'] else rep['reason']}")
    rows.append((rho,100*rep.get('translation_rms_m',0)))
X=np.concatenate(pool_f);Y=np.concatenate(pool_s)
r=np.array([a for a,_ in rows]); print('pooled rho',round(float(np.corrcoef(X,Y)[0,1]),3),'median per-episode rho',round(float(np.nanmedian(r)),3),'positive',int(np.nansum(r>0)),'/30')
