#!/usr/bin/env python3
"""Throwaway diagnostic: can released RGB backgrounds constrain scene geometry?

No ground truth, external room calibration, or monocular depth values are used.
Initial focal estimates come from the current Track 1 RGB pipeline. This does not
modify reconstruction outputs and is not a submission method.
"""
from pathlib import Path
import itertools,json
import cv2
import numpy as np

cv2.setNumThreads(4)
root=Path('/work/mnidgu2s/project1/repo'); out=root/'work/spikes/background_geometry'
out.mkdir(parents=True,exist_ok=True)
episodes=[0,3,4,6,9,12,15,18,21,24,27,28]
sift=cv2.SIFT_create(nfeatures=7000,contrastThreshold=.015)
images={}; records=[]
for ep in episodes:
 tag=f'episode_{ep:06d}'
 cap=cv2.VideoCapture(str(root/f'data/track_1/videos/chunk-000/observation.images.exo_camera/{tag}.mp4'))
 ok,im=cap.read(); cap.release()
 if not ok: continue
 h,w=im.shape[:2]; size=(w//2,h//2)
 image=cv2.resize(im,size)
 allowed=np.full((h//2,w//2),255,np.uint8)
 for kind,index in [('person','0'),('object','1')]:
  mask=cv2.imread(str(root/f'outputs/{kind}/{tag}/sam2_masks/{index}/000000.png'),0)
  if mask is None: raise RuntimeError(f'missing exclusion mask {tag}/{kind}')
  mask=cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)>127
  allowed[cv2.dilate(mask.astype(np.uint8),np.ones((13,13),np.uint8))>0]=0
 key,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),allowed)
 camera=json.loads((root/f'outputs/camera/{tag}/intrinsics.json').read_text())
 K=np.array([[camera['fx']/2,0,camera['cx']/2],[0,camera['fy']/2,camera['cy']/2],[0,0,1.]])
 points=np.asarray([k.pt for k in key],float)
 images[ep]=(points,desc,K,h//2)
 print('features',ep,len(points),flush=True)
matcher=cv2.BFMatcher(cv2.NORM_L2)
for a,b in itertools.combinations(images,2):
 pa,da,Ka,ha=images[a]; pb,db,Kb,hb=images[b]
 forward=matcher.knnMatch(da,db,k=2); reverse=matcher.knnMatch(db,da,k=2)
 back={m.queryIdx:m.trainIdx for m,n in reverse if m.distance<.75*n.distance}
 pairs=[m for m,n in forward if m.distance<.75*n.distance and back.get(m.trainIdx)==m.queryIdx]
 if len(pairs)<35: continue
 ua=np.asarray([pa[m.queryIdx] for m in pairs]); ub=np.asarray([pb[m.trainIdx] for m in pairs])
 norm=lambda uv,K: (np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T)[:,:2]
 xa,xb=norm(ua,Ka),norm(ub,Kb)
 E,inliers=cv2.findEssentialMat(xa,xb,np.eye(3),method=cv2.RANSAC,prob=.999,threshold=.002,maxIters=3000)
 if E is None or E.shape!=(3,3): continue
 count,R,t,positive=cv2.recoverPose(E,xa,xb,np.eye(3),mask=inliers)
 record=dict(episodes=[a,b],mutual_matches=len(pairs),essential_inliers=int(inliers.sum()),positive_depth=int(count))
 # Parallax distinguishes a real baseline from identical-view matches.
 rays_a=np.c_[xa,np.ones(len(xa))]; rays_b=np.c_[xb,np.ones(len(xb))]@R
 rays_a/=np.linalg.norm(rays_a,axis=1)[:,None]; rays_b/=np.linalg.norm(rays_b,axis=1)[:,None]
 angle=np.degrees(np.arccos(np.clip((rays_a*rays_b).sum(1),-1,1)))
 keep=positive.ravel()>0
 record['parallax_deg_median']=float(np.median(angle[keep])) if keep.any() else 0.
 record['usable_baseline']=bool(count>=40 and record['parallax_deg_median']>=1.)
 if record['usable_baseline']:
  X=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[R,t],xa.T,xb.T)
  X=(X[:3]/X[3]).T
  floor=keep&(ua[:,1]>.62*ha)&(ub[:,1]>.62*hb)&np.isfinite(X).all(1)
  Q=X[floor]; record['floor_candidates']=len(Q)
  if len(Q)>=30:
   rng=np.random.default_rng(0); best=np.zeros(len(Q),bool); normal=None
   for _ in range(1000):
    q=Q[rng.choice(len(Q),3,replace=False)]; n=np.cross(q[1]-q[0],q[2]-q[0]); length=np.linalg.norm(n)
    if length<1e-9: continue
    n/=length
    if n[1]>0: n=-n
    if -n[1]<.8: continue
    threshold=.01*np.median(np.linalg.norm(Q,axis=1))
    mask=np.abs((Q-q[0])@n)<threshold
    if mask.sum()>best.sum(): best=mask; normal=n
   record['floor_inliers']=int(best.sum())
   if best.sum()>=25:
    centered=Q[best]-Q[best].mean(0); _,singular,V=np.linalg.svd(centered,full_matrices=False)
    n=V[-1]; n=n if n[1]<0 else -n
    record['floor_normal_camera_a']=n.tolist(); record['floor_spread_ratio']=float(singular[1]/max(singular[0],1e-12))
 records.append(record); print(json.dumps(record),flush=True)
(out/'report.json').write_text(json.dumps(records,indent=2)+'\n')
print('usable pairs',sum(r['usable_baseline'] for r in records),'floor estimates',sum('floor_normal_camera_a' in r for r in records),flush=True)
