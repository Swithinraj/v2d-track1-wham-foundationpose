# Floor normals in the measured camera: session-averaged MoGe normals mapped through the camera change.
import json,numpy as np
planes=json.load(open('local_recon/kappa_floor.json')); kn=json.load(open('outputs/camera_measured/k_new.json')); ids=json.load(open('config/camera/episode_cameras.json'))
out={}
tags=sorted(planes)
N=np.array([planes[t]['normal'] for t in tags]); N/=np.linalg.norm(N,axis=1,keepdims=True)
for i,t in enumerate(tags):
    # Average over episodes of the same rig camera whose normal agrees within 1 degree (same camera mounting).
    same=[j for j,o in enumerate(tags) if ids[o]['camera']==ids[t]['camera'] and np.degrees(np.arccos(np.clip(N[i]@N[j],-1,1)))<1.0]
    n=N[same].mean(0); n/=np.linalg.norm(n)
    o=json.load(open(f'outputs/camera/{t}/intrinsics.json')); k=kn[t]
    A=np.array([[1,0,(o['cx']-k['cx'])/o['fx']],[0,1,(o['cy']-k['cy'])/o['fx']],[0,0,k['fx']/o['fx']]])
    m=np.linalg.inv(A).T@n; m/=np.linalg.norm(m)
    out[t]=dict(normal=m.tolist(),normal_moge=n.tolist(),support=len(same),spread_deg=float(np.degrees(np.arccos(np.clip(N[same]@n,-1,1))).max()))
json.dump(out,open('work/analysis/floor_planes_measured.json','w'),indent=1)
for t in tags[:30:6]: print(t,ids[t]['camera'][:5],[round(x,4) for x in out[t]['normal']],out[t]['support'],round(out[t]['spread_deg'],2))
