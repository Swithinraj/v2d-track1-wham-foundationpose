# Does the body face where it walks? A yaw error at the alignment frame swings the whole trajectory. No ground truth.
import json,sys,numpy as np
sys.path.insert(0,'scripts')
from floor_root import whittaker
planes=json.load(open('outputs/floor/planes_measured.json')); S=json.load(open('submissions/episodes_base_BMQ3KGM/scored_frames.json'))
bodies='work/analysis/bodies_Mtp_in'
print('ep  walking  n_frames  median_yaw_offset(deg)  spread(p25,p75)  | first-scored-frame: walking? distance moved(cm)')
all_med=[]
for ep in range(30):
    tag=f'episode_{ep:06d}'; f=np.array(S[str(ep)]); J=np.load(f'{bodies}/{tag}.npz')['joints'].astype(float); up=np.array(planes[tag]['normal'])
    pel=J[:,1]; pel=pel-(pel@up)[:,None]*up
    w=np.zeros(len(J)); w[f]=1
    smooth=whittaker(np.where(w[:,None]>0,pel,0.),w+1e-6,50.)
    v=np.zeros_like(smooth); v[1:-1]=(smooth[2:]-smooth[:-2])/2
    speed=np.linalg.norm(v,axis=1)
    hip=J[:,18]-J[:,2]; hip=hip-(hip@up)[:,None]*up
    forward=np.cross(hip,up); forward/=np.maximum(np.linalg.norm(forward,axis=1,keepdims=True),1e-9)
    sel=f[speed[f]>.012]    # above 36 cm/s
    if len(sel)<10:
        print(f"{ep:2d}   no walking in scored frames ({len(sel)})"); continue
    vd=v[sel]/speed[sel][:,None]
    cross=np.cross(forward[sel],vd)@up; dot=np.einsum('ij,ij->i',forward[sel],vd)
    ang=np.degrees(np.arctan2(cross,dot))
    # Forward or backward walking: fold to the nearer direction.
    backward=np.abs(ang)>90
    a=np.where(backward,ang-np.sign(ang)*180,ang)
    med=np.median(a); all_med.append(med)
    an=f[0]; d=np.ptp(pel[f],axis=0); moved=np.linalg.norm(pel[f]-pel[an],axis=1).max()
    print(f"{ep:2d}   {len(sel):4d}   back {100*backward.mean():3.0f}%   {med:+6.1f}   ({np.percentile(a,25):+6.1f},{np.percentile(a,75):+6.1f}) | speed@anchor {100*speed[an]:4.1f} cm/f  moved {100*moved:4.0f}")
m=np.array(all_med); print('episodes',len(m),'median',round(float(np.median(m)),1),'mean abs',round(float(np.abs(m).mean()),1))
