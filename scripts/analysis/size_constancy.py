# One physical object keeps one size. Do size differences between episodes predict the floor's depth correction?
import json,sys,numpy as np,trimesh
base=sys.argv[1] if len(sys.argv)>1 else 'submissions/episodes_cand_Mtp_in'
floor=json.load(open(sys.argv[2] if len(sys.argv)>2 else 'submissions/episodes_cand_MF_in/object_floor_scale_report.json'))
tasks={json.loads(l)['episode_index']:json.loads(l)['tasks'][0].lower() for l in open('data/track_1/meta/episodes.jsonl')}
KEYS=[('hula hoop','hoop'),('big red bowl','bowl'),('iron','iron'),('white desk','desk'),('pan','pan'),('block','block'),('roller','roller'),('foam roll','foam'),('stool','stool'),('laptop cart','cart')]
def kind(t):
    for k,n in KEYS:
        if k in t: return n
    return '?'
rows=[]
for ep in range(30):
    tag=f'episode_{ep:06d}'; d=np.load(f'{base}/{tag}.npz'); m=trimesh.load(f'{base}/{tag}_object.glb',force='mesh',process=False)
    size=float(np.linalg.norm(m.extents))*float(d['object_scale'])
    rows.append((ep,kind(tasks[ep]),size,float(d['object_scale']),floor[tag]['kappa'] if floor[tag]['accepted'] else np.nan))
kinds=sorted({r[1] for r in rows})
print('kind  ep  size(m diag)  size/median  implied_kappa(median/size)  floor_kappa')
X,Y=[],[]
out={}
for k in kinds:
    g=[r for r in rows if r[1]==k]; med=np.median([r[2] for r in g])
    for r in g:
        imp=med/r[2]; out[f'episode_{r[0]:06d}']=imp
        print(f"{k:6s} {r[0]:2d}  {r[2]:.3f}  {r[2]/med:.3f}  {imp:.3f}  {r[4]:.3f}")
    fl=[(med/r[2],r[4]) for r in g if np.isfinite(r[4])]
    if len(fl)>=2:
        a=np.array(fl); X+=list(np.log(a[:,0])-np.log(a[:,0]).mean()); Y+=list(np.log(a[:,1])-np.log(a[:,1]).mean())
X,Y=np.array(X),np.array(Y)
print('pairs within objects',len(X),'corr(implied, floor)',round(float(np.corrcoef(X,Y)[0,1]),3) if len(X)>2 else None,'slope',round(float((X*Y).sum()/(X*X).sum()),3) if len(X)>2 else None)
json.dump(out,open('work/analysis/size_implied_kappa.json','w'),indent=1)
