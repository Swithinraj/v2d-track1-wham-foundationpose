#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --job-name=v2d-p1-refit
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/refit_%j.log
set -uo pipefail
source pipeline/env.sh
set +e
export TMPDIR=/work/mnidgu2s/project1/.tmp
P=.venvs/submission/bin/python
run() { # GPU LAM
  rm -rf submissions/episodes_refit_l$2 work/analysis/full_refit$2
  in_container mhr_converter.sif CUDA_VISIBLE_DEVICES=$1 python -u /track1/scripts/human_refit.py --input /track1/submissions/episodes_cand_Mtp_in --out /track1/submissions/episodes_refit_l$2 --lam $2 | tail -1
  in_container mhr_converter.sif CUDA_VISIBLE_DEVICES=$1 python -u /track1/scripts/mhr_bodies.py --episodes_dir /track1/submissions/episodes_refit_l$2 --out /track1/work/analysis/full_refit$2 | tail -1
}
run 0 30 & run 1 100 & wait
$P - <<'PY'
import json,numpy as np
S=json.load(open('outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json'))
J=(1,2,18,35,3,19,36,4,20,37,8,24,110,74,38,113,75,39,76,40,78,42)
def acc(p,f):
    v=[]
    for b in np.split(f,np.flatnonzero(np.diff(f)!=1)+1):
        if len(b)>=3: v.append(np.linalg.norm(np.diff(p[b],2,axis=0),axis=-1).mean(1))
    return np.concatenate(v).mean()*100
base={ep:np.load(f'work/analysis/full_Mtpin/episode_{ep:06d}.npz')['joints'][:,list(J)].astype(float) for ep in range(30)}
print('mesh lam | self-ACC-H | mean joint move cm, p95 cm | anchor joint move mm')
for name in ['Mtpin','refit30','refit100']:
    a,m,q,n=[],[],[],[]
    for ep in range(30):
        f=np.array(S[str(ep)]); x=np.load(f'work/analysis/full_{name}/episode_{ep:06d}.npz')['joints'][:,list(J)].astype(float)
        a.append(acc(x,f)); d=np.linalg.norm(x[f]-base[ep][f],axis=-1); m.append(d.mean()*100); q.append(np.percentile(d,95)*100); n.append(d[0].mean()*1000)
    print(f"{name:9s} {np.mean(a):.5f}  {np.mean(m):.3f} {np.mean(q):.3f}  {np.mean(n):.2f}",flush=True)
PY
echo PROBEDONE
