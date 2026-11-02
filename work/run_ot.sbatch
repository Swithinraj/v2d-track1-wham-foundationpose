#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=v2d-p1-ot
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/ot_%j.log
set -euo pipefail
source pipeline/env.sh
export TMPDIR=/work/mnidgu2s/project1/.tmp
P=.venvs/submission/bin/python; S=submissions; PL=outputs/floor/planes_moge.json
$P - <<'PY'
import json
planes=json.load(open('outputs/floor/planes_measured.json'))
json.dump({k:{'normal':v['normal_moge']} for k,v in planes.items()},open('outputs/floor/planes_moge.json','w'),indent=1)
PY
[[ -f work/analysis/meshes/BMQ3KGM_old/episode_000029.npy ]] || in_container mhr_converter.sif python -u /track1/scripts/ensemble_meshes.py --episodes_dir /track1/$S/episodes_base_BMQ3KGM --out /track1/work/analysis/meshes/BMQ3KGM_old | tail -1
rm -rf $S/episodes_baseold_lv
$P scripts/floor_root.py --input $S/episodes_base_BMQ3KGM --bodies work/analysis/bodies_BMQ3KGM --planes $PL --out $S/episodes_baseold_lv --lam 1e4 --beta 0 | tail -1
build() {  # NAME BETA FLOOR_OFFSET
  local n=$1 b=$2 h=$3
  rm -rf $S/episodes_person_$n $S/episodes_baseold_k_$n $S/episodes_cand_$n
  $P scripts/floor_root.py --input $S/episodes_cand_Mtp_in --bodies work/analysis/bodies_Mtp_in --planes outputs/floor/planes_measured.json --out $S/episodes_person_$n --lam 1e4 --beta $b | tail -1
  $P scripts/object_floor_scale.py --input $S/episodes_baseold_lv --planes $PL --levels $S/episodes_baseold_lv/floor_root_report.json --floor_offset $h --out $S/episodes_baseold_k_$n | tail -1
  $P scripts/object_transfer.py --people $S/episodes_person_$n --people_meshes work/analysis/meshes/BMQ3KGM --objects $S/episodes_baseold_k_$n --objects_meshes work/analysis/meshes/BMQ3KGM_old --out $S/episodes_cand_$n
}
build OTa 0.5 0.02
build OTb 0.7 0.04
$P - <<'PY'
import json,numpy as np
for n in ('OTa','OTb'):
    r=json.load(open(f'submissions/episodes_baseold_k_{n}/object_floor_scale_report.json'))
    k=[v['kappa'] for v in r.values() if v['accepted']]
    print(n,'floor scale: objects',len(k),'median',round(float(np.median(k)),4),'mean',round(float(np.mean(k)),4),'range',round(min(k),3),round(max(k),3))
t=json.load(open('submissions/episodes_cand_OTa/object_transfer_report.json'))
print('transfer: scale median',round(float(np.median([v['scale'] for v in t.values()])),4),'rotation deg median',round(float(np.median([v['rotation_deg'] for v in t.values()])),2))
PY
echo OTBUILT
