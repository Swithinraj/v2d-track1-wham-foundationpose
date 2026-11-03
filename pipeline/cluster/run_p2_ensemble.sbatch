#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=v2d-p1-p2e
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/p2_ensemble_%j.log
# P2E: P2's recipe (ensemble person, floor slide lam 1e4 beta 0.55, no mesh smoothing; P2 is the ACC-H best) on the
# E3W5 ensemble, which adds CARI4D's CoCoNet person at weight 0.5 on its 19 episodes (E2 elsewhere). Averaging partly
# independent estimators cancels noise without damping their shared motion, unlike the shrinks that raised ACC-H.
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo OMP_NUM_THREADS=4
source pipeline/env.sh
trap 'echo "P2E_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
S=submissions; A=work/analysis; T=/track1
for d in $S/episodes_ens_E3W5all $A/bodies_HE3W5 $A/meshes/person_HE3W5 $S/episodes_person_HE3W5 $S/episodes_cand_P2E; do
  [[ -e $d ]] && { echo "$d exists; refusing to overwrite"; exit 2; }
done
mkdir -p $S/episodes_ens_E3W5all
for e in $(seq 0 29); do t=$(printf 'episode_%06d' "$e"); src=$S/episodes_ens_E3W5; [[ -f $src/$t.npz ]] || src=$S/episodes_ens_E2
  cp -l "$src/$t.npz" $S/episodes_ens_E3W5all/; [[ -f "$src/${t}_object.glb" ]] && cp -l "$src/${t}_object.glb" $S/episodes_ens_E3W5all/ || cp -l "$S/episodes_ens_E2/${t}_object.glb" $S/episodes_ens_E3W5all/
done
run() { in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 python -u "$@"; }
run $T/scripts/floor_bodies.py --episodes_dir $T/$S/episodes_ens_E3W5all --out $T/$A/bodies_HE3W5 | tail -1
"$VENV_PYTHON" scripts/floor_root.py --input $S/episodes_ens_E3W5all --bodies $A/bodies_HE3W5 --planes outputs/floor/planes_measured.json \
  --out $S/episodes_person_HE3W5 --lam 1e4 --beta 0.55 | tail -1
run $T/scripts/ensemble_meshes.py --episodes_dir $T/$S/episodes_person_HE3W5 --out $T/$A/meshes/person_HE3W5 | tail -1
"$VENV_PYTHON" scripts/object_transfer.py --people $S/episodes_person_HE3W5 --people_meshes $A/meshes/person_HE3W5 \
  --objects $S/episodes_baseold_ksz --objects_meshes $A/meshes/BMQ3KGM_old --out $S/episodes_cand_P2E | tail -1
echo P2E_BUILT
