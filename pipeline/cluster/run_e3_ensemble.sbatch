#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --job-name=v2d-p1-e3
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/e3_ensemble_%j.log
# P4C: P4's recipe on a three-member human. E2 averaged two SAM 3D Body estimates (CD-H -0.10); here CARI4D's CoCoNet person
# joins them with a fixed third of the weight on the 19 episodes it finished. Replacing our person by CoCoNet helped on 6
# episodes (C1) and hurt on 19 (N19H); a fixed-weight mean is bounded by its members and gains when errors differ.
# The other 11 episodes reuse E2R exactly. Then the floor slide (lam 3e4, beta 0.55), root mix 0.3 toward a0 and OTa
# objects carried to the person, as for P4.
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo OMP_NUM_THREADS=4
source pipeline/env.sh
trap 'echo "E3_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
echo "E3_START node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} commit=$(git -C /work/mnidgu2s/project1/vpq-vitpose rev-parse HEAD)"
C4="3 9 10 12 13 14 15 16 17 19 21 22 23 24 25 26 27 28 29"
S=submissions; A=work/analysis; T=/track1
for d in $S/episodes_ens_E3 $S/episodes_refit_E3 $S/episodes_refit_E3all $A/bodies_HE3R $A/bodies_HE3Rall $A/meshes/E3R19 \
         $A/meshes/E3R $S/episodes_person_P4C $S/episodes_personm_P4C $S/episodes_cand_P4C; do
  [[ -e $d ]] && { echo "$d exists; refusing to overwrite"; exit 2; }
done
run() { in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 python -u "$@"; }
run $T/scripts/human_ensemble.py --out $T/$S/episodes_ens_E3 --member BMQ3KGM=$T/$S/episodes_cand_Mtp_in:0.34 \
  --member ktcp=$T/$S/episodes_base_ktcp:0.33 --member c4=$T/$S/episodes_c4tp_N19H:0.33 --episodes $C4 | tail -1
run $T/scripts/human_refit.py --input $T/$S/episodes_ens_E3 --out $T/$S/episodes_refit_E3 --lam 100 | tail -1
run $T/scripts/floor_bodies.py --episodes_dir $T/$S/episodes_refit_E3 --out $T/$A/bodies_HE3R | tail -1
run $T/scripts/ensemble_meshes.py --episodes_dir $T/$S/episodes_refit_E3 --out $T/$A/meshes/E3R19 | tail -1
mkdir -p $S/episodes_refit_E3all $A/bodies_HE3Rall $A/meshes/E3R
for e in $(seq 0 29); do
  t=$(printf 'episode_%06d' "$e")
  if [[ -f $S/episodes_refit_E3/$t.npz ]]; then src=$S/episodes_refit_E3; b=$A/bodies_HE3R; m=$A/meshes/E3R19
  else src=$S/episodes_refit_E2; b=$A/bodies_HE2R; m=$A/meshes/E2R; fi
  cp -l "$src/$t.npz" "$src/${t}_object.glb" $S/episodes_refit_E3all/
  cp -l "$b/$t.npz" $A/bodies_HE3Rall/; cp -l "$m/$t.npy" $A/meshes/E3R/
done
"$VENV_PYTHON" scripts/floor_root.py --input $S/episodes_refit_E3all --bodies $A/bodies_HE3Rall --planes outputs/floor/planes_measured.json \
  --out $S/episodes_person_P4C --lam 3e4 --beta 0.55 | tail -1
"$VENV_PYTHON" scripts/human_root_mix.py --base $S/episodes_person_P4C --base_meshes $A/meshes/E3R --other_meshes $A/meshes/a0 \
  --out $S/episodes_personm_P4C --weight 0.3 | tail -1
"$VENV_PYTHON" scripts/object_transfer.py --people $S/episodes_personm_P4C --people_meshes $A/meshes/E3R --objects $S/episodes_baseold_k_OTa \
  --objects_meshes $A/meshes/BMQ3KGM_old --out $S/episodes_cand_P4C | tail -1
echo E3_BUILT
