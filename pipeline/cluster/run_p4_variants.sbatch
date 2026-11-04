#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --job-name=v2d-p1-p4var
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/p4_variants_%j.log
# P4T: P4's recipe with the current (time-axis corrected) root mix; CPU only, separates that fix from P4C's ensemble.
# P4L300: the same on E2 refitted after mesh smoothing lam 300 instead of 100 (lam 100 gave CD-H -0.11; heavier untested).
# Both then get the 13 table objects x1.01 (CD-O's best so far).
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo OMP_NUM_THREADS=4
export TRACK1_EXTRA_BINDS=/work/mnidgu2s/project1/vpq-vitpose
source pipeline/env.sh
trap 'echo "P4VAR_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
S=submissions; A=work/analysis; T=/track1; CODE=/work/mnidgu2s/project1/vpq-vitpose
TAB="1 3 4 6 12 13 14 15 16 18 19 20 22"
echo "P4VAR_START node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} commit=$(git -C $CODE rev-parse HEAD)"
for d in $S/episodes_person_P4T $S/episodes_personm_P4T $S/episodes_cand_P4T $S/episodes_refit_E2L300 $A/bodies_HE2L300 \
         $A/meshes/E2L300 $S/episodes_person_P4L300 $S/episodes_personm_P4L300 $S/episodes_cand_P4L300; do
  [[ -e $d ]] && { echo "$d exists; refusing to overwrite"; exit 2; }
done
recipe() { # NAME REFIT_DIR BODIES MESHES
  "$VENV_PYTHON" scripts/floor_root.py --input "$2" --bodies "$3" --planes outputs/floor/planes_measured.json --out $S/episodes_person_$1 --lam 3e4 --beta 0.55 | tail -1
  "$VENV_PYTHON" scripts/human_root_mix.py --base $S/episodes_person_$1 --base_meshes "$4" --other_meshes $A/meshes/a0 --out $S/episodes_personm_$1 --weight 0.3 | tail -1
  "$VENV_PYTHON" scripts/object_transfer.py --people $S/episodes_personm_$1 --people_meshes "$4" --objects $S/episodes_baseold_k_OTa \
    --objects_meshes $A/meshes/BMQ3KGM_old --out $S/episodes_cand_$1 | tail -1
  "$VENV_PYTHON" $CODE/scripts/object_camera_scale.py --input $S/episodes_cand_$1 --out $S/episodes_cand_${1}tab101 --scale 1.01 --episodes $TAB | tail -1
}
recipe P4T $S/episodes_refit_E2 $A/bodies_HE2R $A/meshes/E2R
run() { in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 python -u "$@"; }
run $T/scripts/human_refit.py --input $T/$S/episodes_ens_E2 --out $T/$S/episodes_refit_E2L300 --lam 300 | tail -1
run $T/scripts/floor_bodies.py --episodes_dir $T/$S/episodes_refit_E2L300 --out $T/$A/bodies_HE2L300 | tail -1
run $T/scripts/ensemble_meshes.py --episodes_dir $T/$S/episodes_refit_E2L300 --out $T/$A/meshes/E2L300 | tail -1
recipe P4L300 $S/episodes_refit_E2L300 $A/bodies_HE2L300 $A/meshes/E2L300
echo P4VAR_BUILT
