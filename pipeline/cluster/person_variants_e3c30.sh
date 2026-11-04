#!/bin/bash
# Person-recipe variants on the all-30 ensemble refit (CPU): floor slide BETA and root-mix WEIGHT toward a0, retuned
# because both were fitted on the earlier human. Usage: [REFIT=E3C30] person_variants_e3c30.sh NAME BETA WEIGHT
# REFIT names the refit set built by run_ensemble_variant.sbatch (episodes_refit_<REFIT>all, bodies, meshes).
# Kaggle so far: mix 0.15 12.96979, 0.3 12.97765, 0.45 13.01137; beta 0.45 13.01232, 0.55 12.97765, 0.65 13.00200.
set -euo pipefail
cd /work/mnidgu2s/project1/repo
source pipeline/env.sh
export TMPDIR=/work/mnidgu2s/project1/.tmp
NAME="$1"; BETA="$2"; WEIGHT="$3"; R="${REFIT:-E3C30}"; S=submissions; A=work/analysis; CODE=/work/mnidgu2s/project1/vpq-vitpose
for d in $S/episodes_person_$NAME $S/episodes_personm_$NAME $S/episodes_cand_$NAME $S/episodes_cand_${NAME}tab; do
  [[ -e $d ]] && { echo "$d exists; refusing to overwrite"; exit 2; }; done
"$VENV_PYTHON" scripts/floor_root.py --input $S/episodes_refit_${R}all --bodies $A/bodies_${R}all --planes outputs/floor/planes_measured.json \
  --out $S/episodes_person_$NAME --lam 3e4 --beta "$BETA" | tail -1
"$VENV_PYTHON" scripts/human_root_mix.py --base $S/episodes_person_$NAME --base_meshes $A/meshes/${R}all --other_meshes $A/meshes/a0 \
  --out $S/episodes_personm_$NAME --weight "$WEIGHT" | tail -1
"$VENV_PYTHON" scripts/object_transfer.py --people $S/episodes_personm_$NAME --people_meshes $A/meshes/${R}all --objects $S/episodes_baseold_k_OTa \
  --objects_meshes $A/meshes/BMQ3KGM_old --out $S/episodes_cand_$NAME | tail -1
"$VENV_PYTHON" $CODE/scripts/object_camera_scale.py --input $S/episodes_cand_$NAME --out $S/episodes_cand_${NAME}tab \
  --factors $CODE/pipeline/cluster/object_factors_t1023_f1008.json | tail -1
