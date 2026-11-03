#!/bin/bash
# Person-recipe variants on the all-30 ensemble refit (CPU): floor slide BETA and root-mix WEIGHT toward a0, retuned
# because both were fitted on the earlier human. Usage: person_variants_e3c30.sh NAME BETA WEIGHT  (repo checkout)
set -euo pipefail
cd /work/mnidgu2s/project1/repo
source pipeline/env.sh
export TMPDIR=/work/mnidgu2s/project1/.tmp
NAME="$1"; BETA="$2"; WEIGHT="$3"; S=submissions; A=work/analysis; CODE=/work/mnidgu2s/project1/vpq-vitpose
for d in $S/episodes_person_$NAME $S/episodes_personm_$NAME $S/episodes_cand_$NAME $S/episodes_cand_${NAME}tab; do
  [[ -e $d ]] && { echo "$d exists; refusing to overwrite"; exit 2; }; done
"$VENV_PYTHON" scripts/floor_root.py --input $S/episodes_refit_E3C30all --bodies $A/bodies_E3C30all --planes outputs/floor/planes_measured.json \
  --out $S/episodes_person_$NAME --lam 3e4 --beta "$BETA" | tail -1
"$VENV_PYTHON" scripts/human_root_mix.py --base $S/episodes_person_$NAME --base_meshes $A/meshes/E3C30all --other_meshes $A/meshes/a0 \
  --out $S/episodes_personm_$NAME --weight "$WEIGHT" | tail -1
"$VENV_PYTHON" scripts/object_transfer.py --people $S/episodes_personm_$NAME --people_meshes $A/meshes/E3C30all --objects $S/episodes_baseold_k_OTa \
  --objects_meshes $A/meshes/BMQ3KGM_old --out $S/episodes_cand_$NAME | tail -1
"$VENV_PYTHON" $CODE/scripts/object_camera_scale.py --input $S/episodes_cand_$NAME --out $S/episodes_cand_${NAME}tab \
  --factors $CODE/pipeline/cluster/object_factors_t1023_f1008.json | tail -1
