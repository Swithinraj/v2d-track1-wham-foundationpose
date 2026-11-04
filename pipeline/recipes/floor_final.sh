#!/bin/bash
# Candidates built from a scored submission file (see floor.md). The file is the exact record of what Kaggle scored, so
# each change here is measured against a known score.
# Usage (inside one Slurm allocation with CONTACT_GPUS GPUs): floor_final.sh SUBMISSION.parquet NAME
#   FLOOR_HUMAN=1      slide the person along the camera ray from floor contact (scripts/floor_root.py)
#   FLOOR_OBJECTS=1    scale each object about the camera onto the person's floor (scripts/object_floor_scale.py)
#   FLOOR_LAM          floor_root.py smoothing (default 1e4), FLOOR_BETA fraction applied (default 1)
# Writes submissions/episodes_cand_NAME (after the contact stage), then pack it with pack_candidate.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
SOURCE="$1"; NAME="$2"; S=submissions; A=outputs/floor
BASE="${S}/episodes_scored_${NAME}"; TP="${S}/episodes_tp_${NAME}"; IN="episodes_cand_${NAME}_in"
rm -rf "${BASE}" "${TP}" "${S:?}/${IN}" "${S}/${IN}_h"
"${VENV_PYTHON}" scripts/parquet_episodes.py "${SOURCE}" "${BASE}"
"${HOST_PYTHON}" scripts/measured_camera.py --mode "${TRUECAM_MODE:-tcp}"
in_container mhr_converter.sif python -u /track1/scripts/true_cam_params.py --input "/track1/${BASE}" --out "/track1/${TP}" \
  --old_root /track1 --k_new /track1/outputs/camera_measured/k_new.json
CURRENT="${TP}"
if [[ "${FLOOR_HUMAN:-0}" == 1 || "${FLOOR_OBJECTS:-0}" == 1 ]]; then
  "${VENV_PYTHON}" scripts/floor_planes.py --moge config/floor/kappa_floor.json --out "${A}/planes_measured.json"
  in_container mhr_converter.sif python -u /track1/scripts/floor_bodies.py --episodes_dir "/track1/${TP}" --out "/track1/${A}/bodies_${NAME}"
  # Always measured; the person's slide is only kept when FLOOR_HUMAN=1. The objects need the person's floor level.
  BETA="${FLOOR_BETA:-1}"; [[ "${FLOOR_HUMAN:-0}" == 1 ]] || BETA=0
  "${VENV_PYTHON}" scripts/floor_root.py --input "${TP}" --bodies "${A}/bodies_${NAME}" --planes "${A}/planes_measured.json" \
    --out "${S}/${IN}_h" --lam "${FLOOR_LAM:-1e4}" --beta "${BETA}"
  CURRENT="${S}/${IN}_h"
fi
if [[ "${FLOOR_OBJECTS:-0}" == 1 ]]; then
  "${VENV_PYTHON}" scripts/object_floor_scale.py --input "${CURRENT}" --planes "${A}/planes_measured.json" \
    --levels "${S}/${IN}_h/floor_root_report.json" --out "${S}/${IN}"
else
  cp -a "${CURRENT}" "${S}/${IN}"
fi
bash pipeline/recipes/contact_stage_split.sh "${IN}" "episodes_cand_${NAME}"
