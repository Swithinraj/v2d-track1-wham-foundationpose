#!/bin/bash
# Video Depth Anything qn series for the human depth trend (recipes/trend.md, candidate BMQ3KGMh): the SAM 3D Body
# identity step (scripts/sam3d_body_identity.py, unchanged) re-run with VDA maps in place of MoGe-2's. The raw SAM 3D
# Body results are RGB-only and read as they are; only the depth the step samples changes, so OUT's prepare.npz differs
# from the MoGe-2 one only in its depth series (frames, r, b, qn, depth_scale).
#   VDA maps: outputs/object_vda/<episode>/depth (scripts/vda_depth.py; VDA_PYTHON, VDA_REPO, VDA_WEIGHTS)
# Usage: pipeline/stages/vda_prep.sh RAW_DIR OUT_DIR
#   RAW_DIR: raw SAM 3D Body dir (<episode>/raw_<episode>.npz): outputs/sam3d_body (full) or BM_ROOT/outputs/sam3d_body
#   OUT_DIR: new dir for <episode>/prepare.npz; a work root OUT_DIR.wk serves as the step's /track1
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
RAW="$(cd "$1" && pwd)"; OUT="$(mkdir -p "$2" && cd "$2" && pwd)"; W="${OUT}.wk"
for e in $(seq 0 29); do
  tag="$(episode_tag "${e}")"; v="${TRACK1_ROOT}/outputs/object_vda/${tag}/depth"
  if [[ ! -s "${v}/$(printf '%06d' $(( $(episode_field "${e}" length) - 1 ))).png" ]]; then
    require_file "${VDA_REPO:-}/video_depth_anything/video_depth.py" "set VDA_REPO to a Video-Depth-Anything checkout"
    "${VDA_PYTHON:?set VDA_PYTHON}" scripts/vda_depth.py --root "${TRACK1_ROOT}" --vda_repo "${VDA_REPO}" \
      --weights "${VDA_WEIGHTS:?set VDA_WEIGHTS (metric_video_depth_anything_vitl.pth)}" --episodes "${e}"
  fi
  [[ "$(find "${v}" -maxdepth 1 -name '*.png' | wc -l)" -eq "$(episode_field "${e}" length)" ]] || { echo "${tag}: VDA maps incomplete" >&2; exit 4; }
  mkdir -p "${W}/outputs/object/${tag}"
  ln -sfn "${v}" "${W}/outputs/object/${tag}/depth"
  ln -sfn "${TRACK1_ROOT}/outputs/object/${tag}/sam2_masks" "${W}/outputs/object/${tag}/sam2_masks"
done
mkdir -p "${W}/outputs/keypoints2d_sam"
ln -sfn "${TRACK1_ROOT}/scripts" "${W}/scripts"; ln -sfn "${TRACK1_ROOT}/outputs/person" "${W}/outputs/person"
apptainer exec --no-home --cleanenv --bind "${W}:/track1" --bind "${TRACK1_ROOT}:${TRACK1_ROOT}:ro" --bind "${RAW}:${RAW}:ro" \
  --bind "${OUT}:${OUT}" "${CONTAINERS}/v2d_cari4d.sif" env HOME=/tmp python -u /track1/scripts/sam3d_body_identity.py \
  --raw_dir "${RAW}" --run_dir "${OUT}" --episodes $(seq 0 29)
for e in $(seq 0 29); do require_file "${OUT}/$(episode_tag "${e}")/prepare.npz"; done
echo "PASS: VDA qn series (identity step on VDA maps) -> ${OUT}"
