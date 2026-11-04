#!/bin/bash
# Candidate VPQ3: VPQ2 + restored image-plane motion (lateral band), BEFORE the contact stage (see vpq.md, "Version VPQ3").
# Input: episodes_vpq2_in (vpq2_final.sh: VPQ's pre-contact human + depth smoothing + object smoothing), the raw input
# human episodes_repro_depth and its bodies outputs/human_refine_vp/bodies_base; pinned in vpq3_inputs.sha256.
#   1. mask visibility of the raw human's joints (actor mask, object mask)                     mask_visibility.py
#   2. per vertex in image coordinates: VPQ2 + w * [S_100 - S_300](raw uv) * z, depth kept, first scored frame pinned;
#      w = ViTPose confidence x raw/ViTPose agreement x mask visibility (upper body); pose refit  lateral_band_refit.py
#   3. VPQ's contact stage: contact solve, arm-correction smoothing (sigma 4), hand-only re-solve
# Usage: pipeline/recipes/vpq3_final.sh [OUTPUT]   (default episodes_cand_VPQ3)
#   OBJ_PYTHON: interpreter with numpy and OpenCV for mask_visibility.py (default VENV_PYTHON).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
OUT="${1:-episodes_cand_VPQ3}"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx
for d in "${S}/episodes_vpq2_in" "${S}/episodes_repro_depth" "${TRACK1_ROOT}/outputs/human_refine_vp/bodies_base" "${TRACK1_ROOT}/${KP}"; do
  require_file "${d}/episode_000029.npz" "pinned input, see pipeline/recipes/vpq.md"
done
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP}"

"${OBJ_PYTHON:-${VENV_PYTHON}}" scripts/mask_visibility.py --bodies outputs/human_refine_vp/bodies_base --out outputs/mask_visibility_vp
in_container mhr_converter.sif python -u /track1/scripts/lateral_band_refit.py --input /track1/submissions/episodes_vpq2_in \
  --raw /track1/submissions/episodes_repro_depth --visibility /track1/outputs/mask_visibility_vp --out /track1/submissions/episodes_vpq3_in \
  --mode band --lam_lat 100
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh episodes_vpq3_in episodes_vpq3_in episodes_vpq3_afw
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved /track1/submissions/episodes_vpq3_afw \
  --base /track1/submissions/episodes_vpq3_in --out /track1/submissions/episodes_vpq3_sd4 --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh episodes_vpq3_sd4 episodes_vpq3_sd4 "${OUT}"
echo "PASS: candidate VPQ3 -> submissions/${OUT}"
