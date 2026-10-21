#!/bin/bash
# Candidate VPQ2: VPQ with temporal refinement of the human's camera depth and stronger object translation smoothing,
# both BEFORE VPQ's contact stage (see vpq.md, "Version VPQ2"). Input: episodes_vpq_ac, VPQ's human after hand fix and
# arm clearance (vpq_final.sh steps 1-3), pinned in vpq2_inputs.sha256.
#   1. MHR bodies of episodes_vpq_ac                                                   mhr_bodies.py
#   2. depth smoothing per episode: extra depth LAM = 300 * (s_z^2 / s_xy^2 - 1) from the raw human's camera-frame
#      noise, first scored frame pinned, body moved along the camera ray                depth_smooth_root.py
#   3. object translation smoothing, lambda 100, all episodes                            physics_objects.py
#   4. VPQ's contact stage: contact solve, arm-correction smoothing (sigma 4), hand-only re-solve
# Usage: pipeline/recipes/vpq2_final.sh [OUTPUT]   (default episodes_cand_VPQ2)
#   OBJ_PYTHON: interpreter with numpy, PIL, OpenCV and trimesh for physics_objects.py (default VENV_PYTHON).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
OUT="${1:-episodes_cand_VPQ2}"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx
for d in "${S}/episodes_vpq_ac" "${TRACK1_ROOT}/outputs/human_refine_vp/bodies_base" "${TRACK1_ROOT}/${KP}"; do
  require_file "${d}/episode_000029.npz" "pinned input, see pipeline/recipes/vpq.md"
done
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP}"

in_container mhr_converter.sif python -u /track1/scripts/mhr_bodies.py --episodes_dir /track1/submissions/episodes_vpq_ac \
  --out /track1/outputs/bodies_vpq_ac
rm -rf "${S:?}/episodes_vpq2_dz" "${S:?}/episodes_vpq2_in"
"${VENV_PYTHON}" scripts/depth_smooth_root.py --final "${S}/episodes_vpq_ac" --final_bodies outputs/bodies_vpq_ac \
  --raw outputs/human_refine_vp/bodies_base --out "${S}/episodes_vpq2_dz"
"${OBJ_PYTHON:-${VENV_PYTHON}}" scripts/physics_objects.py --input "${S}/episodes_vpq2_dz" --out "${S}/episodes_vpq2_in" \
  --floor outputs/physics/kappa_floor.json --no_rest --no_gravity --lam 100
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh episodes_vpq2_in episodes_vpq2_in episodes_vpq2_afw
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved /track1/submissions/episodes_vpq2_afw \
  --base /track1/submissions/episodes_vpq2_in --out /track1/submissions/episodes_vpq2_sd4 --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh episodes_vpq2_sd4 episodes_vpq2_sd4 "${OUT}"
echo "PASS: candidate VPQ2 -> submissions/${OUT}"
