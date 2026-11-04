#!/bin/bash
# Candidate VPQ, final steps (GPU), from the inputs pinned in vpq_inputs.sha256 (see vpq.md):
#   1. human refinement driven by tracked ViTPose-H body landmarks, no MediaPipe   human_refine.sh, HUMAN_KP2D_DIR
#   2. that human with the VPQ objects                                             combine_human_objects.py
#   3. hand fix, arm clearance on the grip episodes, hand fix there again (as Afr2b) hand_fix.sh, arm_clear.py
#   4. contact solve with AFw's settings on the ViTPose landmarks                  contact_solve.sh
#   5. temporal low-pass of the arm part of that correction (sigma 4 frames)       smooth_arm_correction.py
#   6. hand-only re-solve (no scored joint moves)                                  contact_solve.sh --no_arm
# Usage: pipeline/recipes/vpq_final.sh [OUTPUT]   (episode directory under submissions/, default episodes_cand_VPQ)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
OUT="${1:-episodes_cand_VPQ}"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx
for d in "${S}/episodes_repro_depth" "${S}/episodes_repro" "${S}/episodes_objects_vpq" "${TRACK1_ROOT}/${KP}"; do
  require_file "${d}/episode_000029.npz" "pinned input, see pipeline/recipes/vpq.md"
done
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP}"

rm -rf "${S:?}/episodes_rdv"; cp -a "${S}/episodes_repro_depth" "${S}/episodes_rdv"
REFINE_LAM=300 TRACK1_ALIGNMENT_DIR=outputs/alignment_sam HUMAN_REFINE_WORK=human_refine_vp HUMAN_KP2D_DIR="${TRACK1_ROOT}/${KP}" \
  bash pipeline/stages/human_refine.sh episodes_rdv episodes_repro_depth episodes_repro episodes_rdv_refine_vp
rm -rf "${S:?}/episodes_vpq_base"
"${VENV_PYTHON}" scripts/combine_human_objects.py "${S}/episodes_rdv_refine_vp" "${S}/episodes_objects_vpq" "${S}/episodes_vpq_base"
bash pipeline/stages/hand_fix.sh episodes_vpq_base episodes_vpq_hf
# arm clearance on the grip episodes, then the hand fix again there (as for Afr2b); arm_clear.py and hand_fix.sh write
# only their episodes, so their outputs start as copies of their inputs
fresh_copy() { rm -rf "${S:?}/${2:?}"; cp -a "${S}/${1:?}" "${S}/${2}"; }
ARM_EPS="9 27 28 29 22 7 4 3 17 15"
"${VENV_PYTHON}" scripts/budget_object_meshes.py --episodes "${S}/episodes_vpq_hf" --out outputs/armclear_vpq_meshes
fresh_copy episodes_vpq_hf episodes_vpq_arm
in_container mhr_converter.sif python -u /track1/scripts/arm_clear.py --input /track1/submissions/episodes_vpq_hf \
  --out /track1/submissions/episodes_vpq_arm --meshes /track1/outputs/armclear_vpq_meshes --episodes ${ARM_EPS}
fresh_copy episodes_vpq_arm episodes_vpq_ac; fresh_copy episodes_vpq_arm episodes_vpq_ac_pass1
HAND_FIX_EPISODES="${ARM_EPS}" bash pipeline/stages/hand_fix.sh episodes_vpq_arm episodes_vpq_ac
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh episodes_vpq_ac episodes_vpq_ac episodes_vpq_afw
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved /track1/submissions/episodes_vpq_afw \
  --base /track1/submissions/episodes_vpq_ac --out /track1/submissions/episodes_vpq_sd4 --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh episodes_vpq_sd4 episodes_vpq_sd4 "${OUT}"
echo "PASS: candidate VPQ -> submissions/${OUT}"
