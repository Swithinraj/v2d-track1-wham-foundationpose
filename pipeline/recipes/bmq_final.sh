#!/bin/bash
# Candidate BMQ: VPQ2 with the human re-inferred by SAM 3D Body in BODY mode with the actor's SAM 2 mask as mask prompt
# (the organizers' per-view estimate_mhr_params path) instead of FULL mode with a box prompt (see bmq.md). Everything
# after the raw inference is VPQ/VPQ2 code with VPQ2's settings; objects are VPQ's (episodes_objects_vpq).
#   0. BM_ROOT: a work root with its own outputs/{sam3d_body,alignment_sam,hands,keypoints2d_sam,human_refine_bm} and
#      submissions/; every other entry links to TRACK1_ROOT (inputs, weights, person masks, camera, ViTPose landmarks).
#   1. raw inference, body mode + mask prompt, then identity / refit / hands      pipeline/stages/sam3d_body.sh
#   2. episodes, depth trend, human refinement (lambda 300, ViTPose landmarks)     build_track1_episodes_sam.py,
#                                                                                  sam3d_body_depth_trend.py, human_refine.sh
#   3. VPQ's objects, hand fix, arm clearance on the grip episodes, hand fix       (vpq_final.sh steps 2-3)
#   4. VPQ2's steps: depth smoothing (raw = the BM refinement's base bodies), object smoothing lambda 100,
#      contact solve, arm-correction smoothing (sigma 4), hand-only re-solve       (vpq2_final.sh)
# The scored file's raw inference ran in v2d_sam3d_body.sif (the default below); full mode + box in that image vs the
# reference image changes the scored joints by 0.09 cm (noise-like), the body-mode change is 1.8-2.0 cm.
# Usage: pipeline/recipes/bmq_final.sh [OUTPUT]   (default episodes_cand_BMQ)
#   BM_ROOT (default ${TRACK1_ROOT}/work/bm_root), SAM3D_BODY_IMAGE (default v2d_sam3d_body.sif),
#   OBJ_PYTHON: interpreter with numpy, PIL, OpenCV and trimesh for physics_objects.py (default VENV_PYTHON).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
OUT="${1:-episodes_cand_BMQ}"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx
BM="${BM_ROOT:-${TRACK1_ROOT}/work/bm_root}"
for d in "${S}/episodes_objects_vpq" "${TRACK1_ROOT}/${KP}"; do
  require_file "${d}/episode_000029.npz" "pinned input, see pipeline/recipes/vpq.md"
done
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP}"
OWN="sam3d_body alignment_sam hands keypoints2d_sam human_refine_bm"

# 0. work root
mkdir -p "${BM}/outputs" "${BM}/submissions"
for e in "${TRACK1_ROOT}"/* "${TRACK1_ROOT}"/.venvs; do
  n=$(basename "${e}"); [[ "${n}" == outputs || "${n}" == submissions || "${n}" == work ]] || ln -sfn "${e}" "${BM}/${n}"
done
for e in "${TRACK1_ROOT}"/outputs/*; do
  n=$(basename "${e}"); [[ " ${OWN} " == *" ${n} "* ]] || ln -sfn "${e}" "${BM}/outputs/${n}"
done
# 1-2. BM human in the work root (its links resolve inside the containers through a read-only bind of TRACK1_ROOT)
export TRACK1_EXTRA_BINDS="${TRACK1_EXTRA_BINDS:-} ${TRACK1_ROOT}"
for ep in $(seq 0 29); do
  TRACK1_ROOT="${BM}" SAM3D_BODY_IMAGE="${SAM3D_BODY_IMAGE:-v2d_sam3d_body.sif}" \
    SAM3D_BODY_ARGS="--inference_type body --mask_prompt" bash pipeline/stages/sam3d_body.sh "${ep}"
done
(
  export TRACK1_ROOT="${BM}"; cd "${BM}"
  in_container v2d_cari4d.sif TRACK1_ROOT=/track1 python /track1/scripts/build_track1_episodes_sam.py --out /track1/submissions/episodes_bm
  in_container v2d_cari4d.sif TRACK1_ROOT=/track1 python /track1/scripts/sam3d_body_depth_trend.py --input /track1/submissions/episodes_bm \
    --out /track1/submissions/episodes_bm_depth --alpha 0.75
  rm -rf submissions/episodes_bmd; cp -a submissions/episodes_bm_depth submissions/episodes_bmd
  REFINE_LAM=300 TRACK1_ALIGNMENT_DIR=outputs/alignment_sam HUMAN_REFINE_WORK=human_refine_bm HUMAN_KP2D_DIR="${BM}/${KP}" \
    bash pipeline/stages/human_refine.sh episodes_bmd episodes_bm_depth episodes_bm episodes_bm_refine
)
rm -rf "${S:?}/episodes_bm_refine"; cp -a "${BM}/submissions/episodes_bm_refine" "${S}/episodes_bm_refine"
mkdir -p outputs/human_refine_bm; rm -rf outputs/human_refine_bm/bodies_base
cp -a "${BM}/outputs/human_refine_bm/bodies_base" outputs/human_refine_bm/bodies_base
# 3. objects, hand fix, arm clearance, hand fix (as vpq_final.sh)
fresh_copy() { rm -rf "${S:?}/${2:?}"; cp -a "${S}/${1:?}" "${S}/${2}"; }
rm -rf "${S:?}/episodes_bmq_base"
"${VENV_PYTHON}" scripts/combine_human_objects.py "${S}/episodes_bm_refine" "${S}/episodes_objects_vpq" "${S}/episodes_bmq_base"
bash pipeline/stages/hand_fix.sh episodes_bmq_base episodes_bmq_hf
ARM_EPS="9 27 28 29 22 7 4 3 17 15"
"${VENV_PYTHON}" scripts/budget_object_meshes.py --episodes "${S}/episodes_bmq_hf" --out outputs/armclear_bmq_meshes
fresh_copy episodes_bmq_hf episodes_bmq_arm
in_container mhr_converter.sif python -u /track1/scripts/arm_clear.py --input /track1/submissions/episodes_bmq_hf \
  --out /track1/submissions/episodes_bmq_arm --meshes /track1/outputs/armclear_bmq_meshes --episodes ${ARM_EPS}
fresh_copy episodes_bmq_arm episodes_bmq_ac; fresh_copy episodes_bmq_arm episodes_bmq_ac_pass1
HAND_FIX_EPISODES="${ARM_EPS}" bash pipeline/stages/hand_fix.sh episodes_bmq_arm episodes_bmq_ac
# 4. VPQ2's steps
in_container mhr_converter.sif python -u /track1/scripts/mhr_bodies.py --episodes_dir /track1/submissions/episodes_bmq_ac \
  --out /track1/outputs/bodies_bmq_ac
rm -rf "${S:?}/episodes_bmq_dz" "${S:?}/episodes_bmq_in"
"${VENV_PYTHON}" scripts/depth_smooth_root.py --final "${S}/episodes_bmq_ac" --final_bodies outputs/bodies_bmq_ac \
  --raw outputs/human_refine_bm/bodies_base --out "${S}/episodes_bmq_dz"
"${OBJ_PYTHON:-${VENV_PYTHON}}" scripts/physics_objects.py --input "${S}/episodes_bmq_dz" --out "${S}/episodes_bmq_in" \
  --floor outputs/physics/kappa_floor.json --no_rest --no_gravity --lam 100
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh episodes_bmq_in episodes_bmq_in episodes_bmq_afw
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved /track1/submissions/episodes_bmq_afw \
  --base /track1/submissions/episodes_bmq_in --out /track1/submissions/episodes_bmq_sd4 --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh episodes_bmq_sd4 episodes_bmq_sd4 "${OUT}"
echo "PASS: candidate BMQ -> submissions/${OUT}"
