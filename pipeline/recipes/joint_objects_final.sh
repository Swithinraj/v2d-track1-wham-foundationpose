#!/bin/bash
# Candidate "<base> + joint objects": the jointly trained human-object refiner (scripts/joint_train.py, trained on FORM-HOI
# sequences processed by this pipeline; models in config/joint_objects) corrects only the OBJECTS of a candidate's
# pre-contact episodes (the human is kept), then the identical contact stage of band_fusion_final.sh. See joint_objects.md.
#   BMQ3-K + GRU + joint objects:
#     joint_objects_final.sh episodes_cand_BMQ3KG_gru_in episodes_bm_depth mask_visibility_bm config/joint_objects/v6.pt episodes_cand_BMQ3KGN
# Usage: pipeline/recipes/joint_objects_final.sh IN RAW VISIBILITY MODEL OUTPUT
#        (IN must have been through the contact stage once, which builds outputs/contact_sdf/IN; gru_final.sh does)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
IN="$1"; RAW="$2"; VIS="$3"; MODEL="$4"; OUT="$5"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx; W="${OUT}_jo"
C="work/joint_objects/${OUT}"
for d in "${S}/${IN}" "${S}/${RAW}"; do require_file "${d}/episode_000029.npz" "build the base candidate first"; done
require_file "${TRACK1_ROOT}/outputs/contact_sdf/${IN}/episode_000029.npz" "run the base candidate's contact stage first"
require_file "${TRACK1_ROOT}/${MODEL}"
mkdir -p "${C}"; rm -rf "${S:?}/${W}_in"
# image evidence for visual models: our hand points / object track (no ground truth) -> hand and object crops
in_container mhr_converter.sif python -u /track1/scripts/joint_deploy_prep.py "/track1/submissions/${IN}" "/track1/${C}/prep" track1
in_container v2d_sam3d_body.sif python -u /track1/scripts/joint_cache_visual.py "/track1/${C}/prep" "/track1/${C}/prep" \
  "/track1/${C}/visual" /track1 --workers 8
in_container mhr_converter.sif python -u /track1/scripts/joint_deploy.py --input "/track1/submissions/${IN}" \
  --raw "/track1/submissions/${RAW}" --visibility "/track1/outputs/${VIS}" --sdf "/track1/outputs/contact_sdf/${IN}" \
  --visual_cache "/track1/${C}/visual" --model "/track1/${MODEL}" --out "/track1/submissions/${W}_in" \
  --obj_smooth 8 --opt_obj_only --no_hand_head --objects_only
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP}"
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh "${W}_in" "${W}_in" "${W}_afw"
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved "/track1/submissions/${W}_afw" \
  --base "/track1/submissions/${W}_in" --out "/track1/submissions/${W}_sd4" --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh "${W}_sd4" "${W}_sd4" "${OUT}"
echo "PASS: joint-objects candidate -> submissions/${OUT}"
