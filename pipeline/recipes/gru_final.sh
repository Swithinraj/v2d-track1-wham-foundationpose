#!/bin/bash
# Candidate "<base> + GRU": the learned temporal refiner (scripts/gru_fusion_refit.py with config/gru_motion, trained by
# scripts/gru_train.py on FORM-HOI sequences processed by this pipeline) on a candidate's pre-contact human, then the
# identical contact stage of band_fusion_final.sh (contact solve with ViTPose landmarks, arm-correction smoothing
# sigma 4, hand-only re-solve). See gru.md.
#   BMQ3-K + GRU:  gru_final.sh episodes_bmq3k_in episodes_bm_depth mask_visibility_bm episodes_cand_BMQ3KG
# Usage: pipeline/recipes/gru_final.sh IN RAW VISIBILITY OUTPUT   (episode directory names under submissions/,
#        the mask-visibility directory under outputs/)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
IN="$1"; RAW="$2"; VIS="$3"; OUT="$4"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx; W="${OUT}_gru"
for d in "${S}/${IN}" "${S}/${RAW}"; do require_file "${d}/episode_000029.npz" "build the base candidate first"; done
require_file "${TRACK1_ROOT}/config/gru_motion/refiner_motion.pt"
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP}"
rm -rf "${S:?}/${W}_in"
in_container mhr_converter.sif python -u /track1/scripts/gru_fusion_refit.py --input "/track1/submissions/${IN}" \
  --raw "/track1/submissions/${RAW}" --visibility "/track1/outputs/${VIS}" --model /track1/config/gru_motion/refiner_motion.pt \
  --norm /track1/config/gru_motion/refiner_motion_norm.pt --out "/track1/submissions/${W}_in"
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh "${W}_in" "${W}_in" "${W}_afw"
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved "/track1/submissions/${W}_afw" \
  --base "/track1/submissions/${W}_in" --out "/track1/submissions/${W}_sd4" --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh "${W}_sd4" "${W}_sd4" "${OUT}"
echo "PASS: GRU candidate -> submissions/${OUT}"
