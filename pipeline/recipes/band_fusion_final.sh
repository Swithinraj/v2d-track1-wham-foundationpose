#!/bin/bash
# Candidate "<base> + E2": band fusion (scripts/band_fusion_refit.py, config/band_fusion_coef.json, strength 0.5) on a
# candidate's pre-contact human, then the identical contact stage (contact solve with ViTPose landmarks, arm-correction
# smoothing sigma 4, hand-only re-solve). See band_fusion.md.
#   BMQ3-K + E2:  band_fusion_final.sh episodes_bmq3k_in episodes_bm_depth episodes_cand_BMQ3KE
#   VPQ3 + E2:    band_fusion_final.sh episodes_vpq3_in episodes_repro_depth episodes_cand_VPQ3E
# Usage: pipeline/recipes/band_fusion_final.sh IN RAW OUTPUT   (episode directory names under submissions/)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
IN="$1"; RAW="$2"; OUT="$3"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx; W="${OUT}_e2"
for d in "${S}/${IN}" "${S}/${RAW}"; do require_file "${d}/episode_000029.npz" "build the base candidate first"; done
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP}"
rm -rf "${S:?}/${W}_in"
in_container mhr_converter.sif python -u /track1/scripts/band_fusion_refit.py --input "/track1/submissions/${IN}" \
  --raw "/track1/submissions/${RAW}" --coef /track1/config/band_fusion_coef.json --strength 0.5 --out "/track1/submissions/${W}_in"
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh "${W}_in" "${W}_in" "${W}_afw"
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved "/track1/submissions/${W}_afw" \
  --base "/track1/submissions/${W}_in" --out "/track1/submissions/${W}_sd4" --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh "${W}_sd4" "${W}_sd4" "${OUT}"
echo "PASS: band fusion candidate -> submissions/${OUT}"
