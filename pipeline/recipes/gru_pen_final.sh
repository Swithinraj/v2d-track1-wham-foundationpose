#!/bin/bash
# Candidate "<GRU candidate> + PEN": gru_final.sh's contact stage with a 10x penetration weight in both contact solves
# (--w_pen 2e6; default 2e5), on the GRU's pre-contact output. Hand points inside the object cost more than in the
# production stage, so the solver clears more penetrating frames with the same arm / hand variables; objects are
# unchanged. See gru.md, "Contact-stage penetration weight".
#   BMQ3-K + GRU + PEN:  gru_pen_final.sh episodes_cand_BMQ3KG_gru_in episodes_cand_BMQ3KGP
# Usage: pipeline/recipes/gru_pen_final.sh GRU_IN OUTPUT   (episode directory names under submissions/; GRU_IN is the
#        <OUTPUT>_gru_in directory written by gru_final.sh)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
IN="$1"; OUT="$2"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx; W="${OUT}_pen"
require_file "${S}/${IN}/episode_000029.npz" "run pipeline/recipes/gru_final.sh first"
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP} --w_pen 2e6"
rm -rf "${S:?}/${W}_in"; cp -r "${S}/${IN}" "${S}/${W}_in"
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh "${W}_in" "${W}_in" "${W}_afw"
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved "/track1/submissions/${W}_afw" \
  --base "/track1/submissions/${W}_in" --out "/track1/submissions/${W}_sd4" --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh "${W}_sd4" "${W}_sd4" "${OUT}"
echo "PASS: GRU + PEN candidate -> submissions/${OUT}"
