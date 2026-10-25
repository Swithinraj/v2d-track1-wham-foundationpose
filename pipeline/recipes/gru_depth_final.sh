#!/bin/bash
# Candidate "<base> + GRU v2 + learned depth fusion + PEN": the temporal refiner v2 fine-tuned with GT-free per-frame
# depth cues (scripts/gru_depth_train.py; MoGe depth at the torso, apparent size vs ViTPose, foot / floor contact) and a
# depth head that moves the whole body along the camera ray (body size unchanged), at 0.75 of its predicted strength
# (shrinkage chosen on a validation group), on the pre-contact human; then the contact stage with a 20x penetration
# weight (--w_pen 4e6). Objects unchanged. See gru.md, "Learned depth fusion".
#   BMQ3-K + depth-GRU + PEN:  gru_depth_final.sh episodes_bmq3k_in episodes_bm_depth mask_visibility_bm BM_ROOT episodes_cand_BMQ3KGD
# Usage: pipeline/recipes/gru_depth_final.sh IN RAW VISIBILITY SCALE_ROOT OUTPUT
#   SCALE_ROOT: the root whose outputs/sam3d_body/<episode>/prepare.npz holds the depth scale IN's human is expressed in
#   (for BMQ3-K the body-mode run, BM_ROOT of bmq_final.sh); it must lie inside TRACK1_ROOT or TRACK1_EXTRA_BINDS.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
IN="$1"; RAW="$2"; VIS="$3"; SR="$(cd "$4" && pwd)"; OUT="$5"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx; W="${OUT}_gd"
for d in "${S}/${IN}" "${S}/${RAW}"; do require_file "${d}/episode_000029.npz" "build the base candidate first"; done
require_file "${SR}/outputs/sam3d_body/episode_000029/prepare.npz" "depth scale root"
require_file "${TRACK1_ROOT}/config/gru_depth/depth_best.pt"
if [[ "${SR}" == "${TRACK1_ROOT}"* ]]; then
  CSR="/track1${SR#"${TRACK1_ROOT}"}"                                  # inside the repository: its container path
else
  CSR="${SR}"; bound=""
  for b in ${TRACK1_EXTRA_BINDS:-}; do [[ "${SR}" == "${b}"* ]] && bound=y; done
  [[ -n "${bound}" ]] || { echo "SCALE_ROOT ${SR} is not inside TRACK1_ROOT or TRACK1_EXTRA_BINDS" >&2; exit 3; }
fi
rm -rf "${S:?}/${W}_in"
in_container mhr_converter.sif python -u /track1/scripts/gru_depth_refit.py --input "/track1/submissions/${IN}" \
  --raw "/track1/submissions/${RAW}" --visibility "/track1/outputs/${VIS}" --init /track1/config/gru_motion_v2 \
  --depth_model /track1/config/gru_depth/depth_best.pt --scale_root "${CSR}" --eps_scale 0.75 --out "/track1/submissions/${W}_in"
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP} --w_pen 4e6"
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh "${W}_in" "${W}_in" "${W}_afw"
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved "/track1/submissions/${W}_afw" \
  --base "/track1/submissions/${W}_in" --out "/track1/submissions/${W}_sd4" --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh "${W}_sd4" "${W}_sd4" "${OUT}"
echo "PASS: GRU v2 + depth fusion + PEN candidate -> submissions/${OUT}"
