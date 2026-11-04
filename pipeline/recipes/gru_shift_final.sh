#!/bin/bash
# Candidate "<base> + GRU (depth off) + common-translation selector + PEN": on the pre-contact episodes of the depth-GRU
# candidate with the depth correction off (gru_depth_final.sh with EPS_SCALE=0, which leaves them as <its OUTPUT>_gd_in),
# a learned selector (scripts/corr_select3.py, frozen config/corr_select3_T/selector3.pt) decides per 6-s block whether to
# move the human AND the object by the same small vector along the human's camera ray (keep unless it predicts a gain for
# both and no ACC / PEN regression); the per-frame shifts (smoothed, tapered in from the anchor) are applied to the human's
# root translation and the object translation (scripts/apply_shift.py; hand-object geometry unchanged), then the contact
# stage with --w_pen 4e6. See gru.md, "Common-translation selector".
#   BMQ3-K: gru_shift_final.sh episodes_bmq3k_in episodes_bm_depth mask_visibility_bm BM_ROOT episodes_cand_BMQ3KGM_gd_in episodes_cand_BMQ3KGMT
# Usage: pipeline/recipes/gru_shift_final.sh IN RAW VISIBILITY SCALE_ROOT PRE_CONTACT OUTPUT
#   IN / RAW / VISIBILITY / SCALE_ROOT as for gru_depth_final.sh (the selector reads IN, the input of the motion head);
#   PRE_CONTACT: the pre-contact episodes that get shifted (BMQ3-K + GRU with the depth correction off).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
IN="$1"; RAW="$2"; VIS="$3"; SR="$(cd "$4" && pwd)"; PRE="$5"; OUT="$6"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx
W="${OUT}_sel"; SEL=outputs/selector_${OUT}
for d in "${S}/${IN}" "${S}/${RAW}" "${S}/${PRE}"; do require_file "${d}/episode_000029.npz" "build the base candidate first"; done
require_file "${SR}/outputs/sam3d_body/episode_000029/prepare.npz" "depth scale root"
require_file "${TRACK1_ROOT}/config/corr_select3_T/selector3.pt"
if [[ "${SR}" == "${TRACK1_ROOT}"* ]]; then
  CSR="/track1${SR#"${TRACK1_ROOT}"}"
else
  CSR="${SR}"; bound=""
  for b in ${TRACK1_EXTRA_BINDS:-}; do [[ "${SR}" == "${b}"* ]] && bound=y; done
  [[ -n "${bound}" ]] || { echo "SCALE_ROOT ${SR} is not inside TRACK1_ROOT or TRACK1_EXTRA_BINDS" >&2; exit 3; }
fi
rm -rf "${SEL:?}" "${S:?}/${W}_in" "${S:?}/${W}_afw" "${S:?}/${W}_sd4"; mkdir -p "${SEL}/cues"
# 1. GT-free depth cues of IN (the motion head's input), exactly as the depth-GRU deployment computes them
in_container mhr_converter.sif python -u /track1/scripts/gru_depth_refit.py --input "/track1/submissions/${IN}" \
  --raw "/track1/submissions/${RAW}" --visibility "/track1/outputs/${VIS}" --init /track1/config/gru_motion_v2 \
  --depth_model /track1/config/gru_depth/depth_best.pt --scale_root "${CSR}" --out "/track1/${SEL}/refit_scratch" \
  --save_cues "/track1/${SEL}/cues_raw" --cues_only
for f in "${SEL}"/cues_raw/episode_*.npz; do cp "${f}" "${SEL}/cues/chal_$(basename "${f}")"; done
# 2. GT-free human + object caches of IN (valid = scored frames, as in training)
in_container mhr_converter.sif python -u /track1/scripts/obj_cache_free.py --input "/track1/submissions/${IN}" \
  --raw "/track1/submissions/${RAW}" --root_dir /track1 --scale_root "${CSR}" --name chal --valid_scored \
  --scored /track1/outputs/hand_fix/episodes_repro_depth_refine_meshes/scored_frames.json --out "/track1/${SEL}/cache"
# 3. the frozen selector -> per-frame displacements (its printed metrics are meaningless here: no ground truth)
in_container mhr_converter.sif python -u /track1/scripts/corr_select3.py "/track1/${SEL}" --init /track1/config/gru_motion_v2 \
  --eval_only /track1/config/corr_select3_T/selector3.pt --test "chal=/track1/${SEL}/cache/human:/track1/${SEL}/cache/object:/track1/${SEL}/cues:chal" \
  --dump_shifts "/track1/${SEL}/shifts"
# 4. human (root translation) and object (translation) moved by the same per-frame vector
in_container mhr_converter.sif python -u /track1/scripts/apply_shift.py --input "/track1/submissions/${PRE}" \
  --shifts "/track1/${SEL}/shifts/chal" --prefix chal --out "/track1/submissions/${W}_in"
# 5. contact stage with the 20x penetration weight
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP} --w_pen 4e6"
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh "${W}_in" "${W}_in" "${W}_afw"
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved "/track1/submissions/${W}_afw" \
  --base "/track1/submissions/${W}_in" --out "/track1/submissions/${W}_sd4" --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh "${W}_sd4" "${W}_sd4" "${OUT}"
echo "PASS: GRU (depth off) + common-translation selector + PEN candidate -> submissions/${OUT}"
