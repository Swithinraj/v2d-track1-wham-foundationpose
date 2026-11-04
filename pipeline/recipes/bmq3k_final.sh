#!/bin/bash
# Candidate BMQ3-K: BMQ-K + VPQ3's restored 1-2 Hz image-plane motion (lateral band refit), i.e. VPQ3's recipe on the
# body-mode human with consistently scaled objects (see bmq.md, "Version BMQ3-K"). The band's raw reference is the
# body-mode human before refinement (BM_ROOT episodes_bm_depth, the counterpart of VPQ3's episodes_repro_depth) and the
# visibility comes from the body-mode refinement's base bodies.
#   1. mask visibility of the body-mode base bodies                                   mask_visibility.py
#   2. lateral band refit (lambda 100, evidence-weighted, depth kept, first scored frame pinned)  lateral_band_refit.py
#   3. BMQ/VPQ2's contact stage: contact solve, arm-correction smoothing (sigma 4), hand-only re-solve
# Needs bmq_final.sh (BM_ROOT, outputs/human_refine_bm/bodies_base) and bmqk_final.sh step 1 (submissions/episodes_bmqk_in).
# Usage: pipeline/recipes/bmq3k_final.sh [OUTPUT]   (default episodes_cand_BMQ3K; BM_ROOT as in bmq_final.sh)
#   OBJ_PYTHON: interpreter with numpy and OpenCV for mask_visibility.py (default VENV_PYTHON).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
OUT="${1:-episodes_cand_BMQ3K}"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx
BM="${BM_ROOT:-${TRACK1_ROOT}/work/bm_root}"
for d in "${S}/episodes_bmqk_in" "${BM}/submissions/episodes_bm_depth" "${TRACK1_ROOT}/outputs/human_refine_bm/bodies_base"; do
  require_file "${d}/episode_000029.npz" "run pipeline/recipes/bmq_final.sh and bmqk_final.sh first"
done
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP}"

rm -rf "${S:?}/episodes_bm_depth" "${S:?}/episodes_bmq3k_in"; cp -a "${BM}/submissions/episodes_bm_depth" "${S}/episodes_bm_depth"
"${OBJ_PYTHON:-${VENV_PYTHON}}" scripts/mask_visibility.py --bodies outputs/human_refine_bm/bodies_base --out outputs/mask_visibility_bm
in_container mhr_converter.sif python -u /track1/scripts/lateral_band_refit.py --input /track1/submissions/episodes_bmqk_in \
  --raw /track1/submissions/episodes_bm_depth --visibility /track1/outputs/mask_visibility_bm --out /track1/submissions/episodes_bmq3k_in \
  --mode band --lam_lat 100
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh episodes_bmq3k_in episodes_bmq3k_in episodes_bmq3k_afw
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved /track1/submissions/episodes_bmq3k_afw \
  --base /track1/submissions/episodes_bmq3k_in --out /track1/submissions/episodes_bmq3k_sd4 --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh episodes_bmq3k_sd4 episodes_bmq3k_sd4 "${OUT}"
echo "PASS: candidate BMQ3-K -> submissions/${OUT}"
