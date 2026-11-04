#!/bin/bash
# Candidate BMQ-K: BMQ with the objects re-expressed in the body-mode run's depth scale (see bmq.md, "Version BMQ-K").
# BMQ re-derived the MoGe/SAM depth scale s from the body-mode raw output (median +0.5 %) but kept VPQ's objects, which
# the episode builder had expressed with the full-mode s (object_translation = t_co / s, object_scale = s_co / s). The
# human moved along the camera rays with s and the objects did not, so the first-frame registration (fitted on the human)
# moved the objects (CD-O +0.26 on Kaggle with byte-identical objects). Here the objects follow s, as the builder would:
#   1. objects x s_full / s_body (translation and scale, about the camera) on BMQ's pre-contact episodes   rescale_objects.py
#   2. BMQ/VPQ2's contact stage: contact solve, arm-correction smoothing (sigma 4), hand-only re-solve
# Needs bmq_final.sh steps 0-4 (submissions/episodes_bmq_in and BM_ROOT/outputs/sam3d_body).
# Usage: pipeline/recipes/bmqk_final.sh [OUTPUT]   (default episodes_cand_BMQK; BM_ROOT as in bmq_final.sh)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
OUT="${1:-episodes_cand_BMQK}"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx
BM="${BM_ROOT:-${TRACK1_ROOT}/work/bm_root}"
require_file "${S}/episodes_bmq_in/episode_000029.npz" "run pipeline/recipes/bmq_final.sh first"
require_file "${BM}/outputs/sam3d_body/episode_000029/prepare.npz" "run pipeline/recipes/bmq_final.sh first"
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP}"

rm -rf "${S:?}/episodes_bmqk_in"
"${VENV_PYTHON}" scripts/rescale_objects.py --input "${S}/episodes_bmq_in" --out "${S}/episodes_bmqk_in" \
  --s_old outputs/sam3d_body --s_new "${BM}/outputs/sam3d_body"
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh episodes_bmqk_in episodes_bmqk_in episodes_bmqk_afw
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved /track1/submissions/episodes_bmqk_afw \
  --base /track1/submissions/episodes_bmqk_in --out /track1/submissions/episodes_bmqk_sd4 --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh episodes_bmqk_sd4 episodes_bmqk_sd4 "${OUT}"
echo "PASS: candidate BMQ-K -> submissions/${OUT}"
