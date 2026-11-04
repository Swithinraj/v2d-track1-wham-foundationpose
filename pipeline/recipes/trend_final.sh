#!/bin/bash
# Candidate BMQ3KGM with a configurable human depth trend (see trend.md). On FORM-HOI the inherited MoGe-2 trend
# (alpha 0.75) is a net loss: the per-frame depth ratio it follows tracks the person's true distance change poorly.
# Same stages and settings as BMQ3KGM (bmq_final.sh steps 2-4 without its contact stage, bmqk_final.sh step 1,
# bmq3k_final.sh steps 1-2, EPS_SCALE=0 gru_depth_final.sh); only sam3d_body_depth_trend.py changes, in both modes:
#   TREND_ALPHA       trend strength (BMQ3KGM 0.75; BMQ3KGMa0 0; BMQ3KGMh 0.25)
#   TREND_PREP_FULL   optional dirs <episode>/prepare.npz whose qn series the trend reads instead of MoGe-2's, full and
#   TREND_PREP_BM     body mode (BMQ3KGMh: the identity step re-run on Video Depth Anything maps, stages/vda_prep.sh);
#                     inside TRACK1_ROOT or TRACK1_EXTRA_BINDS
# The full-mode human enters the objects only through the gravity kappa (ipf30.md): kappa is recomputed on the
# full-mode human with the new trend and on the alpha-0.75 one, and VPQ's objects are re-scaled about the camera by
# the ratio (scripts/kappa_rescale_objects.py) -> episodes_objects_tr.
# It rewrites BMQ's intermediate directories (episodes_bm*, episodes_bmq*, outputs/human_refine_bm, ...): run it in a
# copy of the root when BMQ3KGM's intermediates are still needed.
# Usage: pipeline/recipes/trend_final.sh [OUTPUT]   (after run_all.sh and bmq_final.sh; default episodes_cand_BMQ3KGMtr)
#   BM_ROOT (default ${TRACK1_ROOT}/work/bm_root), OBJ_PYTHON as in bmq_final.sh
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
OUT="${1:-episodes_cand_BMQ3KGMtr}"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx
BM="${BM_ROOT:-${TRACK1_ROOT}/work/bm_root}"; ALPHA="${TREND_ALPHA:?set TREND_ALPHA}"
for d in "${S}/episodes_repro" "${S}/episodes_repro_depth" "${S}/episodes_objects_vpq" "${TRACK1_ROOT}/${KP}"; do
  require_file "${d}/episode_000029.npz" "run run_all.sh first (pinned inputs: pipeline/recipes/vpq.md)"
done
require_file "${BM}/outputs/sam3d_body/episode_000029/prepare.npz" "run pipeline/recipes/bmq_final.sh first"
grep -q '"alpha": 0.75' "${S}/episodes_repro_depth/sam_depth_trend_report.json" || { echo "episodes_repro_depth is not the alpha-0.75 trend" >&2; exit 3; }
cpath() {  # a host directory's path inside the containers
  local d; d="$(cd "$1" && pwd)"
  if [[ "${d}" == "${TRACK1_ROOT}"* ]]; then echo "/track1${d#"${TRACK1_ROOT}"}"; return; fi
  for b in ${TRACK1_EXTRA_BINDS:-}; do [[ "${d}" == "${b}"* ]] && { echo "${d}"; return; }; done
  echo "${d} is not inside TRACK1_ROOT or TRACK1_EXTRA_BINDS" >&2; exit 3
}
PF=(); PB=()
[[ -n "${TREND_PREP_FULL:-}" ]] && PF=(--prep_dir "$(cpath "${TREND_PREP_FULL}")")
[[ -n "${TREND_PREP_BM:-}" ]] && PB=(--prep_dir "$(cpath "${TREND_PREP_BM}")")
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP}"
fresh_copy() { rm -rf "${S:?}/${2:?}"; cp -a "${S}/${1:?}" "${S}/${2}"; }

# 1. full-mode trend -> gravity kappa -> objects re-scaled
rm -rf "${S:?}/episodes_repro_depth_tr" "${S:?}/episodes_kg_tr" "${S:?}/episodes_kg_a075"
in_container v2d_cari4d.sif python /track1/scripts/sam3d_body_depth_trend.py --input /track1/submissions/episodes_repro \
  --out /track1/submissions/episodes_repro_depth_tr --alpha "${ALPHA}" "${PF[@]}"
for a in tr:episodes_repro_depth_tr a075:episodes_repro_depth; do
  in_container mhr_converter.sif python -u /track1/scripts/kappa_gravity.py --input "/track1/submissions/${a#*:}" \
    --out "/track1/submissions/episodes_kg_${a%%:*}" --floor /track1/outputs/physics/kappa_floor.json
done
"${VENV_PYTHON}" scripts/kappa_rescale_objects.py --objects "${S}/episodes_objects_vpq" --kappa_new "${S}/episodes_kg_tr" \
  --kappa_old "${S}/episodes_kg_a075" --out "${S}/episodes_objects_tr"
# 2. body-mode episodes, trend, human refinement (bmq_final.sh step 2)
export TRACK1_EXTRA_BINDS="${TRACK1_EXTRA_BINDS:-} ${TRACK1_ROOT}"
(
  export TRACK1_ROOT="${BM}"; cd "${BM}"
  in_container v2d_cari4d.sif TRACK1_ROOT=/track1 python /track1/scripts/build_track1_episodes_sam.py --out /track1/submissions/episodes_bm
  in_container v2d_cari4d.sif TRACK1_ROOT=/track1 python /track1/scripts/sam3d_body_depth_trend.py --input /track1/submissions/episodes_bm \
    --out /track1/submissions/episodes_bm_depth --alpha "${ALPHA}" "${PB[@]}"
  rm -rf submissions/episodes_bmd; cp -a submissions/episodes_bm_depth submissions/episodes_bmd
  REFINE_LAM=300 TRACK1_ALIGNMENT_DIR=outputs/alignment_sam HUMAN_REFINE_WORK=human_refine_bm HUMAN_KP2D_DIR="${BM}/${KP}" \
    bash pipeline/stages/human_refine.sh episodes_bmd episodes_bm_depth episodes_bm episodes_bm_refine
)
rm -rf "${S:?}/episodes_bm_refine"; cp -a "${BM}/submissions/episodes_bm_refine" "${S}/episodes_bm_refine"
mkdir -p outputs/human_refine_bm; rm -rf outputs/human_refine_bm/bodies_base
cp -a "${BM}/outputs/human_refine_bm/bodies_base" outputs/human_refine_bm/bodies_base
# 3. objects, hand fix, arm clearance, hand fix (bmq_final.sh step 3)
rm -rf "${S:?}/episodes_bmq_base"
"${VENV_PYTHON}" scripts/combine_human_objects.py "${S}/episodes_bm_refine" "${S}/episodes_objects_tr" "${S}/episodes_bmq_base"
bash pipeline/stages/hand_fix.sh episodes_bmq_base episodes_bmq_hf
ARM_EPS="9 27 28 29 22 7 4 3 17 15"
"${VENV_PYTHON}" scripts/budget_object_meshes.py --episodes "${S}/episodes_bmq_hf" --out outputs/armclear_bmq_meshes
fresh_copy episodes_bmq_hf episodes_bmq_arm
in_container mhr_converter.sif python -u /track1/scripts/arm_clear.py --input /track1/submissions/episodes_bmq_hf \
  --out /track1/submissions/episodes_bmq_arm --meshes /track1/outputs/armclear_bmq_meshes --episodes ${ARM_EPS}
fresh_copy episodes_bmq_arm episodes_bmq_ac; fresh_copy episodes_bmq_arm episodes_bmq_ac_pass1
HAND_FIX_EPISODES="${ARM_EPS}" bash pipeline/stages/hand_fix.sh episodes_bmq_arm episodes_bmq_ac
# 4. depth smoothing + object smoothing (bmq_final.sh step 4 without its contact stage)
in_container mhr_converter.sif python -u /track1/scripts/mhr_bodies.py --episodes_dir /track1/submissions/episodes_bmq_ac \
  --out /track1/outputs/bodies_bmq_ac
rm -rf "${S:?}/episodes_bmq_dz" "${S:?}/episodes_bmq_in"
"${VENV_PYTHON}" scripts/depth_smooth_root.py --final "${S}/episodes_bmq_ac" --final_bodies outputs/bodies_bmq_ac \
  --raw outputs/human_refine_bm/bodies_base --out "${S}/episodes_bmq_dz"
"${OBJ_PYTHON:-${VENV_PYTHON}}" scripts/physics_objects.py --input "${S}/episodes_bmq_dz" --out "${S}/episodes_bmq_in" \
  --floor outputs/physics/kappa_floor.json --no_rest --no_gravity --lam 100
# 5. objects in the body-mode depth scale (bmqk_final.sh step 1)
rm -rf "${S:?}/episodes_bmqk_in"
"${VENV_PYTHON}" scripts/rescale_objects.py --input "${S}/episodes_bmq_in" --out "${S}/episodes_bmqk_in" \
  --s_old outputs/sam3d_body --s_new "${BM}/outputs/sam3d_body"
# 6. mask visibility + lateral band (bmq3k_final.sh steps 1-2)
"${OBJ_PYTHON:-${VENV_PYTHON}}" scripts/mask_visibility.py --bodies outputs/human_refine_bm/bodies_base --out outputs/mask_visibility_bm
rm -rf "${S:?}/episodes_bm_depth" "${S:?}/episodes_bmq3k_in"; cp -a "${BM}/submissions/episodes_bm_depth" "${S}/episodes_bm_depth"
in_container mhr_converter.sif python -u /track1/scripts/lateral_band_refit.py --input /track1/submissions/episodes_bmqk_in \
  --raw /track1/submissions/episodes_bm_depth --visibility /track1/outputs/mask_visibility_bm --out /track1/submissions/episodes_bmq3k_in \
  --mode band --lam_lat 100
# 7. motion head (depth correction off) + contact stage w_pen 4e6
EPS_SCALE=0 bash pipeline/recipes/gru_depth_final.sh episodes_bmq3k_in episodes_bm_depth mask_visibility_bm "${BM}" "${OUT}"
echo "PASS: candidate BMQ3KGM, depth trend alpha ${ALPHA}${TREND_PREP_BM:+ (qn: ${TREND_PREP_FULL:-MoGe-2} / ${TREND_PREP_BM})} -> submissions/${OUT}"
