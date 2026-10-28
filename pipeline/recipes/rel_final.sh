#!/bin/bash
# Candidate "<base> + depth-reliability module" (see rel.md): before the contact stage, a small learned module predicts
# per frame a bounded correction of the OBJECT's distance along its camera ray (and of the human's, which it leaves at
# zero in practice) from GT-free evidence (MoGe-2 depth at our human / object vs our reconstruction, apparent sizes,
# ViTPose residuals, occlusion, motion, hand-object distance). The correction is smoothed again (sigma 20 frames) and
# applied in the motion-head step of gru_depth_final.sh (depth head off), then the same contact stage (w_pen 4e6).
#   BMQ3KGM + module:  rel_final.sh episodes_bmq3k_in episodes_cand_BMQ3KGMR   (after bmq3k_final.sh / trend_final.sh)
# Usage: pipeline/recipes/rel_final.sh IN OUTPUT
#   IN: the motion head's input (BMQ3-K pre-contact human + objects); needs submissions/episodes_bm_depth and
#   outputs/mask_visibility_bm from the same build. BM_ROOT (default ${TRACK1_ROOT}/work/bm_root): the body-mode root whose
#   outputs/sam3d_body holds IN's depth scale. REL_SIGMA (default 20). OBJ_PYTHON: numpy, PIL, OpenCV (as bmq_final.sh).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "${TRACK1_ROOT}"
IN="$1"; OUT="$2"; S="${TRACK1_ROOT}/submissions"; KP=outputs/kp2d_vpx; W="work/rel/${OUT}"; P=rel; PY="${OBJ_PYTHON:-${VENV_PYTHON}}"
BM="$(cd "${BM_ROOT:-${TRACK1_ROOT}/work/bm_root}" && pwd)"
for d in "${S}/${IN}" "${S}/episodes_bm_depth" "${TRACK1_ROOT}/outputs/mask_visibility_bm"; do require_file "${d}/episode_000029.npz" "build the base candidate first"; done
require_file "${BM}/outputs/sam3d_body/episode_000029/prepare.npz" "body-mode root (bmq_final.sh)"
require_file "${TRACK1_ROOT}/config/rel_module/rel_best.pt"
if [[ "${BM}" == "${TRACK1_ROOT}"* ]]; then CSR="/track1${BM#"${TRACK1_ROOT}"}"; else
  CSR="${BM}"; bound=""; for b in ${TRACK1_EXTRA_BINDS:-}; do [[ "${BM}" == "${b}"* ]] && bound=y; done
  [[ -n "${bound}" ]] || { echo "BM_ROOT ${BM} is not inside TRACK1_ROOT or TRACK1_EXTRA_BINDS" >&2; exit 3; }
fi
rm -rf "${W:?}"; mkdir -p "${W}"
# 1. GT-free evidence: our human / object / ViTPose fields, MoGe background lock, depth cues
in_container mhr_converter.sif python -u /track1/scripts/rel_deploy_prep.py "/track1/submissions/${IN}" "/track1/${W}" "${P}"
"${PY}" scripts/moge_lock.py --root . --joints "${W}/human" --prefix "${P}" --out "${W}/lock" --scale_root "${BM}"
"${PY}" scripts/depth_cues.py --joints "${W}/human" --prefix "${P}" --root . --scale_root "${BM}" --out "${W}/cues"
"${PY}" scripts/rel_cache.py --root . --mode bm --prefix "${P}" --out "${W}/rel" --no_vda --deploy \
  --dirs "human=${W}/human,object=${W}/object,visual=${W}/visual,cues=${W}/cues,lock=${W}/lock"
# 2. per-frame corrections, smoothed again
in_container mhr_converter.sif python -u /track1/scripts/rel_train.py --cache "/track1/${W}/rel" --val "${P}" \
  --eval_only /track1/config/rel_module/rel_best.pt --out "/track1/${W}" --dump "/track1/${W}/dump" --deploy
"${PY}" scripts/rel_smooth.py "${W}/dump" --sigma "${REL_SIGMA:-20}"
# 3. motion head with the module's corrections (depth head off) + contact stage w_pen 4e6 (as gru_depth_final.sh)
G="${OUT}_gd"; rm -rf "${S:?}/${G}_in"
in_container mhr_converter.sif python -u /track1/scripts/gru_rel_refit.py --input "/track1/submissions/${IN}" \
  --raw /track1/submissions/episodes_bm_depth --visibility /track1/outputs/mask_visibility_bm --init /track1/config/gru_motion_v2 \
  --depth_model /track1/config/gru_depth/depth_best.pt --scale_root "${CSR}" --eps_scale 1 \
  --rel_dir "/track1/${W}/dump" --rel_prefix "${P}" --out "/track1/submissions/${G}_in"
AFW_ARGS="--w_prior_arm 2000 --w_img 20 --w_cap 1000 --nsub 0 --w_2d_hand 0 --pen_frames_only --freeze_anchor \
--w_con 200000 --w_rig 20000 --w_hdisp 0.3 --kp2d_dir ${KP} --w_pen 4e6"
CONTACT_ARGS="${AFW_ARGS}" bash pipeline/stages/contact_solve.sh "${G}_in" "${G}_in" "${G}_afw"
in_container mhr_converter.sif python -u /track1/scripts/smooth_arm_correction.py --solved "/track1/submissions/${G}_afw" \
  --base "/track1/submissions/${G}_in" --out "/track1/submissions/${G}_sd4" --sigma 4
CONTACT_ARGS="${AFW_ARGS} --no_arm" bash pipeline/stages/contact_solve.sh "${G}_sd4" "${G}_sd4" "${OUT}"
echo "PASS: base + depth-reliability module -> submissions/${OUT}"
