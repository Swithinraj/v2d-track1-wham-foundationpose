#!/bin/bash
# Assemble all 30 episodes and pack the Track 1 submission with the official kit.
#   scripts/build_track1_episodes.py     -> submissions/episodes_wham_fp/ (episode NPZ + object mesh)
#   DEPTH_TREND=1 (default)              -> human depth drift removed with MoGe's trend (DEPTH_TREND_ALPHA,
#                                           default 0.75) -> submissions/episodes_wham_fp_depth/
#   SMOOTH_SYMMETRIC=1 (default)         -> centre of symmetry-tracked objects smoothed (SMOOTH_SYMMETRIC_LAM,
#                                           default 100) -> <previous dir>_smooth/
#   HUMAN_REFINE=1 (default, needs a GPU) -> human motion smoothing, object-anchored hands, MediaPipe/RTS wrist
#                                           fusion and lateral root RTS (stages/human_refine.sh, REFINE_LAM,
#                                           default 100) -> <previous dir>_refine/
#   RESOLVE_PEN=1 (optional)             -> hand points + penetration push -> <previous dir>_pen/
#   official eval_reconstruction.py      -> submissions/*.parquet
# COMMIT_URL=https://github.com/<owner>/<repo>/commit/<sha> makes the file uploadable; without it a
# clearly named local validation file is written. POSES=smoothed|raw selects the FoundationPose output.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
require_file "${VENV_PYTHON}" "run pipeline/setup/setup_venv.sh"
KIT="${TRACK1_ROOT}/official/v2d_submission_kit"
EPISODES_NAME="episodes_wham_fp"

"${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/build_track1_episodes.py" --out "${TRACK1_ROOT}/submissions/${EPISODES_NAME}" \
  --poses "${POSES:-smoothed}"
if [[ "${DEPTH_TREND:-1}" == 1 ]]; then
  for e in $(seq 0 29); do
    require_file "${TRACK1_ROOT}/outputs/depth_trend/$(episode_tag "${e}").npz" "run pipeline/stages/depth_trend.sh ${e}"
  done
  in_container wham.sif python /track1/scripts/depth_trend_correction.py apply \
    --input "/track1/submissions/${EPISODES_NAME}" --out "/track1/submissions/${EPISODES_NAME}_depth" \
    --alpha "${DEPTH_TREND_ALPHA:-0.75}"
  EPISODES_NAME="${EPISODES_NAME}_depth"
fi
if [[ "${SMOOTH_SYMMETRIC:-1}" == 1 ]]; then
  in_container wham.sif python /track1/scripts/smooth_symmetric_objects.py \
    --input "/track1/submissions/${EPISODES_NAME}" --out "/track1/submissions/${EPISODES_NAME}_smooth" \
    --lam "${SMOOTH_SYMMETRIC_LAM:-100}"
  EPISODES_NAME="${EPISODES_NAME}_smooth"
fi
if [[ "${HUMAN_REFINE:-1}" == 1 ]]; then
  [[ "${DEPTH_TREND:-1}" == 1 ]] || { echo "HUMAN_REFINE=1 expects DEPTH_TREND=1" >&2; exit 2; }
  bash "${TRACK1_ROOT}/pipeline/stages/human_refine.sh" "${EPISODES_NAME}" episodes_wham_fp_depth episodes_wham_fp \
    "${EPISODES_NAME}_refine"
  EPISODES_NAME="${EPISODES_NAME}_refine"
fi
if [[ "${RESOLVE_PEN:-0}" == 1 ]]; then
  in_container mhr_converter.sif python /track1/scripts/extract_mhr_hands.py \
    --episodes_dir "/track1/submissions/${EPISODES_NAME}"
  "${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/resolve_hand_penetration.py" --input "${TRACK1_ROOT}/submissions/${EPISODES_NAME}" \
    --out "${TRACK1_ROOT}/submissions/${EPISODES_NAME}_pen" --workers "${PEN_WORKERS:-8}"
  EPISODES_NAME="${EPISODES_NAME}_pen"
fi
EPISODES_DIR="${TRACK1_ROOT}/submissions/${EPISODES_NAME}"

if [[ -n "${COMMIT_URL:-}" ]]; then
  OUT="${TRACK1_ROOT}/submissions/track1_wham_fp_$(date +%Y%m%d_%H%M).parquet"
  COMMIT="${COMMIT_URL}"
else
  OUT="${TRACK1_ROOT}/submissions/track1_LOCAL_CHECK_do_not_upload.parquet"
  COMMIT="https://github.com/local-check/not-for-upload/commit/0000000000000000000000000000000000000000"
fi
cd "${KIT}"
"${VENV_PYTHON}" eval_reconstruction.py --episodes "${EPISODES_DIR}" \
  --sample data/track_1_sample_submission.parquet --commit "${COMMIT}" --out "${OUT}"
echo "PASS: packed ${OUT}"
