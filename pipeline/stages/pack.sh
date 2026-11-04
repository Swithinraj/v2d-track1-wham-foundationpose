#!/bin/bash
# Assemble all 30 episodes and pack the Track 1 submission with the official kit. The human is SAM 3D Body
# (pipeline/stages/sam3d_body.sh) in the static camera's frame; objects are FoundationPose tracks.
#   scripts/build_track1_episodes_sam.py -> submissions/episodes_sam/ (objects mapped with the MoGe/SAM depth scale;
#                                           hula-hoop centre smoothing included)
#   DEPTH_TREND=1 (default)              -> human root follows MoGe's depth trend (DEPTH_TREND_ALPHA, default 0.75)
#                                           -> _depth/
#   HUMAN_REFINE=1 (default, GPU)        -> mesh-smoothing refit (REFINE_LAM, default 300), object-anchored hands,
#                                           MediaPipe/RTS wrist fusion, lateral root RTS (stages/human_refine.sh) -> _refine/
#   OBJECT_FOLLOW_HUMAN=1 (optional, needs DEPTH_TREND=0) -> object depth follows the human's depth trend -> _objtrend/
#   HAND_FIX=1 (default, GPU)            -> hand-side contact fix: fingers/wrist rotations only (stages/hand_fix.sh)
#                                           -> _handfix/
#   RESOLVE_PEN=1 (optional)             -> hand points + object penetration push -> _pen/
#   official eval_reconstruction.py      -> submissions/*.parquet
# COMMIT_URL=https://github.com/<owner>/<repo>/commit/<sha> makes the file uploadable; without it a
# clearly named local validation file is written. PACK_EPISODES=<dir under submissions/> only packs an existing episodes
# dir (e.g. after HAND_FIX=0 and hand_fix.sh split across GPUs with HAND_FIX_EPISODES). RUN_NAME (default sam) names the
# episode dirs (submissions/episodes_<RUN_NAME>...) and the refinement work dir, so two builds can run side by side.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
require_file "${VENV_PYTHON}" "run pipeline/setup/setup_venv.sh"
KIT="${TRACK1_ROOT}/official/v2d_submission_kit"
RUN="${RUN_NAME:-sam}"

if [[ -n "${PACK_EPISODES:-}" ]]; then
  EPISODES_NAME="${PACK_EPISODES}"
else
  for e in $(seq 0 29); do
    require_file "${TRACK1_ROOT}/outputs/sam3d_body/$(episode_tag "${e}")/refit.npz" "run pipeline/stages/sam3d_body.sh ${e}"
  done
  in_container v2d_cari4d.sif python /track1/scripts/build_track1_episodes_sam.py --out "/track1/submissions/episodes_${RUN}"
  EPISODES_NAME="episodes_${RUN}"
  if [[ "${DEPTH_TREND:-1}" == 1 ]]; then
    in_container v2d_cari4d.sif python /track1/scripts/sam3d_body_depth_trend.py --input "/track1/submissions/episodes_${RUN}" \
      --out "/track1/submissions/episodes_${RUN}_depth" --alpha "${DEPTH_TREND_ALPHA:-0.75}"
    EPISODES_NAME="episodes_${RUN}_depth"
  fi
  if [[ "${HUMAN_REFINE:-1}" == 1 ]]; then
    TRACK1_ALIGNMENT_DIR="${TRACK1_ROOT}/outputs/alignment_sam" REFINE_LAM="${REFINE_LAM:-300}" HUMAN_REFINE_WORK="human_refine_${RUN}" \
      bash "${TRACK1_ROOT}/pipeline/stages/human_refine.sh" "${EPISODES_NAME}" "${EPISODES_NAME}" "episodes_${RUN}" "${EPISODES_NAME}_refine"
    EPISODES_NAME="${EPISODES_NAME}_refine"
  fi
  if [[ "${OBJECT_FOLLOW_HUMAN:-0}" == 1 ]]; then
    [[ "${DEPTH_TREND:-1}" == 0 ]] || { echo "OBJECT_FOLLOW_HUMAN=1 expects DEPTH_TREND=0 (human at SAM 3D Body's own depth)" >&2; exit 2; }
    "${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/object_follow_human.py" --input "${TRACK1_ROOT}/submissions/${EPISODES_NAME}" \
      --out "${TRACK1_ROOT}/submissions/${EPISODES_NAME}_objtrend"
    EPISODES_NAME="${EPISODES_NAME}_objtrend"
  fi
  if [[ "${HAND_FIX:-1}" == 1 ]]; then
    bash "${TRACK1_ROOT}/pipeline/stages/hand_fix.sh" "${EPISODES_NAME}" "${EPISODES_NAME}_handfix"
    EPISODES_NAME="${EPISODES_NAME}_handfix"
  fi
  if [[ "${RESOLVE_PEN:-0}" == 1 ]]; then
    in_container mhr_converter.sif python /track1/scripts/extract_mhr_hands.py \
      --episodes_dir "/track1/submissions/${EPISODES_NAME}"
    "${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/resolve_hand_penetration.py" --input "${TRACK1_ROOT}/submissions/${EPISODES_NAME}" \
      --out "${TRACK1_ROOT}/submissions/${EPISODES_NAME}_pen" --workers "${PEN_WORKERS:-8}"
    EPISODES_NAME="${EPISODES_NAME}_pen"
  fi
fi
EPISODES_DIR="${TRACK1_ROOT}/submissions/${EPISODES_NAME}"

if [[ -n "${COMMIT_URL:-}" ]]; then
  OUT="${TRACK1_ROOT}/submissions/track1_sam3d_$(date +%Y%m%d_%H%M).parquet"
  COMMIT="${COMMIT_URL}"
else
  OUT="${TRACK1_ROOT}/submissions/track1_LOCAL_CHECK_do_not_upload${RUN_NAME:+_${RUN_NAME}}.parquet"
  COMMIT="https://github.com/local-check/not-for-upload/commit/0000000000000000000000000000000000000000"
fi
cd "${KIT}"
"${VENV_PYTHON}" eval_reconstruction.py --episodes "${EPISODES_DIR}" \
  --sample data/track_1_sample_submission.parquet --commit "${COMMIT}" --out "${OUT}"
echo "PASS: packed ${OUT}"
