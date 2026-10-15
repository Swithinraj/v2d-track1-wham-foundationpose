#!/bin/bash
# Assemble all 30 episodes and pack the Track 1 submission with the official kit.
#   scripts/build_track1_episodes.py     -> submissions/episodes_wham_fp/ (episode NPZ + object mesh)
#   RESOLVE_PEN=1 (optional)             -> hand points + penetration push -> submissions/episodes_wham_fp_pen/
#   official eval_reconstruction.py      -> submissions/*.parquet
# COMMIT_URL=https://github.com/<owner>/<repo>/commit/<sha> makes the file uploadable; without it a
# clearly named local validation file is written. POSES=smoothed|raw selects the FoundationPose output.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
require_file "${VENV_PYTHON}" "run pipeline/setup/setup_venv.sh"
KIT="${TRACK1_ROOT}/official/v2d_submission_kit"
EPISODES_DIR="${TRACK1_ROOT}/submissions/episodes_wham_fp"

"${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/build_track1_episodes.py" --out "${EPISODES_DIR}" --poses "${POSES:-smoothed}"
if [[ "${RESOLVE_PEN:-0}" == 1 ]]; then
  in_container mhr_converter.sif python /track1/scripts/extract_mhr_hands.py \
    --episodes_dir /track1/submissions/episodes_wham_fp
  "${VENV_PYTHON}" "${TRACK1_ROOT}/scripts/resolve_hand_penetration.py" --input "${EPISODES_DIR}" \
    --out "${TRACK1_ROOT}/submissions/episodes_wham_fp_pen" --workers "${PEN_WORKERS:-8}"
  EPISODES_DIR="${TRACK1_ROOT}/submissions/episodes_wham_fp_pen"
fi

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
