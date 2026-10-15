#!/bin/bash
#SBATCH --job-name=v2d_t1_pack
#SBATCH --partition=hpc3,hpc1,hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/pack_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/pack_%j.err

# Joint alignment + official packing for the WHAM + FoundationPose baseline:
#   scripts/build_track1_episodes.py      -> submissions/episodes_wham_fp/episode_*.npz + meshes
#   scripts/extract_mhr_hands.py          -> outputs/hands/ (MHR image)
#   scripts/resolve_hand_penetration.py   -> submissions/episodes_wham_fp_pen/ (opt-in: RESOLVE_PEN=1)
#   official eval_reconstruction.py       -> Track 1 Parquet (same packer/validation as Kaggle upload)
# COMMIT_URL must be the public GitHub commit of the code that produced the run. Without it the
# job still packs, as a clearly named local validation file that must not be uploaded.
set -euo pipefail
source /work/smoses2s/nvidea_chal/TRACK1/scripts/common.sh
load_python
ROOT="${TRACK1_ROOT}"
PY="${ROOT}/.venvs/submission/bin/python"
KIT="${ROOT}/official/v2d_submission_kit"
EPISODES_DIR="${ROOT}/submissions/episodes_wham_fp"
mkdir -p "${TMPDIR}"

"${PY}" "${ROOT}/scripts/build_track1_episodes.py" --out "${EPISODES_DIR}" --poses "${POSES:-smoothed}"
if [[ "${RESOLVE_PEN:-0}" == 1 ]]; then
  apptainer exec --no-home --cleanenv --bind "${ROOT}:/track1" "${ROOT}/containers/mhr_converter.sif" \
    env HOME=/tmp python /track1/scripts/extract_mhr_hands.py --episodes_dir /track1/submissions/episodes_wham_fp
  "${PY}" "${ROOT}/scripts/resolve_hand_penetration.py" --input "${EPISODES_DIR}" \
    --out "${ROOT}/submissions/episodes_wham_fp_pen" --workers 12
  EPISODES_DIR="${ROOT}/submissions/episodes_wham_fp_pen"
fi

if [[ -n "${COMMIT_URL:-}" ]]; then
  OUT="${ROOT}/submissions/track1_wham_fp_$(date +%Y%m%d_%H%M).parquet"
  COMMIT="${COMMIT_URL}"
else
  OUT="${ROOT}/submissions/track1_LOCAL_CHECK_do_not_upload.parquet"
  COMMIT="https://github.com/local-check/not-for-upload/commit/0000000000000000000000000000000000000000"
fi
cd "${KIT}"
"${PY}" eval_reconstruction.py \
  --episodes "${EPISODES_DIR}" \
  --sample data/track_1_sample_submission.parquet \
  --commit "${COMMIT}" \
  --out "${OUT}"
echo "PASS: packed ${OUT}"
