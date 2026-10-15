#!/bin/bash
#SBATCH --job-name=v2d_t1_depth
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/depth_%A_%a.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/depth_%A_%a.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
if [[ -n "${EPISODE:-}" ]]; then
  EPISODE="${EPISODE}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  EPISODE="$(( ${EPISODE_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))"
else
  EPISODE=""
fi
[[ "${EPISODE}" =~ ^[0-9]+$ ]] && ((EPISODE >= 0 && EPISODE <= 29)) || {
  echo "Set EPISODE to an integer from 0 through 29." >&2; exit 2;
}
PADDED="$(printf '%06d' "${EPISODE}")"
VIDEO="${ROOT}/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_${PADDED}.mp4"
OUT="${ROOT}/outputs/object/episode_${PADDED}"
WEIGHTS="${ROOT}/weights/object/moge2"
mkdir -p "${OUT}/depth" "${OUT}/intrinsics"
test -s "${VIDEO}"
test -s "${WEIGHTS}/model.pt"

apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_cari4d.sif" \
  python -m v2d.moge.lib.video_to_depth \
  --video_path "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_${PADDED}.mp4" \
  --depth_folder "/track1/outputs/object/episode_${PADDED}/depth" \
  --intrinsics_folder "/track1/outputs/object/episode_${PADDED}/intrinsics" \
  --weights_path /track1/weights/object/moge2 \
  --batch_size "${MOGE_BATCH_SIZE:-4}"

EXPECTED="$(python3 -c "import json; print(next(x['length'] for x in map(json.loads, open('${ROOT}/data/track_1/meta/episodes.jsonl')) if x['episode_index']==${EPISODE}))")"
DEPTH_COUNT="$(find "${OUT}/depth" -maxdepth 1 -name '*.png' | wc -l)"
INTRINSICS_COUNT="$(find "${OUT}/intrinsics" -maxdepth 1 -name '*.json' | wc -l)"
[[ "${DEPTH_COUNT}" -eq "${EXPECTED}" && "${INTRINSICS_COUNT}" -eq "${EXPECTED}" ]] || {
  echo "Depth output incomplete: expected=${EXPECTED} depth=${DEPTH_COUNT} intrinsics=${INTRINSICS_COUNT}" >&2; exit 3;
}
echo "PASS: MoGe-2 metric depth episode ${PADDED}, ${EXPECTED} frames"
