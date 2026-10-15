#!/bin/bash
#SBATCH --job-name=v2d_t1_cari4d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/cari4d_%A_%a.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/cari4d_%A_%a.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
RECON="${ROOT}/third_party/video_to_data/reconstruction"
EPISODE="${EPISODE:-${SLURM_ARRAY_TASK_ID:-}}"

if [[ ! "${EPISODE}" =~ ^[0-9]+$ ]] || ((EPISODE < 0 || EPISODE > 29)); then
  echo "Set EPISODE to an integer from 0 through 29." >&2
  exit 2
fi

EP_PADDED="$(printf '%06d' "${EPISODE}")"
INPUT_DIR="${ROOT}/inputs/episode_${EP_PADDED}"
VIDEO_SOURCE="${ROOT}/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_${EP_PADDED}.mp4"
VIDEO="${INPUT_DIR}/episode_${EP_PADDED}.0.color.mp4"
MASKS="${INPUT_DIR}/masks_k0.h5"
MESH="${INPUT_DIR}/object.glb"
OUTPUT="${ROOT}/outputs/cari4d/episode_${EP_PADDED}"

for path in "${VIDEO_SOURCE}" "${MASKS}" "${MESH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 3
  fi
done

# CARI4D enforces its capture naming contract; preserve the public video bytes via a symlink.
if [[ ! -e "${VIDEO}" ]]; then
  ln -s "${VIDEO_SOURCE}" "${VIDEO}"
fi

EXPECTED_FRAMES="$(python3 - "${ROOT}/data/track_1/meta/episodes.jsonl" "${EPISODE}" <<'PY'
import json, sys
path, wanted = sys.argv[1], int(sys.argv[2])
with open(path) as handle:
    for line in handle:
        item = json.loads(line)
        if int(item['episode_index']) == wanted:
            print(int(item['length']))
            break
    else:
        raise SystemExit(f'episode {wanted} absent from metadata')
PY
)"

cd "${RECON}"
export APPTAINER_CACHEDIR="${ROOT}/containers/apptainer-cache"
export APPTAINER_TMPDIR="${ROOT}/containers/apptainer-tmp"
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

if [[ ! -f "${ROOT}/containers/v2d_cari4d.sif" ]]; then
  echo "Missing ${ROOT}/containers/v2d_cari4d.sif; complete the build job first." >&2
  exit 4
fi

echo "episode=${EP_PADDED} frames=${EXPECTED_FRAMES} node=${SLURM_NODELIST}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

apptainer exec --nv --no-home --cleanenv --env HOME=/tmp \
  --bind "${ROOT}:/track1" \
  "${ROOT}/containers/v2d_cari4d.sif" \
  python -m v2d.cari4d.lib.run_inference \
  --video_path "/track1/inputs/episode_${EP_PADDED}/episode_${EP_PADDED}.0.color.mp4" \
  --mask_h5_path "/track1/inputs/episode_${EP_PADDED}/masks_k0.h5" \
  --object_mesh_path "/track1/inputs/episode_${EP_PADDED}/object.glb" \
  --weights_path /track1/weights/cari4d \
  --output_dir "/track1/outputs/cari4d/episode_${EP_PADDED}" \
  --expected_frames "${EXPECTED_FRAMES}" \
  --skip_weight_download

echo "PASS: CARI4D episode ${EP_PADDED} completed"

