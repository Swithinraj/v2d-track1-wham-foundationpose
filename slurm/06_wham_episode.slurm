#!/bin/bash
#SBATCH --job-name=v2d_t1_wham
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/wham_%A_%a.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/wham_%A_%a.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
WHAM="${ROOT}/third_party/WHAM"
ASSETS="${ROOT}/weights/wham"
IMAGE="${ROOT}/containers/wham.sif"
if [[ -n "${EPISODE:-}" ]]; then
  EPISODE="${EPISODE}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  EPISODE="$(( ${EPISODE_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))"
else
  EPISODE=""
fi

if [[ ! "${EPISODE}" =~ ^[0-9]+$ ]] || ((EPISODE < 0 || EPISODE > 29)); then
  echo "Set EPISODE to an integer from 0 through 29." >&2
  exit 2
fi

EP_PADDED="$(printf '%06d' "${EPISODE}")"
VIDEO="${ROOT}/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_${EP_PADDED}.mp4"
OUTPUT_BASE="${ROOT}/outputs/wham"
RUNTIME="${ROOT}/.cache/wham_runtime"
DPVO_EXTENSIONS="${ROOT}/.cache/dpvo_extensions/lib"
FRAMES="$(python3 -c "import json; print(next(x['length'] for x in map(json.loads, open('${ROOT}/data/track_1/meta/episodes.jsonl')) if x['episode_index']==${EPISODE}))")"
MESH_OUTPUT="${OUTPUT_BASE}/episode_${EP_PADDED}/wham_smpl_world.npz"

python3 "${ROOT}/scripts/validate_wham_setup.py"
[[ -s "${VIDEO}" ]] || { echo "Missing ${VIDEO}" >&2; exit 3; }
if [[ -s "${MESH_OUTPUT}" ]] && python3 -c "import numpy as np; z=np.load('${MESH_OUTPUT}'); v=z['vertices']; assert v.shape==(${FRAMES},6890,3); assert np.isfinite(v).all()"; then
  echo "PASS: reusing validated WHAM mesh for episode ${EP_PADDED} (${FRAMES} frames)"
  exit 0
fi
for module in cuda_corr cuda_ba lietorch_backends; do
  test -n "$(find "${DPVO_EXTENSIONS}" -maxdepth 1 -type f -name "${module}*.so" -print -quit 2>/dev/null)" || {
    echo "Missing cluster-compatible DPVO extension ${module}; run slurm/16_build_dpvo_extensions.slurm" >&2
    exit 6
  }
done

mkdir -p "${OUTPUT_BASE}" "${RUNTIME}" "${RUNTIME}/home" "${RUNTIME}/matplotlib"
# The workspace is mounted at /track1 inside Apptainer. Absolute host links under
# /work/... are therefore broken in the container; keep these links relative to
# the runtime directory so they resolve in both namespaces.
ln -sfn ../../third_party/WHAM/configs "${RUNTIME}/configs"
ln -sfn ../../weights/wham/checkpoints "${RUNTIME}/checkpoints"
ln -sfn ../../weights/wham/dataset "${RUNTIME}/dataset"
# DetectionModel derives its ViTPose/YOLO paths from the WHAM source tree,
# independently of the current working directory.
ln -sfn ../../weights/wham/checkpoints "${WHAM}/checkpoints"
ln -sfn ../../weights/wham/dataset "${WHAM}/dataset"

echo "episode=${EP_PADDED} node=${SLURM_NODELIST}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

apptainer exec --nv --no-home --cleanenv \
  --bind "${ROOT}:/track1" --pwd /track1/.cache/wham_runtime \
  "${IMAGE}" \
  env HOME=/track1/.cache/wham_runtime/home \
      MPLCONFIGDIR=/track1/.cache/wham_runtime/matplotlib \
      PYTHONPATH=/track1/.cache/dpvo_extensions/lib:/track1/third_party/WHAM \
  python /track1/third_party/WHAM/demo.py \
  --video "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_${EP_PADDED}.mp4" \
  --output_pth /track1/outputs/wham \
  --save_pkl

RESULT="${OUTPUT_BASE}/episode_${EP_PADDED}/wham_output.pkl"
[[ -s "${RESULT}" ]] || { echo "WHAM did not produce ${RESULT}" >&2; exit 4; }
apptainer exec --nv --no-home --cleanenv \
  --bind "${ROOT}:/track1" --pwd /track1/.cache/wham_runtime \
  "${IMAGE}" \
  env HOME=/track1/.cache/wham_runtime/home \
      MPLCONFIGDIR=/track1/.cache/wham_runtime/matplotlib \
      PYTHONPATH=/track1/.cache/dpvo_extensions/lib:/track1/third_party/WHAM \
  python /track1/scripts/export_wham_world_mesh.py \
  --input "/track1/outputs/wham/episode_${EP_PADDED}/wham_output.pkl" \
  --output "/track1/outputs/wham/episode_${EP_PADDED}/wham_smpl_world.npz" \
  --frames "${FRAMES}"
[[ -s "${MESH_OUTPUT}" ]] || { echo "WHAM mesh export missing: ${MESH_OUTPUT}" >&2; exit 5; }
echo "PASS: WHAM episode ${EP_PADDED} completed with dense ${FRAMES}-frame SMPL mesh"
