#!/bin/bash
#SBATCH --job-name=v2d_t1_wham_mhr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/wham_mhr_%A_%a.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/wham_mhr_%A_%a.err

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
IMAGE="${ROOT}/containers/mhr_converter.sif"
INPUT="${ROOT}/outputs/wham/episode_${PADDED}/wham_smpl_world.npz"
OUTPUT="${ROOT}/outputs/human/episode_${PADDED}/mhr_human.npz"
RUNTIME="${ROOT}/.cache/mhr_runtime"

python3 "${ROOT}/scripts/validate_mhr_setup.py"
[[ -s "${INPUT}" ]] || { echo "Missing WHAM mesh input: ${INPUT}" >&2; exit 3; }
if [[ -s "${OUTPUT}" ]] && python3 -c "import numpy as np; z=np.load('${OUTPUT}'); n=np.load('${INPUT}')['vertices'].shape[0]; assert z['pose'].shape==(n,136); assert z['scales'].shape==(68,); assert z['shape'].shape==(45,); assert all(np.isfinite(z[k]).all() for k in ('pose','scales','shape'))"; then
  echo "PASS: reusing validated WHAM -> MHR output for episode ${PADDED}"
  exit 0
fi
mkdir -p "$(dirname "${OUTPUT}")" "${RUNTIME}/home"

apptainer exec --nv --no-home --cleanenv \
  --bind "${ROOT}:/track1" "${IMAGE}" \
  env HOME=/track1/.cache/mhr_runtime/home \
  python /track1/scripts/convert_wham_to_mhr.py \
  --input "/track1/outputs/wham/episode_${PADDED}/wham_smpl_world.npz" \
  --output "/track1/outputs/human/episode_${PADDED}/mhr_human.npz" \
  --smpl-faces /track1/weights/wham/dataset/body_models/smpl_faces.npy \
  --mapping /track1/third_party/MHR/tools/mhr_smpl_conversion/assets/smpl2mhr_mapping.npz \
  --mhr-model /track1/weights/mhr/assets/mhr_model.pt \
  --official-converter /track1/official/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py \
  --batch-size "${MHR_MAP_BATCH_SIZE:-64}" \
  --model-batch "${MHR_MODEL_BATCH_SIZE:-256}"

python3 -c "import numpy as np; z=np.load('${OUTPUT}'); assert z['pose'].shape[1:]==(136,); assert z['scales'].shape==(68,); assert z['shape'].shape==(45,)"
echo "PASS: WHAM -> MHR Track 1 human parameters episode ${PADDED}"
