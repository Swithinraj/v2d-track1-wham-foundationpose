#!/bin/bash
#SBATCH --job-name=v2d_t1_trellis_mesh
#SBATCH --partition=gpu
#SBATCH --exclude=wr14,wr15,wr16,wr17,wr18,wr19,wr20,wr21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/trellis_mesh_%A_%a.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/trellis_mesh_%A_%a.err

# One generated metric mesh per Track 1 object (array index = sorted object index):
#   1. scripts/trellis_object_mesh.py  clean-view selection + TRELLIS candidates (each episode's
#                                      best view and all views together, SEEDS) + depth-extent scale
#   2. scripts/select_object_mesh.py   FoundationPose fit in all three cameras picks the candidate,
#                                      refines its scale -> inputs/object_meshes/<object>.glb
# The SAM3D image's nvdiffrast and flash-attn are built for sm_80+, so V100 nodes
# (wr14-wr19) and the unverified wr20/wr21 are excluded.
set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
mapfile -t OBJECTS < <(python3 -c "
import json
rows = [json.loads(l) for l in open('${ROOT}/data/track_1/meta/episodes_metadata.jsonl') if l.strip()]
print('\n'.join(sorted({r['object'] for r in rows})))")
if [[ -n "${OBJECT:-}" ]]; then
  OBJECT="${OBJECT}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  OBJECT="${OBJECTS[$(( ${OBJECT_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))]}"
else
  echo "Set OBJECT or submit as an array with OBJECT_OFFSET." >&2; exit 2
fi
printf '%s\n' "${OBJECTS[@]}" | grep -qx "${OBJECT}" || { echo "Unknown object ${OBJECT}" >&2; exit 2; }

for path in "${ROOT}/containers/v2d_sam3d.sif" "${ROOT}/containers/v2d_cari4d.sif" \
  "${ROOT}/weights/trellis/TRELLIS-image-large/pipeline.json" \
  "${ROOT}/weights/trellis/TRELLIS-image-large/REVISION"; do
  [[ -s "${path}" ]] || { echo "Missing ${path}; run slurm/17_download_trellis_weights.slurm" >&2; exit 3; }
done

echo "object=${OBJECT} node=${SLURM_NODELIST:-local}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# SKIP_GENERATION=1 re-runs only the selection on existing candidates.
if [[ "${SKIP_GENERATION:-0}" != 1 ]]; then
  apptainer exec --nv --no-home --cleanenv \
    --bind "${ROOT}:/track1" --pwd /track1 "${ROOT}/containers/v2d_sam3d.sif" \
    env HOME=/tmp TORCH_HOME=/track1/weights/trellis/torch_home \
        HF_HUB_OFFLINE=1 ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    python -u /track1/scripts/trellis_object_mesh.py --object "${OBJECT}" --seeds ${SEEDS:-1 2}
fi

apptainer exec --nv --no-home --cleanenv \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_cari4d.sif" \
  env HOME=/tmp MPLCONFIGDIR=/tmp/mpl python -u /track1/scripts/select_object_mesh.py --object "${OBJECT}"
