#!/bin/bash
#SBATCH --job-name=v2d_t1_cari_inspect
# The CPU partition is backlogged for weeks on this cluster. This short job uses
# a GPU node only as a compute-node allocation; checkpoint tensors stay on CPU.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/cari4d_inspect_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/cari4d_inspect_%j.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
CHECKPOINT="${ROOT}/weights/cari4d/cari4d/2026-08-25-09-35-57/step200000.pth"
OUTPUT="${ROOT}/weights/cari4d/checkpoint_inventory.json"

test -s "${CHECKPOINT}"
apptainer exec --no-home --cleanenv --env HOME=/tmp \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_cari4d.sif" \
  python /track1/scripts/inspect_cari4d_checkpoint.py \
  --checkpoint /track1/weights/cari4d/cari4d/2026-08-25-09-35-57/step200000.pth \
  --output /track1/weights/cari4d/checkpoint_inventory.json
test -s "${OUTPUT}"
echo "PASS: CARI4D checkpoint inventory written to ${OUTPUT}"
