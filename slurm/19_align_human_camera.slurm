#!/bin/bash
#SBATCH --job-name=v2d_t1_align
#SBATCH --partition=hpc3,hpc1,hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/align_%j.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/align_%j.err

# WHAM world -> MoGe-2 camera rigid transform and MoGe/SMPL depth scale for each episode
# (scripts/align_human_camera.py). CPU only. EPISODES defaults to all 30.
set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
EPISODES="${EPISODES:-$(seq -s ' ' 0 29)}"
failed=()
for episode in ${EPISODES}; do
  if ! apptainer exec --no-home --cleanenv --env HOME=/tmp \
      --bind "${ROOT}:/track1" "${ROOT}/containers/wham.sif" \
      python /track1/scripts/align_human_camera.py --episode "${episode}"; then
    failed+=("${episode}")
  fi
done
if ((${#failed[@]})); then
  echo "FAILED episodes: ${failed[*]}" >&2
  exit 1
fi
echo "PASS: alignment for episodes ${EPISODES}"
