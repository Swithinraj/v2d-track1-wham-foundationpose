#!/bin/bash
set -euo pipefail

ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd /work/smoses2s/nvidea_chal

# WR limits arrays to five tasks, so six offset chunks cover episodes 0..29.
jid_build="$(sbatch --parsable "${ROOT}/slurm/07_build_object_tools.slurm")"
jid_weights="$(sbatch --parsable --dependency="afterok:${jid_build}" \
  "${ROOT}/slurm/08_download_object_weights.slurm")"

depth_jobs=()
mask_jobs=()
for offset in 0 5 10 15 20 25; do
  depth_jobs+=("$(sbatch --parsable --dependency="afterok:${jid_weights}" --array=0-4 \
    --export="ALL,EPISODE_OFFSET=${offset}" "${ROOT}/slurm/09_object_depth.slurm")")
  mask_jobs+=("$(sbatch --parsable --dependency="afterok:${jid_weights}" --array=0-4 \
    --export="ALL,EPISODE_OFFSET=${offset}" "${ROOT}/slurm/10_object_masks.slurm")")
done

printf 'build=%s\nweights=%s\ndepth=%s\nmasks=%s\n' \
  "${jid_build}" "${jid_weights}" "${depth_jobs[*]}" "${mask_jobs[*]}"
