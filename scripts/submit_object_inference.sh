#!/bin/bash
set -euo pipefail

ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd /work/smoses2s/nvidea_chal
for image in v2d_cari4d.sif v2d_sam2.sif v2d_grounding_dino.sif; do
  [[ -s "${ROOT}/containers/${image}" ]] || { echo "Missing ${image}" >&2; exit 2; }
done

jid_weights="$(sbatch --parsable "${ROOT}/slurm/08_download_object_weights.slurm")"
depth_jobs=()
mask_jobs=()
for offset in 0 5 10 15 20 25; do
  depth_jobs+=("$(sbatch --parsable --dependency="afterok:${jid_weights}" --array=0-4 \
    --export="ALL,EPISODE_OFFSET=${offset}" "${ROOT}/slurm/09_object_depth.slurm")")
  mask_jobs+=("$(sbatch --parsable --dependency="afterok:${jid_weights}" --array=0-4 \
    --export="ALL,EPISODE_OFFSET=${offset}" "${ROOT}/slurm/10_object_masks.slurm")")
done

printf 'weights=%s\ndepth=%s\nmasks=%s\n' \
  "${jid_weights}" "${depth_jobs[*]}" "${mask_jobs[*]}"
