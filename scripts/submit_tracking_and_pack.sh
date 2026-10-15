#!/bin/bash
# FoundationPose tracking for all 30 episodes (six 5-task arrays; WR MaxArraySize=5), then the
# joint assembly + official packing job once every tracking task succeeds.
# Requires inputs/object_meshes/<object>.glb for all ten objects (slurm/18_trellis_object_mesh.slurm).
# Pass COMMIT_URL=https://github.com/<owner>/<repo>/commit/<sha> for an uploadable Parquet;
# without it the pack job writes a local validation file only.
set -euo pipefail

ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd /work/smoses2s/nvidea_chal
python3 "${ROOT}/scripts/validate_object_setup.py"

track_jobs=()
for offset in 0 5 10 15 20 25; do
  track_jobs+=("$(sbatch --parsable --array=0-4 --export="ALL,EPISODE_OFFSET=${offset}" \
    "${ROOT}/slurm/11_foundationpose_episode.slurm")")
done
dependency="afterok:$(IFS=:; echo "${track_jobs[*]}")"
jid_pack="$(sbatch --parsable --dependency="${dependency}" \
  --export="ALL,COMMIT_URL=${COMMIT_URL:-}" "${ROOT}/slurm/20_pack_track1.slurm")"

printf 'foundationpose=%s\npack=%s\n' "${track_jobs[*]}" "${jid_pack}"
