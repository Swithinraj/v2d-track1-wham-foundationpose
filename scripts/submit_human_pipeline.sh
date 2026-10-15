#!/bin/bash
set -euo pipefail

ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd /work/smoses2s/nvidea_chal

DPVO_EXTENSIONS="${ROOT}/.cache/dpvo_extensions/lib"
for module in cuda_corr cuda_ba lietorch_backends; do
  test -n "$(find "${DPVO_EXTENSIONS}" -maxdepth 1 -type f -name "${module}*.so" -print -quit 2>/dev/null)" || {
    echo "Missing cluster-compatible DPVO extension ${module}." >&2
    echo "Run: sbatch ${ROOT}/slurm/16_build_dpvo_extensions.slurm" >&2
    exit 3
  }
done

# Refuse to create 60 doomed jobs when the separately licensed SMPL asset is absent.
SMPL_MODEL="${ROOT}/weights/wham/dataset/body_models/smpl/SMPL_NEUTRAL.pkl"
SMPL_FACES="${ROOT}/weights/wham/dataset/body_models/smpl_faces.npy"
[[ -s "${SMPL_MODEL}" ]] || {
  echo "Missing licensed SMPL model: ${SMPL_MODEL}" >&2
  exit 2
}
if [[ ! -s "${SMPL_FACES}" ]]; then
  apptainer exec --no-home --cleanenv \
    --bind "${ROOT}:/track1" "${ROOT}/containers/wham.sif" \
    python /track1/scripts/prepare_smpl_faces.py \
      --model /track1/weights/wham/dataset/body_models/smpl/SMPL_NEUTRAL.pkl \
      --output /track1/weights/wham/dataset/body_models/smpl_faces.npy
fi
python3 "${ROOT}/scripts/validate_wham_setup.py"
python3 "${ROOT}/scripts/validate_mhr_setup.py"

wham_jobs=()
mhr_jobs=()
for offset in 0 5 10 15 20 25; do
  wham_job="$(sbatch --parsable --array=0-4 \
    --export="ALL,EPISODE_OFFSET=${offset}" "${ROOT}/slurm/06_wham_episode.slurm")"
  wham_jobs+=("${wham_job}")
  mhr_jobs+=("$(sbatch --parsable --dependency="afterok:${wham_job}" --array=0-4 \
    --export="ALL,EPISODE_OFFSET=${offset}" "${ROOT}/slurm/13_wham_to_mhr_episode.slurm")")
done

printf 'wham=%s\nmhr=%s\n' "${wham_jobs[*]}" "${mhr_jobs[*]}"
