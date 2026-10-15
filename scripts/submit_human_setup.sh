#!/bin/bash
set -euo pipefail

ROOT=/work/smoses2s/nvidea_chal/TRACK1
cd /work/smoses2s/nvidea_chal

jid_wham="$(sbatch --parsable "${ROOT}/slurm/04_build_wham.slurm")"
jid_mhr="$(sbatch --parsable "${ROOT}/slurm/12_build_mhr_converter.slurm")"
jid_assets="$(sbatch --parsable --dependency="afterok:${jid_wham}" \
  "${ROOT}/slurm/05_download_wham_assets.slurm")"

printf 'wham_build=%s\nmhr_build=%s\npublic_assets=%s\n' \
  "${jid_wham}" "${jid_mhr}" "${jid_assets}"
printf 'After these pass, place licensed SMPL_NEUTRAL.pkl under weights/wham/dataset/body_models/smpl/.\n'
