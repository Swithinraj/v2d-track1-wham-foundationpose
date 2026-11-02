#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=v2d-p1-penzero
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/pen_zero_%j.log
# A PEN-0 version of a candidate (pipeline/stages/pen_zero.sh): objects pushed out of every dense hand vertex, human
# untouched, so CD-H and ACC-H keep the input's values. Usage: IN=episodes_cand_X OUT=episodes_cand_XZ sbatch this.
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo OMP_NUM_THREADS=4 PEN_ZERO_WORKERS=8
source pipeline/env.sh
trap 'echo "PENZERO_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
[[ -e submissions/${OUT} ]] && { echo "submissions/${OUT} exists; refusing to overwrite"; exit 2; }
echo "PENZERO_START node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} in=${IN} out=${OUT}"
bash pipeline/stages/pen_zero.sh "${IN}" "${OUT}" | tail -2
