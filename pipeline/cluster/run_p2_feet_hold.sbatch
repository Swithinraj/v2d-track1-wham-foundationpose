#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=v2d-p1-p2-feet
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/p2_feet_hold_%j.log
# P2F: P2 (the ACC-H best) with planted feet held still (scripts/feet_hold.py), then the per-frame MHR refit.
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo OMP_NUM_THREADS=4
export TRACK1_EXTRA_BINDS=/work/mnidgu2s/project1/vpq-vitpose
source pipeline/env.sh
trap 'echo "P2F_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
CODE=/work/mnidgu2s/project1/vpq-vitpose
echo "P2F_START node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} commit=$(git -C $CODE rev-parse HEAD)"
in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 TRACK1_ROOT=/track1 \
  PYTHONPATH=/track1/scripts python -u "$CODE/scripts/feet_hold.py" --input /track1/submissions/episodes_cand_P2 \
  --out /track1/submissions/episodes_cand_P2F | tail -1
echo P2F_BUILT
