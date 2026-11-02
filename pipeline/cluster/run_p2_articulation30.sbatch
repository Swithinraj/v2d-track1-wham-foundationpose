#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=v2d-p1-p2-artic30
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/p2_artic30_%j.log
# P2A30: P2 (the ACC-H best) with articulation acceleration scaled by 0.3, root motion kept (scripts/articulation_shrink.py).
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo OMP_NUM_THREADS=4
export TRACK1_EXTRA_BINDS=/work/mnidgu2s/project1/vpq-vitpose
source pipeline/env.sh
trap 'echo "P2A_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
CODE=/work/mnidgu2s/project1/vpq-vitpose
echo "P2A_START node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES} commit=$(git -C $CODE rev-parse HEAD)"
in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 TRACK1_ROOT=/track1 \
  PYTHONPATH=/track1/scripts python -u "$CODE/scripts/articulation_shrink.py" --input /track1/submissions/episodes_cand_P2 --k 0.3 \
  --out /track1/submissions/episodes_cand_P2A30 | tail -1
echo P2A_BUILT
