#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=v2d-p1-p2-full60
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/p2_full60_%j.log
# P2S60: P2 with every vertex's acceleration scaled by 0.6 (root included): human_refit.py targets V + s (V - V_s) with
# s = -0.4 and V_s the lam 1e4 smooth, i.e. 0.6 V + 0.4 V_s, then the per-frame refit. The kit's table puts the ground
# truth's own joint acceleration at 0.116-0.134 cm/frame^2, about the leaderboard's ACC-H, so this maps the full-body curve.
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo OMP_NUM_THREADS=4
source pipeline/env.sh
trap 'echo "P2S_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
[[ -e submissions/episodes_cand_P2S60 ]] && { echo "exists; refusing to overwrite"; exit 2; }
in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
  python -u /track1/scripts/human_refit.py --input /track1/submissions/episodes_cand_P2 --out /track1/submissions/episodes_cand_P2S60 \
  --lam 1e4 --sharpen -0.4 | tail -1
echo P2S_BUILT
