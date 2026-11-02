#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=v2d-p1-geometry-cache
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/geometry_root_cache_%j.log
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo
export TRACK1_EXTRA_BINDS=/work/mnidgu2s/project1/vpq-vitpose
source pipeline/env.sh
trap 'echo "GEOMETRY_CACHE_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
echo "GEOMETRY_CACHE_START node=$(hostname)"
in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
  python -u /work/mnidgu2s/project1/vpq-vitpose/scripts/pva_mhr_joints.py --root /track1 \
  --episodes-dir /track1/submissions/episodes_cand_P3 \
  --out /track1/work/geometry_root/joints_P3_all --episodes {0..29}
