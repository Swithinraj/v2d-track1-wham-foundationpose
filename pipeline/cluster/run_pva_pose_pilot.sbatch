#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=v2d-p1-pva-pose
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/pva_pose_%j.log
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo
export TRACK1_EXTRA_BINDS=/work/mnidgu2s/project1/vpq-vitpose
source pipeline/env.sh
trap 'echo "PVA_POSE_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
echo "PVA_POSE_START node=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES}"
in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
  python -u /work/mnidgu2s/project1/vpq-vitpose/scripts/pva_pose_refine.py \
  --root /track1 --base /track1/submissions/episodes_cand_P2 --predictions /track1/work/pva/predictions \
  --out /track1/work/pva/pose_pilot --episode 9 --steps 100
