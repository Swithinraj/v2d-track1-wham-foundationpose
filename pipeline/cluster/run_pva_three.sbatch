#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=v2d-p1-pva-three
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/pva_three_%j.log
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo
export TRACK1_EXTRA_BINDS="/work/mnidgu2s/project1/vpq-vitpose /work/mnidgu2s/project1/research/HTD-Refine /work/mnidgu2s/project1/research/pva_site"
source pipeline/env.sh
trap 'echo "PVA_THREE_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
echo "PVA_THREE_START node=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES}"
in_container v2d_cari4d.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
  PYTHONPATH=/work/mnidgu2s/project1/research/pva_site \
  python -u /work/mnidgu2s/project1/vpq-vitpose/scripts/run_pva_track1.py \
  --upstream /work/mnidgu2s/project1/research/HTD-Refine \
  --model /work/mnidgu2s/project1/research/HTD-Refine/inputs/checkpoints/pvanet.pt \
  --inputs /track1/work/pva/inputs --out /track1/work/pva/predictions --episodes 9 13 24
in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
  python -u /work/mnidgu2s/project1/vpq-vitpose/scripts/pva_mhr_joints.py \
  --root /track1 --episodes-dir /track1/submissions/episodes_cand_P2 \
  --out /track1/work/pva/joints_P2 --episodes 9 13 24
