#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=v2d-p1-pva-pilot
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/pva_pilot_%j.log
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp
export TRACK1_ROOT=/work/mnidgu2s/project1/repo
export TRACK1_EXTRA_BINDS="/work/mnidgu2s/project1/vpq-vitpose /work/mnidgu2s/project1/research/HTD-Refine /work/mnidgu2s/project1/research/pva_site"
source pipeline/env.sh
echo "PVA_PILOT_START node=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES}"
trap 'echo "PVA_PILOT_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
in_container v2d_cari4d.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 \
  PYTHONPATH=/work/mnidgu2s/project1/research/pva_site \
  python -u /work/mnidgu2s/project1/vpq-vitpose/scripts/run_pva_track1.py \
  --upstream /work/mnidgu2s/project1/research/HTD-Refine \
  --model /work/mnidgu2s/project1/research/HTD-Refine/inputs/checkpoints/pvanet.pt \
  --inputs /track1/work/pva/pilot_inputs --out /track1/work/pva/pilot_predictions --episodes 9
