#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=v2d-p1-geometry-pilot
#SBATCH --chdir=/work/mnidgu2s/project1/vpq-vitpose
#SBATCH --output=/work/mnidgu2s/project1/logs/geometry_root_pilot_%j.log
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
trap 'echo "GEOMETRY_PILOT_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
echo "GEOMETRY_PILOT_START node=$(hostname)"
.venvs/submission/bin/python -u scripts/geometry_root_pilot.py --root /work/mnidgu2s/project1/repo \
  --out /work/mnidgu2s/project1/repo/work/geometry_root/pilot --episodes 9 13 24
