#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=v2d-p1-object-audit
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/refined_object_audit_%j.log
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo
source pipeline/env.sh
trap 'echo "OBJECT_AUDIT_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
echo "OBJECT_AUDIT_START node=$(hostname)"
in_container v2d_cari4d.sif env OMP_NUM_THREADS=4 python -u /track1/work/analysis/audit_refined_objects.py
