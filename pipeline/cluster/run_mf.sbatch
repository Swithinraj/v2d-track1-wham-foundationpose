#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=v2d-p1-mf
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/mf_%j.log
set -euo pipefail
source pipeline/env.sh
export TMPDIR=/work/mnidgu2s/project1/.tmp
bash pipeline/recipes/contact_stage_split.sh episodes_cand_MF_in episodes_cand_MF
