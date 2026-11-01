#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --job-name=v2d-p1-kh
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/kh_%j.log
set -euo pipefail
source pipeline/env.sh
export TMPDIR=/work/mnidgu2s/project1/.tmp
in_container mhr_converter.sif python -u /track1/scripts/mhr_bodies.py --episodes_dir /track1/submissions/episodes_base_ktcp --out /track1/work/analysis/hands_ktcp --hands-only | tail -1
.venvs/submission/bin/python work/analysis/kappa_hand.py submissions/episodes_cand_Mtp_in work/analysis/hands_ktcp work/analysis/kappa_hand_ktcp.json work/analysis/meshes/ktcp work/analysis/meshes/BMQ3KGM
