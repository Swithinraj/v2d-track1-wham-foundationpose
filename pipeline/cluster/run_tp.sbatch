#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --job-name=v2d-p1-tp
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/tp_%j.log
set -euo pipefail
source pipeline/env.sh
export TMPDIR=/work/mnidgu2s/project1/.tmp
rm -rf submissions/episodes_cand_Mtp_in
in_container mhr_converter.sif python -u /track1/scripts/true_cam_params.py --input /track1/submissions/episodes_base_BMQ3KGM \
  --out /track1/submissions/episodes_cand_Mtp_in --old_root /track1 --k_new /track1/outputs/camera_measured/k_new.json
bash pipeline/recipes/contact_stage_split.sh episodes_cand_Mtp_in episodes_cand_Mtp
