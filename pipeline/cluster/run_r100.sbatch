#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --job-name=v2d-p1-r100
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/r100_%j.log
set -euo pipefail
source pipeline/env.sh
export TMPDIR=/work/mnidgu2s/project1/.tmp
P=.venvs/submission/bin/python
rm -rf submissions/episodes_cand_R100S_in
$P scripts/motion_smooth.py --input submissions/episodes_refit_l100 --out submissions/episodes_cand_R100S_in --lam_object 10000
CONTACT_GPUS=2 bash pipeline/recipes/contact_stage_split.sh episodes_cand_R100S_in episodes_cand_R100S
