#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=v2d-p1-cv3
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/cv3_%j.log
export TMPDIR=/work/mnidgu2s/project1/.tmp CONTACT_GPUS=1
rm -rf submissions/episodes_V3_in; cp -a submissions/episodes_cand_V3 submissions/episodes_V3_in
bash pipeline/recipes/contact_stage_split.sh episodes_V3_in episodes_cand_V3C
