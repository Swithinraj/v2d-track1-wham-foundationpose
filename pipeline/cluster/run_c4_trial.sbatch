#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=v2d-p1-c4t
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/c4t_%j.log
set -uo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp
bash pipeline/recipes/cari4d_episode.sh 28 0
