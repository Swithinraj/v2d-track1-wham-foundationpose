#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=66
#SBATCH --mem=360G
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00
#SBATCH --job-name=v2d-p1-c4all
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/c4all_%j.log
export TMPDIR=/work/mnidgu2s/project1/.tmp
bash pipeline/recipes/cari4d_all.sh 3 2
