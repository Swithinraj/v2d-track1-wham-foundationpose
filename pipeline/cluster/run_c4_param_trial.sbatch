#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=v2d-p1-c4p
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/c4p_%j.log
export TMPDIR=/work/mnidgu2s/project1/.tmp
time bash pipeline/recipes/cari4d_to_params.sh 28 coconet 300 0
