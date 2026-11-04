#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=v2d-p1-c4f
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/c4final.log
export TMPDIR=/work/mnidgu2s/project1/.tmp CONTACT_GPU=0
bash pipeline/recipes/cari4d_candidate.sh C3 coconet 300 0.55 && OWN_OBJECTS=1 bash pipeline/recipes/cari4d_candidate.sh C4 coconet 300 0.55 && echo FINALBUILT
