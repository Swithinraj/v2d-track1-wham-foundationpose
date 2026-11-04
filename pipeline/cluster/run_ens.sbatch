#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --job-name=v2d-p1-ens
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/ens_%j.log
set -uo pipefail
source pipeline/env.sh
set +e
export TMPDIR=/work/mnidgu2s/project1/.tmp
S=/track1/submissions; OUT=$S/episodes_ens_E2
rm -rf submissions/episodes_ens_E2
M="--member BMQ3KGM=$S/episodes_cand_Mtp_in:0.5 --member ktcp=$S/episodes_base_ktcp:0.5"
A=$(seq -s ' ' 0 2 29); B=$(seq -s ' ' 1 2 29)
in_container mhr_converter.sif CUDA_VISIBLE_DEVICES=0 python -u /track1/scripts/human_ensemble.py --out $OUT $M --episodes $A | tail -2 &
in_container mhr_converter.sif CUDA_VISIBLE_DEVICES=1 python -u /track1/scripts/human_ensemble.py --out $OUT $M --episodes $B | tail -2 &
wait
ls submissions/episodes_ens_E2/*.npz | wc -l
