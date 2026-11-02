#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --job-name=v2d-p1-ew
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/ew_%j.log
set -uo pipefail
source pipeline/env.sh
set +e
export TMPDIR=/work/mnidgu2s/project1/.tmp
S=/track1/submissions; OUT=$S/episodes_ens_EW30
rm -rf submissions/episodes_ens_EW30
M="--member BMQ3KGM=$S/episodes_cand_Mtp_in:0.7 --member h2t=$S/episodes_tp_h2t:0.3"
A=$(seq -s ' ' 0 2 29); B=$(seq -s ' ' 1 2 29)
in_container mhr_converter.sif CUDA_VISIBLE_DEVICES=0 python -u /track1/scripts/human_ensemble.py --out $OUT $M --episodes $A | tail -1 &
in_container mhr_converter.sif CUDA_VISIBLE_DEVICES=1 python -u /track1/scripts/human_ensemble.py --out $OUT $M --episodes $B | tail -1 &
wait
ls submissions/episodes_ens_EW30/*.npz | wc -l
set -e
CONTACT_GPUS=2 bash pipeline/recipes/contact_stage_split.sh episodes_ens_EW30 episodes_cand_EW30
