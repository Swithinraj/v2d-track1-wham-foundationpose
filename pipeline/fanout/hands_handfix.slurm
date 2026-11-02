#!/bin/bash
#SBATCH --job-name=hands-hf
#SBATCH --account=rwth2243
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hands-hf-%j.out
#SBATCH --error=logs/hands-hf-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ujjwalpatil20@gmail.com

# Dense hand vertices of the baseline chain's human (mhr_bodies.py --hands-only, metres), for
# the PEN safety check of the depth-bias-corrected object track (scripts/pen_check.py): the
# human is identical in base and corrected episodes, so one hands dir serves both.
set -euo pipefail
cd /hpcwork/mdk96850/v2d-track1
apptainer exec --cleanenv --nv --bind /hpcwork/mdk96850/v2d-track1:/track1 \
  containers/mhr_converter.sif python /track1/scripts/mhr_bodies.py \
  --episodes_dir /track1/submissions/episodes_sam_handfix --out /track1/outputs/hands_handfix --hands-only
ls outputs/hands_handfix/*.npz | wc -l
echo "PASS: hands_handfix computed"
