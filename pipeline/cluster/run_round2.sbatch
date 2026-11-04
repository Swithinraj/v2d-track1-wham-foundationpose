#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256G
#SBATCH --gres=gpu:3
#SBATCH --time=03:00:00
#SBATCH --job-name=v2d-p1-r2
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/r2_%j.log
set -euo pipefail
source pipeline/env.sh
export TMPDIR=/work/mnidgu2s/project1/.tmp CONTACT_GPUS=3
P=.venvs/submission/bin/python; S=submissions; TP=$S/episodes_cand_Mtp_in; B=work/analysis/bodies_Mtp_in; PL=outputs/floor/planes_measured.json
# ML3: person slide with lighter smoothing (lam 3000); objects untouched: the CD-O baseline on the measured camera
rm -rf $S/episodes_cand_ML3_in; $P scripts/floor_root.py --input $TP --bodies $B --planes $PL --out $S/episodes_cand_ML3_in --lam 3e3 | tail -1
bash pipeline/recipes/contact_stage_split.sh episodes_cand_ML3_in episodes_cand_ML3
# FB15: person slide at 1.5 strength; object floor scale with the real floor 2 cm below the lowest vertex
rm -rf $S/episodes_FB15_h $S/episodes_cand_FB15_in; $P scripts/floor_root.py --input $TP --bodies $B --planes $PL --out $S/episodes_FB15_h --lam 1e4 --beta 1.5 | tail -1
$P scripts/object_floor_scale.py --input $S/episodes_FB15_h --planes $PL --levels $S/episodes_FB15_h/floor_root_report.json --floor_offset 0.02 --out $S/episodes_cand_FB15_in | tail -1
bash pipeline/recipes/contact_stage_split.sh episodes_cand_FB15_in episodes_cand_FB15
echo ROUND2BUILT
