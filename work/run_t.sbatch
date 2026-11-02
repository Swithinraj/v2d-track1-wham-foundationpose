#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --job-name=v2d-p1-t
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/t_%j.log
set -euo pipefail
source pipeline/env.sh
export TMPDIR=/work/mnidgu2s/project1/.tmp
P=.venvs/submission/bin/python; S=submissions; PL=outputs/floor/planes_measured.json; LV=submissions/episodes_cand_MF_h/floor_root_report.json
# T_G97: the human unchanged, every object 3 % closer (global dose)
rm -rf $S/episodes_cand_G97_in; $P scripts/object_floor_scale.py --input $S/episodes_cand_Mtp_in --planes $PL --levels $LV --out $S/episodes_cand_G97_in --fixed_kappa 0.97 | tail -1
bash pipeline/recipes/contact_stage_split.sh episodes_cand_G97_in episodes_cand_G97
# T_WF: WHAM root mix 0.25; objects with floor evidence scaled, others at the floor group's mean
rm -rf $S/episodes_mix_w25 $S/episodes_cand_WF_in
$P scripts/human_root_mix.py --base $S/episodes_cand_Mtp_in --base_meshes work/analysis/meshes/BMQ3KGM --other_meshes work/analysis/meshes/h2t --out $S/episodes_mix_w25 --weight 0.25
$P scripts/object_floor_scale.py --input $S/episodes_mix_w25 --planes $PL --levels $LV --out $S/episodes_cand_WF_in --abstain_kappa 0.974 | tail -1
bash pipeline/recipes/contact_stage_split.sh episodes_cand_WF_in episodes_cand_WF
