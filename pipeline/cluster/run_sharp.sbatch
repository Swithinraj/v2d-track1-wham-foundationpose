#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=v2d-p1-sh
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/sh_%j.log
set -uo pipefail
source pipeline/env.sh
set +e
export TMPDIR=/work/mnidgu2s/project1/.tmp
P=.venvs/submission/bin/python; S=submissions; A=work/analysis; PM=outputs/floor/planes_measured.json
[[ -f $S/episodes_sharp30/episode_000029.npz ]] || in_container mhr_converter.sif python -u /track1/scripts/human_refit.py --input /track1/$S/episodes_cand_Mtp_in --out /track1/$S/episodes_sharp30 --lam 100 --sharpen 0.3 | tail -1
[[ -f $A/bodies_Sh/episode_000029.npz ]] || in_container mhr_converter.sif python -u /track1/scripts/floor_bodies.py --episodes_dir /track1/$S/episodes_sharp30 --out /track1/$A/bodies_Sh | tail -1
rm -rf $S/episodes_person_Sh $S/episodes_objects_Sh $S/episodes_cand_S30
$P scripts/floor_root.py --input $S/episodes_sharp30 --bodies $A/bodies_Sh --planes $PM --out $S/episodes_person_Sh --lam 1e4 --beta 0.55 | tail -1
in_container mhr_converter.sif python -u /track1/scripts/ensemble_meshes.py --episodes_dir /track1/$S/episodes_person_Sh --out /track1/$A/meshes/person_Sh | tail -1
$P scripts/motion_smooth.py --input $S/episodes_baseold_k_OTa --out $S/episodes_objects_Sh --lam_object 100 --sharpen 0.3 | tail -1
$P scripts/object_transfer.py --people $S/episodes_person_Sh --people_meshes $A/meshes/person_Sh --objects $S/episodes_objects_Sh --objects_meshes $A/meshes/BMQ3KGM_old --out $S/episodes_cand_S30
echo SHARPBUILT
