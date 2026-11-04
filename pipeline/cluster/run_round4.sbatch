#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=v2d-p1-r4
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=logs/r4_%j.log
set -uo pipefail
source pipeline/env.sh
set +e
export TMPDIR=/work/mnidgu2s/project1/.tmp
P=.venvs/submission/bin/python; S=submissions; A=work/analysis; PM=outputs/floor/planes_measured.json; PO=outputs/floor/planes_moge.json
container() { in_container mhr_converter.sif python -u "$@" | tail -1; }
bodies() { [[ -f $A/bodies_$1/episode_000029.npz ]] || container /track1/scripts/floor_bodies.py --episodes_dir /track1/$2 --out /track1/$A/bodies_$1; }
meshes() { [[ -f $A/meshes/$1/episode_000029.npy ]] || container /track1/scripts/ensemble_meshes.py --episodes_dir /track1/$2 --out /track1/$A/meshes/$1; }
person() { # NAME INPUT_DIR BETA: floor slide on a person variant
  bodies $1 $2
  rm -rf $S/episodes_person_$1
  $P scripts/floor_root.py --input $2 --bodies $A/bodies_$1 --planes $PM --out $S/episodes_person_$1 --lam 1e4 --beta $3 | tail -1
  meshes person_$1 $S/episodes_person_$1
}
carry() { # NAME PERSON OBJECTS_DIR OBJECTS_MESHES
  rm -rf $S/episodes_cand_$1
  $P scripts/object_transfer.py --people $S/episodes_person_$2 --people_meshes $A/meshes/person_$2 --objects $3 --objects_meshes $4 --out $S/episodes_cand_$1
}
# People: mesh smoothing, ensemble, and both, each with the floor slide at 0.55
[[ -f $S/episodes_refit_E2/episode_000029.npz ]] || container /track1/scripts/human_refit.py --input /track1/$S/episodes_ens_E2 --out /track1/$S/episodes_refit_E2 --lam 100
person H100 $S/episodes_refit_l100 0.55
person HE2 $S/episodes_ens_E2 0.55
person HE2R $S/episodes_refit_E2 0.55
# Objects: BMQ3KGM's with the floor scale at 2 cm (OTa's), and a0's the same way
OA=$S/episodes_baseold_k_OTa
bodies a0old $S/episodes_base_a0; meshes a0_old $S/episodes_base_a0
rm -rf $S/episodes_a0old_lv $S/episodes_a0old_k
$P scripts/floor_root.py --input $S/episodes_base_a0 --bodies $A/bodies_a0old --planes $PO --out $S/episodes_a0old_lv --lam 1e4 --beta 0 | tail -1
$P scripts/object_floor_scale.py --input $S/episodes_a0old_lv --planes $PO --levels $S/episodes_a0old_lv/floor_root_report.json --floor_offset 0.02 --out $S/episodes_a0old_k | tail -1
# Objects: relative size constancy among table episodes
rm -rf $S/episodes_baseold_ksz
$P scripts/object_relative_size.py --input $OA --floor $OA/object_floor_scale_report.json --out $S/episodes_baseold_ksz | tail -1
carry P1 H100 $S/episodes_a0old_k $A/meshes/a0_old
carry P2 HE2 $S/episodes_baseold_ksz $A/meshes/BMQ3KGM_old
carry P3 HE2R $OA $A/meshes/BMQ3KGM_old
echo ROUND4BUILT
