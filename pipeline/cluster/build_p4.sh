#!/bin/bash
# P4: P3's person (mesh ensemble, mesh smoothing) + slide lam 30000 + root mix toward a0; OTa objects. Runs inside the allocation.
cd /work/mnidgu2s/project1/repo
source pipeline/env.sh
set +e
export TMPDIR=/work/mnidgu2s/project1/.tmp
P=.venvs/submission/bin/python; S=submissions; A=work/analysis; OA=$S/episodes_baseold_k_OTa; PM=outputs/floor/planes_measured.json
[[ -f $A/meshes/E2R/episode_000029.npy ]] || in_container mhr_converter.sif CUDA_VISIBLE_DEVICES=1 python -u /track1/scripts/ensemble_meshes.py --episodes_dir /track1/$S/episodes_refit_E2 --out /track1/$A/meshes/E2R | tail -1
build() { # NAME BETA LAM WEIGHT
  rm -rf $S/episodes_person_$1 $S/episodes_personm_$1 $S/episodes_cand_$1
  $P scripts/floor_root.py --input $S/episodes_refit_E2 --bodies $A/bodies_HE2R --planes $PM --out $S/episodes_person_$1 --lam $3 --beta $2 | tail -1
  $P scripts/human_root_mix.py --base $S/episodes_person_$1 --base_meshes $A/meshes/E2R --other_meshes $A/meshes/a0 --out $S/episodes_personm_$1 --weight $4 | tail -1
  $P scripts/object_transfer.py --people $S/episodes_personm_$1 --people_meshes $A/meshes/E2R --objects $OA --objects_meshes $A/meshes/BMQ3KGM_old --out $S/episodes_cand_$1 | tail -1
}
build P4 0.55 3e4 0.3
build P5 0.55 3e4 0.5
echo P4BUILT
