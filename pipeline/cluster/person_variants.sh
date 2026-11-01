#!/bin/bash
# Person-only variants on the measured camera, carrying OTa's objects. Host CPU: the anchor frame never moves.
cd /work/mnidgu2s/project1/repo
P=.venvs/submission/bin/python; S=submissions; A=work/analysis; PM=outputs/floor/planes_measured.json; OA=$S/episodes_baseold_k_OTa
variant() {  # NAME BETA LAM [OTHER_MESHES WEIGHT]
  local n=$1 b=$2 l=$3 other=${4:-} w=${5:-0} src
  rm -rf $S/episodes_person_$n $S/episodes_personm_$n $S/episodes_cand_$n
  $P scripts/floor_root.py --input $S/episodes_cand_Mtp_in --bodies $A/bodies_Mtp_in --planes $PM --out $S/episodes_person_$n --lam $l --beta $b 2>&1 | tail -1 | cut -c1-80
  src=$S/episodes_person_$n
  if [[ -n "$other" ]]; then
    $P scripts/human_root_mix.py --base $src --base_meshes $A/meshes/BMQ3KGM --other_meshes $other --out $S/episodes_personm_$n --weight $w 2>&1 | tail -1 | cut -c1-90
    src=$S/episodes_personm_$n
  fi
  $P scripts/object_transfer.py --people $src --people_meshes $A/meshes/BMQ3KGM --objects $OA --objects_meshes $A/meshes/BMQ3KGM_old --out $S/episodes_cand_$n 2>&1 | tail -1 | cut -c1-80
}
variant V1 0.55 3e4
variant V2 0.75 3e4
variant V3 0.55 1e4 $A/meshes/a0 0.3
variant V4 0.55 1e4 $A/meshes/ktcp 0.5
echo VARIANTSBUILT
