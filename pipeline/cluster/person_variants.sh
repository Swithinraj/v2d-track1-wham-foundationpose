#!/bin/bash
# Person-only variants on the measured camera, carrying OTa's objects. Host CPU: the anchor frame never moves.
cd /work/mnidgu2s/project1/repo
P=.venvs/submission/bin/python; S=submissions; A=work/analysis; PM=outputs/floor/planes_measured.json; OA=$S/episodes_baseold_k_OTa
variant() {  # NAME BETA LAM [OTHER WEIGHT]...  (root mixes add up: each is measured against the same base meshes)
  local n=$1 b=$2 l=$3 src next k=0; shift 3
  rm -rf $S/episodes_person_$n $S/episodes_personm${n}_* $S/episodes_cand_$n
  $P scripts/floor_root.py --input $S/episodes_cand_Mtp_in --bodies $A/bodies_Mtp_in --planes $PM --out $S/episodes_person_$n --lam $l --beta $b 2>&1 | tail -1 | cut -c1-80
  src=$S/episodes_person_$n
  while (( $# >= 2 )); do
    next=$S/episodes_personm${n}_$k
    $P scripts/human_root_mix.py --base $src --base_meshes $A/meshes/BMQ3KGM --other_meshes $A/meshes/$1 --out $next --weight $2 2>&1 | tail -1 | cut -c1-90
    src=$next; k=$((k + 1)); shift 2
  done
  $P scripts/object_transfer.py --people $src --people_meshes $A/meshes/BMQ3KGM --objects $OA --objects_meshes $A/meshes/BMQ3KGM_old --out $S/episodes_cand_$n 2>&1 | tail -1 | cut -c1-80
}
[[ $# -gt 0 ]] && "$@"
