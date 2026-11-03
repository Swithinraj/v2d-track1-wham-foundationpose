#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --job-name=v2d-p1-ensvar
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/ensvar_%x_%j.log
# A variant of P4C (run_e3_ensemble.sbatch, CD-H 13.10770): ensemble weights W="BMQ3KGM ktcp c4" and mesh smoothing LAM,
# then P4's recipe and the table objects at TABLE about the camera. Usage: NAME=.. W="a b c" LAM=.. TABLE=.. sbatch this.
# ENS=existing ensemble dir name to reuse (skips the ensemble step when only LAM changes).
# C4DIR / C4LIST (or C4LIST_FILE): CARI4D person directory and its episodes (default the 19 of episodes_c4tp_N19H). FACTORS: JSON of
# per-episode object scales (object_camera_scale.py --factors) used instead of TABLE.
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo OMP_NUM_THREADS=4
export TRACK1_EXTRA_BINDS=/work/mnidgu2s/project1/vpq-vitpose
source pipeline/env.sh
trap 'echo "ENSVAR_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
[[ -n "${C4LIST_FILE:-}" && -z "${C4LIST:-}" ]] && C4LIST="$(cat "$C4LIST_FILE")"
C4="${C4LIST:-3 9 10 12 13 14 15 16 17 19 21 22 23 24 25 26 27 28 29}"; C4DIR="${C4DIR:-episodes_c4tp_N19H}"; TAB="1 3 4 6 12 13 14 15 16 18 19 20 22"
S=submissions; A=work/analysis; T=/track1; CODE=/work/mnidgu2s/project1/vpq-vitpose
read -r W1 W2 W3 <<< "$W"
echo "ENSVAR_START name=$NAME w=$W lam=$LAM table=${TABLE:-} factors=${FACTORS:-} c4=$C4DIR ens=${ENS:-new} commit=$(git -C $CODE rev-parse HEAD)"
run() { in_container mhr_converter.sif env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" OMP_NUM_THREADS=4 python -u "$@"; }
ENSDIR=${ENS:-episodes_ens_$NAME}
if [[ -z "${ENS:-}" ]]; then
  run $T/scripts/human_ensemble.py --out $T/$S/$ENSDIR --member BMQ3KGM=$T/$S/episodes_cand_Mtp_in:$W1 \
    --member ktcp=$T/$S/episodes_base_ktcp:$W2 --member c4=$T/$S/$C4DIR:$W3 --episodes $C4 | tail -1
fi
run $T/scripts/human_refit.py --input $T/$S/$ENSDIR --out $T/$S/episodes_refit_$NAME --lam $LAM | tail -1
run $T/scripts/floor_bodies.py --episodes_dir $T/$S/episodes_refit_$NAME --out $T/$A/bodies_$NAME | tail -1
run $T/scripts/ensemble_meshes.py --episodes_dir $T/$S/episodes_refit_$NAME --out $T/$A/meshes/$NAME | tail -1
# The 11 episodes CARI4D did not finish: E2 refitted with the same LAM (E2R for 100, E2L300 for 300).
case "$LAM" in 100) R=episodes_refit_E2; B=bodies_HE2R; M=E2R;; 300) R=episodes_refit_E2L300; B=bodies_HE2L300; M=E2L300;; *) echo "no E2 refit for lam $LAM"; exit 2;; esac
mkdir -p $S/episodes_refit_${NAME}all $A/bodies_${NAME}all $A/meshes/${NAME}all
for e in $(seq 0 29); do
  t=$(printf 'episode_%06d' "$e")
  if [[ -f $S/episodes_refit_$NAME/$t.npz ]]; then src=$S/episodes_refit_$NAME; b=$A/bodies_$NAME; m=$A/meshes/$NAME; else src=$S/$R; b=$A/$B; m=$A/meshes/$M; fi
  cp -l "$src/$t.npz" "$src/${t}_object.glb" $S/episodes_refit_${NAME}all/; cp -l "$b/$t.npz" $A/bodies_${NAME}all/; cp -l "$m/$t.npy" $A/meshes/${NAME}all/
done
"$VENV_PYTHON" scripts/floor_root.py --input $S/episodes_refit_${NAME}all --bodies $A/bodies_${NAME}all --planes outputs/floor/planes_measured.json \
  --out $S/episodes_person_$NAME --lam 3e4 --beta 0.55 | tail -1
"$VENV_PYTHON" scripts/human_root_mix.py --base $S/episodes_person_$NAME --base_meshes $A/meshes/${NAME}all --other_meshes $A/meshes/a0 \
  --out $S/episodes_personm_$NAME --weight 0.3 | tail -1
"$VENV_PYTHON" scripts/object_transfer.py --people $S/episodes_personm_$NAME --people_meshes $A/meshes/${NAME}all --objects $S/episodes_baseold_k_OTa \
  --objects_meshes $A/meshes/BMQ3KGM_old --out $S/episodes_cand_$NAME | tail -1
if [[ -n "${FACTORS:-}" ]]; then
  "$VENV_PYTHON" $CODE/scripts/object_camera_scale.py --input $S/episodes_cand_$NAME --out $S/episodes_cand_${NAME}tab --factors "$FACTORS" | tail -1
  cp "$FACTORS" $S/episodes_cand_${NAME}tab/factors.json
else
  "$VENV_PYTHON" $CODE/scripts/object_camera_scale.py --input $S/episodes_cand_$NAME --out $S/episodes_cand_${NAME}tab --scale $TABLE --episodes $TAB | tail -1
fi
echo ENSVAR_BUILT
