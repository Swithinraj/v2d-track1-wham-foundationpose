#!/bin/bash
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:3
#SBATCH --time=01:30:00
#SBATCH --job-name=v2d-p1-c30
#SBATCH --chdir=/work/mnidgu2s/project1/repo
#SBATCH --output=/work/mnidgu2s/project1/logs/c4_c30_%j.log
# CARI4D's CoCoNet person (mesh smoothing lam 300, as N19H) on every episode whose CoCoNet stage exists: the 19 already
# converted plus those run_cari4d_coconet11.sbatch finished. Writes submissions/episodes_c4tp_C30 (measured camera) and
# work/cari4d/c30_episodes.txt for the ensemble job (run_ensemble_variant.sbatch with C4DIR/C4LIST).
set -euo pipefail
export TMPDIR=/work/mnidgu2s/project1/.tmp TRACK1_ROOT=/work/mnidgu2s/project1/repo OMP_NUM_THREADS=4
source pipeline/env.sh
trap 'echo "C30_EXIT status=$? time=$(date -u +%FT%TZ)"' EXIT
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NEW=$(ls work/cari4d/coconet_done 2>/dev/null | sort -n | tr '\n' ' ')
echo "C30_START new episodes: ${NEW}"
for d in submissions/episodes_c4_C30 submissions/episodes_c4tp_C30 work/cari4d/camroot/C30 work/cari4d/select/C30; do
  [[ -e $d ]] && { echo "$d exists; refusing to overwrite"; exit 2; }
done
convert() { local gpu="$1"; shift; for e in "$@"; do bash pipeline/recipes/cari4d_to_params.sh "$e" coconet 300 "$gpu" | tail -1; done; }
read -r -a N <<< "${NEW}"
A=(); B=(); C=(); for i in "${!N[@]}"; do case $((i % 3)) in 0) A+=("${N[$i]}");; 1) B+=("${N[$i]}");; 2) C+=("${N[$i]}");; esac; done
convert "${GPUS[0]}" "${A[@]}" & convert "${GPUS[1]:-${GPUS[0]}}" "${B[@]}" & convert "${GPUS[2]:-${GPUS[0]}}" "${C[@]}" &
wait
X=work/cari4d/export/coconet_l300; P=work/cari4d/params/coconet_l300; SUB=work/cari4d/select/C30
mkdir -p "$SUB/export" "$SUB/params"
EPS=()
for e in $(seq 0 29); do
  t=$(episode_tag "$e")
  [[ -f "$P/$t.npz" && -f "$X/$t.object.npz" ]] || continue
  cp -l "$X/$t.object.npz" "$X/${t}_object.glb" "$SUB/export/"; cp -l "$P/$t.npz" "$SUB/params/"; EPS+=("$e")
done
echo "${EPS[*]}" > work/cari4d/c30_episodes.txt
echo "C30 episodes (${#EPS[@]}): ${EPS[*]}"
"$VENV_PYTHON" scripts/cari4d_episodes.py --export "$SUB/export" --params "$SUB/params" --out submissions/episodes_c4_C30
"$VENV_PYTHON" scripts/cari4d_camroot.py --export "$SUB/export" --k_measured outputs/camera_measured/k_new.json --out work/cari4d/camroot/C30
in_container mhr_converter.sif python -u /track1/scripts/true_cam_params.py --input /track1/submissions/episodes_c4_C30 \
  --out /track1/submissions/episodes_c4tp_C30 --old_root /track1/work/cari4d/camroot/C30 --k_new /track1/work/cari4d/camroot/C30/k_new.json | tail -1
echo C30_BUILT
