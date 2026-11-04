#!/bin/bash
# All episodes through CARI4D on the GPUs of ONE allocation. Workers claim episodes atomically, and an episode starts once its
# masks have fully arrived, so the run can begin while masks are still restoring. Finished stages are reused on a rerun.
# Usage (inside one allocation): cari4d_all.sh [GPUS (default 4)] [WORKERS_PER_GPU (default 2)]
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
set +e
cd "${TRACK1_ROOT}"
GPUS="${1:-4}"; PER_GPU="${2:-2}"; Q="work/cari4d/queue"; mkdir -p "${Q}"
mapfile -t ORDER < <("${VENV_PYTHON}" - 2>/dev/null <<'PY'
import json
meta = [json.loads(line) for line in open("data/track_1/meta/episodes.jsonl")]
for row in sorted(meta, key=lambda r: r["length"]):      # shortest first: early results, early warning
    print(row["episode_index"], row["length"])
PY
)
ready() {   # EPISODE LENGTH
  local tag; tag="$(episode_tag "$1")"
  [[ "$(find "outputs/object/${tag}/sam2_masks/1" -name '*.png' 2>/dev/null | wc -l)" == "$2" ]] &&
    [[ "$(find "outputs/person/${tag}/sam2_masks/0" -name '*.png' 2>/dev/null | wc -l)" == "$2" ]]
}
worker() {  # GPU
  local gpu="$1" idle=0 line episode length claimed
  while (( idle < 720 )); do
    claimed=0
    for line in "${ORDER[@]}"; do
      read -r episode length <<< "${line}"
      [[ -f "${Q}/${episode}.done" || -d "${Q}/${episode}.claim" ]] && continue
      ready "${episode}" "${length}" || continue
      mkdir "${Q}/${episode}.claim" 2>/dev/null || continue
      claimed=1; idle=0
      if bash pipeline/recipes/cari4d_episode.sh "${episode}" "${gpu}" > "logs/c4_${episode}.log" 2>&1; then
        touch "${Q}/${episode}.done"; echo "DONE ${episode} gpu ${gpu} $(date -u +%H:%M)"
      else
        touch "${Q}/${episode}.failed"; echo "FAILED ${episode} gpu ${gpu} $(date -u +%H:%M)"
      fi
      break
    done
    # Stop when nothing is left to claim; otherwise wait for masks.
    (( $(ls "${Q}" | grep -c '\.claim$') >= ${#ORDER[@]} )) && break
    (( claimed == 0 )) && { sleep 30; idle=$((idle + 1)); }
  done
}
for k in $(seq 0 $((GPUS - 1))); do
  for w in $(seq 1 "${PER_GPU}"); do worker "${k}" & sleep 7; done
done
wait
echo "ALL WORKERS FINISHED: $(ls "${Q}" | grep -c '\.done$') done, $(ls "${Q}" | grep -c '\.failed$') failed"
