#!/bin/bash
# Re-express the GRU training caches in the measured rig camera, group by group (CPU; numpy only).
#
# The training caches (ext/gru_cache_export.py format: <group>_episode_XXXXXX.npz with
# ours / raw / gt / conf / vis / valid / anchor) live in ONE directory shared by all recording-
# disjoint groups, and episode numbers collide across groups — so each group is re-expressed with
# its own k_new (built from the group's episodes_metadata.jsonl labels by build_k_new.py) and the
# others are copied through unchanged (--prefix in true_cam_cache.py).
#
# Groups come from a TSV manifest, one per line:  <group_name><TAB><group_root>
# where <group_root> is the processed pipeline checkout of that group (must contain
# outputs/camera/episode_*/intrinsics.json and data/track_1/meta/episodes_metadata.jsonl).
# The cache files of a group are named <group_name>_episode_XXXXXX.npz in CACHE_DIR.
#
# Output: OUT_CACHE (default TRACK1_ROOT/outputs/gru_cache_tc — INSIDE the repo, because the
# training container mounts only the repo) with every cache re-expressed (tcp mode), plus
# per-group anchor-residual reports true_cam_cache_<group>.json (the gt mapping is
# approximate; a materially larger residual means that group needs a second look).
#
# Usage: prepare_caches.sh GROUPS.tsv   (host python3 needs only numpy; ~minutes)
#   CACHE_DIR    dir with the original caches           (env, required)
#   OUT_CACHE    output dir                             (env, default TRACK1_ROOT/outputs/gru_cache_tc)
#   TRACK1_ROOT  repo root                              (env, default: two levels up from here)
#   HOST_PYTHON  host interpreter                       (env, default python3)
set -euo pipefail
TRACK1_ROOT="${TRACK1_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CACHE_DIR="${CACHE_DIR:?set CACHE_DIR to the dir holding <group>_episode_XXXXXX.npz caches}"
OUT_CACHE="${OUT_CACHE:-${TRACK1_ROOT}/outputs/gru_cache_tc}"
GROUPS="${1:?usage: prepare_caches.sh GROUPS.tsv (group_name<TAB>group_root per line)}"
PY="${HOST_PYTHON:-python3}"
"${PY}" -c "import numpy" 2>/dev/null || { echo "HOST_PYTHON needs numpy (or set HOST_PYTHON)" >&2; exit 3; }

[[ -d "${CACHE_DIR}" ]] || { echo "CACHE_DIR ${CACHE_DIR} missing" >&2; exit 3; }
mkdir -p "${OUT_CACHE}"

n_ok=0; n_group=0
while IFS=$'\t' read -r GROUP ROOT; do
  [[ -z "${GROUP}" || "${GROUP}" =~ ^# ]] && continue
  n_group=$((n_group + 1))
  echo "== group ${GROUP} (${ROOT})" >&2
  require() { [[ -s "$1" ]] || { echo "  missing $1 — fix GROUPS.tsv for ${GROUP}" >&2; return 1; }; }
  # per-episode MoGe intrinsics of the group (the OLD camera)
  CAM_DIR="${ROOT}/outputs/camera"
  META="${ROOT}/data/track_1/meta/episodes_metadata.jsonl"
  n_cache=$(ls "${CACHE_DIR}/${GROUP}_episode_"*.npz 2>/dev/null | wc -l)
  if [[ "${n_cache}" -eq 0 ]]; then echo "  no caches named ${GROUP}_episode_*.npz in ${CACHE_DIR}" >&2; exit 3; fi
  if ! require "${CAM_DIR}/episode_000000/intrinsics.json" || ! require "${META}"; then exit 3; fi
  "${TRACK1_ROOT}/pipeline/gru_retrain/build_k_new.py" --camera "${CAM_DIR}" --meta "${META}" \
    --mode tcp --out "${OUT_CACHE}/k_new_${GROUP}.json"
  "${PY}" "${TRACK1_ROOT}/scripts/true_cam_cache.py" \
    --cache "${CACHE_DIR}" --out "${OUT_CACHE}" --old "${CAM_DIR}" \
    --k_new "${OUT_CACHE}/k_new_${GROUP}.json" --prefix "${GROUP}"
  n_ok=$((n_ok + 1))
done < "${GROUPS}"

echo "== copied the groups NOT in the manifest unchanged" >&2
# any cache file of an unlisted group was already copied through by --prefix runs; verify counts
a=$(ls "${CACHE_DIR}"/*_episode_??????.npz 2>/dev/null | wc -l)
b=$(ls "${OUT_CACHE}"/*_episode_??????.npz 2>/dev/null | wc -l)
[[ "${a}" -eq "${b}" ]] || { echo "count mismatch: ${a} in / ${b} out" >&2; exit 3; }
echo "PASS: ${n_ok}/${n_group} groups re-expressed into ${OUT_CACHE} (${b} caches); per-group anchor residuals in ${OUT_CACHE}/true_cam_cache_<group>.json"
