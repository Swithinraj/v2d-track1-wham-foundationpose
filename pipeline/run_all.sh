#!/bin/bash
# Run the whole Track 1 pipeline on this machine, stage by stage, in dependency order.
# Prerequisite: the one-time setup in README.md (sources, containers, data, weights incl. the gated SAM 3D Body and
# SAM 3D Objects models, venvs). Completed per-episode/per-object outputs are skipped, so the script can be re-run
# after an interruption. On a cluster, submit the same stage scripts as one task per episode (or object) instead;
# the order below is the dependency order.
#
# Usage: pipeline/run_all.sh            (COMMIT_URL=... for an uploadable Parquet)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
STAGES="${TRACK1_ROOT}/pipeline/stages"
EPISODES=$(seq 0 29)

done_count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l; }

for e in ${EPISODES}; do
  tag="$(episode_tag "${e}")"
  frames="$(episode_field "${e}" length)"
  out="${TRACK1_ROOT}/outputs/object/${tag}"
  [[ "$(done_count "${out}/depth" '*.png')" -eq "${frames}" ]] || bash "${STAGES}/object_depth.sh" "${e}"
  [[ -s "${out}/mask_audit.json" ]] || bash "${STAGES}/object_masks.sh" "${e}"
  [[ "$(done_count "${TRACK1_ROOT}/outputs/person/${tag}/sam2_masks/0" '*.png')" -eq "${frames}" ]] \
    || bash "${STAGES}/person.sh" "${e}"
  [[ -s "${TRACK1_ROOT}/outputs/mediapipe/${tag}.npz" ]] || bash "${STAGES}/keypoints2d.sh" "${e}"
  [[ -s "${TRACK1_ROOT}/outputs/hands/${tag}.npz" ]] || bash "${STAGES}/sam3d_body.sh" "${e}"
done

while read -r object; do
  [[ -s "${TRACK1_ROOT}/inputs/object_meshes/${object}.glb" ]] || bash "${STAGES}/object_mesh.sh" "${object}"
done < <(all_objects)

for e in ${EPISODES}; do
  tag="$(episode_tag "${e}")"
  frames="$(episode_field "${e}" length)"
  [[ "$(done_count "${TRACK1_ROOT}/outputs/object/${tag}/foundationpose/smoothed" '*.json')" -eq "${frames}" ]] \
    || bash "${STAGES}/track_object.sh" "${e}"
  [[ "$(done_count "${TRACK1_ROOT}/outputs/object/${tag}/foundationpose/symmetric" '*.json')" -eq "${frames}" ]] \
    || bash "${STAGES}/track_symmetric.sh" "${e}"     # no-op unless the object is in config/symmetric_objects.json
done

bash "${STAGES}/pack.sh"
