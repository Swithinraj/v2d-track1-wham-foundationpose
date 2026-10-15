#!/bin/bash
# FoundationPose tracking (forward and backward from the prompt's reference frame) and ESKF/RTS
# smoothing for one episode -> outputs/object/episode_X/foundationpose/{raw,smoothed}.
# Usage: track_object.sh EPISODE   (REREGISTER_IOU_THRESHOLD, default 0.25)
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
EPISODE="${1:-}"; require_episode "${EPISODE}"
TAG="$(episode_tag "${EPISODE}")"
OUT="${TRACK1_ROOT}/outputs/object/${TAG}"
OBJECT="$(episode_field "${EPISODE}" object)"
REFERENCE="$(printf '%06d' "$(tr -d '[:space:]' < "${OUT}/prompt/reference_frame.txt")")"
for path in "${TRACK1_ROOT}/inputs/object_meshes/${OBJECT}.glb" "${OUT}/intrinsics/${REFERENCE}.json" \
    "${OUT}/depth/${REFERENCE}.png" "${OUT}/sam2_masks/1/${REFERENCE}.png"; do
  require_file "${path}"
done
mkdir -p "${OUT}/foundationpose/raw" "${OUT}/foundationpose/smoothed"

in_container v2d_cari4d.sif python -m v2d.foundation_pose.lib.run_video_to_poses \
  --video_path "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/${TAG}.mp4" \
  --depth_folder "/track1/outputs/object/${TAG}/depth" \
  --masks_folder "/track1/outputs/object/${TAG}/sam2_masks/1" \
  --camera_intrinsics_path "/track1/outputs/object/${TAG}/intrinsics/${REFERENCE}.json" \
  --mesh_path "/track1/inputs/object_meshes/${OBJECT}.glb" \
  --poses_dir "/track1/outputs/object/${TAG}/foundationpose/raw" \
  --weights_dir /track1/weights/object/foundationpose \
  --backend nvlabs_pytorch \
  --reference_frame "$((10#${REFERENCE}))" \
  --reregister_iou_thresh "${REREGISTER_IOU_THRESHOLD:-0.25}" \
  --register_iteration 10 --track_iteration 5

in_container v2d_cari4d.sif python -m v2d.foundation_pose.lib.run_ekf_smoothing \
  --poses_dir "/track1/outputs/object/${TAG}/foundationpose/raw" \
  --mesh_path "/track1/inputs/object_meshes/${OBJECT}.glb" \
  --intrinsics_path "/track1/outputs/object/${TAG}/intrinsics/${REFERENCE}.json" \
  --weights_dir /track1/weights/object/foundationpose \
  --backend nvlabs_pytorch \
  --masks_folder "/track1/outputs/object/${TAG}/sam2_masks/1" \
  --output_dir "/track1/outputs/object/${TAG}/foundationpose/smoothed"

EXPECTED="$(episode_field "${EPISODE}" length)"
COUNT="$(find "${OUT}/foundationpose/smoothed" -maxdepth 1 -name '*.json' | wc -l)"
[[ "${COUNT}" -eq "${EXPECTED}" ]] || { echo "Expected ${EXPECTED} poses, found ${COUNT}" >&2; exit 4; }
echo "PASS: FoundationPose + RTS smoothing ${TAG}, object=${OBJECT}"
