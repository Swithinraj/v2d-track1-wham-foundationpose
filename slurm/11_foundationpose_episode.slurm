#!/bin/bash
#SBATCH --job-name=v2d_t1_fp
#SBATCH --partition=gpu
#SBATCH --exclude=wr14,wr15,wr16,wr17,wr18,wr19,wr20,wr21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/work/smoses2s/nvidea_chal/TRACK1/logs/foundationpose_%A_%a.out
#SBATCH --error=/work/smoses2s/nvidea_chal/TRACK1/logs/foundationpose_%A_%a.err

set -euo pipefail
ROOT=/work/smoses2s/nvidea_chal/TRACK1
if [[ -n "${EPISODE:-}" ]]; then
  EPISODE="${EPISODE}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  EPISODE="$(( ${EPISODE_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))"
else
  EPISODE=""
fi
[[ "${EPISODE}" =~ ^[0-9]+$ ]] && ((EPISODE >= 0 && EPISODE <= 29)) || {
  echo "Set EPISODE to an integer from 0 through 29." >&2; exit 2;
}
PADDED="$(printf '%06d' "${EPISODE}")"
OUT="${ROOT}/outputs/object/episode_${PADDED}"
OBJECT="$(python3 -c "import json; print(next(x['object'] for x in map(json.loads, open('${ROOT}/data/track_1/meta/episodes_metadata.jsonl')) if x['episode_index']==${EPISODE}))")"
MESH_ROOT="${MESH_ROOT:-${ROOT}/inputs/object_meshes}"
MESH="${MESH_ROOT}/${OBJECT}.glb"
REFERENCE="$(tr -d '[:space:]' < "${OUT}/prompt/reference_frame.txt")"
INTRINSICS="${OUT}/intrinsics/$(printf '%06d' "${REFERENCE}").json"
WEIGHTS="${ROOT}/weights/object/foundationpose"
POSES="${OUT}/foundationpose/raw"
SMOOTHED="${OUT}/foundationpose/smoothed"

for path in "${MESH}" "${INTRINSICS}" "${OUT}/depth/$(printf '%06d' "${REFERENCE}").png" \
  "${OUT}/sam2_masks/1/$(printf '%06d' "${REFERENCE}").png"; do
  [[ -s "${path}" ]] || { echo "Missing required FoundationPose input: ${path}" >&2; exit 3; }
done
mkdir -p "${POSES}" "${SMOOTHED}"

apptainer exec --nv --no-home --cleanenv \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_cari4d.sif" \
  env HOME=/tmp MPLCONFIGDIR=/tmp/mpl python -m v2d.foundation_pose.lib.run_video_to_poses \
  --video_path "/track1/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_${PADDED}.mp4" \
  --depth_folder "/track1/outputs/object/episode_${PADDED}/depth" \
  --masks_folder "/track1/outputs/object/episode_${PADDED}/sam2_masks/1" \
  --camera_intrinsics_path "/track1/outputs/object/episode_${PADDED}/intrinsics/$(printf '%06d' "${REFERENCE}").json" \
  --mesh_path "/track1/inputs/object_meshes/${OBJECT}.glb" \
  --poses_dir "/track1/outputs/object/episode_${PADDED}/foundationpose/raw" \
  --weights_dir /track1/weights/object/foundationpose \
  --backend nvlabs_pytorch \
  --reference_frame "${REFERENCE}" \
  --reregister_iou_thresh "${REREGISTER_IOU_THRESHOLD:-0.25}" \
  --register_iteration 10 --track_iteration 5

apptainer exec --nv --no-home --cleanenv \
  --bind "${ROOT}:/track1" "${ROOT}/containers/v2d_cari4d.sif" \
  env HOME=/tmp MPLCONFIGDIR=/tmp/mpl python -m v2d.foundation_pose.lib.run_ekf_smoothing \
  --poses_dir "/track1/outputs/object/episode_${PADDED}/foundationpose/raw" \
  --mesh_path "/track1/inputs/object_meshes/${OBJECT}.glb" \
  --intrinsics_path "/track1/outputs/object/episode_${PADDED}/intrinsics/$(printf '%06d' "${REFERENCE}").json" \
  --weights_dir /track1/weights/object/foundationpose \
  --backend nvlabs_pytorch \
  --masks_folder "/track1/outputs/object/episode_${PADDED}/sam2_masks/1" \
  --output_dir "/track1/outputs/object/episode_${PADDED}/foundationpose/smoothed"

EXPECTED="$(python3 -c "import json; print(next(x['length'] for x in map(json.loads, open('${ROOT}/data/track_1/meta/episodes.jsonl')) if x['episode_index']==${EPISODE}))")"
COUNT="$(find "${SMOOTHED}" -maxdepth 1 -name '*.json' | wc -l)"
[[ "${COUNT}" -eq "${EXPECTED}" ]] || { echo "Expected ${EXPECTED} poses, found ${COUNT}" >&2; exit 4; }
echo "PASS: FoundationPose + RTS smoothing episode ${PADDED}, object=${OBJECT}"
