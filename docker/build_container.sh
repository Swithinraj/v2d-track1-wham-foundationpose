#!/usr/bin/env bash
# Build, verify and save the submission container from the directory docker/collect_container.py wrote. Needs Docker, the
# NVIDIA container toolkit, zstd, the challenge dataset, and disk for the whole image (see docs/SUBMISSION.md).
#
#   DATASET=/path/to/dataset bash docker/build_container.sh /path/to/container_build
#   REPRODUCE="3,12" DATASET=... bash docker/build_container.sh ...    also reconstruct those episodes on the GPU
#
# DATASET holds track_1/. Writes the tarball and SEND.txt, with the checksum and image id for the report, next to context/.
set -euo pipefail
BUILD=$(realpath "${1:?usage: DATASET=... bash docker/build_container.sh <container_build dir>}")
DATASET=$(realpath "${DATASET:?set DATASET to the challenge dataset root}")
IMAGE=izus_systems-v2d-track1-submission:final
TARBALL=$BUILD/Izus_systems-v2d-track1-submission.tar.zst

docker build -t "$IMAGE" "$BUILD/context"

echo "--- verify (no network)"
docker run --rm --network none -v "$DATASET":/data:ro "$IMAGE" /submission/run.sh verify --dataset-root /data

echo "--- pack (no network)"
OUT=$BUILD/reproduced
mkdir -p "$OUT"
docker run --rm --network none -v "$OUT":/output "$IMAGE" /submission/run.sh pack --label track1 --output /output

if [ -n "${REPRODUCE:-}" ]; then
  echo "--- reproduce episodes $REPRODUCE (no network, no extra privileges)"
  docker run --rm --gpus all --network none -v "$DATASET":/data:ro -v "$OUT":/output "$IMAGE" \
    /submission/run.sh reproduce --label track1 --dataset-root /data --output /output --episodes "$REPRODUCE"
fi

echo "--- save"
docker save "$IMAGE" | zstd -T0 > "$TARBALL"
{
  echo "image   $IMAGE"
  echo "image id $(docker image inspect --format '{{.Id}}' "$IMAGE")"
  echo "tarball $(basename "$TARBALL")"
  echo "sha256  $(sha256sum "$TARBALL" | cut -d' ' -f1)"
} | tee "$BUILD/SEND.txt"
echo "Upload $TARBALL with link sharing on, so it downloads without requesting access, and submit the link plus SEND.txt"
echo "with the technical report."
