#!/bin/bash
# Generated metric mesh for one object -> inputs/object_meshes/<object>.glb (+ .provenance.json).
#   1. scripts/trellis_object_mesh.py  clean-view selection + TRELLIS candidates (each episode's best
#                                      view alone and all three together, SEEDS) + depth-extent scale
#   2. scripts/select_object_mesh.py   FoundationPose fit in all three cameras picks the candidate and
#                                      refines its scale within 0.75-1.33x
# Needs all three episodes' depth and masks. Usage: object_mesh.sh OBJECT
#   SEEDS="1 2"; SKIP_GENERATION=1 re-runs only the selection on existing candidates.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
OBJECT="${1:-}"
all_objects | grep -qx "${OBJECT}" || { echo "Object must be one of: $(all_objects | tr '\n' ' ')" >&2; exit 2; }
require_file "${TRACK1_ROOT}/weights/trellis/TRELLIS-image-large/pipeline.json" "run pipeline/setup/download_weights.sh"

if [[ "${SKIP_GENERATION:-0}" != 1 ]]; then
  in_container v2d_sam3d.sif TORCH_HOME=/track1/weights/trellis/torch_home HF_HUB_OFFLINE=1 \
    ATTN_BACKEND=flash_attn SPCONV_ALGO=native \
    python -u /track1/scripts/trellis_object_mesh.py --object "${OBJECT}" --seeds ${SEEDS:-1 2}
fi
in_container v2d_cari4d.sif python -u /track1/scripts/select_object_mesh.py --object "${OBJECT}"
