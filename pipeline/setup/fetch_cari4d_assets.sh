#!/bin/bash
# The two Git LFS assets the CARI4D image holds only as pointers, from the upstream commit the image was built from.
# The pointers' sha256 values are recorded here, so a changed download fails.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; D="${ROOT}/weights/cari4d/lib_mhr_assets"; mkdir -p "${D}"
COMMIT=193382a184308f6800331f3db829f76dc4294cfe
BASE="https://media.githubusercontent.com/media/nvidia-isaac/video_to_data/${COMMIT}/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/assets"
while read -r sha name; do
  curl -sSL -o "${D}/${name}" "${BASE}/${name}"
  echo "${sha}  ${D}/${name}" | sha256sum -c -
done <<LIST
a026fe82599609ee82814b4c18eb0335fa70925740dea5e1c1e1c0930d5edcf7 mhr_collision_proxy_4000v.npz
65e467ae534281c8c5370b76d672c80cf99b95bc73af9c3ac64d5bea6c7f60c8 mhr_hand_surface_spec.npz
LIST
