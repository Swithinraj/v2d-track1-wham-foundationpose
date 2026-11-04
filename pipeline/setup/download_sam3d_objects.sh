#!/bin/bash
# SAM 3D Objects weights (gated: request access at https://huggingface.co/facebook/sam-3d-objects) in the layout of the
# toolkit's downloader -> weights/sam3d_objects/{hf-download, hf_home (MoGe v1), torch_home/hub/checkpoints (DINOv2)}.
# HF_TOKEN or HF_TOKEN_FILE must hold a token of an account with approved access (never printed). The MoGe v1 and DINOv2
# files are fetched outside v2d_sam3d.sif: its `hf` CLI fails on compressed responses ("process() takes no keyword
# arguments" in httpx's decoder).
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
TOKEN="${HF_TOKEN:-}"
[[ -z "${TOKEN}" && -n "${HF_TOKEN_FILE:-}" ]] && TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
[[ -n "${TOKEN}" ]] || { echo "Set HF_TOKEN or HF_TOKEN_FILE (gated repo facebook/sam-3d-objects)" >&2; exit 2; }
W="${TRACK1_ROOT}/weights/sam3d_objects"
mkdir -p "${W}/hf-download" "${W}/hf_home/hub" "${W}/torch_home/hub/checkpoints"
W="$(cd "${W}" && pwd -P)"   # bind the real directory (weights/ may hold links)

if [[ ! -s "${W}/hf-download/checkpoints/pipeline.yaml" ]]; then
  APPTAINERENV_HF_TOKEN="${TOKEN}" apptainer exec --no-home --cleanenv --bind "${W}:/weights" "${CONTAINERS}/v2d_sam3d.sif" \
    env HOME=/tmp hf download facebook/sam-3d-objects --revision "${SAM3D_OBJECTS_REVISION}" --local-dir /weights/hf-download
fi
apptainer exec --no-home --cleanenv --bind "${W}:/weights" "${CONTAINERS}/v2d_cari4d.sif" env HOME=/tmp HF_HUB_DISABLE_XET=1 \
  python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Ruicheng/moge-vitl', revision='${MOGE1_VITL_REVISION}', cache_dir='/weights/hf_home/hub'))"
for name in dinov2_vitl14_reg4_pretrain.pth dinov2_vitb14_reg4_pretrain.pth; do
  model="${name%%_reg4*}"
  [[ -s "${W}/torch_home/hub/checkpoints/${name}" ]] || curl -fsSL -o "${W}/torch_home/hub/checkpoints/${name}" \
    "https://dl.fbaipublicfiles.com/dinov2/${model}/${name}"
done
D="${W}/torch_home/hub/facebookresearch_dinov2_main"   # the code torch.hub.load('facebookresearch/dinov2') uses
if [[ ! -s "${D}/hubconf.py" ]]; then
  git clone -q "${DINOV2_GIT_URL}" "${D}"
  git -C "${D}" checkout -q "${DINOV2_GIT_COMMIT}"
fi
require_file "${D}/hubconf.py"
require_file "${W}/hf-download/checkpoints/pipeline.yaml"
require_file "${W}/hf-download/checkpoints/slat_generator.ckpt"
ls "${W}"/hf_home/hub/models--Ruicheng--moge-vitl/snapshots/*/model.pt >/dev/null
require_file "${W}/torch_home/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
echo "PASS: SAM 3D Objects weights in weights/sam3d_objects"
