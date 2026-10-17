#!/bin/bash
# Optional: SAM 3D Body weights for the experimental human branch (gated repo; request access on
# Hugging Face first). HF_TOKEN or HF_TOKEN_FILE must be set; the token is never printed.
# Layout (what the toolkit's v2d_sam3d_body expects):
#   weights/sam3d_body/sam-3d-body-dinov3/{model.ckpt, model_config.yaml, assets/mhr_model.pt}
#   weights/sam3d_body/moge-2-vitb-normal/model.pt
#   weights/sam3d_body/torch_home/hub/facebookresearch_dinov3_main   (DINOv3 code for torch.hub)
# assets/mhr_model.pt is the same file as weights/mhr/assets/mhr_model.pt (MHR_MODEL_SHA256) and is
# copied from there when present.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
TOKEN="${HF_TOKEN:-}"
[[ -n "${TOKEN}" || -z "${HF_TOKEN_FILE:-}" ]] || TOKEN="$(tr -d '\r\n' < "${HF_TOKEN_FILE}")"
[[ -n "${TOKEN}" ]] || { echo "Set HF_TOKEN or HF_TOKEN_FILE (gated repo ${SAM3D_BODY_REPO})" >&2; exit 2; }
W="${TRACK1_ROOT}/weights/sam3d_body"
mkdir -p "${W}/sam-3d-body-dinov3/assets" "${W}/moge-2-vitb-normal" "${W}/torch_home/hub"

hf_get() {  # hf_get REPO REVISION FILE DEST
  [[ -s "$4" ]] && return 0
  curl -sSfL -H "Authorization: Bearer ${TOKEN}" -o "$4.part" "https://huggingface.co/$1/resolve/$2/$3"
  mv "$4.part" "$4"
}
hf_get "${SAM3D_BODY_REPO}" "${SAM3D_BODY_REVISION}" model_config.yaml "${W}/sam-3d-body-dinov3/model_config.yaml"
hf_get "${SAM3D_BODY_REPO}" "${SAM3D_BODY_REVISION}" model.ckpt "${W}/sam-3d-body-dinov3/model.ckpt"
MHR="${W}/sam-3d-body-dinov3/assets/mhr_model.pt"
if [[ ! -s "${MHR}" && -s "${TRACK1_ROOT}/weights/mhr/assets/mhr_model.pt" ]]; then
  cp "${TRACK1_ROOT}/weights/mhr/assets/mhr_model.pt" "${MHR}"
fi
hf_get "${SAM3D_BODY_REPO}" "${SAM3D_BODY_REVISION}" assets/mhr_model.pt "${MHR}"
echo "${MHR_MODEL_SHA256}  ${MHR}" | sha256sum -c --quiet
hf_get "${MOGE2_VITB_REPO}" "${MOGE2_VITB_REVISION}" model.pt "${W}/moge-2-vitb-normal/model.pt"
D="${W}/torch_home/hub/facebookresearch_dinov3_main"
if [[ ! -s "${D}/hubconf.py" ]]; then
  git clone "${DINOV3_GIT_URL}" "${D}"
  git -C "${D}" checkout -q "${DINOV3_GIT_COMMIT}"
fi
for f in sam-3d-body-dinov3/model.ckpt sam-3d-body-dinov3/model_config.yaml sam-3d-body-dinov3/assets/mhr_model.pt \
         moge-2-vitb-normal/model.pt torch_home/hub/facebookresearch_dinov3_main/hubconf.py; do
  require_file "${W}/${f}"
done
echo "PASS: SAM 3D Body weights in weights/sam3d_body (revision ${SAM3D_BODY_REVISION})"
