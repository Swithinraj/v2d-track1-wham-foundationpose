#!/bin/bash
# `apptainer exec` for the submission container, installed as /usr/local/bin/apptainer.
#
# The organizers run `docker run --gpus all` with no extra privileges, where real Apptainer cannot start: it fails with
# "Failed to create user namespace". The pipeline's images are therefore unpacked to /opt/envs/<image> (one directory per
# SIF, unchanged) and run with proot, which needs no privileges. This takes exactly the call in_container() makes in
# pipeline/env.sh, so the pipeline code is the code at the submitted commit.
# ponytail: proot adds a cost per system call; GPU work is unaffected, Python imports and file IO are slower.
set -euo pipefail
ENVS="${APPTAINER_SHIM_ENVS:-/opt/envs}"
[[ "${1:-}" == exec ]] || { echo "apptainer shim: only 'exec' is supported, got: $*" >&2; exit 2; }
shift

ARGS=(-b /dev -b /proc -b /sys -b /etc/resolv.conf)
NV=0; PWD_IN=/
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nv) NV=1; shift ;;
    --no-home|--cleanenv) shift ;;
    --bind) IFS=: read -r src dst _ <<< "$2"; ARGS+=(-b "${src}:${dst:-$src}"); shift 2 ;;
    --pwd) PWD_IN="$2"; shift 2 ;;
    -*) echo "apptainer shim: unsupported option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
IMAGE="$(basename "$1" .sif)"; shift
ROOT="${ENVS}/${IMAGE}"
[[ -d "${ROOT}" ]] || { echo "apptainer shim: no unpacked image ${ROOT}" >&2; exit 3; }

# What --nv does: the driver libraries the NVIDIA runtime injected into Docker, visible at one path inside the image.
LIB=""
if [[ "${NV}" == 1 ]]; then
  LIB=/.nv
  while read -r library; do ARGS+=(-b "${library}:${LIB}/$(basename "${library}")"); done \
    < <(ldconfig -p | grep -E 'lib(nvidia|cuda|nvcuvid|nvoptix)' | awk '{print $NF}' | sort -u)
fi
# The image's own environment, as Apptainer would apply it; --cleanenv: nothing from the outside except HOME-less defaults.
exec env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/tmp PROOT_NO_SECCOMP="${PROOT_NO_SECCOMP:-}" \
  proot -r "${ROOT}" "${ARGS[@]}" -w "${PWD_IN}" /bin/sh -c '
    for f in /.singularity.d/env/*.sh; do [ -f "$f" ] && . "$f"; done
    [ -z "$1" ] || export LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    shift; exec "$@"' sh "${LIB}" "$@"
