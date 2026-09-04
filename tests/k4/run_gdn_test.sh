#!/bin/bash
set -e
MOUNTS=$(tr "\n" " " < $REPO_ROOT/patches/MOUNTS.txt)
flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --entrypoint python3 \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  -e HIP_VISIBLE_DEVICES=1 -e VLLM_LOGGING_LEVEL=WARNING \
  $MOUNTS -v $REPO_ROOT/k4:/k4 \
  local/q38fn-rocm10:try1 "$@"
