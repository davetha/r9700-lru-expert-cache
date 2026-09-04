#!/bin/bash
# Degraded-mount smoke: mount ONLY the two files that cross-reference other patched
# files, so their guarded imports must take the fallback instead of raising.
set -e
MOUNTS=$(grep -E 'indexer_qsa.py|hotcold/r4d_mxfp4_moe.py' $REPO_ROOT/patches/MOUNTS.txt | tr '\n' ' ')
echo "mounts: $MOUNTS"
flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --entrypoint python3 \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  -e HIP_VISIBLE_DEVICES=1 -e VLLM_LOGGING_LEVEL=INFO \
  $MOUNTS -v $REPO_ROOT/k4:/k4 \
  local/q38fn-rocm10:try1 /k4/smoke_noguard.py
