#!/usr/bin/env bash
# Same recipe as k1/lru/build.sh: hipcc from the ROCm 10 SDK inside local/q38fn-rocm10:k1build.
set -euo pipefail
OUT=${1:-/w/k1/hotprobe/libmoeprobe.so}
SDK=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core
/opt/rocm/bin/hipcc -O3 -std=c++17 -fPIC --offload-arch=gfx1201 \
  --rocm-device-lib-path="$SDK/lib/llvm/amdgcn/bitcode" \
  -shared /w/k1/hotprobe/moe_read_probe.hip -o "$OUT"
nm -D "$OUT" | grep -q " T moe_read_probe$" || { echo "MISSING EXPORT"; exit 1; }
echo "built $OUT"; ls -l "$OUT"
