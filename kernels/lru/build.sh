#!/usr/bin/env bash
# Build librlu.so (LRU expert-cache kernels) for gfx1201 with the ROCm 10 hipcc that ships
# inside the server image. Runs in local/q38fn-rocm10:k1build (= :try1 + g++ + the
# libamdhip64.so link symlink); the resulting .so loads fine in :try1, which needs no
# compiler to dlopen it.
set -euo pipefail
OUT=${1:-/w/k1/lru/librlu.so}
SDK=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core
/opt/rocm/bin/hipcc -O3 -std=c++17 -fPIC --offload-arch=gfx1201 \
  --rocm-device-lib-path="$SDK/lib/llvm/amdgcn/bitcode" \
  -shared /w/k1/lru/r4d_lru.hip -o "$OUT"
rc=$?
if [ $rc -ne 0 ]; then echo "BUILD FAILED rc=$rc"; exit $rc; fi
echo "built $OUT"
ls -l "$OUT"
# All three entry points must be present. A build that silently loses one would show up
# only once a server is already running -- as a dlopen failure, or worse, as the fused
# path quietly never engaging.
for sym in r4d_lru_manage r4d_lru_gather r4d_lru_fused; do
  nm -D "$OUT" | grep -q " T $sym$" || { echo "MISSING EXPORT: $sym"; exit 1; }
done
echo "exports OK: r4d_lru_manage r4d_lru_gather r4d_lru_fused"
