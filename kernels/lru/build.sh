#!/usr/bin/env bash
# Build librlu.so (LRU expert-cache kernels) for gfx1201 with the ROCm 10 hipcc that ships
# inside the server image.
#
# One .so carries all three entry points -- r4d_lru_manage, r4d_lru_gather and
# r4d_lru_fused -- so VLLM_R4D_LRU=1 and VLLM_R4D_LRU_FUSE=1 both work against it.
#
# Run it in the *build* image (docker/Dockerfile.build = the runtime image + g++ + the
# libamdhip64.so link symlink). The resulting .so loads fine in the runtime image, which
# needs no compiler to dlopen it:
#
#   docker build -t q38fn-rocm10:build -f docker/Dockerfile.build docker/
#   docker run --rm -v "$PWD:/repo" --entrypoint bash q38fn-rocm10:build \
#     -c '/repo/kernels/lru/build.sh /repo/build/kernels/librlu.so'
#
# The runtime image's /opt/rocm is a symlink to the pip ROCm 10 SDK; SDK below must match.
# Pass a second argument to build a different source (e.g. r4d_lru_pre_victim.hip, which
# tests/lru/test_victim_equiv.py needs as its OLD_LIB).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-$HERE/librlu.so}
SRC=${2:-$HERE/r4d_lru.hip}
SDK=${SDK:-/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core}
GFX_ARCH=${GFX_ARCH:-gfx1201}
mkdir -p "$(dirname "$OUT")"
${HIPCC:-/opt/rocm/bin/hipcc} -O3 -std=c++17 -fPIC --offload-arch="$GFX_ARCH" \
  --rocm-device-lib-path="$SDK/lib/llvm/amdgcn/bitcode" \
  -shared "$SRC" -o "$OUT"
echo "built $OUT"
ls -l "$OUT"
# All three entry points must be present. A build that silently loses one would show up
# only once a server is already running -- as a dlopen failure, or worse, as the fused
# path quietly never engaging.
for sym in r4d_lru_manage r4d_lru_gather r4d_lru_fused; do
  nm -D "$OUT" | grep -q " T $sym$" || { echo "MISSING EXPORT: $sym"; exit 1; }
done
echo "exports OK: r4d_lru_manage r4d_lru_gather r4d_lru_fused"
