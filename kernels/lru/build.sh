#!/usr/bin/env bash
# Build librlu.so (LRU expert-cache kernels) for gfx1201 with the ROCm 10 hipcc that ships
# inside the server image.
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
set -euo pipefail
OUT=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/librlu.so}
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/r4d_lru.hip
SDK=${SDK:-/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core}
GFX_ARCH=${GFX_ARCH:-gfx1201}
mkdir -p "$(dirname "$OUT")"
${HIPCC:-/opt/rocm/bin/hipcc} -O3 -std=c++17 -fPIC --offload-arch="$GFX_ARCH" \
  --rocm-device-lib-path="$SDK/lib/llvm/amdgcn/bitcode" \
  -shared "$SRC" -o "$OUT"
echo "built $OUT"
ls -l "$OUT"
nm -D "$OUT" | grep -E "r4d_lru_(manage|gather|fused)" || { echo "MISSING EXPORTS"; exit 1; }
