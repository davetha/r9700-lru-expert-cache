#!/usr/bin/env bash
# Build libhcqfp8sk.so (small-M fp8 block-scaled skinny GEMM) for gfx1201 with the ROCm 10
# hipcc that ships inside the server image. Same recipe as kernels/lru/build.sh.
#
#   docker run --rm -v "$PWD:/repo" --entrypoint bash q38fn-rocm10:build \
#     -c '/repo/kernels/experimental/fp8skinny/build.sh /repo/build/kernels/libhcqfp8sk.so'
#
# EXPERIMENTAL. This kernel is correct but not worth shipping on -- 1.01x in isolation and a
# dead heat in the running server. See README.md before spending time on it.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-$HERE/libhcqfp8sk.so}
SRC=${2:-$HERE/hcq_fp8skinny.hip}
SDK=${SDK:-/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core}
GFX_ARCH=${GFX_ARCH:-gfx1201}
mkdir -p "$(dirname "$OUT")"
${HIPCC:-/opt/rocm/bin/hipcc} -O3 -std=c++17 -fPIC --offload-arch="$GFX_ARCH" \
  --rocm-device-lib-path="$SDK/lib/llvm/amdgcn/bitcode" \
  -shared "$SRC" -o "$OUT"
echo "built $OUT"
ls -l "$OUT"
for sym in hcq_gemm_fp8blk_nt_m16 hcq_gemm_fp8blk_nt_m16_max_m hcq_stream_probe; do
  nm -D "$OUT" | grep -q " T $sym$" || { echo "MISSING EXPORT: $sym"; exit 1; }
done
echo "exports OK: hcq_gemm_fp8blk_nt_m16 hcq_gemm_fp8blk_nt_m16_max_m hcq_stream_probe"
