#!/usr/bin/env bash
# Rebuild the patched vLLM files under build/vllm/ from the fork image's own copies.
#
# We do not redistribute any file from tcclaviger/vllm:DevQwenNextFlash. This script
# copies each original out of the image, applies our unified diff to it, and drops our
# new-from-scratch files in alongside. It then writes build/MOUNTS.txt, the -v lines the
# launcher bind-mounts over /app/vllm/vllm.
#
#   ./patches/apply_patches.sh              apply
#   ./patches/apply_patches.sh --dry-run    verify every diff applies, write nothing
#
# Env:
#   IMAGE      fork image to take originals from (default tcclaviger/vllm:DevQwenNextFlash)
#   MOE_MODE   lru (default) or static -- which r4d_mxfp4_moe.py variant to apply
#   BUILD      output dir (default <repo>/build)
#   WITH_FP8SK 1 also applies the experimental fp8 skinny-GEMM dispatcher. Off by default:
#              the kernel is a measured non-win (see kernels/experimental/fp8skinny/README.md)
#              and none of the results in the top-level README were produced with it mounted.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
IMAGE=${IMAGE:-tcclaviger/vllm:DevQwenNextFlash}
MOE_MODE=${MOE_MODE:-lru}
BUILD=${BUILD:-$REPO_ROOT/build}
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

VDIR=/app/vllm/vllm          # where vllm lives inside the image
PD=$REPO_ROOT/patches

# vllm-relative path <- diff file. One per line.
PATCHED="
ir/op.py|ir/op.py.diff
model_executor/kernels/linear/mxfp4/r4dhip.py|model_executor/kernels/linear/mxfp4/r4dhip.py.diff
model_executor/layers/fused_moe/modular_kernel.py|model_executor/layers/fused_moe/modular_kernel.py.diff
model_executor/layers/layernorm.py|model_executor/layers/layernorm.py.diff
model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py|model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py.diff
model_executor/layers/rotary_embedding/mrope.py|model_executor/layers/rotary_embedding/mrope.py.diff
model_executor/layers/utils.py|model_executor/layers/utils.py.diff
model_executor/models/qwen2_moe.py|model_executor/models/qwen2_moe.py.diff
models/qwen4_exp/amd/indexer_qsa.py|models/qwen4_exp/amd/indexer_qsa.py.diff
models/qwen4_exp/amd/model.py|models/qwen4_exp/amd/model.py.diff
models/qwen4_exp/amd/mtp.py|models/qwen4_exp/amd/mtp.py.diff
third_party/flash_linear_attention/ops/fused_sigmoid_gating.py|third_party/flash_linear_attention/ops/fused_sigmoid_gating.py.diff
model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py|model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py.diff
model_executor/layers/fused_moe/experts/r4d_mxfp4_moe.py|model_executor/layers/fused_moe/experts/r4d_mxfp4_moe.py.MOE_MODE.diff
"

# Opt-in, off by default. VLLM_HC_FP8SK gates it to the closed kernel at runtime anyway.
if [ "${WITH_FP8SK:-0}" = "1" ]; then
  PATCHED="$PATCHED
model_executor/kernels/linear/scaled_mm/fp8hip.py|model_executor/kernels/linear/scaled_mm/fp8hip.py.diff
"
fi

# Files that are entirely ours: repo path <- vllm-relative destination.
NEWFILES="
kernels/draft_w4_lmhead.py|model_executor/kernels/draft_w4_lmhead.py
kernels/fused_silu_mul_quant.py|model_executor/layers/fused_silu_mul_quant.py
kernels/fused_gate_mul.py|model_executor/layers/fused_gate_mul.py
"

case "$MOE_MODE" in lru|static) ;; *) echo "MOE_MODE must be lru or static" >&2; exit 2 ;; esac

command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
command -v patch  >/dev/null || { echo "patch not found"  >&2; exit 1; }

ORIG=$(mktemp -d); trap 'rm -rf "$ORIG"' EXIT
OUT=$BUILD/vllm

echo "image:    $IMAGE"
echo "moe mode: $MOE_MODE"
[ $DRY = 1 ] && echo "DRY RUN - nothing will be written to $BUILD"

# 1. pull every original out of the image in a single container run
rels=""
while IFS='|' read -r rel dif; do
  [ -z "$rel" ] && continue
  rels="$rels $rel"
done <<< "$(echo "$PATCHED" | sed '/^$/d')"

# shellcheck disable=SC2086
docker run --rm --entrypoint bash "$IMAGE" -c \
  "cd $VDIR && tar cf - $rels" | tar xf - -C "$ORIG"

for rel in $rels; do
  [ -s "$ORIG/$rel" ] || { echo "MISSING in image: $rel" >&2; exit 1; }
done

# 2. apply each diff
fail=0
while IFS='|' read -r rel dif; do
  [ -z "$rel" ] && continue
  dif=${dif/MOE_MODE/$MOE_MODE}
  [ -f "$PD/$dif" ] || { echo "missing diff $PD/$dif" >&2; exit 1; }
  work=$ORIG/_stage; rm -rf "$work"; mkdir -p "$work/vllm/$(dirname "$rel")"
  cp "$ORIG/$rel" "$work/vllm/$rel"
  if [ $DRY = 1 ]; then
    if ( cd "$work" && patch --dry-run -p1 --forward -s < "$PD/$dif" ); then
      echo "  ok (dry)  $rel"
    else
      echo "  FAILED    $rel  <- $dif"; fail=1
    fi
  else
    ( cd "$work" && patch -p1 --forward -s < "$PD/$dif" ) || { echo "  FAILED    $rel"; fail=1; continue; }
    mkdir -p "$OUT/$(dirname "$rel")"
    cp "$work/vllm/$rel" "$OUT/$rel"
    echo "  patched   $rel"
  fi
done <<< "$(echo "$PATCHED" | sed '/^$/d')"
[ $fail = 0 ] || { echo "one or more diffs did not apply" >&2; exit 1; }

[ $DRY = 1 ] && { echo "all diffs apply cleanly"; exit 0; }

# 3. drop in the files that are entirely ours
while IFS='|' read -r src dst; do
  [ -z "$src" ] && continue
  mkdir -p "$OUT/$(dirname "$dst")"
  cp "$REPO_ROOT/$src" "$OUT/$dst"
  echo "  new       $dst"
done <<< "$(echo "$NEWFILES" | sed '/^$/d')"

# 4. the -v lines
: > "$BUILD/MOUNTS.txt"
while IFS='|' read -r rel dif; do
  [ -z "$rel" ] && continue
  echo "-v $OUT/$rel:$VDIR/$rel:ro" >> "$BUILD/MOUNTS.txt"
done <<< "$(echo "$PATCHED" | sed '/^$/d')"
while IFS='|' read -r src dst; do
  [ -z "$src" ] && continue
  echo "-v $OUT/$dst:$VDIR/$dst:ro" >> "$BUILD/MOUNTS.txt"
done <<< "$(echo "$NEWFILES" | sed '/^$/d')"

echo
echo "wrote $BUILD/MOUNTS.txt:"
cat "$BUILD/MOUNTS.txt"
