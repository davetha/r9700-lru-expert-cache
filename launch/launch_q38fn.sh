#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next on 2x R9700 with the device-side LRU expert cache and the
# kernel-count patches. Every gate that measured a win is on. Mirrors the production
# launcher: 15 GB of expert slots per rank, max-num-batched-tokens 4096, MTP-4, 256K ctx.
# Measured across the arms at these settings: prose ~90-96, JSON ~123-131, code ~103-119
# tok/s, prefill ~3400-3540, needle 9/9 at 256K.
#
# Do not raise the slot budget to 16 GB. It fits only sometimes -- the engine sizes the KV
# cache from what is left, and arm t8 died on exactly this ("2.4 GiB KV cache is needed,
# ... available 2.12 GiB") while an earlier 16 GB arm came up fine. NBT 8192 is worse still
# (0.83 GiB left). 15 GB / NBT 4096 is the largest pair that came up every time.
#
# NOTE: this does NOT pin VLLM_GEMMA_NORM_FUSED, so the patched layernorm.py picks its own
# default (currently 2 = fp32-weight fused norm). Every arm in the README's table ran
# BEFORE that default existed, i.e. with the fused norm off. Export
# VLLM_GEMMA_NORM_FUSED=0 to reproduce them exactly.
#
#   $1  LRU slot budget in GB of expert weights per rank (default 15)
#   $2  max model len (default 262144)
#
# Everything else is env. Defaults assume you ran patches/apply_patches.sh and
# kernels/lru/build.sh into <repo>/build.
#
#   REPO_ROOT    repo checkout           (default: parent of this script)
#   BUILD        patched tree + kernels  (default: $REPO_ROOT/build)
#   MOUNTS_FILE  -v lines                (default: $BUILD/MOUNTS.txt)
#   MODELS_DIR   host dir holding the checkpoint, mounted at /models
#   MODEL        model path inside the container
#   PROFILE_DIR  host dir holding hot_profile.json, mounted at /hot
#   IMG          runtime image           (default local/q38fn-rocm10:try1)
#   PORT         host port               (default 8057)
#   GPUS         HIP_VISIBLE_DEVICES     (default 1,2)
#   VRAM_CARDS   /sys/class/drm cards backing those GPUs, for the VRAM-drain wait
#   LRU_LIB      librlu .so inside the container (default /build/kernels/librlu.so)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$HERE/.." && pwd)}
BUILD=${BUILD:-$REPO_ROOT/build}
MOUNTS_FILE=${MOUNTS_FILE:-$BUILD/MOUNTS.txt}
IMG=${IMG:-local/q38fn-rocm10:try1}
MODELS_DIR=${MODELS_DIR:-/mnt/llm-storage}
MODEL=${MODEL:-/models/q38fn-heretic2-mxfp4-fp8}
PROFILE_DIR=${PROFILE_DIR:-$REPO_ROOT/profiles}
PORT=${PORT:-8057}
GPUS=${GPUS:-1,2}
VRAM_CARDS=${VRAM_CARDS:-"card2 card3"}
LRU_LIB=${LRU_LIB:-/build/kernels/librlu.so}
NAME=${NAME:-q38fn-mxfp4}
HOT_GB=${1:-15.0}
# chat template inside the container; templates/fetch.sh downloads the froggeric fixed template.
# Set CHAT_TEMPLATE= (empty) to use the checkpoint's own template.
CHAT_TEMPLATE=${CHAT_TEMPLATE-/templates/qwen_fixed_chat_template.jinja}
MAXLEN=${2:-262144}

[ -f "$MOUNTS_FILE" ] || { echo "no $MOUNTS_FILE - run patches/apply_patches.sh first" >&2; exit 1; }

docker rm -f "$NAME" >/dev/null 2>&1 || true
# docker rm -f can return while the old container is still tearing down; wait for the name to free
for i in $(seq 1 60); do docker inspect "$NAME" >/dev/null 2>&1 || break; docker rm -f "$NAME" >/dev/null 2>&1; sleep 2; done
# and VRAM is released lazily after that, so wait for the cards to drain too
for i in $(seq 1 90); do
  busy=0
  for c in $VRAM_CARDS; do
    u=$(cat "/sys/class/drm/$c/device/mem_info_vram_used" 2>/dev/null || echo 0)
    [ "$u" -ge 1610612736 ] && busy=1
  done
  [ "$busy" = 0 ] && { echo "GPU VRAM free after ${i}s"; break; }
  sleep 1
done

PM=$(tr '\n' ' ' < "$MOUNTS_FILE")

docker run -d --name "$NAME" --ipc host --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --group-add video --shm-size 32g --device /dev/kfd --device /dev/dri \
  -e HIP_VISIBLE_DEVICES="$GPUS" \
  -e VLLM_ROCM_USE_AITER=0 \
  -e VLLM_PLE_CPU_OFFLOAD=1 \
  -e VLLM_R4D_HOT_PROFILE=/hot/hot_profile.json -e VLLM_R4D_HOT_GB="$HOT_GB" \
  -e VLLM_R4D_LRU="${VLLM_R4D_LRU:-1}" -e VLLM_R4D_LRU_FUSE="${VLLM_R4D_LRU_FUSE:-1}" \
  -e R4D_LRU_LIB="$LRU_LIB" -e VLLM_R4D_SHARE_A8="${VLLM_R4D_SHARE_A8:-0}" \
  -e VLLM_GDN_STRIDED_QKV="${VLLM_GDN_STRIDED_QKV:-1}" \
  -e VLLM_FUSED_SHARED_GATE="${VLLM_FUSED_SHARED_GATE:-1}" \
  -e VLLM_FUSED_SILU_QUANT="${VLLM_FUSED_SILU_QUANT:-1}" \
  -e VLLM_QSA_ROPE_GATHER="${VLLM_QSA_ROPE_GATHER:-1}" \
  -e VLLM_UVA_OFFLOAD_EMBED=1 -e VLLM_UVA_OFFLOAD_VISUAL=1 \
  -e VLLM_DRAFT_W4_LMHEAD="${VLLM_DRAFT_W4_LMHEAD:-1}" \
  -e VLLM_R4D_MOE_CFG1="${VLLM_R4D_MOE_CFG1:-1,2,4}" -e VLLM_R4D_MOE_CFG2="${VLLM_R4D_MOE_CFG2:-1,1,1}" \
  -v "$PROFILE_DIR:/hot:ro" -v "$MODELS_DIR:/models" -v "$BUILD:/build:ro" -v "$REPO_ROOT/templates:/templates:ro" \
  -p "$PORT:8000" $PM ${EXTRA_DOCKER_ARGS:-} \
  "$IMG" "$MODEL" \
    --served-model-name q38fn-mxfp4 \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --cpu-offload-gb 40 --cpu-offload-params experts \
    --gpu-memory-utilization "${UTIL:-0.97}" \
    --max-model-len "$MAXLEN" \
    --max-num-seqs "${NSEQ:-4}" --max-num-batched-tokens "${NBT:-4096}" \
    --enable-prefix-caching --enable-chunked-prefill \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --limit-mm-per-prompt.image 8 --limit-mm-per-prompt.video 1 --mm-processor-cache-gb .5 \
    --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_N:-4}${SPEC_EXTRA:-}}" \
    ${EXTRA_VLLM_ARGS:-} \
    ${CHAT_TEMPLATE:+--chat-template $CHAT_TEMPLATE} \
    --host 0.0.0.0 --port 8000 >/dev/null
echo "$NAME launching on :$PORT (hot ${HOT_GB} GB/rank, ctx ${MAXLEN})"
