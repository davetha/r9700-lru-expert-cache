# Quickstart

Get the measured configuration running on 2x AMD Radeon AI PRO R9700 (gfx1201). Five steps,
about an hour of which is the model download.

For why any of it works, read the [README](README.md). This page is just the commands.

---

## 1. The model (118 GB)

```bash
hf download davetha/q38fn-heretic2-mxfp4-fp8 \
    --local-dir /mnt/llm-storage/q38fn-heretic2-mxfp4-fp8
```

MXFP4 experts with FP8 attention/MTP/PLE, Heretic-decensored from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). **This is the
exact checkpoint every number in the README's results table was measured on.** A different
Flash-Next quantization will run, but will not reproduce those figures.

## 2. Images

```bash
git clone https://github.com/davetha/r9700-lru-expert-cache
cd r9700-lru-expert-cache

docker build -t local/q38fn-rocm10:try1  -f docker/Dockerfile       docker/
docker build -t local/q38fn-rocm10:build -f docker/Dockerfile.build docker/

# sanity: does the runtime image see both cards?
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  -v "$PWD:/repo" --entrypoint python3 local/q38fn-rocm10:try1 /repo/docker/probe.py
```

## 3. Kernels and patches

```bash
./prebuilt/install.sh                  # prebuilt gfx1201 kernels; skips the HIP build
./patches/apply_patches.sh --dry-run   # verify every diff applies to your image FIRST
./patches/apply_patches.sh
./templates/fetch.sh                   # fixed Qwen chat template (not vendored)
```

Building the kernels yourself instead of `prebuilt/install.sh` is two extra commands — see
[README → Build](README.md#build).

## 4. Find your GPU indices

**Do not copy indices from the README.** They are whatever your machine assigns, and on a
mixed-GPU host the R9700s are usually *not* 0 and 1. Ask the driver:

```bash
python3 - <<'PY'
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"HIP index {i}: {torch.cuda.get_device_name(i)}  {p.total_memory/2**30:.1f} GiB")
PY

# and the matching /sys/class/drm cards (R9700 = device id 0x7551):
for d in /sys/class/drm/card*/device; do
  [ "$(cat $d/device 2>/dev/null)" = "0x7551" ] && echo "R9700: $(basename $(dirname $d))"
done
```

Use the two R9700 indices for `GPUS`, and their two cards for `VRAM_CARDS`.

<details>
<summary>Worked example: a host with 2x MI210 + 2x R9700</summary>

```
HIP index 0: AMD Instinct MI210          64.0 GiB
HIP index 1: AMD Radeon AI PRO R9700     31.9 GiB
HIP index 2: AMD Radeon AI PRO R9700     31.9 GiB
HIP index 3: AMD Instinct MI210          64.0 GiB
R9700: card2
R9700: card3
```

so `GPUS=1,2` and `VRAM_CARDS="card2 card3"` — **not** `0,1` / `"card1 card2"`. Taking the
defaults there would put one rank on an MI210 and the run would be meaningless.
</details>

## 5. Launch

```bash
export MODELS_DIR=/mnt/llm-storage        # host dir holding the checkpoint
export MODEL=/models/q38fn-heretic2-mxfp4-fp8
export GPUS=1,2                           # from step 4
export VRAM_CARDS="card2 card3"           # from step 4
./launch/launch_q38fn.sh 15 262144        # 15 GB expert slots per rank, 256K context
```

Server comes up on **:8057**.

---

## Check it actually engaged

```
r4d LRU expert cache: ON (lib ..., thresh 0.50, max_inserts 64, grid 8x16)
r4d LRU: layer 0 -> 257 slots warm-started from the profile hot set,
         read-through above 128 distinct experts/step
```

**If you see `r4d unavailable`, a mounted patch raised on import and the MoE silently fell
back to the stock path — any numbers you collect are invalid.** This fails quietly, so check
for it rather than assuming. `launch/run_arm_bench.sh` fails loud on both conditions and is
the safer way to benchmark.

## What you should get

Launcher defaults reproduce the `fp8head` arm:

| | prose | JSON | code |
|---|---|---|---|
| static hot set (baseline) | 60.2 | 68.6 | 89.1 |
| **this repo, launcher defaults** | **93.1** | **138.8** | **128.5** |

Single stream, greedy, MTP-4, 256K context, 15 GB expert slots per rank. Full table and
per-arm breakdown in the [README](README.md#results).

## Thinking on/off

Per request, not a launch flag. Flash-Next thinks by default:

```json
{"model": "...", "messages": [...],
 "chat_template_kwargs": {"enable_thinking": false}}
```

`reasoning_effort` and `preserve_thinking` are honoured too.

## Before you launch

- **Free the cards.** The launcher waits for VRAM to drain on `VRAM_CARDS`, but another
  inference server holding those GPUs will OOM the run. Stop it first.
- **Host RAM.** The experts live in host memory and are read over PCIe through UVA — budget
  for the checkpoint on top of what the GPUs hold.
- **`--dry-run` the patches.** They are reconstructed against a specific fork image; if the
  image moved, the dry run tells you before the launch does.
