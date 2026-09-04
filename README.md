# r9700-lru-expert-cache

Qwen3.8-Flash-Next decode on 2x AMD Radeon AI PRO R9700 (gfx1201): a device-side LRU
expert cache plus a set of kernel-count patches for [tcclaviger](https://hub.docker.com/r/tcclaviger/vllm)'s
vLLM fork, running on a ROCm 10 image.

The model's experts do not fit in 64 GB of VRAM, so they live in host memory and are read
over PCIe through UVA. The stock arrangement pins a fixed hot set at load time. This repo
makes the contents of those same VRAM buffers mutable: which expert sits in which slot is
decided on the GPU between decode steps, by two HIP kernels that run ahead of
`moe_align_block_size`. Same slots, same bytes, same GEMM.

Measured effect on the production trace: PCIe expert traffic falls from 432 MB/step to
86 MB/step (-80%), and the MoE grouped GEMM falls from 21.1 ms/step to 4.2 ms/step.

## Results

Single stream, greedy, MTP-4, 256K context, 15 GB of expert slots per rank, `max-num-seqs 4`,
`max-num-batched-tokens 2048`. tok/s is best-of-3 from `bench/ab3.py`.

| arm | prose | JSON | code | ms/step | prefill | source |
|---|---|---|---|---|---|---|
| baseline (static hot set, ROCm 10 image) | 60.2 | 68.6 | 89.1 | 44.3 / 54.9 / 38.0 | 3066 | `ab3_r10_base_h15` |
| + LRU expert cache | 76.7 | 112.6 | 94.1 | | | `ab3_lru1` |
| + W4 draft LM head | 84.5 | 119.4 | 100.3 | | | `ab3_w4head` |
| + GDN strided QKV, fused shared gate (c4) | 89.4 | 122.2 | 105.8 | 30.1 / 31.2 / 29.9 | 3191 | `ab3_c4` |
| + fused SiLU-quant, QSA rope gather (c7) | 93.2 | 125.9 | 107.4 | 30.2 / 30.3 / 29.6 | 3220 | `ab3_c7` |
| **c8 — c7 on the v2 victim kernel (what this repo builds)** | **91.8** | **124.1** | **106.2** | 30.0 / 30.8 / 29.4 | 3174 | `ab3_c8_v2` |
| **t8b — c8 at NBT 4096 with the gate GEMV on r4d (launcher default)** | **91.4** | **130.8** | **118.7** | 29.5 / 29.6 / 30.4 | 3539 | `ab3_t8b` |

`c7` and `c8` differ only in which build of the LRU kernel was loaded. `c7` ran the older
`librlu_fused.so`; `c8` ran `librlu_v2.so`, whose rewritten victim selection is what
`kernels/lru/r4d_lru.hip` now contains. The 1.4 / 1.8 / 1.2 tok/s gap between them is
inside the run-to-run spread (see caveats): the v2 kernel is a wash on single-stream
throughput and exists to remove a latency cliff that only bites at B>=4 (below).

**`t8b` is what `launch/launch_q38fn.sh` reproduces.** It is `c8` plus
`max-num-batched-tokens 4096` and the K3 dispatcher change that routes the shared-expert
gate GEMV to r4d. Read the prose column honestly: the intermediate arm `t1` (NBT 4096, gate
still on hipBLASLt) measured 96.2 / 122.8 / 103.4, so the gate change bought +8.0 JSON and
+15.3 code but appears to cost ~5 on prose. Prose is the noisiest of the three probes on
this box — `t1`'s three runs were 71.7 / 96.2 / 95.7 against `t8b`'s 85.6 / 90.7 / 91.4 —
so treat the prose delta as unresolved rather than as a measured regression.

### The 52 hipBLASLt kernels were a 2560 -> 1 GEMV

`shared_expert_gate` is `ReplicatedLinear(2560, 1, bias=False)`: a single output column.
The skinny-GEMM path rejected it, so `F.linear` landed on hipBLASLt, which split K 32 ways
over that one column — the `Cijk_..._MT16x16x32` kernels K3's dispatch census counted at
**52/step**, 17-23 us each. Letting the r4d bf16 kernel take it instead costs **3.4 us**
and is `torch.equal` to `F.linear` (0 of 7,449,600 elements differ across 6 real weights,
6 token counts and 5 input scales), for about **-0.7 ms/step**. The `Cijk` line is simply
absent from the `t8b` kernel breakdown, where `r4d_gemm_bf16_nt_m64` picks up the extra
48 calls (292 -> 340).

The dispatcher does this through a fall-through (`VLLM_HC_R4D_BF16_FT`): the main skinny
branch still requires `n >= 3`, but anything with `m >= 1` and `n <= 32` falls through to
r4d rather than to hipBLASLt. Wide batches, where hipBLASLt is genuinely the better kernel,
are above `FT_MAX_N` and keep the old path.

### The closed fp8 GEMM buys nothing

`VLLM_DISABLE_FP8HIP=1` swaps the image's closed `libfp8hip_gemm.so` W8A8 kernel for vLLM's
own Triton fallback. Arm `t4` measures **99.9 / 123.9 / 113.1** against `t1`'s
96.2 / 122.8 / 103.4 on the same stack — no worse, and nominally better on prose and code.
`fp8hip_gemm_w8a8_tiled` is 2.8 ms/step, ~15% of kernel-busy time, so this is worth
knowing: none of that time is buying anything the open path does not already give you.
Neither arm was repeated, so read it as "same cost", not as a win for Triton.

Concurrency, aggregate tok/s across streams (`bench/concurrent_bench.py`):

| streams | static hot set | LRU only | full stack (c8) |
|---|---|---|---|
| B=1 |  76.6 |  94.0 | 105.7 |
| B=2 |  88.6 | 114.9 | — |
| B=4 | 126.6 | 151.8 | 165.7 |

### Why the v2 victim kernel matters at B=4

The old victim search was a serial argmin per insert, and its cost had a cliff: at S=257
slots it jumped from 18.6 us at 13 inserts to 43.4 us at 14, then climbed ~3 us/insert to
224.9 us at 64. B=4 already runs 5-13 inserts/layer/step — immediately below the cliff —
so a step that crossed it paid 52 layers x 25 us = **+1.3 ms/step**. The rewrite ranks all
slots against one LDS key array instead, which is a fixed ~3.7 us regardless of insert
count. Measured `us/call`, S=257 (`tests/lru/bench_victim.py`):

| inserts | 0 | 1 | 2 | 4 | 8 | 13 | 14 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| old (serial argmin) | 6.90 | 8.67 | 9.50 | 11.39 | 14.24 | 18.61 | **43.4** | 86.0 | 224.9 |
| v2 (batched ranking) | 6.97 | 8.61 | 9.21 | 10.57 | 11.14 | 11.12 | **11.05** | 11.92 | 12.97 |

Ranking is a regression below ~4 inserts, so short lists keep the serial argmin (`NSER 4`);
both paths live in the same function and produce identical state. That equality is the
whole point of `tests/lru/test_victim_equiv.py`, which compares `table` / `map_cold` /
`slot_expert` / `slot_stamp` / `miss` / `n_miss` byte for byte between the two libraries
over 19 cases (production B=1 and B=4, at and past the cliff, zero misses, capped by
`max_inserts`, `max_inserts` 1 and 1024, empty slots, all-equal stamps, nothing-evictable
on both paths, S=1024, read-through) plus a 3-perturbation negative control, 3/3 caught.

The cliff itself was never root-caused. It sits at k=14 for S=257 and S=320, k=16 for
S=200, is absent at S=129, and does not move with E.

### MTP depth

`num_speculative_tokens` is not a one-way dial. Same stack, MTP-3 vs MTP-4:

| | prose | JSON | code |
|---|---|---|---|
| MTP-3 (`ab3_c2_mtp3`) | **94.0** | 115.7 | 93.1 |
| MTP-4 (`ab3_c4`) | 89.4 | **122.2** | **105.8** |

Prose prefers 3, JSON and code prefer 4, and MTP-6 (`ab3_c2_mtp6`) is worse than both on
prose. The measured MTP-3 -> MTP-4 step costs +2.6 ms/step, and K1's offline replay
(`tools/routecap/mtp_prefetch.py`) prices only ~0.29 ms of that as cold expert traffic
(the 5th row adds 8.0 MB/step at 28 GB/s) — about 11%. The other ~2.3 ms is the 5th row's
own GEMM and attention work. Each of the 4 speculative rows costs only ~0.3-0.5 ms/step of
PCIe traffic because the LRU already flattens the routing-width penalty: misses per layer
per step grow from 0.69 to 1.45 for 5x the rows, since the extra rows mostly re-route to
experts the target row already pulled. **Deeper MTP is not priced by the expert cache** —
tune it against your own traffic mix.

Long context is intact: `bench/needle.py` is **9/9** at 32K / 128K / 200K x depth 10/50/90%,
the longest prompt being 256,389 tokens.

### Caveats

* **Restart-to-restart nondeterminism.** Greedy output on this box is stable within a server
  instance and diverges across restarts. The same baseline configuration, re-run later in the
  session, measured 58.5 / 66.1 / 76.6 (`ab3_lru0ctl`) and 60.8 / 68.7 / 81.3 (`ab3_lru0ctl_b`)
  against the 60.2 / 68.6 / 89.1 headline. Treat single-arm differences under ~5% as noise, and
  run your own base-vs-base control before attributing anything.
* **Single stream.** Every number in the first table is one request at a time. The concurrency
  table is the only multi-stream measurement.
* **The W4 draft head trades acceptance for time.** It removes ~2.8 ms/step but lowers the
  speculative acceptance rate by roughly 5-9% relative (0.429 -> 0.421 prose, 0.744 -> 0.706
  JSON, 0.551 -> 0.530 code). It is net positive here; it may not be on your traffic. The
  later arms partly recover it: `c7` measures 0.462 / 0.704 / 0.545 against `c4`'s
  0.421 / 0.704 / 0.538 — identical on JSON, better on prose and code.
* **`VLLM_GEMMA_NORM_FUSED` defaults to 2 and every arm from `combo1` onward (`c4`..`c8`,
  `t1`..`t8b`) ran with it on** — the `c4` and `c7` profiles contain 32 `vllm::rms_norm_kernel`
  launches per step, which only exist on the fused path. Mode 2 casts to fp32 around the fused
  kernel so it matches the stock decomposition up to reduction order (~3e-6 of elements differ
  by one bf16 ulp; none at decode row counts). Mode 1 (bf16 weight) perturbs ~30% of elements
  and is not the default; `0` is the stock 10-kernel path. See `docs/PATCHES.md` #1.
* The ablation that rules out "the machinery, not the policy": `VLLM_R4D_LRU_MAX_INSERTS=0`
  (cache on, inserts disabled) measures 61.1 / 70.5 / 81.4 — indistinguishable from the
  cache-off control.
* `docs/REVIEW.md` is an internal read-only review written a few minutes before the `c4` arm
  ran. Its C4 finding ("`VLLM_R4D_LRU_FUSE=1` has never been in an arm") is stale: `c4`, `c5`
  and `c6` all ran with the fused kernel. The rest of it stands.

## How it works

```
                 host memory (UVA)                       VRAM, per MoE layer
        w1/w2 + scales for all E experts        S slots (S ~= 182-344 at 15 GB/rank)
                       |                                       ^
                       |  lru_gather_k: 16 B/lane,             |
                       |  grid (CHUNKS x LANES), only          |
                       |  the missing experts                  |
                       +---------------------------------------+
                                                               |
   topk_ids ---> lru_manage_k (one workgroup) -----------------+
                   mark experts routed this step
                   if distinct > S*THRESH: read through, change nothing
                   else: refresh stamps, enumerate misses in expert-id order,
                         evict min (stamp, slot) among slots NOT routed this step
                   rewrite table[E] (expert -> slot) and map_cold[E] in place
                       |
                       +--> moe_align(topk_ids, table)     -> resident grouped GEMM
                       +--> moe_align(topk_ids, map_cold)  -> UVA fallback GEMM
                                                              (both calls always kept)
```

`table` *is* the fork's existing `map_hot` tensor, now mutable, so `moe_align_block_size`
needs no change. Every state tensor is allocated once in `_build_hot` and never reallocated,
so a captured HIP graph keeps adapting: only the contents change between replays, never a
pointer. No atomics, no host sync — both TP ranks evolve identically because misses are
enumerated by expert id and victims by a total order on `(stamp, slot)`.

`VLLM_R4D_LRU_FUSE=1` folds the manager and the two `moe_align_block_size` calls into a
single kernel (`lru_fused_k`). See `docs/FUSE.md`.

Full rationale and the trace analysis that predicted the win: `docs/LOCALITY.md`.
The kernel-count patches that make up the rest: `docs/PATCHES.md` and `docs/KERNEL_HITLIST.md`.

## Prerequisites

**Hardware.** 2x AMD Radeon AI PRO R9700 (gfx1201, 32 GB each). The expert spill runs over
PCIe, so host memory bandwidth and PCIe topology dominate: this was measured on an EPYC Gen4
board where the two cards are on separate root ports with no peer link. The all-reduce is
`r4d_ar_oneshot_2rank_exact` over PCIe, ~0.9 ms/step. You need enough host RAM to hold the
full expert set (~30 GiB here) plus the KV cache spill.

**Software.**

* Docker.
* `tcclaviger/vllm:DevQwenNextFlash` — the vLLM fork image. It carries prebuilt `_C`/`_moe_C`,
  the closed `r4d.so` / `libfp8hip_gemm.so` pybind modules, and `rfi_hip`. Not distributed here.
* The ROCm 10 SDK, installed into that image from AMD's pip index
  (`https://stable.repo.amd.com/rocm/whl-next`) by `docker/Dockerfile`. Python 3.12 and the
  torch 2.11 ABI are kept so the fork's prebuilt extensions still load.
* A Qwen3.8-Flash-Next MXFP4 checkpoint. The measurements used a locally quantized
  `q38fn-heretic2-mxfp4-fp8` (MXFP4 experts, FP8 attention/MTP/PLE).

## Build

```bash
git clone https://github.com/davetha/r9700-lru-expert-cache
cd r9700-lru-expert-cache

# 1. the ROCm 10 runtime image, and the same image plus a compiler
docker build -t local/q38fn-rocm10:try1  -f docker/Dockerfile       docker/
docker build -t local/q38fn-rocm10:build -f docker/Dockerfile.build docker/
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  -v "$PWD:/repo" --entrypoint python3 local/q38fn-rocm10:try1 /repo/docker/probe.py

# 2. the LRU kernels  ->  build/kernels/librlu.so
docker run --rm -v "$PWD:/repo" --entrypoint bash local/q38fn-rocm10:build \
  -c '/repo/kernels/lru/build.sh /repo/build/kernels/librlu.so'

# 3. reconstruct the patched vLLM files  ->  build/vllm/ and build/MOUNTS.txt
./patches/apply_patches.sh --dry-run     # verify every diff applies to your image
./patches/apply_patches.sh
```

`apply_patches.sh` copies each original file out of the fork image, applies our unified diff,
and writes the `-v` bind-mount lines. No file from the fork is redistributed here.

`MOE_MODE=static ./patches/apply_patches.sh` selects the pre-LRU static hot/cold MoE instead,
which is the baseline arm.

### libr4d (optional)

The closed `r4d.so` in the image is built from a source library published at
<https://codeberg.org/StillDeadcode/libr4d>. Our LRU kernels do **not** link against it — they
are self-contained HIP and need only `hip/hip_runtime.h`. You need a libr4d checkout only to
run the open-vs-closed dense-GEMM comparison in `tests/k1/cmp_dense.py`:

```bash
git clone https://codeberg.org/StillDeadcode/libr4d build/libr4d   # measured at 5dc6302
docker run --rm -v "$PWD:/repo" --entrypoint bash local/q38fn-rocm10:build \
  -c 'cd /repo/build/libr4d && GFX_ARCH=gfx1201 ./build.sh'
```

## Run

```bash
export MODELS_DIR=/mnt/llm-storage          # host dir holding the checkpoint
export MODEL=/models/q38fn-heretic2-mxfp4-fp8
export GPUS=0,1                             # HIP_VISIBLE_DEVICES
export VRAM_CARDS="card1 card2"             # /sys/class/drm cards for the drain wait
./launch/launch_q38fn.sh 15 262144          # 15 GB of expert slots/rank, 256K ctx
```

The gates that measured a win are on by default: LRU + fused kernel, the W4 draft head,
GDN strided QKV, the fused shared gate, the fused SiLU-quant, the QSA rope gather, and the
UVA offload of `embed_tokens` and the vision tower. Every one is individually overridable.

Server comes up on `:8057`. Startup lines worth grepping for:

```
r4d LRU expert cache: ON (lib ..., thresh 0.50, max_inserts 64, grid 8x16)
r4d LRU: layer 0 -> 257 slots warm-started from the profile hot set,
         read-through above 128 distinct experts/step
hot experts: budget ...
```

If `r4d unavailable` appears, a mounted patch raised on import and the MoE silently fell back —
the numbers are invalid. `launch/run_arm_bench.sh` fails loud on both conditions.

`launch/launch_q38fn_prof.sh` is the same launcher with the torch profiler wired up and every
gate defaulted **off**; it is what produced the baseline arms.

### Environment knobs

| variable | default | meaning |
|---|---|---|
| `VLLM_R4D_LRU` | `0` | `1` enables the cache. `0` is byte-for-byte the static behaviour. |
| `VLLM_R4D_LRU_FUSE` | `0` | `1` uses `lru_fused_k`: manager + both `moe_align_block_size` calls in one kernel. |
| `R4D_LRU_LIB` | — | path to `librlu.so`. Required when the cache is on. |
| `VLLM_R4D_LRU_THRESH` | `0.5` | a step routing more than this fraction of a layer's slots does no inserts and reads through instead (keeps prefill chunks and wide batches from thrashing the cache) |
| `VLLM_R4D_LRU_MAX_INSERTS` | `64` | hard cap on inserts per layer per forward; also the miss-buffer size. `0` = ablation (cache machinery on, policy off). |
| `VLLM_R4D_LRU_CHUNKS` | `8` | gather grid.x (slab split) |
| `VLLM_R4D_LRU_LANES` | `16` | gather grid.y (concurrent experts) |
| `VLLM_R4D_HOT_PROFILE` | — | routing-statistics JSON; chooses the per-layer slot budget and the warm start |
| `VLLM_R4D_HOT_GB` | — | total expert-weight budget per rank, in GB |
| `VLLM_R4D_SHARE_A8` | `1` | share the FP8 activation between the shared expert and the routed MoE. Unverified win; `0` in every measured arm. |
| `VLLM_GEMMA_NORM_FUSED` | `2` | dispatch the fused `rms_norm` in eager regions: -288 kernels/step. `2` keeps the `+1` in fp32 and skips the residual case, so only the final cast rounds; `1` is the original variant, which **perturbs numerics**; `0` is the stock path. Not in any measured arm — read `docs/PATCHES.md` #1 first. |
| `VLLM_GDN_STRIDED_QKV` | `0` | strided QKV into the GDN linear attention: -144 copies/step, bit-identical |
| `VLLM_FUSED_SHARED_GATE` | `0` | fuse the shared-expert gate multiply: -48/step, bit-identical |
| `VLLM_FUSED_SILU_QUANT` | `0` | fuse SiLU-mul with the FP8 activation quant: -52/step, bit-identical. Neutral on its own in `c6`; on in `c7`/`c8`. |
| `VLLM_DRAFT_W4_LMHEAD` | `0` | W4 LM head for the MTP draft iterations: -2.8 ms/step, costs ~5-9% relative acceptance |
| `VLLM_MOE_OUTPUT_ALIAS` | `1` | let the r4d MoE write straight into the caller's output buffer: -48 copies/step |
| `VLLM_UVA_OFFLOAD_EMBED` | `0` | keep `embed_tokens` in host memory |
| `VLLM_UVA_OFFLOAD_VISUAL` | `0` | keep the vision tower in host memory |
| `VLLM_HC_R4D_BF16` | `1` | route skinny bf16 linears to the r4d GEMM instead of `wvSplitK` / hipBLASLt. `0` restores the stock dispatch. See `docs/HC_QUANT.md`. |
| `VLLM_HC_R4D_BF16_MIN_N` | `3` | minimum output width for the main skinny branch. `n=1,2` are left to the fall-through below. |
| `VLLM_HC_R4D_BF16_FT` | `1` | the fall-through: send anything the main branch rejected to r4d rather than to hipBLASLt. This is what captures the m=1 `shared_expert_gate` GEMV. `0` disables it. |
| `VLLM_HC_R4D_BF16_FT_MIN_M` | `1` | fall-through lower bound on M |
| `VLLM_HC_R4D_BF16_FT_MAX_N` | `32` | fall-through upper bound on N. Above this, hipBLASLt is the better kernel and keeps the call. |
| `VLLM_HC_R4D_BF16_FT_MIN_N` | `1` | fall-through lower bound on N |
| `VLLM_HC_R4D_BF16_DEBUG` | `0` | log every dispatch decision once per distinct `(route, n, m, k)`. This is how the 52/step gate GEMV was found. |
| `VLLM_HC_R4D_BF16_WAVES` | `192` | wave-occupancy target the skinny heuristic aims at |
| `VLLM_HC_R4D_BF16_CFG` | — | force a `"WV,SK"` config, overriding the heuristic. Debug only. |
| `VLLM_DISABLE_FP8HIP` | `0` | `1` replaces the image's closed `libfp8hip_gemm.so` W8A8 kernel with vLLM's Triton fallback. Measured same cost (arm `t4`). |
| `VLLM_QSA_ROPE_GATHER` | `0` | fold the cos/sin gather into the mrope Triton kernel (hitlist #12). On in `c7`/`c8`. Degrades to a warning if the patched `mrope.py` is not mounted. |

## Generating a hot profile

`VLLM_R4D_HOT_PROFILE` wants per-layer routing counts. Capture them from real traffic:

```bash
# 1. run the server with routing capture on; routes_rank<N>.npz lands in $ROUTECAP_DIR
EXTRA_DOCKER_ARGS="-e ROUTECAP=1 -e ROUTECAP_DIR=/w/artifacts" \
  MOUNTS_FILE=build/MOUNTS.txt ./launch/launch_q38fn.sh 15 262144
#    ... drive it with your own traffic, or bench/traffic.py ...

# 2. how much coverage a given slot budget buys
python3 bench/cov_curve.py artifacts/routes_rank0.npz

# 3. water-fill the budget across layers -> hot_profile.json
python3 tools/routecap/build_hot_profile.py artifacts/routes_rank0.npz > profiles/hot_profile.json
```

`profiles/hot_profile.json` in this repo is the one used for every measurement: derived routing
statistics only, no weights and no prompt text. `tools/routecap/locality_sim.py` replays a
capture against the LRU policy offline (and against Belady, as an upper bound) if you want to
pick `THRESH` / `MAX_INSERTS` without a GPU.

## Benchmarks and tests

```bash
./launch/run_arm_bench.sh c4 15 -e VLLM_R4D_LRU=1 -e VLLM_R4D_LRU_FUSE=1 ...   # one A/B arm
python3 bench/ab3.py out.json          # best-of-3 tok/s on prose / JSON / code + prefill
python3 bench/needle.py                # 9-point needle-in-a-haystack to 256K
python3 bench/concurrent_bench.py      # B=1/2/4 aggregate throughput
```

`tests/README.md` says which tests need a GPU, how to take the GPU lock, and how to run each.
The short version, for the LRU kernels:

```bash
flock -w 3600 gpu.lock docker run --rm --ipc host --group-add video \
  --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 \
  -v "$PWD:/w" --entrypoint bash local/q38fn-rocm10:build -c \
  'cd /w/tests/lru && python3 test_lru.py && python3 test_numerics_lru.py &&
   NLAYER=4 python3 test_graph_lru.py && python3 test_trace_replay.py'
```

## What is not here

* No binaries. The closed `r4d.so`, `libfp8hip_gemm.so` and `gdn_hip_C*.so` belong to the fork
  image and are not redistributed; our own `librlu.so` / `cold_gather.so` ship as source plus a
  build script.
* No fork source. Every modification to a fork file is a unified diff against the file as it
  exists in `tcclaviger/vllm:DevQwenNextFlash`; `apply_patches.sh` fetches the originals.
* No captures or weights: `routes_rank*.npz` (74 MB each), profiler traces, `tensor_headers.json`,
  and the checkpoint itself. `profiles/hot_profile.json` (226 KB of derived counts) is the one
  data artifact included.

## Credits

* **libr4d** — the gfx1201 kernel library the fork's closed `r4d.so` is built from, by
  StillDeadcode / tcclaviger: <https://codeberg.org/StillDeadcode/libr4d>. Our LRU kernels sit
  in front of its grouped MXFP4 GEMM and do not modify it.
* **The vLLM fork** — `tcclaviger/vllm:DevQwenNextFlash`, which is what makes Qwen3.8-Flash-Next
  run on RDNA4 at all: <https://hub.docker.com/r/tcclaviger/vllm>. The hot/cold expert split,
  the UVA offload path and the r4d MoE integration are theirs. Everything in `patches/` is a
  diff on top of their work.
* **vLLM** — Apache-2.0, <https://github.com/vllm-project/vllm>.
* The LRU cache, the kernel-count patches, the tests and the analysis in `docs/` were produced
  in this repository.

## License

Apache-2.0 — see `LICENSE`. The unified diffs under `patches/` are derivative works of
Apache-2.0 vLLM code and of the fork's modifications to it; they are distributed on the same
terms and carry no copy of the files they patch.
