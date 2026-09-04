# k1 — open libr4d toolchain + MoE GEMM ABI

Host `big`, all paths absolute. GPU work held `flock $REPO/gpu.lock`, GPUs 1,2 only.

## Status

| # | deliverable | state |
|---|---|---|
| 1 | `build_open.sh` → `r4d_open.so` (+ `_fast`, `_fma`) | DONE |
| 2 | open vs shipped dense kernel: correctness + speed | DONE |
| 3 | `MOE_ABI.md` | DONE |
| 4 | `moe_ref_harness.py` | DONE |

## 1. From-source build

`$REPO/k1/build_open.sh` (+ `_build_inner.sh`, the in-container half).
Builds three variants of the public checkout at `$REPO/libr4d` (HEAD 5dc6302),
each in its own scratch copy (`k1/build_off`, `k1/build_fma`, `k1/build_fast`) so the
pristine checkout never gets object files and variants cannot share a stale `.o`:

| output | flags | wall |
|---|---|---|
| `$REPO/k1/r4d_open.so` | author's: `-O3 -std=c++17 -fPIC --offload-arch=gfx1201 -Wno-unused-result -ffp-contract=off` (+`-mcumode` on the GDN unit only) | 15.0 s |
| `$REPO/k1/r4d_open_fast.so` | `-ffp-contract=fast`, no extra `-mcumode` | 15.1 s |
| `$REPO/k1/r4d_open_fma.so` | `-ffp-contract=fast -mcumode` on every unit | 15.0 s |

Two changes to the environment were REQUIRED — the stock image `local/q38fn-rocm10:try1`
cannot compile any `.hip` host code at all:

1. **`local/q38fn-rocm10:k1build`** = try1 + `apt g++ bc` + a `libamdhip64.so` symlink
   (`k1/Dockerfile.k1build`). Base has no libstdc++ headers, and its bundled libc++ is
   unusable (no `__config_site`, no `libc++.so`); and only `libamdhip64.so.7` ships while
   the hip link line names the unversioned path.
2. **`--rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode`** (injected through `HIPCC`).
   The SDK puts the device bitcode under `lib/llvm/`, not where clang probes from `ROCM_PATH`.

Neither touches a codegen flag. `libr4d/build.sh` and `Makefile` are used unmodified.

## 2. Dense kernel: open vs shipped

`k1/cmp_dense.py` loads shipped + all three open builds into ONE process over the ctypes
surface, so identical device buffers feed every variant and outputs are compared bit-for-bit.
`k1/test_edge_dense.py` runs libr4d's own test grid (tail-block N=48/N=80, M=1/5) the same way.

**Result: the from-source build is BIT-IDENTICAL to `/app/r4dhip/r4d.so`** — zero differing
bf16 words on 32 shape×M cells and 33 shape×config edge cells. So is `_fast`, and so is
`_fma`. `relerr(shipped vs exact torch dequant)` = 0.9–1.8e-3 everywhere (bf16 output rounding).

Speed, best-of-25 µs, GPU 1 under the lock (`k1/cmp_dense.out`, `k1/test_edge_dense.out`):

| shape (N,K) | M | shipped | open | open_fast | open_fma |
|---|---|---|---|---|---|
| moe.gate_up 640,2560 | 8 | 37.7 | 37.8 | 35.3 | 42.2 |
| moe.gate_up 640,2560 | 64 | 49.8 | 49.9 | 49.5 | 62.5 |
| moe.down 2560,320 | 8 | 36.4 | 36.4 | 39.2 | 37.0 |
| attn.q 3072,2560 | 64 | 51.7 | 50.8 | 52.5 | 66.3 |
| attn.o 2560,3072 | 64 | 54.5 | 53.1 | 54.8 | 67.4 |
| b:mlp.gate_up 17408,5120 | 64 | 187.4 | 189.0 | 192.1 | 248.3 |
| b:attn.q 6144,5120 | 64 | 100.2 | 100.0 | 96.5 | 119.1 |

**`-ffp-contract=off` costs nothing.** `open` ≈ `open_fast` ≈ shipped inside run-to-run noise
(±5%), and both are bit-identical, i.e. this kernel has no contraction site to lose (as the
author's build.sh comment claims). The `_fma` regression is **`-mcumode`**, not contraction:
up to +33% (attn.q M=64 100→119 µs, mlp.gate_up M=64 187→248 µs) on the large-N shapes.
Do not put `-mcumode` outside the GDN unit.

### Harness trap found
`libr4d/test_mxfp4_gemm.py` run in this image prints `FAILURES: 0` **while skipping every
case**. `import r4d` (the pybind module) before torch touches the device leaves
`torch.cuda.current_stream()` raising `invalid argument to getCurrentStream`; the test
catches `Exception` per case, prints `SKIP (...)`, and exits 0. `k1/test_edge_dense.py` runs
the same grid over ctypes and does not trip it.

## 3. `MOE_ABI.md`

`$REPO/k1/MOE_ABI.md`. Reconstructed from the ctypes prototypes, the fork's
argument preparation, the registry table read out of the shipped binary
(`import r4d; r4d.kernels()`), the public dense sister kernel, and live probes.

Settled facts: `_block()==16` and `_group()==32` are compile-time constants (`mov eax,imm;
ret` in the binary), not queries; there is no `MB` argument and no M cap; `EM` is the sorted
buffer CAPACITY and the live length is `num_post_pad` read on device; `sorted_ids` pads with
the sentinel `Mtk`; `expert_ids == -1` skips a block; unnamed rows of `c` are never written;
GEMM2 passes `top_k=1`; the routed-weight pointer may be a literal 0. **The per-expert weight,
scale and reference-exponent layouts are byte-identical to the dense kernel's** — only a
leading expert dimension is added (strides `N*K/2`, `(K/32)*N`, `N`).

Probed live and passing: baseline writes exactly the live `sorted_ids` rows; `expert_ids=-1`
skips its block and leaves everything else unchanged; `num_post_pad` is honoured on device;
the `grid.x` tail (N/16 not a multiple of `WV*NPW`) is clamped, no fault.

Not established (stated as such in the doc): the grid decomposition, whether out-of-range
activation rows are clamped or masked, pointer alignment beyond 256 B, `expert_ids >= E`
behaviour, `EM` overflow points.

## 4. `moe_ref_harness.py`

`$REPO/k1/moe_ref_harness.py` (output `k1/moe_ref_harness.out`). Standalone —
imports torch only; `moe_align_block_size` is reimplemented in torch and cross-checked against
vllm's C op (**MATCHES vllm exactly**; note vllm's op is not stable within an expert, so the
harness compares block contents, not order). `R4D_LIB=<path>` picks the library under test.

Accuracy vs an exact torch dequant reference, shipped kernel, E=512 top_k=10:
`max_rel` (vs `|ref|max`) **1.9e-3 … 2.6e-3**, tail case 3.3e-3.
**Tolerance for a replacement: max_rel <= ~3e-3.**

Speed, best-of-25 µs, GPU 1 under the lock, VRAM weight vs whole expert stack in pinned host
memory reached through a UVA device pointer (the cold-expert case):

| case | N | K | mtk | µs VRAM | µs UVA | ratio |
|---|---|---|---|---|---|---|
| gate_up | 640 | 2560 | 10 | 44.3 | 326.0 | 7.4x |
| gate_up | 640 | 2560 | 80 | 98.7 | 2198.2 | 22.3x |
| gate_up | 640 | 2560 | 160 | 252.4 | 3952.7 | 15.7x |
| down | 2560 | 320 | 10 | 35.1 | 181.6 | 5.2x |
| down | 2560 | 320 | 160 | 117.5 | 1921.2 | 16.4x |

The UVA result is bit-identical to the VRAM result — a cold expert costs only time.

## Follow-up: cold-expert transfer study (COLD_TRANSFER.md)

| item | status |
|---|---|
| 5. transfer methods a/b/c + dual-GPU, gate_up & down, nsel 10/25/50/80 | DONE |
| 6. closed kernel on UVA (d) vs staged (e), end to end | DONE |

**Result: no win available; stage-then-compute would COST 3.3-9.4%.**
Host->device is capped at **28.4 GB/s per GPU** because the on-card PCIe switch uplink to the
EPYC 74F3 (Milan, Gen4-only) root port negotiates 16 GT/s x16 `(downgraded)` -- the endpoints

## Follow-up: cold-expert transfer study (COLD_TRANSFER.md)

| item | status |
|---|---|
| 5. transfer methods a/b/c + dual-GPU, gate_up & down, nsel 10/25/50/80 | DONE |
| 6. closed kernel on UVA (d) vs staged (e), end to end | DONE |

**Result: no win available; stage-then-compute would COST 3.3-9.4%.**
Host->device is capped at **28.4 GB/s per GPU** because the on-card PCIe switch uplink to the
EPYC 74F3 (Milan, Gen4-only) root port negotiates 16 GT/s x16 `(downgraded)` -- the endpoint's
"32 GT/s x16" is only the card-internal link. 55-60 GB/s is unreachable on this host.

- The closed kernel reading pinned host memory already runs at 26.7-28.3 GB/s (94-100% of the
  ceiling) and hides all of its compute behind the transfer: `uva us` == `stage us` to within
  0.3% at every size. Staging first serializes what is currently overlapped.
- Custom gather kernel (`cold_gather.hip`, 16-byte grid-stride loads off the UVA pointer) is
  the fastest transfer at 28.4 GB/s, but only 0.4-1% over `hipMemcpyBatchAsync` and over a
  plain contiguous copy of the same bytes. Per-expert `hipMemcpyAsync` in a loop is the worst
  (23.8 GB/s gate_up, 21.0 GB/s down) -- submit overhead.
- Both GPUs pull simultaneously at full rate: 56.8 GB/s aggregate, no root-complex contention.
- Staged output verified bit-identical to the UVA output in all 8 rows.
- New files: `cold_transfer.py`, `cold_gather.hip`/`.so`, `probe_peak.py`, `cold_transfer.out`,
  `COLD_TRANSFER.md`.

---

## Task C — expert locality / dynamic cache study (done)

Traces captured by team-lead's `routecap` server arm: `routes_rank0.npz` / `routes_rank1.npz`,
2376 steps, `lost_to_wrap = 1`, both ranks bit-identical. Segmented into 15 generations
(9 ab3 + the 6 labelled traffic.py prompts) by modal step width, not by the step counter.

Deliverable: `LOCALITY.md`. Headline:

* Static profile hot set @15 GB measures **23.31% distinct-miss / 562.6 MB/step / 19.8 ms/step**
  of PCIe at the 28.4 GB/s ceiling from `COLD_TRANSFER.md`. This agrees with the independent
  torch profile (~96 cold MoE calls/step at p75 150 us inside a 23 ms MoE GEMM budget).
* Warm-started per-layer **LRU at the same 12207 slots: 4.68% / 64.6 MB/step / 2.3 ms** (-88%
  of cold bytes). Belady bound is 2.60%, so LRU captures 86% of the achievable win.
* **Calibration**: profile advertises 0.851 coverage at 15 GB; measured routing-weighted
  coverage on real traffic is 0.782 -> cold fraction 0.218 vs 0.149 predicted, **1.46x**.
  Range across the 6 prompts is 10.6% (code) to 31.4% (json). Not a layer-permutation artifact
  (`align_check.py`: diagonal overlap 0.782 vs 0.678 best off-diagonal, argmax on the diagonal
  for 44/48 layers).
* **Iso-quality VRAM**: LRU needs ~7.0 GB/rank to match static@15 GB -> ~8 GB/rank freed.
* LFU is the wrong policy (+4.06 pp across context switches, stale scores). Pinning half the
  slots to the profile (hybrid) is 24% worse than pure LRU.
* Concurrency (synthetic merge of B independent generations): at B=4 LRU is still 2.5x better
  than static (9.98% vs 24.78%).
* **Free static win, independent of the above**: `hot_profile.json` has 48 layers, so the MTP
  head's MoE (router index 48) gets no hot set at all and every MTP expert read is cold today.
  Pinning that whole layer costs 0.63 GB/rank and removes ~1.1-1.7 ms/step.

Not established: any end-to-end measurement, B>1 on real concurrent traffic, and the cost of
the eviction bookkeeping (must be one fused kernel/step — the box already pays a 24% dispatch
tax at 3490 kernels/step).

Tools added: `locality_sim.py` (CPU/numpy, `POLICIES=`/`GB=`/`TRAFFIC_TABLE=1`/`SYNTH=iid`),
`align_check.py`, `mtp.py`, `probe_trace.py`. Raw output in `locality.out`,
`locality_sweep.out`, `locality_null.out`. Nothing committed; no foreign containers touched;
no GPU work in this task (CPU only, no lock needed).

---

## Task D — device-side LRU expert cache (built, validated, not yet server-tested)

`patches/hotcold/r4d_mxfp4_moe_lru.py` (insert-only diff vs `r4d_mxfp4_moe.py`; team-lead's
census / side-stream / guarded-import lines are byte-identical, verified with difflib) plus
`k1/lru/librlu.so` (`lru_manage`, `lru_gather`, gfx1201, ROCm 10 hipcc). Gated on
`VLLM_R4D_LRU=1`; with it off the module is behaviourally today's static patch. Full design,
knobs and validation table in `k1/lru/README.md`.

Validation, all under the GPU lock on one R9700:
* policy vs an independent numpy LRU: 6 suites, 1230 steps, every field every step — PASS
* real r4d GEMM, (resident + fallback) vs one all-UVA call, 13.3 inserts/step — bit-identical
* HIP graph: 20 replays, cache keeps adapting, invariants hold, gathered bytes identical
* full 2330-step production trace x 48 layers through the real kernel with the real 15 GB hot
  set: static 23.33% / 432.1 MB/step -> LRU 4.62% / 85.6 MB/step (-80.2%), within 0.3% of
  `locality_sim.py`
* cost: manager 5.91 us/layer, empty gather 3.41, both 8.98 -> 431 us/step over 48 layers,
  i.e. a 3.7% tax on a 12.2 ms/step saving; gather hits the 28.4 GB/s COLD_TRANSFER ceiling

**Correction folded back into LOCALITY.md**: the first version billed PCIe bytes using the
post-warm-up miss/step, which drops every segment shorter than the warm-up and reweights the
mix. All byte/ms figures are now all-inclusive: static 431.9 MB/step (15.2 ms), LRU 86.8
(3.1 ms), saving 12.2 ms/step, upper bound ~1.23x decode — not the 17.5 ms / 1.39x first
reported. The kernel replay is what caught it. Section 6 (MTP layer allegedly all-cold) is
RETRACTED per K2: the MTP module is a plain ModuleList the offloader never wraps.

Timings were taken with team-lead's `q38fn-mxfp4` arm resident on the same card (idle);
gather bandwidth matches the clean ceiling but the manager figure wants a re-measure on a
quiet GPU. Nothing committed; no foreign containers touched; only `k1/**` and the new
`patches/hotcold/r4d_mxfp4_moe_lru.py` written.

---

## Task E — concurrency harness, per-layer slot allocation, MAX_INSERTS (2026-09-04)

### E.1 `k1/concurrent_bench.py` (for the coordinator to run under the lock)
Fires B simultaneous DIFFERENT-prompt streams (8-prompt pool, mixed code/prose/json) at :8057
and reports aggregate + per-stream tok/s, TTFT, and the engine's ms/step.

The trap it avoids: `vllm:spec_decode_num_drafts_total` counts one draft per RUNNING REQUEST
per engine step, so with B in flight it advances by ~B per step. A 0.25 s background sampler
reads it together with `vllm:num_requests_running` and converts
`engine_steps(window) = delta_drafts / running`; the headline "steady" ms/step uses only
windows where `running == B` at both endpoints, so ramp-up/ramp-down cannot bias it.
Token counts come from the server's `usage` block (stream_options.include_usage), never from
counting SSE chunks -- vLLM coalesces ~2.7 tok/chunk under MTP.

    B=1,2,4 N=500 REPS=1 python3 concurrent_bench.py out.json     # env: BASE, SAMPLE, WARMUP

Metric names verified present on the live 8057 server. Not run by me: firing benchmark traffic
would have polluted the coordinator's in-flight LRU=0 control.

### E.2 `k1/slot_alloc.py` -- is the per-layer slot split still right under LRU?
Per-layer miss(C) curves for ALL C in one pass via STEP-GRANULAR stack distances (Mattson with
a Fenwick tree), which is exactly the kernel's semantics: every expert routed in a step becomes
equally-MRU and none can be evicted by its own step. Warm start is modelled by prepending the
profile hot set in reverse rank order, so at cap C the surviving warm entries are ranks 0..C-1
-- what `_build_lru` actually loads. Allocation solved by concave-hull greedy on the cumulative
gain curves (optimal for the hull relaxation, integral at every hull vertex).
Cross-check vs the exact LRU replay at the shipped split: 5.19% vs 4.94% (+0.25 pp; the gap IS
batch-vs-access granularity, and batch is what the kernel does).

Fit on the 9 ab3 generations, scored OUT OF SAMPLE on the 6 traffic.py prompts (925 steps):

    slots   GB     profile   equal    demand   lru-opt(fit)   lru-opt(score, in-sample bound)
     4000   4.92    28.95%   28.43%   27.92%     27.62%          27.27%
     6000   7.37    18.63%   18.50%   17.37%     16.93%          16.74%
     8000   9.83    11.75%   12.26%   11.31%     10.95%          10.79%
    10000  12.29     7.74%    8.44%    7.56%      7.36%           7.20%
    12207  15.00     5.19%    5.80%    5.14%      5.06%           4.89%
    14000  17.20     3.92%    4.40%    3.86%      3.83%           3.70%
    16000  19.66     2.87%    3.20%    2.80%      2.80%           2.72%

EQUAL SLOTS IS WORSE than the profile water-fill at every budget above 6000 (+4 to +12%).
The optimal split correlates 0.850 with the profile split, mean |delta| 22.7 slots at 12207.
Per-layer vectors for every scheme and budget: `k1/slot_alloc.json`.

### E.3 Price of VRAM after LRU (lru-opt(fit) allocation, 28.4 GB/s PCIe ceiling)
    16000 slots  51.0 MB/step  1.80 ms      10000  134.0 MB/step  4.72 ms
    14000        69.8          2.46          8000  199.4          7.02
    12207 (now)  92.1          3.24          6000  308.3         10.86
Weight budget x1.19 = actual VRAM. Dropping 12207 -> 10000 frees ~3.2 GB/rank of real VRAM and
costs +1.5 ms/step (~4% of a 35 ms step). The curve is far flatter than the pre-LRU one
(~2 ms/GB): the marginal GB is now worth 0.4-0.8 ms/step.

### E.4 MAX_INSERTS and the ascending-expert-id truncation bias
Merged traces, B independent generations unioned into one step, shipped split, THRESH=0.5:

    distinct experts/layer/step:  B=1 mean 31.4 max 50 | B=2 42.2/96 | B=4 69.1/139 | B=8 84.2/133
    per-(layer,step) MISS count:  B=1 p50 1 p90 4 p99 8 max 30
                                  B=2 p50 2 p90 6 p99 13 max 30
                                  B=4 p50 5 p90 13 p99 23 max 37
                                  B=8 p50 6 p90 14 p99 23 max 37
    steps with >64 misses: 0.00% at every B. miss% is flat from MAX_INSERTS=32 to inf
    (B=4: 9.34% at 32/64/128/inf, 9.31% at 16). Mean resident expert id 254.7-256.8
    (unbiased 255.5) -- the ascending-id truncation never fires, so there is no id bias.
    B=8 is 1 merged group only (few long-enough generations); treat it as indicative.

Read-through gate: fires on 0.12% of (layer,step) at B=4, 0.33% at B=8, never at B<=2.

VERDICTS: keep MAX_INSERTS=64 (32 would also do; 16 is the first value that costs anything).
Keep THRESH=0.5. Keep the profile water-fill split at 12207 -- worth only -2.6%; switch to
`lru-opt` only if the budget drops, where it is worth -6.8% at 8000 and -9.1% at 6000.

### E.5 Numerics: hypothesis for the LRU=1 vs LRU=0 text divergence
The "does it matter which of the two calls computes a given (token,k) row?" question is
ALREADY answered by test_numerics_lru: it compares the two-call split against ONE all-UVA call
over the same routing while the LRU kernels reshuffle the partition every step, and it was
bit-identical. So the r4d grouped GEMM is partition-invariant -- at E=64, S=24, mtk=40.
What is NOT covered is production scale: E=512, S~257, and the real mtk values (50 decode,
10 draft, 200 at B=4, 20480 prefill), where each align carries an order of magnitude more
expert blocks. `k1/lru/test_numerics_prod.py` runs exactly those; it needs a GPU window.
Second point, more likely to be the whole story: greedy output on this server is NOT
reproducible across restarts at identical config, so an LRU=1 vs LRU=0 text difference is not
evidence of anything until the base-vs-base control lands.

## Task F -- LRU numerics audit (2026-09-04)

**Verdict: no correctness defect in the LRU cache. The greedy/logprob divergence is
restart-to-restart nondeterminism**, confirmed at the server level by team-lead (two fresh
LRU=0 servers differ from each other exactly as LRU=0 differs from LRU=1; the LRU=1 hot-17
server is bit-identical to the second LRU=0 control). My probe-file analysis had already put
LRU=1 vs LRU=0 at ~2x the restart noise floor with *uniform* scaling and no tail, and code
prompts at or below the control -- an amplified accumulation-order effect, not a wrong-slot
defect. (The first lru1 probe set was additionally polluted: it was taken while the 256K
needle test shared the server, so those decode steps ran batched at a different M.)

### GPU verification, under `flock $REPO/gpu.lock` on GPU 1, image `local/q38fn-rocm10:k1build`

`k1/lru/test_numerics_prod.py` -- production shapes, real geometry, PASSED bit-identical:
gate_up (N=640 K=2560) and down (N=2560 K=320), E=512 top_k=10 slots=257, at M=5 (decode
mtp4, mtk=50), M=1 (draft), M=20 (B=4 decode) x 30 steps, plus M=2048 prefill. Split
(resident + fallback) == one all-UVA call, bit-for-bit, on every step, including the padded
`-1` rows and the `_into` y-buffer semantics. mtk=50 is not a multiple of 16, and `tw` is
passed on the down leg: both covered.

`k1/lru/test_slots_prod.py` -- real routing from `routes_rank0.npz` layer 7, 120 real decode
steps, 303 inserts, sources in pinned host memory reached by device pointer (== how
`--cpu-offload-params` presents them, so the gather really crosses PCIe). PASSED:
- (a) every inserted (expert, slot): all six buffers byte-equal to the UVA source
      (w1, w2, w1_ws_t, w1_wref, w2_ws_t, w2_wref; per-expert 819200/409600/51200/640/25600/2560 B,
      all 16 B multiples, expert dim leading in the repacked layout as `_lru_step` assumes)
- (b) full residency audit every 10 steps: *every* slot, not just this step's inserts, so a
      slot that went stale ten steps ago is caught
- (c) table/map_cold complementarity over all E: no expert in both, none in neither,
      `table[slot_expert[s]] == s`, slot_expert injective, exactly S resident
- (d) split vs all-UVA bit-for-bit with real routing

**Negative control (this is why "PASSED" means something):** `FAULT=<step> FAULT_BYTES=<n>`
scribbles a just-inserted slot. 1 byte -> (a) fires at that step and (b) fires at every later
audit; (d) does NOT fire, because a single flipped mxfp4 nibble out of a K=2560 dot product is
absorbed by bf16 rounding. 1280 bytes -> all three fire. So the split-vs-all-UVA numerics check
alone was never sufficient to catch a corrupt slot; the byte-level residency audit is the one
that would have.

## Task G -- slot_stamp warm start (2026-09-04)

**Applied** to `patches/hotcold/r4d_mxfp4_moe_lru.py` (and the identical `k1/lru/` copy).
`_build_lru` seeded every warm slot with stamp 0, so the kernel's `(stamp, slot)` tiebreak
evicted the lowest slot index -- which, since `hot_ids_for_layer` returns sorted ids, is the
lowest *expert id*. Arbitrary. Now stamps are seeded by profile rank via a new
`profile_rank_for_layer()`: `slot_stamp = S - argsort(argsort(rank))`, giving distinct stamps
in [1, S] with the most-frequent expert highest, and `step` starts at S so the first real step
stamps S+1 and any hit outranks every warm start. Stamps stay non-negative, so
`(stamp << 20) | slot` neither overflows nor relies on shifting a negative (which the previous
"seed negative ranks" idea would have).

Measured in simulation on the real traces (`k1/stamp_ab2.py`, 15 generations, water-fill
split, MAX_INSERTS=64):

    B    zero (shipped)   profile rank
    1        4.469%         4.349%   -2.69%
    2        5.978%         5.851%   -2.13%
    4        9.219%         9.111%   -1.17%
    first 20 steps, B=1: 7.982% -> 7.201%  (-9.78%)

**And it caught a bug in my own simulator.** `slot_alloc.KernelLRU` seeded
`stamp = -(len(init) - j)` with j=0 the hottest, i.e. it evicted the *hottest* profile expert
first -- the exact inverse of the intended policy, and 1.3-2.9% worse than even the shipped
zero-stamp behaviour. Fixed. This means the Task E MAX_INSERTS numbers were computed under the
worst of the three policies, so they *over*-state misses; the conclusions (MAX_INSERTS=64,
THRESH=0.5, keep the water-fill split) hold a fortiori. The allocation curves used
`curves()`/stack distances, which warm-start correctly in reverse rank order, so the
allocation table itself is unaffected.

Kernel re-verified with the production initialization (`STAMP=rank` in `test_slots_prod.py`:
distinct non-zero stamps, `step` starting at S): PASSED, 305 inserts over 120 steps.

`VLLM_R4D_LRU=0` is unchanged: `diff r4d_mxfp4_moe.py r4d_mxfp4_moe_lru.py` removes zero lines
from the base, and every LRU call site is gated (`if _LRU_ON` in `_build_hot`, `"lru" in h` in
`_apply_split`).


## Task H (step 0) -- what one kernel launch actually costs, 2026-09-04

Before writing the fused kernel, measured the thing the estimate rested on.
`k1/lru/step0_nodecost.py` captures HIP graphs of
`52 x [lru_manage, lru_gather, k x (moe_align(table), moe_align(map_cold), tw cast)]`
plus 2886 filler nodes (15.6 us elementwise kernels, interleaved) for k = 0..3, so node
counts 2990/3250/3510/3770 bracket the real ~3250-node decode step, and times replay.
Round-robin over k; 0.2% spread within an arm. Ran under the GPU lock with team-lead's
server resident, so the 52 layers share one set of slot buffers (16.5 GB otherwise) --
timing harness, every layer still has its own table/stamps/miss list.

  k=0 45609 us | k=1 46376 | k=2 47143 | k=3 47914  ->  **2.95 us/node**, dead linear.
  control arm, same node count but one-element kernels: **2.40 us/node**.

So a launch node costs 2.40 us of pure dispatch + 0.55 us of the removed kernels' own work.
Production's median inter-kernel *gap* (3.72 us) overstates the *marginal* cost by 26%.

Second correction, from reading the installed vLLM: `prepare_finalize/no_dp_ep.py`
`topk_indices_dtype()` returns None, so `fused_topk` allocates `topk_ids` int32 and
`topk_weights` fp32. Both `.to()` calls in the patch are no-ops: no hidden 8th launch at
`_lru_step`, and the fp32 cast was never a kernel. Bookkeeping is **6 launches, not 7**,
and fusion removes **4, not 5**.

Net: 4 x 52 x 2.95 = **0.61 ms/step gross, ~0.5 ms net** = 1.5% at B=1, 0.6% at B=4.
Above my own 0.4 ms go/no-go bar, so: proceed. The generally useful number from this is
2.4 us per launch anywhere -- the step's ~3250 nodes carry ~7.8 ms of dispatch (23%).


## Task H -- fused routing bookkeeping, SHIPPED behind a flag, 2026-09-04

`lru_fused_k` (k1/lru/r4d_lru.hip -> **k1/lru/librlu_fused.so**) does the manager AND both
moe_align_block_size outputs in one workgroup: **6 launches per MoE layer -> 2**. The two
existing kernels are unchanged; the new .so is a superset of librlu.so, kept separate so
nothing overwrites a .so a live server has mmapped.

Enable with `VLLM_R4D_LRU=1 VLLM_R4D_LRU_FUSE=1 R4D_LRU_LIB=/w/build/kernels/librlu.so`.
Default off. FUSE=1 against a .so without the symbol raises instead of falling back.
`VLLM_R4D_LRU_FUSE_MAX` (default 2048 (token,k) slots) keeps prefill on the old path.

**Measured 0.478 ms/step** (k1/lru/bench_fused.py: 46073 -> 45595 us at 3198 -> 2990 nodes),
9.2 us/layer, against the 0.499 ms pure-dispatch ceiling. 1.4% at B=1, 0.6% at B=4.
First cut was only 0.249 ms: two blkExScan calls = 32 barriers = ~7 us/layer. One chunked
two-level dual scan (3 barriers) plus LDS-resident placement keys recovered the rest.

**Validated** (k1/lru/test_fused.py, all PASS + working negative control): manager state
bit-identical to r4d_lru_manage; npad/expert_ids exact and sorted_ids per-expert-identical
vs vllm's own op; hot+cold partition [0,mk); forced read-through and zero-miss steps; 12 HIP
graph replays == 12 eager calls; mk = 10/50/200/400/2050/20480. End-to-end
`FUSE=1 test_numerics_prod.py` is bit-identical to the all-UVA reference.

**Comparator correction worth keeping:** vllm's moe_align is unstable *within an expert*, not
just within a block -- when an expert spans several blocks the split of its tokens across
them varies too. A per-block set comparison passes at mk=50 and produces false failures at
mk>=400. Compare per expert.

**Generator caught up:** k1/lru/gen_lru_py.py now emits the Task G stamp warm-start (which had
only been hand-applied to the generated file -- a regeneration would have silently reverted
it) and the fused path. Regenerating reproduces patches/hotcold/r4d_mxfp4_moe_lru.py exactly.
Base vs LRU diff is still insertion-only except four lines: the two moe_align calls now sit
in an `else:`. With VLLM_R4D_LRU=0 nothing there executes differently.

---

## Task I(a) -- the victim loop: 0.9 us/insert, but a cliff at 14 inserts. Fixed.

`bench_victim.py` (new). The manager is restored from a pristine packed state buffer and
relaunched inside a HIP graph, so no host launch cost is in the number; the routing is
built to miss exactly k experts, so every call does exactly k inserts and `t(k) - t(0)`
is the marginal cost of an insert. Eager launches cost ~47 us of CPU each and hid the
whole kernel -- the first two attempts measured the CPU, not the GPU.

**Answer to the question as asked: no, it is 0.9 us/insert, under your 2 us bar.**
But the cost is not linear. `us/call`, S=257, mk=200, min of 7 rounds:

| inserts | 0 | 1 | 2 | 4 | 8 | 13 | **14** | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| old (serial argmin) | 6.90 | 8.67 | 9.50 | 11.39 | 14.24 | 18.61 | **43.4** | 86.0 | 224.9 |

(k = 1..4 for the old kernel come from a separate low-k run, 7.12 us at k=0; the two
runs agree to ~0.35 us. At k <= 4 the two kernels are within noise of each other.)
| new | 6.97 | 8.61 | 9.21 | 10.57 | 11.14 | 11.12 | **11.05** | 11.92 | 12.97 |

Past 13 inserts the old kernel's time jumps by 25-32 us in one step and then keeps
climbing at ~3 us/insert. The cliff is at k=14 for S=257 and S=320, k=16 for S=200, and
absent at S=129 up to k=16; it does NOT move with E (2048/4096 put it at 14 as well) and
it is not a kernel-duration threshold (E=4096 k=12 runs 23 us with no cliff). Not root
caused. It is reproducible across seeds, S, E and three different k-lists.

Why it matters: B=4 already runs 5-13 inserts/layer/step, i.e. immediately below the
cliff. 52 layers x 25 us = **+1.3 ms/step** the moment a step crosses it.

**Fix (`pickVictims` in r4d_lru.hip, both kernels).** The serial loop takes argmin(stamp,
slot) over the evictable slots, installs a missing expert there -- which makes that slot
non-evictable, because the expert it now holds IS routed this step -- and repeats.
Installing changes no other slot's key and no other slot's evictability, so the sequence
it produces is exactly the nins smallest keys of the set that was evictable on entry, in
ascending order. So: build one key per slot in LDS (LLONG_MAX parks the ones that must not
be evicted), rank each evictable slot against that array, hand rank r to miss r. The nins
updates then run in parallel -- victim slots are distinct, missing experts are distinct,
and an evicted expert is by definition not routed this step while every inserted one is,
so no two updates touch the same table/map_cold entry.

Ranking is a fixed ~3.7 us at S=257 (256 threads x S LDS reads) no matter how many
inserts, which is a regression below ~4 inserts, so short lists keep the serial argmin
(`NSER 4`). Both paths are in the same function and produce the same state.

Validated by `test_victim_equiv.py`: old library vs new, same pristine state, comparing
table / map_cold / slot_expert / slot_stamp / miss / n_miss byte for byte, for both
`r4d_lru_manage` and `r4d_lru_fused`, over 19 cases -- production B=1 and B=4, at and past
the cliff, zero misses, capped by max_inserts, max_inserts 1 and 1024, empty slots, stamp
ties (all-equal stamps), nothing-evictable (ev=0) on both paths, S=1024, read-through --
plus a 3-perturbation negative control (3/3 caught). Then `test_fused.py`,
`test_numerics_prod.py` (baseline and FUSE=1, both bit-identical), `test_slots_prod.py`
all pass on the new library, and `bench_fused.py` is unchanged (0.466 vs 0.464 ms/step
saved by the fusion).

**Artifact: `$REPO/k1/lru/librlu_v2.so`** = fused kernel + fixed victim
selection. `librlu.so` and `librlu_fused.so` were NOT touched (both are mmapped by running
servers). Same ABI, so it is a drop-in `R4D_LRU_LIB` swap. Source backup of the pre-change
kernel: `r4d_lru.hip.pre-victim`.

Not changed: `max_inserts > 1024` now returns -1 from both wrappers (`s_vic` is MAXINS
deep; in the fused kernel it aliases `cnt_c[MAXE]`). Production uses 64.

## Task I(b) -- what each MTP draft row costs in cold traffic

`mtp_prefetch.py` (new), offline on `routes_rank0.npz` (2376 steps, 15 generations, modal
width 5 rows = 1 target + 4 drafts). For r = 1..5 keep only the first r rows of every step
and replay the same LRU at the same 15 GB water-filled per-layer caps, warm started from
the profile hot set. Caveat: the token sequence is the one MTP-4 produced; a real MTP-2 run
would generate different text, so this isolates the routing-width effect only.

| rows kept | distinct experts/layer | marginal | LRU miss/layer/step | MB/step | marginal | ms @28 GB/s |
|---|---|---|---|---|---|---|
| 1 (no spec) | 10.0 | -- | 0.692 | 40.8 | -- | 1.46 |
| 2 | 16.2 | +6.2 | 0.957 | 56.5 | +15.6 | 2.02 |
| 3 | 21.8 | +5.5 | 1.148 | 67.7 | +11.3 | 2.42 |
| 4 (MTP-3) | 26.7 | +5.0 | 1.309 | 77.2 | +9.5 | 2.76 |
| 5 (MTP-4) | 31.4 | +4.7 | 1.445 | 85.2 | +8.0 | 3.04 |

The 4 speculative rows add 44.4 MB/step over the target row alone: **1.59 ms/step of the
2-3 ms cold traffic is caused by speculation**, and the 5th row (MTP-3 -> MTP-4) costs
+8.0 MB = **+0.29 ms/step**.

Measured MTP-3 -> MTP-4 under the LRU combo is +2.6 ms/step (29.3 -> 31.9). So cold expert
traffic explains only ~11% of what an extra draft token costs; the other ~2.3 ms is the
GEMM/attention/kernel work of a 5th row. **Deeper MTP is not being priced by the expert
cache** -- the LRU already flattens the routing-width penalty (miss/layer/step grows
sublinearly, 0.69 -> 1.45 for 5x the rows, because the extra rows mostly re-route to
experts the target row already pulled).

## Task I(c) -- prefetch: no predictor beats LRU residency

Scored against the LRU's own miss stream at 15 GB, second half of each generation, layers
1..47, 64,095 misses. Every predictor is filtered to experts NOT resident at prediction
time, so "predict what the cache already holds" scores zero by construction.

    P(e routed at layer L step t | routed at layer L step t-1) = 0.451

| predictor | P=1 | P=2 | P=4 | P=8 | P=16 |
|---|---|---|---|---|---|
| oracle (true miss set) | 46.2% | 70.6% | 91.1% | 99.1% | 100% |
| last step's set, same layer | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| global frequency (profile rank) | 0.4% | 0.8% | 1.5% | 3.0% | 6.4% |
| cross-layer co-occurrence L-1 -> L | 2.0% | 3.4% | 6.0% | 10.0% | 16.2% |

(cover % of misses; the co-occurrence model is C[e_prev, f] trained on the first half of
each generation and scored as sum over the experts layer L-1 routed at THIS step, so it is
available before layer L's router runs.)

"Re-gather what layer L routed last step" covers exactly **0.0%** -- not approximately
zero, exactly zero, because the LRU inserted every one of them at t-1 and nothing has
evicted them by t. Global frequency is worse than useless (the hot set already holds those).
The learned cross-layer model reaches 16% coverage only at P=16, i.e. **44 wasted pulls per
hit** -- 16 speculative expert pulls per layer per step is 10x the real miss traffic.

The reason is in the misses themselves:

| when the missing expert was last routed by this layer | share |
|---|---|
| within the last 16 steps | 0.0% |
| 17-64 steps ago | 17.1% |
| more than 64 steps ago | 16.2% |
| never, in this generation | 66.7% |

Two thirds of the misses are compulsory -- the layer has never routed that expert in this
generation, so no history-based predictor of any kind can name it -- and not one miss is a
recently-evicted expert coming back, so the LRU is not thrashing and a victim buffer would
buy nothing. **No predictor beats LRU residency.** The only prefetch lever left would be
predicting the router's output from the hidden state before the router runs, which is a
model change, not a cache change; the size of that prize is the oracle row, ~2.4-3.0 ms/step.

## Note -- K4 edited r4d_mxfp4_moe_lru.py (gate off by default)

K4 replaced the three post-GEMM1 statements (`ic2 = torch.empty` / `self.activation` /
`ops.scaled_fp8_quant`) in both `apply` and `_apply_split` with one `self._act_quant(...)`
call, plus a module-level `VLLM_FUSED_SILU_QUANT` gate (default 0) and a guarded
`_fused_silu_quant()` accessor. Diffed against their backup `/tmp/lru.bak`: with the gate
unset it is my exact original three lines. Backups: /tmp/lru.bak, /tmp/moe.bak.

Answered their two questions:
- `VLLM_R4D_HOT_SIDE_STREAM` is neutral for the fused kernel. `_gemm_split` ends with
  `cur.wait_stream(side)`, so the current stream has already joined before `_act_quant`
  runs; `ic1` is complete and the fused kernel reads it on the current stream exactly like
  `silu_and_mul` did. `_SIDE` is measured-dead anyway (halved throughput under graph
  capture) and no arm sets it.
- The real invariant their kernel depends on in `_apply_split` is row coverage, not the
  stream: `ic1` is `torch.empty` and the hot/cold GEMMs fill disjoint row sets that are
  exhaustive (every (token,k) slot is in exactly one of map_hot/map_cold; the LRU manager
  preserves that). Flagged it so nobody adds a "skipped rows" mode later.
- `VLLM_R4D_SHARE_A8=0` is not mine. It is not in any .sh; team-lead passes it per-arm on
  the run_arm.sh command line, and 0 is already the default in r4dhip.py.
