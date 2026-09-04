# `hcq_gemm_fp8blk_nt_m16` — small-M fp8 block-scaled skinny GEMM for gfx1201

**Status: experimental, correct, and NOT worth shipping on. Default off.**
It is 1.01x on the production mix in isolation, and a dead heat in the running server.
This document exists because the *measurement method*
that established that is worth more than the kernel, and because the kernel is a working,
graph-safe reference for anyone who wants to do W4 or a fused epilogue on these layers.

Sources: `kernels/experimental/fp8skinny/hcq_fp8skinny.hip`, `kernels/experimental/fp8skinny/build.sh`, `patches/model_executor/kernels/linear/scaled_mm/fp8hip.py.diff` (env-gated dispatcher),
`tests/k3/test_fp8sk.py`, `tests/k3/bench_fp8sk.py`, `tests/k3/bench_roof.py`.
Hardware: 2x AMD Radeon AI PRO R9700 (gfx1201, 64 CU / 32 WGP, 644.6 GB/s nominal, 64 MB
Infinity Cache), ROCm 10, vLLM fork image `local/q38fn-rocm10:try1`, Qwen3.8-Flash-Next TP2.

---

## 1. What it computes

    C[M,N] (bf16) = (A[M,K] * As[M,K/128]) @ (W[N,K] * Ws[N/128,K/128])^T,   M <= 16

the exact operand set the closed `fp8hip_gemm_w8a8_tiled` takes, **byte for byte, including the
pre-shuffled weight**. `shuffle_weight_gfx1201` is

    w.view(N/16, 16, K/32, 2, 2, 8).permute(0, 2, 4, 1, 3, 5)

which places byte `(n, k)`, `n = 16*nt + r`, `k = 32*k32 + 16*d3 + 8*d4 + c`, at
`((nt*(K/32) + k32)*2 + d4)*256 + r*16 + d3*8 + c`. That **is** gfx1201 wave32 fp8 WMMA
fragment order: lane `l` wants row `l&15` and the 8 k of the 16-step at `8*(l>>4)`, so `d4 == l>>4`
and lane `l`'s sixteen contiguous bytes at `r*16` are the `d3=0` and `d3=1` fragments of one 32-K
step. One `global_load_b128` per lane per 32 K feeds two WMMA; the wave reads 512 contiguous
bytes; nothing is staged in LDS. Consequence: **the new kernel needs no second copy of the weight
in VRAM** — it reads the tensor the closed kernel already has. (`tests/k3/test_layout.py` proves the
layout on CPU, no GPU needed.)

Block scales are applied per 128-K chunk, not per element:

    sum_k a_q[m,k]*as[m,kb] * w_q[n,k]*ws[nb,kb]  =  sum_kb as[m,kb]*ws[nb,kb] * sum_{k in kb} a_q*w_q

with the inner sum being 8 WMMA into one fp32 accumulator. That is a regrouping of the same
products — not an approximation — but a different fp32 **association**, so it is not expected to be
bitwise equal to fp8hip, and is not (see §5). A 16-row N tile always lies inside one 128-row scale
block, so `ws` is one scalar per (tile, kb); `as` varies across the 8 accumulator elements of a lane
and is staged into LDS **transposed** at entry so a lane reads its eight values as two `ds_read_b128`.
Split-K is reduced through LDS inside the workgroup — no HBM partial buffer, no counter, no
dependence on wave scheduling, therefore HIP-graph replayable (asserted in the test).
Rows past `M` are clamped on the A load and their activation scale zeroed, so the inner loop is
branch-free.

Tuning knobs are compile-time-shaped, passed at launch: `WV` waves per N direction, `SK` split-K
ways, `NPW` N tiles per wave. Preconditions return negative codes rather than misbehaving:
`-1` M outside 1..16, `-2` N%128, `-3` K%128, `-4` (K/128)%SK, `-5` WV*SK*32>1024, `-6` NPW not in
{1,2,4}, `-7` LDS > 64 KB.

## 2. Why it was written (the hypothesis)

The 96 fp8 GEMMs per decode step per rank are exactly: gdn `in_proj_qkvz` N=8192 K=2560 x36,
gdn `out_proj` N=2560 K=3072 x36, qsa `qkv_proj` N=6656 K=2560 x12, qsa `o_proj` N=2560 K=3072 x12.
(MTP attention is in the checkpoint's ignore list, so it is bf16 and not in this family.)

`fp8hip_gemm_w8a8_tiled` launches **exactly N/128 workgroups** of 256 threads. From the eager
production trace (`prof_eager_h15`, rank 0, 23 steps, `k3/trace_fp8.py <trace> fp8hip`):

| shape (per rank) | grid | calls/step | us/call | weight MB | GB/s | % of 644.6 |
|---|---|---|---|---|---|---|
| `in_proj_qkvz` N=8192 K=2560 | 64 | 36 | 42.46 | 20.97 | 494 | 77% |
| `qkv_proj` N=6656 K=2560 | 52 | 12 | 36.48 | 17.04 | 467 | 72% |
| `out_proj`/`o_proj` N=2560 K=3072 | 20 | 48 | 26.03 | 7.86 | **302** | **47%** |

20 workgroups on a 64-CU part, at less than half the nominal bandwidth, on a shape that is pure
weight streaming at decode M (A is 15 KB, the scales 2 KB). The obvious read is under-occupancy,
and the obvious fix is more workgroups: one 16-wide N tile per wave times a k-split, the
decomposition every skinny GEMM in libr4d uses. Predicted saving ~1.4 ms/step.

**That read was wrong.** See §4.

## 3. The control: a kernel that only reads the bytes

Never benchmark a memory-bound kernel against a nominal spec roofline. 644.6 GB/s is not
reachable by *any* kernel with this access pattern, so "47% of roofline" measured nothing.

`hcq_stream_probe` (same .so) is a kernel that **only reads** the same weight pool, in the same
b128 pattern, over the same byte count, launched in the same HIP graph, with a configurable
workgroup count. That is the number a GEMM on this shape can actually be held to.

Two sizing rules that decide whether the number means anything:

* **The pool must be bigger than the 64 MB Infinity Cache.** A single 8-21 MB weight timed in a
  loop is cache-resident and reports fantasy bandwidth. `bench_fp8sk.py` and `bench_roof.py` both
  allocate a **1 GB pool** of distinct weights (51/63/136 copies depending on shape) and time the
  whole pool inside one graph, dividing by the number of calls.
* **Time inside a HIP graph, with a pre-allocated output.** Python-loop A/B on these shapes
  measures launch overhead, not the kernel.

Do **not** use `torch.sum(W.view(torch.uint8))` as a read probe: it is a reduction with its own
arithmetic, and reports 35 GB/s. That mistake is what made the roofline argument look plausible
for an hour.

Measured control ceilings (`k3/bench_roof.json`, best over a 20..2048 workgroup sweep):

| shape | best us | best wgs | **GB/s** |
|---|---|---|---|
| N=8192 K=2560 | 35.36 | 128 | **593** |
| N=6656 K=2560 | 29.18 | 128 | **584** |
| N=2560 K=3072 | 14.71 | 2048 | **535** |

## 4. Result: both kernels are at the ceiling

Isolated, in-graph, 1 GB pool, M=5 (the production decode row count with MTP-4), best `(WV,SK,NPW)`
per shape (`k3/bench_fp8sk.json`):

| shape | fp8hip us (GB/s) | hcq us (GB/s) | control us (GB/s) | fp8hip % of ctl | hcq % of ctl |
|---|---|---|---|---|---|
| N=8192 K=2560 | 36.65 (572) | 36.75 (571) | 35.36 (593) | 96.5% | 96.2% |
| N=6656 K=2560 | 30.44 (560) | 30.36 (561) | 29.18 (584) | 95.8% | 96.1% |
| N=2560 K=3072 | 16.13 (488) | 15.57 (505) | 14.71 (535) | 91.2% | 94.4% |

Weighted over the 96 calls/step: **fp8hip 2.459 ms, hcq 2.435 ms, delta -0.024 ms/step = 1.010x.**
Best single cell is the 20-workgroup shape at M=16: 16.81 -> 15.86 us, 1.06x. Nothing else moves.

**Why the workgroup-count hypothesis was wrong.** The control probe *does* reproduce the
suspicious number: at **20 workgroups it reaches only ~320 GB/s**, close to the 302 the trace
showed — which is exactly why the premise was believable. But fp8hip's 20 workgroups are not the
probe's 20 workgroups. They are 256 threads each, deeply unrolled, with many loads in flight per
thread, and they reach **488 GB/s**. The gfx1201 memory system is saturated by **outstanding loads
per thread x threads**, not by workgroup count; a grid of 20 fat workgroups and a grid of 128 thin
ones land within a few percent of each other. The new kernel spreads the same work over ~10x more
waves and buys 3.5% on the one shape where fp8hip was furthest from the control, and 0% elsewhere.

The residual 302 -> 488 GB/s difference between the trace and the isolated measurement is **not**
the kernel. It is contention: the eager trace inflates every call, and even the graph-mode
production arm does. In-situ (`prof_c7_rope`, rank 0, 8064 calls) the family runs at
**29.65 us/call mean = 2.846 ms/step**, against 25.61 us/call isolated — **+16%, 0.39 ms/step, 13.6%
of the family's production time, spent sharing the memory system with the rest of the step.**
No GEMM kernel can recover that; only removing other traffic can.

**In-situ confirmation (arm t9 vs t8b, the only measurement that decides it).** Swapped into the
production server behind `VLLM_HC_FP8SK=1`, the 96 calls/step come to **2.81 ms/step with hcq and
2.81 ms/step with fp8hip**; end-to-end `ab3` is **29.2 / 29.3 / 29.8 ms/step** against
**29.5 / 29.6 / 30.4** for stock — inside the run-to-run noise on all three prompts. The isolated
-0.024 ms/step did not survive contact with the rest of the step, exactly as the isolated
measurement predicted it would not. **fp8hip stays.**

**Remaining lever: fewer weight bytes.** These calls are 100% weight-streaming at decode M and both
kernels are within 4-9% of a pure-read kernel, so time is `weight_bytes / ~550 GB/s` and the only
way down is W4 on the attention/GDN linears (libr4d already has `gemm_w4a8_nt_m64` /
`gemm_mxfp4a8_nt_m64`, and this kernel is a working template for the block-scale epilogue).
That would be ~-1.2 ms/step in isolation — **and it is a quality decision, not a kernel decision:**
it is the reverse of the fp8 surgery that put these layers in fp8 in the first place. Not taken.

## 5. Correctness

`tests/k3/test_fp8sk.py` — every legal `(WV,SK,NPW)` at every production shape plus two off-shapes
(512x2560, 256x128), M in {1,2,3,4,5,8,12,16}, against an fp32 oracle computed on the *unshuffled*
weight, and against the closed kernel. **ALL OK.**

* Acceptance criterion: `rel_hcq < 3 * rel_fp8hip + 1e-4`, where `rel = max|y - ref| / max|ref|`.
  Passed at every (shape, M, cfg).
* On the production shapes both kernels sit at `rel` 2.0e-3 .. 3.7e-3 vs the fp32 oracle — that is
  the fp8 quantisation error, and the two kernels are **equally accurate**: in all 18 measured
  (shape, M) rows the two rel figures are bit-identical, i.e. the worst-case element rounds to the
  same bf16 value in both.
* `torch.equal(hcq, fp8hip)` is **False**, by construction (§1: same products, different fp32
  association). `max|hcq - fp8hip|` is the same order as each kernel's own distance from the oracle.
  If you need bit-identity with the closed kernel, this kernel cannot give it to you.
* Precondition codes are asserted against the dispatcher's guard, so the Python guard rejects
  exactly what the kernel rejects.
* HIP-graph replay is asserted `torch.equal` to the eager result, 3 replays.

## 6. Build and run

Build (needs the build image, which is `:try1` plus a compiler; the runtime image dlopens the .so
with no compiler present):

    docker run --rm -v $HOME/rocm10:/w --entrypoint bash \
      local/q38fn-rocm10:k1build /w/tests/k3/build.sh

produces `k3/libhcqfp8sk.so` and verifies the three exports
(`hcq_gemm_fp8blk_nt_m16`, `..._max_m`, `hcq_stream_probe`).

Test / bench — **every GPU run must hold the lock** (`$REPO/GPU_LOCK_PROTOCOL.md`) and
must not run while a q38fn server occupies the GPUs; a shared memory system makes any unlocked
timing here meaningless:

    flock -w 3600 $REPO/gpu.lock docker run --rm --ipc host --group-add video \
      --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1,2 \
      -v $HOME/rocm10:/w --entrypoint python3 local/q38fn-rocm10:try1 /w/tests/k3/test_fp8sk.py

`bench_fp8sk.py` (-> `bench_fp8sk.json`) and `bench_roof.py` (-> `bench_roof.json`) run the same
way. `bench_roof.py` prints the full workgroup sweep to stdout and persists only the best row.

In-server use — `patches/model_executor/kernels/linear/scaled_mm/fp8hip.py.diff` is a drop-in replacement for
`vllm/model_executor/kernels/linear/scaled_mm/fp8hip.py` that keeps fp8hip as the fallback for
everything it does not serve:

    -v $REPO/patches/fp8hip_k3.py:/app/vllm/vllm/model_executor/kernels/linear/scaled_mm/fp8hip.py:ro
    -e VLLM_HC_FP8SK=1                 # default 0 = stock fp8hip, kernel never loaded
    -e VLLM_HC_FP8SK_MAX_M=16          # anything larger falls through to fp8hip
    -e FP8SK_LIB=/w/build/kernels/libhcqfp8sk.so
    -e VLLM_HC_FP8SK_DEBUG=1           # log which calls route where

The dispatcher routes only `(N,K)` present in `_HCQ_CFG_TABLE` (the three production shapes, with
the winning `(WV,SK,NPW)` per shape) and only when M, dtypes, shapes and contiguity all check out;
everything else, including any shape it has not been tuned for, falls through unchanged.
`tests/k3/smoke_patch.py` exercises the guard on CPU (11 cases, no GPU).

## 7. If you are reading this to decide whether to reuse it

Reuse the **method** (§3) unconditionally: pure-read control kernel, pool past the cache, timed in
a graph. Reuse the **kernel** if you want a fp8 block-scaled small-M GEMM you can modify — a fused
epilogue, a W4 weight, a different scale granularity — because it is correct, graph-safe, reads the
existing shuffled weight, and has its preconditions pinned. Do not reuse it expecting speed: on
this hardware, at these shapes, there is 4-9% of headroom to a kernel that does nothing but read
the bytes, and the closed kernel already has most of it.
