# k3 — wvSplitK skinny bf16 GEMM on AMD R9700 (gfx1201), decode shapes of q38fn

Host `big`, GPU 1 of the R9700 pair, image `local/q38fn-rocm10:try1`
(torch 2.11.0+rocm10.0.0), all runs under `flock $REPO/gpu.lock`.
`multi_processor_count` = **32** (64 CUs / 32 WGPs; torch reports WGPs on RDNA).

## Verdict (read this first)

1. **`VLLM_SKINNY_CU_COUNT` is not a lever.** The best single CuCount for the
   production call mix is **32 — exactly what `num_compute_units()` already
   returns**. Per-shape optimal tuning is worth **1.4 % (47 µs/step)** at n=5,
   and 0.3 % at n=1. The `grid=[32,1,1]` in the profile is not a bug.
2. **Do NOT set `VLLM_ROCM_USE_SKINNY_GEMM=0`.** hipBLASLt is *slower* than
   wvSplitK on every one of the 249 production calls: +3.3 % at n=5, +16.6 % at
   n=1 when both are inside a HIP graph, and ~3× slower in eager mode (see §5).
3. **The kernel is already at the memory roofline.** The 249 calls read
   **1.47 GB of bf16 weights per decode step** and take 3.41 ms → **431 GB/s**,
   vs a measured cold-read ceiling of ~560 GB/s on this card. Subtracting the
   ~3–4 µs fixed per-kernel cost, the kernels themselves run at ~90 % of
   achievable bandwidth. There is no dispatch fix worth shipping.
4. **The real lever is bytes, not kernels.** 1.33 GB of that 1.47 GB is the
   hyper-connection `input_mix_weight_down/up` pair, which sits in the
   checkpoint's quantization `ignore` list and therefore runs bf16. Quantizing
   it is worth ~2 ms/step; nothing in the GEMM dispatch is worth more than
   50 µs. See §6.

## 1. The 249 calls/step, exactly

From `vllm/models/qwen4_exp/amd/{model,hyperconnection}.py`, not from the
safetensors alone — the loader **merges** `input_mix_weight_down` [320,10240]
with `block_inject_weight` [4,10240] plus 12 rows of alignment padding into one
`MergedColumnParallelLinear(..., disable_tp=True)` of **M=336**:

| shape (M=out, K=in) | source | calls/step |
|---|---|---|
| 336 × 10240 | `{attn,mlp}_hyper_connection` down+inject merged, 48 layers × 2 + mtp × 2 | 98 |
| 320 × 10240 | `hyper_connection_mixer` down (`use_combine=False`, unmerged) | 2 |
| 10240 × 320 | `input_mix_weight_up` (ReplicatedLinear) | 100 |
| 512 × 2560  | MoE router `mlp.gate`, 48 layers + mtp | 49 |
| **total** | | **249** |

This reproduces the profiled count exactly. **None of these are TP-split** —
`input_mix_weight_up` and the mixer are `ReplicatedLinear`, and the merged down
projection passes `disable_tp=True`. So TP=2 does not change the shapes; both
ranks run the identical 249 GEMMs. (The `/2` variants were benchmarked anyway
and are in `bench_results.json` / `cold_results.json`.)

`linear_attn.in_proj_a/b` and `indexer.index_qk_proj` do **not** reach this path.

## 2. Warm vs cold — why this needed two passes

Navi48 has a 64 MB Infinity Cache. Replaying one 6.5 MB weight back to back
leaves it MALL-resident (streaming read measured at 1383 GB/s, above the
~640 GB/s DRAM ceiling), which flatters hipBLASLt and produced a *wrong* first
answer. A decode step touches 1.47 GB of these weights, so **none of them are
cache-resident in production**. All headline numbers below are **cold**: every
call in the captured graph reads a different weight copy from a pool sized
> 2× MALL (`bench_cold.py`, `bench_prod.py`). The inversion is large:

| shape, n=5 | warm F.linear | warm wvSplitK | cold F.linear | cold wvSplitK |
|---|---|---|---|---|
| 10240 × 320 (hc_up) | **7.67** | 14.13 | 15.37 | **14.22** |
| 12288 × 2560 (mtp.q) | **43.4** | 102.8 | 109.1 | **104.7** |
| 2560 × 2560 | **9.97** | 24.3 | 24.8 | **24.7** |

Warm, hipBLASLt reaches ~1.4 TB/s and wvSplitK never exceeds ~610 GB/s — the
split-K kernel does not exploit the Infinity Cache at all. That only matters
for a weight small enough to stay resident, which none of these are.

## 3. Cold µs/call, production shapes, n=5 (the MTP-4 decode case)

| shape | ×/step | F.linear | cu16 | cu24 | **cu32** | cu48 | cu64 | cu96 | cu128 | cu320 | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 336×10240 hc_down | 98 | 15.97 | 16.55 | 16.15 | **16.18** | 16.65 | 17.07 | 20.82 | 20.82 | 20.81 | 16.15 @24 |
| 320×10240 mixer | 2 | 15.70 | 16.13 | 15.64 | **16.05** | 16.13 | 16.14 | 19.75 | 19.78 | 19.85 | 15.64 @24 |
| 10240×320 hc_up | 100 | 15.37 | 20.35 | 15.81 | **14.22** | 15.86 | 14.16 | 14.40 | 14.70 | 13.79 | 13.79 @320 |
| 512×2560 router | 49 | 7.96 | 7.64 | 7.53 | **7.53** | 7.74 | 7.55 | 7.66 | 8.42 | 9.60 | 7.53 @24 |

Full 19-point CuCount curves (including non-powers-of-two 8/24/40/56/80/112/224/320)
for n=1..5 are in `prod.log` / `prod_results.json`.

### Weighted step cost of the 249 calls (cold, µs)

| n | cu32 (current) | best single cu | per-shape optimal | F.linear everywhere |
|---|---|---|---|---|
| 1 | 3099 | cu48 3091 (−0.3 %) | 3086 (−0.4 %) | 3615 (**+16.6 %**) |
| 2 | 3161 | cu56 3160 (−0.0 %) | 3150 (−0.3 %) | 3604 (**+14.0 %**) |
| 3 | 3208 | cu32 3208 (0.0 %) | 3202 (−0.2 %) | 3465 (**+8.0 %**) |
| 4 | 3333 | cu32 3333 (0.0 %) | 3282 (−1.5 %) | 3466 (**+4.0 %**) |
| 5 | 3409 | cu32 3409 (0.0 %) | 3361 (−1.4 %) | 3523 (**+3.3 %**) |

The curve is flat from cu16 to cu80 and degrades above cu96 (at n=5, cu256 is
+21 %). Setting CuCount far above the real WGP count makes it worse, not better.

**Projected saving from `VLLM_SKINNY_CU_COUNT`: 0 µs/step.** The per-shape
optimum (which needs a code change, not an env var — cu24 for the K=10240
shapes, cu320 for hc_up) is 47 µs/step at n=5.

## 4. Correctness across CuCount

wvSplitK's max-abs error vs an fp32 reference is **bit-identical across every
CuCount tested**, in 120/120 (shape, n) groups of the warm sweep and across all
19 CuCount values (powers of two and not) in the production sweep. Worst
relative error 3.7e-3 — the same order as `F.linear` on the same input
(3.5e-3); both accumulate in bf16. **CuCount is numerically safe**, including
values that are not the device's CU count and not powers of two.

## 5. Launch overhead vs kernel time (what a HIP graph hides)

n=5, µs/call. `profiler` = `torch.profiler` CUDA kernel duration, eager.

| shape | variant | profiler kernel | in-graph (warm) | in-graph (cold) | eager wall |
|---|---|---|---|---|---|
| 336×10240 | F.linear | 24.5 | 14.4 | 17.4 | **47.7** |
| 336×10240 | wvSplitK@32 | 16.3 | 16.0 | 17.3 | 21.3 |
| 10240×320 | F.linear | 7.5 | 7.7 | 15.4 | **49.3** |
| 10240×320 | wvSplitK@32 | 14.0 | 14.1 | 14.4 | 19.3 |
| 12288×2560 | F.linear | 43.3 | 43.4 | 109.1 | 48.2 |
| 12288×2560 | wvSplitK@32 | 102.4 | 103.1 | 104.8 | 108.4 |

- hipBLASLt costs **~30 µs of host-side dispatch per call** in eager mode
  (47.7 wall − 17.4 in-graph); wvSplitK costs ~4 µs. Across 249 calls that is
  7.5 ms vs 1.0 ms of pure overhead — which is why
  `VLLM_ROCM_USE_SKINNY_GEMM=0` would be a disaster on any path that is not
  cudagraph-captured.
- Inside a HIP graph that difference vanishes and the two are within 3 %.
- The profiler and in-graph numbers agree within 2 % for wvSplitK. They
  disagree for `F.linear` on 336×10240 (24.5 vs 14.4) — hipBLASLt appears to
  pick a different kernel under the profiler; I did not chase this, and no
  conclusion here rests on it.

## 6. Where the time actually goes

Bytes per decode step through these 249 calls:

| shape | ×/step | MB/call | MB/step |
|---|---|---|---|
| 336×10240 | 98 | 6.88 | 674 |
| 10240×320 | 100 | 6.55 | 655 |
| 512×2560 | 49 | 2.62 | 128 |
| 320×10240 | 2 | 6.55 | 13 |
| **total** | 249 | | **1470 MB** |

1470 MB / 3.41 ms = **431 GB/s achieved**. Measured cold ceilings on this card:
560 GB/s (63 MB tensor), 346–360 GB/s (6.5 MB tensor, `torch.sum`). The
residual gap is the **~3–4 µs fixed cost of each kernel** (measured: a 0.12 MB
GEMM still takes 3.4 µs in-graph), i.e. **750–1000 µs/step, 22–29 % of the
3.41 ms, is per-kernel launch/teardown, not arithmetic and not bandwidth.**

Two structural levers, both far larger than anything in the dispatch:

- **Quantize the hyper-connection weights.** 1.33 GB of the 1.47 GB is
  `input_mix_weight_down`/`_up`, held bf16 only because
  `re:.*hyper_connection.*` and `re:.*input_mix.*` are in the checkpoint's
  quantization `ignore` list. At mxfp4 that is ~7× fewer bytes; even accounting
  for the fixed per-call floor the 3.41 ms would fall to ~1.3 ms. A
  `wvSplitK_int4_g` path already exists in `_custom_ops.py`. Whether these
  weights *can* be quantized without quality loss is a separate question — they
  are a rank-320 residual mixer, which is exactly the kind of layer that
  usually gets excluded on purpose.
- **Fuse the 200 hyper-connection GEMMs.** 98+100 calls × ~3.5 µs fixed cost is
  ~700 µs/step of pure overhead. The down and up projections of one layer are
  sequential and rank-320; batching across layers is not possible, but a fused
  down→silu→up kernel would remove one launch and one round trip per layer.

## 7. Unexplained gap — flagging honestly

The decode profile reports 249 calls averaging **31 µs** (≈7.7 ms/step). I
measure the identical 249-call mix at **13.7 µs average (3.41 ms/step)**, a
2.3× gap, with a benchmark that matches the profile's shapes, dtype, launch
geometry (`grid=[32,1,1] block=[32,16,1]`) and cold-cache condition. Candidates
I did **not** test: contention with the MoE expert weight stream for DRAM
bandwidth; TP=2 all-reduce serialization between these layers; the profile's
per-call figure including inter-kernel gaps. I also could **not** reproduce the
"one call per layer at 106 µs" — no production shape takes 106 µs at any
CuCount. The only shapes I measured near 106 µs are `mtp.q_proj` (12288×2560,
104.7 µs) and `ple.key_proj` (10240×2560, 87.9 µs), and both occur **once** per
step, not once per layer. Worth re-checking which kernel that 106 µs bucket is.

## 8. Other variants

- **`ops.LLMM1`** (n=1 only): within noise of wvSplitK (3.39 vs 3.46 µs on
  48×2560; 8.06 vs 8.07 µs on 320×5120). Irrelevant to MTP-4 decode, where
  n=5.
- **aiter**: not installed in `local/q38fn-rocm10:try1` —
  `ModuleNotFoundError: No module named 'aiter'` for both
  `aiter.ops.triton.gemm_a16w16` and `aiter.tuned_gemm`. Untested.
- **Large bf16 shapes outside the 249** (`fc_embedding`, `ple.key_proj`,
  `mtp.q_proj`, `mtp.o_proj`, `lm_head`): hipBLASLt wins by 2.0–2.4× **warm**
  and ties or loses slightly **cold**. `lm_head` (635 MB at TP2) is pure
  bandwidth: 1011 µs either way, 629 GB/s = the card's DRAM limit. Since
  `rocm_unquantized_gemm_impl` only routes `n <= 5` to wvSplitK and these are
  all cold in practice, no change is warranted there either.

## Files

- `$REPO/k3/bench_skinny.py` — main sweep: 24 shapes × n=1..5 ×
  {F.linear, wvSplitK @ 13 CuCounts, LLMM1, aiter}, eager **and** in-HIP-graph
  timing, correctness vs fp32, bandwidth roofline. → `bench_results.json`, `bench.log`
- `$REPO/k3/bench_cold.py` — cold-cache (pool > 2× MALL) rerun +
  `torch.profiler` kernel times. → `cold_results.json`, `cold.log`
- `$REPO/k3/bench_prod.py` — the exact 249-call production mix,
  19 CuCounts, n=1..5, with the per-step projection. → `prod_results.json`, `prod.log`
- `$REPO/k3/analyze.py` → `analysis.txt`
- `$REPO/k3/shapes2.py` — safetensors-header shape enumeration
