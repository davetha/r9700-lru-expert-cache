# k4 — per-decode-step kernel → Python call-site attribution (Qwen3.8-Flash-Next, 2×R9700, TP=2, MTP-4)

Method, data, and every proposal below are derived from two torch-profiler traces of the same
server — one production (HIP graphs + inductor) and one `--enforce-eager` (Python stacks on
every launch) — aligned kernel-for-kernel.

## 0. Method

1. `extract.py` — loads the eager trace, picks a mid-run decode step, and for every
   `cuda_runtime` launch inside it resolves (a) the enclosing `cpu_op` / aten-op stack and
   (b) the full `python_function` stack, by a proper nesting sweep (merge events + query
   points in ts order, pop expired frames *before* pushing — popping only from the top leaks
   finished siblings and produced a 10 780-deep bogus stack on the first attempt).
2. `attrib.py` — classifies each kernel by phase (target fwd / MTP draft / sampler / input
   prep) and by model sub-module, taking the **outermost** matching frame under
   `Qwen4ExpDecoderLayer.forward`, so a kernel launched from `layernorm.py` inside the QSA
   indexer is attributed to QSA and not to a generic "norm" bucket.
3. `graph_census.py` — same census on the production (HIP-graph) trace.
4. `align.py` — `difflib.SequenceMatcher` over the two ordered kernel-name sequences
   (with `__amd_rocclr_copyBuffer` ≡ `Memcpy DtoD` aliasing). **3112 of 3490 production
   kernels (89%) match an eager twin 1:1 and inherit its Python call site.** The 378
   unmatched are inductor products (`triton_poi_fused_*`, `triton_per_fused_*`).
   The 860 eager kernels with no production twin are exactly the ones inductor *did* fuse.
5. `hitlist.py` / `perlayer.py` — the tables below.

Scripts: `k4/{extract,attrib,align,hitlist,perlayer,graph_census,seq_eager,seq_graph,cmp_eager_graph,regions}.py`.
Intermediates: `step_eager_r0.pkl`, `full_eager_r0.pkl`, `full_eager_r0_attr.pkl`, `aligned.pkl`.
Raw output: `HITLIST_RAW.txt`, `align_out.txt`, `attrib_out.txt`.

## 1. Correcting the step budget

The "3048 kernels/step" figure is the **`gpu_user_annotation` window only** — i.e. the target
forward. The full decode iteration (annotation start → next annotation start) is larger:

| | production (graph) | eager |
|---|---|---|
| kernels+memcpy per decode step | **3490** | 3972 |
| — inside the `generation` annotation (target fwd) | 3048 | 3369 |
| — outside it (MTP draft + logits + sampler/rejection) | **442** | 603 |
| step wall (annotation→annotation) | **61.9 ms** | 256.8 ms |
| kernel busy | 47.2 ms | 47.9 ms |
| sum of inter-kernel gaps | **14.7 ms (23.8% of step)** | — |
| median / mean inter-kernel gap | **3.72 / 4.22 µs** | — |
| kernels < 3 µs | 2358 (67.6%) | — |

So the dispatch tax is 14.7 ms, not 13, and there are 442 more kernels per step than the
annotation shows. Every "kernels saved" number below is against **3490**, and every "ms saved"
is `count × 3.72 µs` (gap) `+` the kernels' own measured busy time.

*Caveat: my measured step is 61.9 ms, not the 43 ms in the brief. Both traces are the `h15`
runs. If 43 ms came from a different concurrency, scale the ms figures — the counts hold.*

## 2. Model shape (derived from the trace, not the config)

48 decoder layers = **36 GDN (`linear_attention`) + 12 QSA (`full_attention`)**; every layer
has a MoE block (48 `topkGating`); 2 hyper-connection modules per layer (96); 1 MTP draft
layer run **4×** per step (~110 kernels per draft iteration, 90 of them attributed).

## 3. Where the 3490 kernels are

| phase | sub-module | kernels | busy ms | dispatch ms |
|---|---|---:|---:|---:|
| target fwd | mlp: MoE block | 1104 | 27.16 | 4.11 |
| target fwd | self_attn/QSA (12 layers) | 552 | 1.72 | 2.05 |
| target fwd | hyper_connection (96 modules) | 486 | 3.08 | 1.81 |
| target fwd | linear_attn GDN (36 layers) | 468 | 3.71 | 1.74 |
| MTP draft | self_attn/QSA | 173 | 0.62 | 0.64 |
| MTP draft | MoE block | 64 | 0.58 | 0.24 |
| MTP draft | hyper_connection | 60 | 0.35 | 0.22 |
| target fwd | PLE | 53 | 0.21 | 0.20 |
| input prep | — | 70 | 0.18 | 0.26 |
| MTP draft | logits / all-reduce / other | 63 | 4.36 | 0.23 |
| sampler / rejection | — | 9 | 0.03 | 0.03 |
| *(unattributed: inductor `triton_*_fused_*`)* | | 378 | ~0.6 | 1.41 |

By kernel kind (eager census, which is what the call sites are about):
`a) torch eager aten` 1537 · `b) vLLM custom op` 884 · `c) triton` 466 · `d) closed
r4d/fp8hip/clav/Tensile` 959 · `e) memcpy/memset` 126.

**In production 761 aten kernels survive inductor**, and the alignment shows exactly which
ones (§5). The "280 `elementwise_kernel_manual_unroll` + 254 `vectorized_elementwise`" in the
brief are in fact **331 + 339** per full step, and they are dominated by the QSA indexer's
RMSNorm decomposition and the GDN qkv repack.

## 4. Call-site hitlist (production kernels, sorted by COUNT)

`per` = launches per instance of that module (36 GDN / 12 QSA / 48 MoE / 96 HC).
Times are the **production** (graph-mode) durations.

| n/step | per | busy µs | module | call site (file:line: fn) | kernel |
|---:|---:|---:|---|---|---|
| 192 | 4.00 | 23055 | MoE | `fused_moe/experts/r4d_mxfp4_moe.py:190 _r4d_mxfp4_moe_gemm_into` | `r4d_gemm_moe_mxfp4a8_nt_b16` |
| 192 | 4.00 | 342 | MoE | `_custom_ops.py:1832 scaled_fp8_quant` | `dynamic_per_token_scaled_fp8_quant` |
| 192 | 4.00 | 386 | MoE | `_custom_ops.py:2254 moe_align_block_size` | `moe_align_block_size` + `count_and_sort_expert_tokens` |
| 218 | 1.01×2 | 2975 | HC | `_custom_ops.py:2182 wvSplitK` | `wvSplitK_hf_big_` / `_sml_` |
| 109 | 1.01 | 115 | HC | `models/qwen4_exp/amd/ops/hc.py:108 _hc_silu` | `_hc_silu_kernel` |
| 109 | 1.01 | 138 | HC | `models/qwen4_exp/amd/ops/hc.py:163 _hc_gate_mix` | `_hc_gate_mix_kernel` |
| 108 | 3.00 | 185 | GDN | `mamba/gdn/qwen_gdn_linear_attn.py:778 rearrange_mixed_qkv` | `elementwise_kernel_manual_unroll` (aten::copy_) |
| 103 | 0.99 | 187 | HC | `models/qwen4_exp/amd/ops/hc.py:334 _hc_combine_norm` | `_hc_combine_norm_kernel` |
| 96 | 2.00 | 663 | MoE | `kernels/linear/mxfp4/r4dhip.py:89 _r4d_mxfp4_linear` | `r4d_gemm_mxfp4a8_nt_m64` |
| 96 | 8.00 | 208 | **QSA** | `ir/ops/layernorm.py:9 rms_norm` (aten copy_ / mul) | `vectorized_elementwise` + `manual_unroll` |
| 96 | 8.00 | 140 | **QSA** | `ir/ops/layernorm.py:9 rms_norm` (pow/mean/add/rsqrt) | `vectorized_elementwise` + `reduce_kernel` |
| 72 | 2.00 | 2313 | GDN | `kernels/linear/scaled_mm/fp8hip.py:59 _fp8hip_block_scaled_mm_func` | `fp8hip_gemm_w8a8_tiled` |
| 72 | 2.00 | 124 | GDN | `quantization/utils/fp8_utils.py:534 per_token_group_quant_fp8` | `per_token_group_quant_8bit` |
| 56 | 4.00 | 102 | **QSA** | `rotary_embedding/mrope.py:159 triton_mrope` (`.contiguous()`) | `elementwise_kernel_manual_unroll` |
| 48 | 4.00 | 59 | **QSA** | `layers/layernorm.py:151 forward_native` (`weight.float()+1.0`) | `vectorized_elementwise` ×2 |
| 48 | 1.00 | 970 | MoE | `layers/utils.py:117 rocm_unquantized_gemm_impl` (router gate) | `Cijk_…MT16x16x32` |
| 48 | 1.00 | 359 | MoE | `_custom_ops.py:2402 topk_softmax` | `topkGating` |
| 48 | 1.00 | 681 | MoE | `r4d_all_reduce.py:284 custom_all_reduce` | `r4d_ar_oneshot_2rank_exact` |
| 48 | 1.00 | 137 | MoE | `_custom_ops.py:2245 moe_sum` | `moe_sum_vec_dynamic` |
| 48 | 1.00 | 64 | MoE | `fused_moe/activation.py:196 apply_moe_activation` | `act_and_mul` |
| 48 | 1.00 | 50 | MoE | `fused_moe/topk_weight_and_reduce.py:53 apply` | **`Memcpy DtoD`** |
| 48 | 1.00 | 307 | MoE | `_custom_ops.py:2182 wvSplitK` (shared-expert gate) | `wvSplitK_hf_sml_` |
| 48+48 | 1+1 | 61+88 | MoE | `models/qwen2_moe.py:110 forward` (`sigmoid`, `mul`) | `vectorized_elementwise`, `manual_unroll` |
| 36 | 1.00 | 352 | GDN | `flash_linear_attention/ops/fused_sigmoid_gating.py:181` | `fused_sigmoid_gating_delta_rule_update` |
| 36 | 1.00 | 150 | GDN | `mamba/ops/causal_conv1d.py:1195 causal_conv1d_update` | `conv1d_update_kernel` |
| 36 | 1.00 | 85 | GDN | `qwen_gdn_linear_attn.py:778 rearrange_mixed_qkv` | `CatArrayBatchedCopy_contig` |
| 36 | 1.00 | 35 | GDN | `qwen_gdn_linear_attn.py:1244 _forward_core` | **`Memcpy DtoD`** |
| 36 | 3.00 | 48 | QSA | `amd/ops/qsa.py:1402 qsa_store_cache_rows` | `_store_qsa_rows_kernel` |
| 32 | 2.67 | 53 | QSA | `amd/indexer_qsa.py:30 apply_qsa_rope` (`cache[positions]`) | `index_elementwise` / `vectorized_gather` |
| 28 | 2.33 | 33 | QSA | `rotary_embedding/mrope.py:159 triton_mrope` | `_triton_mrope_forward` |
| 24 | 2.00 | 618 | QSA | `fp8hip.py:59 _fp8hip_block_scaled_mm_func` | `fp8hip_gemm_w8a8_tiled` |

Full table including MTP draft and the eager-only (already-fused) rows: `HITLIST_RAW.txt`.

## 5. The finding that drives most of the count: one `.float()` disables the fused RMSNorm

`GemmaRMSNorm.forward_native`, `model_executor/layers/layernorm.py:157-159`:

```python
weight = self.weight.float() + 1.0                       # <-- fp32 weight, x is bf16
if residual is None:
    return ir.ops.rms_norm(x, weight, self.variance_epsilon)
```

The fused vLLM kernel is registered for that IR op, but it is guarded
(`kernels/vllm_c.py:17-20`):

```python
rms_no_var_size = lambda x, weight, epsilon, variance_size=None: (
    variance_size is None and (weight is None or weight.dtype == x.dtype)
)
@ir.ops.rms_norm.register_impl("vllm_c", supports_args=rms_no_var_size, ...)
```

`weight.dtype (float32) != x.dtype (bfloat16)` → `supports_args` returns False, the `vllm_c`
(and `aiter`, `oink`) providers are all skipped in `IrOp.dispatch` (`ir/op.py:344-360`,
which logs "Skipping provider %s because it does not support ..." at debug level), and
dispatch falls through to `native` — the 8-kernel Python decomposition in
`ir/ops/layernorm.py:9` (`to(f32) · pow · mean · add · rsqrt · mul · mul · to(bf16)`).
Plus the `weight.float() + 1.0` itself is **recomputed every call** — 2 more kernels on a
128-element parameter.

Every *other* RMSNorm in the model (plain `RMSNorm`, bf16 weight) passes the guard and runs
as one kernel. That is exactly the pattern in the trace: the un-fused decomposition appears
only where `GemmaRMSNorm` is used — the QSA indexer (`indexer_qsa.py:116-123` q/k_layernorm)
and the MTP layer. Inside torch.compile regions inductor would fuse the decomposition away,
but it does **not** here: the production graph-mode trace runs it verbatim, in the same order
as eager (sequence alignment: graph indices 1521-1549 map 1:1 onto eager 1427-1455).

Per QSA layer: 2 `GemmaRMSNorm` calls × (8 decomposition + 2 weight-prep) = **20 kernels**.
× 12 QSA layers = 240, plus 20/iteration × 4 MTP draft iterations = 80.
**320 kernels per step, ~434 µs busy, to do 32 RMS norms** — all because of one `.float()`.

## 6. Proposals — top 15 by kernels saved per step

Legend: **P** = Python-only (cheap), **K** = needs a new/modified kernel (expensive),
**N** = changes numerics, gate with a divergence probe.

| # | fix | file:line | kernels saved/step | ms saved/step | cost |
|---:|---|---|---:|---:|---|
| 1 | **Cache `weight + 1` in the parameter's own dtype in `GemmaRMSNorm`.** Replace the per-call `self.weight.float() + 1.0` with a buffer computed once at load, `(w.float()+1).to(w.dtype)` (bf16). This kills the 2 weight-prep kernels *and* makes `weight.dtype == x.dtype`, so `supports_args` passes and the fused `vllm_c` `torch.ops._C.rms_norm` is dispatched instead of the 8-kernel decomposition. **320 → 32 kernels, and it is a Python-only change in one class.** | `layers/layernorm.py:157-159`; guard at `kernels/vllm_c.py:17-20`; users at `indexer_qsa.py:65,116-123` | **−288** | **−1.46** | **P**, **N** (weight multiply moves fp32 → bf16; gate with a divergence probe) |
| 1a | *If #1's numerics are rejected*: keep fp32 weight but still precompute it once. | `layers/layernorm.py:157` | −64 | −0.30 | **P**, no numerics change |
| 2 | **GDN qkv repack.** `rearrange_mixed_qkv` does 3 `reshape(-1)` copies **plus** a `torch.cat` — 4 full-qkv copies per GDN layer. (Its docstring claims torch.compile emits one Triton kernel; the production trace shows 3 `elementwise_manual_unroll` + 1 `CatArrayBatchedCopy`.) Best fix: pass strided q/k/v straight to `fused_sigmoid_gating_delta_rule_update` (FLA kernels take strides). | `mamba/gdn/qwen_gdn_linear_attn.py:778-810` | **−144** | **−0.81** | K (kernel signature) |
| 2a | *Fallback*: one repack kernel `[seq,D] → 3 contiguous blocks` instead of 4 launches. | same | −108 | −0.58 | K (small) |
| 3 | **Dedup the hot/cold `moe_align_block_size`.** `_apply_split` calls it twice on the *same* `topk_ids` with complementary expert maps → 4 kernels/layer. One kernel can emit both subsets' sorted buffers in a single counting pass. | `patches/hotcold/r4d_mxfp4_moe.py:422-425`; `_custom_ops.py:2254` | **−96** | **−0.55** | K |
| 4 | **mrope: stop materialising cos/sin.** `triton_mrope` does `cos.contiguous(); sin.contiguous()` on views produced by `cos_sin.chunk(2,-1)` (2 copies per call), preceded by a `cache[positions]` gather. Pass `cos_sin_cache` + `positions` into the Triton kernel and gather/split inside. | `rotary_embedding/mrope.py:193-196`; `indexer_qsa.py:30-45` | **−88** | **−0.49** | K (small Triton) |
| 4a | *Python-only variant*: keep cos and sin in two separate contiguous caches (2 gathers, 0 copies). | rotary-embedding cache init | −28 | −0.15 | **P** |
| 5 | **Shared expert re-quantises the routed activation.** `qwen3_next.py:200` passes the same tensor as `hidden_states` and as the shared-expert input; the routed path quantises it at `r4d_mxfp4_moe.py:426` and `_r4d_mxfp4_linear` quantises it *again* at `r4dhip.py:105`. Plumb the existing `(qx, xs)` into the shared expert's first GEMM. | `models/qwen3_next.py:200`; `hotcold/r4d_mxfp4_moe.py:426`; `kernels/linear/mxfp4/r4dhip.py:105` | **−48** | **−0.26** | **P** (plumbing) |
| 6 | **Redundant MoE output copy.** `TopKWeightAndReduceNoOP.apply` ends in `output.copy_(fused_expert_output)` — a full hidden-state DtoD per MoE layer. The r4d path already writes `output` via `moe_sum`; the method even has an `if output is fused_expert_output: return output` fast path that never fires. Alias `fused_out` to `output` upstream. | `fused_moe/topk_weight_and_reduce.py:53,75`; `hotcold/r4d_mxfp4_moe.py:441` | **−48** | **−0.23** | **P** |
| 7 | **Shared-expert gate `F.sigmoid(g) * out`** is 2 aten kernels per MoE layer that inductor does not fuse (the shared expert runs on an aux stream ⇒ graph break). One fused sigmoid-mul, or fold it into the `down_proj` epilogue. | `models/qwen2_moe.py:110-114` | **−48** | **−0.25** | K (tiny) |
| 8 | **GDN `_forward_core` DtoD** — one full-state `Memcpy DtoD` per GDN layer (`z_out[:] = z` / state copy). Have the projection split write `z` in place. | `qwen_gdn_linear_attn.py:1234,1244` | **−36** | −0.18 | P, needs a shape check |
| 9 | **`_store_qsa_rows` fires 3× per QSA layer** (raw rows / compressed rows / rope positions). Batch the three stores into one launch. | `models/qwen4_exp/amd/ops/qsa.py:1402` | **−24** | −0.13 | K (small) |
| 10 | **Make the silent IR-op fallback loud.** `IrOp.dispatch` (`ir/op.py:352-360`) drops to the native decomposition on a `supports_args` miss and only logs at `debug`. That is how #1 hid: an 8× kernel-count regression with no warning. Promote it to `warning_once` per (op, provider). | `ir/op.py:352-360` | 0 today | — | **P**, prevents recurrence |

**Totals (proposals 1–9, full versions):** −760 kernels/step = **21.8% of 3490**;
**≈ −4.05 ms/step ≈ −6.5% of a 61.9 ms step** (2.83 ms of dispatch gap + ~1.2 ms of kernel time).
Python-only subset (**1**, 4a, 5, 6, 8): **−448 kernels, ≈ −2.3 ms/step (3.7%)** — and #1 alone
is more than half of that, in one class, with no new kernel.

## 7. Two structural notes

1. **The hot/cold expert patch costs +192 kernels/step** versus the single-call path: +96 from
   the second `moe_align_block_size` and +96 from the second grouped-GEMM launch
   (`_gemm_split`, `hotcold/r4d_mxfp4_moe.py:353-358`). That is 5.5% of all kernels/step,
   bought for the patch's measured +7% / +20%. Proposal #3 gives half of it back for free.
2. **Count is not where the time is.** `r4d_gemm_moe_mxfp4a8_nt_b16` is 192 launches but
   **23.1 ms of the 47.2 ms busy (49%)**; the two per-layer all-reduces are 109 launches /
   1.24 ms. Removing every kernel in §6 leaves the step at ~58 ms. The count lever is worth
   ~6.5%; the remaining headroom is the MoE GEMM and the all-reduce, not dispatch.

## 8. MTP draft loop

442 kernels/step outside the target annotation (360 attributed), **~110 per draft iteration ×4**.
Per iteration: 20 kernels of QSA `GemmaRMSNorm` decomposition (proposal #1 applies — 80/step),
~5 `wvSplitK`, 3 `_store_qsa_rows`, 2 `clav_ag_push` all-gathers, 2 `per_token_group_quant`.
Notably the draft's MoE takes the **Triton `fused_moe_kernel`** path
(`fused_moe/fused_moe.py:763`), not the r4d grouped GEMM — 2 launches/iteration, ~40 µs each.
Nothing in the draft loop is wasteful by a factor; it is the same per-layer overhead run 4×.

## Elementwise census (2026-09-04, `prof_w4head` rank 0, 84 decode steps, c4/c6 gates OFF)

`k4/eattrib.py` groups every `*elementwise_kernel*` in a kineto trace by full
templated name + grid; `k4/eattrib2.py` gives each group a Python call site from
`aligned.pkl` (the eager<->graph alignment). Totals: **489.4 kernels/step, 907 us/step,
102 distinct (name, grid)**. With `CatArrayBatched` folded in it is 741/step in the
older `prof_base_h15` step used for attribution.

| group | /step | us/step | call site | status |
|---|---|---|---|---|
| `direct_copy` bf16 | 150.5 | 281 | GDN qkv repack `qwen_gdn_linear_attn.py:778` (108) + `indexer_qsa.py:150 project_qk` (12) + `clav_all_gather.py:86` (5) | 108+36 cat removed by #2 |
| `MulFunctor` bf16 | 68.9 | 131 | shared expert `qwen2_moe.py:110` (52) + ~17 elsewhere | 52 removed by #7 |
| `sigmoid` bf16 | 52.0 | 71 | same shared-expert site | removed by #7 |
| `index OpaqueType<2>` | 32.4 | 92 | `indexer_qsa.py:30 apply_qsa_rope` cos/sin gathers, 16 calls x 2 | removed by **#12** |
| bf16->fp32 + fp32->bf16 | 62.8 | 85 | #1 mode-2's casts around the fused rms_norm, 32 calls/step | **open** (P-A below) |
| `FillFunctor` bf16 | 18.6 | 18 | `qsa.py:138 forward_qsa` output zero-fill, 16/step | **open** (P-B) |
| `add` bf16 | 8.0 | 11 | residual adds outside compiled regions | open, small |
| PLE index math | ~30 | ~45 | `ple_layer.py:999 _short_conv_dilated_spec` (arange/where/clamp/sub/cat on long) | **open** (P-C) |
| eager input-prep + sampler | ~35 | ~90 | `gdn_attn.py:210 build`, `short_conv_attn.py:243 build`, `model_state.py:65 _prepare_ngram_context`, `rejection_sampler.py:249`, `speculator.py:419/555` | open; outside the graph, so real CPU launches |

**Why inductor leaves them.** Every one of these carries an aten kernel name, not
`triton_poi_*`: they sit in eager regions outside the compiled model region -- the QSA
indexer, the PLE, the GDN wrapper, and the input-prep/sampler code that runs before and
after the graph. The rms_norm casts are a second mechanism: `platforms/rocm.py:1130`
pins the rms_norm IR op to `['native']` under inductor, and the fused impl we call in
its place only takes fp32, so the two casts are a dtype boundary, not a missed fusion.

**Post-c4/c6 remainder: ~277 kernels/step, ~535 us/step**, and ~245/step after #12.

## P-A closed: single-kernel GemmaRMSNorm -- REJECTED (2026-09-04)

-64 launches/step, but not bit-identical to the shipped mode 2: **1 element in 81.8M**
(6 input scales, 639K rows, `probe_gemma10.py`). A 3-scale probe said 0 (`probe_gemma9.py`)
-- widen the input scales and it comes back, so do not trust the narrower run.
The vllm_c arithmetic is fully pinned (`sum/128 + eps`, `rsqrt`, `(x*inv)*w`, all exact);
only the 128-term reduction ASSOCIATION is unknown, and none of 15 candidate orders
reproduces it (best: adjacent-pair balanced tree, 99.74% of rows). Probes 3-10 in `k4/`.
Full write-up: `patches/PATCHES.md`, section "P-A".

## Post-c7 census (2026-09-04, `prof_c7_rope` rank 0, 84 decode steps, ALL gates on)

Arm c7 = the launcher default (LRU + `MOUNTS_COMBO4` + #8 W4 draft head + #2 GDN strided
+ #7 fused shared gate + #11 fused silu-quant + #12 QSA rope gather). Live: 93.2 prose /
125.9 JSON / 107.4 code tok/s, accept rates identical to c4.

**Full decode step (`k4/prof_diff.py`, annotation -> next annotation, rank 0):**

| arm | kernels/step | ms/step |
|---|---|---|
| `prof_lru1` (LRU only, k4 env gates off, `MOUNTS_LRU` mounted) | 3265 | 41.7 |
| `prof_c4` | 2880 | 36.7 |
| `prof_c7_rope` | **2776** | **36.2** |

`prof_lru1 -> prof_c7_rope` is **-489 kernels/step**. -192 of that is K1's fused LRU
manager (`moe_align_block_size` 96 + `count_and_sort_expert_tokens` 96 + `lru_manage_k` 48
-> `lru_fused_k` 48) and the W4 draft head is count-neutral (`wvSplitK_hf_sml_` -4,
`r4d_gemm_w4a16_nt_m64` +4). **The remaining -297/step is #2 + #7 + #11 + #12**, against -301 predicted from their
individual measurements. `prof_lru1` already mounted `MOUNTS_LRU.txt`, so #1 (mode 2),
#4a and #10 are on in *both* arms and do not appear in this delta.

**Elementwise family (`k4/eattrib.py`, same 84 steps, `k4/ediff.py` for the diff):**

| | kernels/step | us/step | distinct name+grid |
|---|---|---|---|
| `prof_w4head` (c4/c6 gates off) | 489.4 | 907.4 | 102 |
| `prof_c7_rope` (all gates on) | **255.0** | **457.8** | 100 |

That is **-234/step and -450 us/step**, against the -244 predicted in the census above.
Per family (`+` = new, `-` = removed):

| family | before/step | after/step | delta | by |
|---|---|---|---|---|
| `direct_copy` bf16 | 153.4 | 22.0 | **-131.4** | #2 GDN strided qkv (-108 copies -36 cat) + #12 (-26 cos/sin `.contiguous()`) |
| `MulFunctor` bf16 | 69.0 | 17.0 | **-52.0** | #7 fused shared gate |
| `sigmoid` bf16 | 52.0 | 0.0 | **-52.0** | #7 |
| `index OpaqueType<2>` | 33.0 | 9.0 | **-24.0** | #12 |
| `direct_copy` int64 | 6.7 | 31.7 | **+25.0** | #12, see below |
| bf16<->fp32 casts | 64.0 | 64.0 | 0 | #1 mode 2's dtype boundary, still open |
| `FillFunctor` bf16 | 19.8 | 19.8 | 0 | P-B (QSA output zero-fill), withdrawn |
| PLE index math (arange/where/clamp/sub/cat on long) | ~30 | ~30 | 0 | P-C, not now |
| everything else | ~61 | ~61 | 0 | eager input-prep + sampler, outside the graph |

### The one new item: #12 pays 25 int64 copies/step for the 26 bf16 copies it removes

`elementwise_kernel_manual_unroll` variants in ONE step (`k4/mu.py`):

| variant | c4 | c7 |
|---|---|---|
| `direct_copy` bf16 | 48 | 22 |
| `direct_copy` int64 | 5 | 30 |
| MulFunctor / Half / int / other | 31 | 31 |

`canonical_qsa_rope_positions` returns `positions.unsqueeze(0).expand(3, -1).transpose(0, 1)...`
-- stride 0 on the broadcast axis -- so the `positions = positions.contiguous()` that
`triton_mrope` needs for its row-major pointer arithmetic materialises a real `[3, T]`
int64 copy on **every** call: ~16 calls/step (12 QSA layers + 4 MTP drafts) plus MTP
re-entries = +25/step. The full-step ledger for #12 is therefore
`-32 vectorized_gather, -24 index_elementwise, -26 bf16 contiguous, +25 int64 = -57/step`,
which is exactly the -56 measured for the c4 -> c7 pair once #11's -48 is taken out. The
patch is still a clear win and the live arm confirms it.

**Open, cheap:** every one of those 25 copies materialises the *same* tensor. Hoisting
one `.contiguous()` per forward (or, better, giving the kernel a positions stride
parameter -- the same trick that fixed #2, where the FLA wrapper's `.contiguous()` was
the whole cost) would recover ~24 launches/step. Not implemented: out of scope for the
patch as adopted, and it needs its own bit-identity probe.

### Remaining elementwise budget after c7

255/step, 458 us/step. The three biggest open blocks are unchanged in kind:
**64/step bf16<->fp32 casts** around the fused rms_norm (#1 mode 2's dtype boundary --
P-A tried to remove them with one Triton kernel and was rejected on numerics, previous section),
**~30/step PLE index math**, **~20/step QSA `FillFunctor`**. All three are eager-region
kernels, not missed inductor fusions.
