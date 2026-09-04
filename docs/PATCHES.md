# k4 kernel-count patches (bind-mountable working copies)

Working copies of fork files, edited in place under `$REPO/patches/`.
Nothing in `/app`, `$REPO/fork_vllm` or `$HOME/hotcold` was touched.
Mount list: `MOUNTS.txt` (one `-v` per line, splice into the launcher).

Baseline for every "per step" number below: the production HIP-graph trace census in
`$REPO/k4/KERNEL_HITLIST.md` — **3490 kernels / 61.9 ms / 47.2 ms busy /
14.7 ms of inter-kernel gap, median gap 3.72 us** for one full decode step (target
forward + 4 MTP draft iterations + logits + sampler).

| # | file | env gate (default) | kernels/step | evidence |
|---|------|--------------------|--------------|----------|
| 1 | `model_executor/layers/layernorm.py` | `VLLM_GEMMA_NORM_FUSED=1` | **-288** | measured 10 -> 1 per call |
| 4a | `model_executor/layers/rotary_embedding/mrope.py`, `models/qwen4_exp/amd/indexer_qsa.py` | none | **-32** | measured 3 -> 2 per call |
| 5 | `model_executor/kernels/linear/mxfp4/r4dhip.py`, `hotcold/r4d_mxfp4_moe.py` | `VLLM_R4D_SHARE_A8=1` | **-52 (unverified)** | reasoned, not yet traced |
| 6 | `model_executor/layers/fused_moe/modular_kernel.py` | `VLLM_MOE_OUTPUT_ALIAS=1` | **-48** | removes a measured DtoD/layer |
| 10 | `ir/op.py` | none | 0 | diagnostics only |
| 2 | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `third_party/flash_linear_attention/ops/fused_sigmoid_gating.py` | `VLLM_GDN_STRIDED_QKV=1` | **-144** | measured 4 -> 0 copies x 36 layers, bit-identical |
| 7 | `model_executor/layers/fused_gate_mul.py`, `model_executor/models/qwen2_moe.py` | `VLLM_FUSED_SHARED_GATE=1` | **-48** | measured 2 -> 1 per MoE layer, bit-identical |
| 8 | `models/qwen4_exp/amd/mtp.py` + K3's `model_executor/kernels/draft_w4_lmhead.py` | `VLLM_DRAFT_W4_LMHEAD=1` | 0 | not a kernel count: -2.8 ms/step measured in production, **net +4-8% tok/s** after the acceptance-rate cost |
| 9 | `models/qwen4_exp/amd/mtp.py` | none (needs `use_local_argmax_reduction` in the speculative config) | small | replaces a 248320-column all-gather with a (value, index) reduction, ~-0.14 ms/step |
| 11 | `model_executor/layers/fused_silu_mul_quant.py`, `hotcold/r4d_mxfp4_moe_lru.py`, `hotcold/r4d_mxfp4_moe.py` | `VLLM_FUSED_SILU_QUANT=1` | **-52** | measured 2 -> 1 per MoE block x 52, bit-identical (proven on device, not argued) |
| 12 | `model_executor/layers/rotary_embedding/mrope.py`, `models/qwen4_exp/amd/indexer_qsa.py` | `VLLM_QSA_ROPE_GATHER=1` | **-32** | the QSA rope's two cos/sin gathers move inside the mrope kernel; bit-identical on 43 device cases |

Total ~**-420 kernels/step (12%)**, ~**-1.9 ms/step (~3%)** at 3.72 us of gap per kernel
plus the kernels' own time. Only #1 is measured end to end; treat the rest as estimates
until a re-profile confirms them.

---

## #1 GemmaRMSNorm: dispatch the fused rms_norm in eager regions

`model_executor/layers/layernorm.py`

**The hitlist's stated root cause was incomplete.** It said the fp32 weight from
`self.weight.float() + 1.0` failed the `weight.dtype == x.dtype` guard in
`kernels/vllm_c.py:17`. That is true but it is not the binding constraint:
`platforms/rocm.py:1130` sets

```python
default = ["native"] if using_inductor else ["vllm_c", "native"]
```

so with inductor on — production — the rms_norm priority list is `['native']` and the
dtype guard is never consulted. Confirmed at startup:

```
[kernel.py:308] Final IR op priority after setting platform defaults:
                IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
```

That default exists so inductor can fuse the decomposition. It works for the norms that
sit inside compiled regions. The QSA indexer's `q_layernorm`/`k_layernorm` do not: they
run in an eager region (that is why their kernels are `elementwise_kernel_manual_unroll`
/ `vectorized_elementwise_kernel`, aten names, not `triton_poi_fused_*`), so "native"
there means ten separately launched kernels per call and nothing ever fuses them.

**Change.** `forward_cuda` now, only when `not torch.compiler.is_compiling()`, resolves
the first supported non-native impl (`aiter`, `vllm_c`, `oink` — `vllm_c` on this box)
and calls it directly with `(1 + w)` cached in `x`'s dtype. Tracing still goes down
`forward_native`, so compiled regions and their fusions are untouched. `forward_native`
itself is unchanged — the fp32 weight is still what inductor sees.

Measured, `H=128` (indexer head dim), gfx1201, `local/q38fn-rocm10:try1`:

```
GPU events / call, no residual : native 10 -> fused 1
GPU events / call, residual    : native 13 -> fused  3   (harness clones excluded)
VLLM_GEMMA_NORM_FUSED=0 -> forward_cuda bit-identical to forward_native: True
```

32 calls/step (12 QSA layers x q+k, plus the MTP draft layer x 4 iterations x q+k)
x 9 kernels = **-288/step**.

### Numerics — read this before enabling in production

The fused kernels all require `weight.dtype == x.dtype`, so `(1 + w)` must be rounded to
bf16. That is a real perturbation, not a rounding footnote. Measured against the
checkpoint's own weights (`k4/gemma_real.py`, `k4/gemma_numerics.py`):

```
tensor                                            |w|max  |1+w|min   max rel  mean rel  bf16 differ%
layers.11.self_attn.indexer.k_layernorm.weight    0.3926   0.6074  1.198e-02 1.730e-03     31.4
layers.15.self_attn.indexer.q_layernorm.weight    0.3984   0.6016  1.105e-02 1.728e-03     31.4
layers.11.self_attn.q_norm.weight                 0.9023   0.0977  1.198e-02 1.966e-03     36.2
```

~30% of output elements move by >= 1 bf16 ulp; mean relative change ~1.7e-3, max ~1.2e-2
(about two ulps). Against an fp64 reference the old path is 1.41e-3 mean rel and the new
one 2.33e-3 — both dominated by bf16 output rounding, but the new path is ~1.65x further
out. The residual variant's residual output is bit-identical.

This is the same magnitude as the llama.cpp ubatch change that moved 0.78% of argmax
tokens. Gate it on a greedy divergence test **with a base-vs-base control** (q38fn greedy
output already diverges 4/5 across restarts) and enough tokens for the power you want:
N >= ln(1-B)/ln(1-p_min). `VLLM_GEMMA_NORM_FUSED=0` restores the stock path bit-exactly.

If the numerics are unacceptable, the accuracy-preserving version of this win is a
Gemma-specific fused kernel that keeps the `+1` in fp32 inside the kernel — a new kernel,
not a Python change.

### Mode 2 (the shipped default) -- measured, 2026-09-04

`VLLM_GEMMA_NORM_FUSED` defaults to **2**: cast x to fp32, call the fused kernel with an
fp32 `(1 + w)` (so its dtype guard passes without rounding the weight), cast back.
10 -> 3 kernels per call, -224/step.

It is **not** bit-identical to stock, and the earlier "same arithmetic, so the text is
unchanged" note in the file header was too strong. `k4/probe_gemma1.py`, H=128, eps 1e-6,
against the stock decomposition:

| rows | elements differing, mode 2 vs stock |
|---|---|
| 1, 20 | none |
| 2048 | 1 in 262144 |
| 65536 | 12 in 8388608 |
| 4096, larger weights | 1 in 524288 |

~3e-6 of elements move by one bf16 ulp, because the fused kernel's fp32 variance
reduction is not torch's `.mean(-1)` tree; once in ~1e6 the fp32 rsqrt lands the other
side of a bf16 rounding boundary. Mode 1 moves 30% of elements, so mode 2 is five orders
of magnitude tighter -- but it is a perturbation, not zero. Decode-sized calls (20 rows)
came back equal in every probe case.

## #4a mrope: pre-split contiguous cos/sin caches

`model_executor/layers/rotary_embedding/mrope.py` (new `cos_sin_halves` helper, used by
`MRotaryEmbedding.forward_cuda`) and `models/qwen4_exp/amd/indexer_qsa.py`
(`apply_qsa_rope`).

`cos_sin_cache[positions].chunk(2, -1)` yields two strided views, which `triton_mrope`
then has to `.contiguous()` (`mrope.py`, "ensure tensors passed into the kernel are
contiguous"): one gather plus two copies. Splitting the cache once, memoized on the rope
module and keyed on the cache tensor's identity, makes it two gathers and no copies.

Measured (`k4/rope_kernels.py`): **3 kernels -> 2**, values bit-identical, at 5/32/512
tokens. ~32 call sites/step (12 QSA `project_qk` + 12 main-attention mrope + MTP draft)
= **-32/step**.

Cost: one extra cache-sized buffer, **32 MB** at this model's `262144 x 64` bf16 cache,
shared by every caller of the rope instance (vLLM caches one instance for all layers).
That is ~15 experts' worth of VRAM — worth it at 0.05% of the weight budget, but it is
the reason to keep this proposal separable from the rest.

## #5 share the fp8 activation between the shared expert and the routed MoE

`model_executor/kernels/linear/mxfp4/r4dhip.py` (new `fp8_quant_shared`) and
`hotcold/r4d_mxfp4_moe.py` (both `apply` and `_apply_split` call sites).

On ROCm `SharedExperts._determine_shared_experts_order` cannot pick
`MULTI_STREAM_OVERLAPPED` (it requires `current_platform.is_cuda()`), so the shared
expert runs `NO_OVERLAP`, i.e. before the routed experts, on the same `hidden_states`
tensor — `MoEPrepareAndFinalizeNoDPEP.prepare` passes it through unquantized for this
quant config. Both then call `ops.scaled_fp8_quant` on it. The helper memoizes the result
on the tensor that owns the storage, keyed on
`(data_ptr, shape, stride, dtype, _version)`, so a recycled allocation or an in-place
write misses and simply re-quantizes.

**Unverified.** The saving is real only if the shared expert's input is the same tensor
object (or a view of the same base) as the routed `hidden_states`. That holds by code
reading but I have not confirmed it in a trace, and I did not want to claim a win I have
not seen. Up to **-52/step** (48 layers + 4 MTP draft iterations) if it hits, zero if it
does not — a miss costs one dict-free attribute lookup. `VLLM_R4D_SHARE_A8=0` disables.

## #6 let the r4d MoE write straight into the caller's output buffer

`model_executor/layers/fused_moe/modular_kernel.py:1335`

The alias that skips `TopKWeightAndReduceNoOP.apply`'s `output.copy_` is already there,
but on ROCm it is gated behind `rocm_aiter_ops.is_fused_moe_enabled()`. The r4d MXFP4
experts are not AITER, so the gate excluded them and every MoE layer paid a full
`__amd_rocclr_copyBuffer` of the hidden state. `_apply_split` ends in
`ops.moe_sum(ic3.view(M, top_k, H), output)`, which writes every row, and the existing
`use_output_alias` check already requires matching shape/dtype/device/contiguity, so the
alias is safe here. Aliasing also *removes* an aliasing hazard: `fused_out` otherwise
shares storage with `workspace13` (`_allocate_buffers`, `_resize_cache(common_workspace,
...)`).

**-48/step.** `VLLM_MOE_OUTPUT_ALIAS=0` restores the AITER-only gate.

## #10 make the IR-op fallback visible

`ir/op.py`, `IrOp.dispatch`

The skip path only logged at `debug`, which is how #1 stayed hidden. Now also
`logger.warning_once`, keyed on `(op name, provider)` only — both plain strings, so the
lru_cache key is cheap and the lazy tensor formatting stays on the `debug` line. Still
inside the existing `not torch.compiler.is_compiling()` guard.

Note this warning will *not* fire for the #1 case: when the priority list is
`['native']`, `dispatch` returns before the loop. It fires when a provider is in the list
and rejects the args.

---

## Not implemented

**#8 GDN `_forward_core_rocm` `z_out[:] = z`** (`qwen_gdn_linear_attn.py:1236`, -36/step).
The shape check passes, but the only way to remove the copy is to have `forward_hip` pass
a slice of `projected_states_qkvz` as the `z` argument instead of a fresh `torch.empty`.
`z` is the `a_or_z_out` parameter of `torch.ops.vllm.qwen_gdn_attention_core`, declared
`mutates_args=["a_or_z_out", "core_attn_out"]` (`:1946-1949`). Aliasing an input into a
mutated argument of a custom op is exactly what functionalization forbids; under
torch.compile it either reintroduces a clone or silently reads stale data. Not worth 0.06%
of a step. The correct fix is upstream: give the op a functional signature that returns z,
or drop `z_out` and let the caller slice.

## Validation performed

- `python3 -m py_compile` on all 9 files under `patches/` — all OK.
- In-container import smoke with every `MOUNTS.txt` bind applied
  (`k4/smoke.sh` -> `k4/smoke_imports.py`): all 9 modules import from the mounted paths,
  patched symbols present, `cos_sin_halves` bit-identical to `chunk()` and memoized,
  `fp8_quant_shared` hits on identity and misses on a new tensor and on an in-place write.
- `k4/gemma_verify.py`: kernel counts and the `VLLM_GEMMA_NORM_FUSED=0` bit-identity check.
- `k4/gemma_numerics.py`, `k4/gemma_real.py`: the bf16-weight numerics tables above.

Not done: no server was started, so nothing here is confirmed against a real decode step.
The next step is a profiled run with these mounts, re-censused with `k4/graph_census.py`,
against the 3490-kernel baseline.

## Cross-patch import rule (2026-09-04)

A patched module must never hard-depend on a symbol that only exists in another patched
module: mounting one without the other raises ImportError, and the fork's MoE loader
swallows that into `Using OCP MX emulation Triton experts ... r4d unavailable: import
failed` — a silent ~20x decode regression, not a crash.

Two such dependencies existed; both are now guarded with try/except + a fallback that
reproduces the stock behaviour:

| file | symbol | source | fallback |
|---|---|---|---|
| `hotcold/r4d_mxfp4_moe.py` | `fp8_quant_shared` | patched `r4dhip.py` | `ops.scaled_fp8_quant(..., use_per_token_if_dynamic=True)` + WARNING (added by team-lead) |
| `models/qwen4_exp/amd/indexer_qsa.py` | `cos_sin_halves` | patched `mrope.py` | inline `_match_cos_sin_cache_dtype(x)` + slice halves (bit-identical to stock `chunk()`) |

Verified by `k4/smoke_degraded.sh` + `k4/smoke_noguard.py`: mounting ONLY those two
files (stock `mrope.py`/`r4dhip.py` underneath) imports cleanly, the r4d WARNING fires,
`R4dMxfp4MoEExperts` is still exported, and the fallback cos/sin is bit-identical to
`chunk()`. Full-mount `k4/smoke.sh` still reports SMOKE OK.

## Inference-tensor / graph-capture audit (2026-09-04)

The k4patch arm died at startup in `fp8_quant_shared` with `RuntimeError: Inference
tensors do not track version counter.` Tensors in the model forward are created under
`torch.inference_mode()` and have no `_version`, so any cache key that reads it raises
on the first call.

**Two sites read `_version`; both are fixed.**

- `r4dhip.fp8_quant_shared` (#5) — the crash. Redesigned and **now DEFAULT OFF**
  (`VLLM_R4D_SHARE_A8=0`). Dropping `_version` from a data_ptr key would have been
  unsound, not just unverified: a freed address is reused by an unrelated tensor, so a
  hit could serve another layer's activation. It is now a single-slot handoff keyed on
  Python **object identity** (`slot[0] is x`), holding a strong ref so the cached object
  cannot be freed and aliased while live, and the consumer **clears** the slot so a hit is
  served at most once. Residual assumption: nothing writes into that tensor in place
  between the two calls. Enable only behind a greedy-divergence gate.
- `GemmaRMSNorm._weight_for` (#1) — same latent crash, not yet reached because the MoE
  faulted first. `_version` bought nothing on a weight (static after load); the key is
  now `(data_ptr, w.dtype, x.dtype, x.device)`.

**Graph capture.** Both surviving memos allocate their cached tensor on first call. A
tensor allocated during HIP-graph capture lives in that graph's private pool and is not
valid for another graph or for eager. vLLM warms up eagerly before capturing, so the
first call is eager in practice — but that is an assumption, so both miss paths now check
`torch.cuda.is_current_stream_capturing()` (0.27 us, miss path only) and skip caching,
with a `warning_once` in the layernorm case rather than silently caching pool memory.

**Clean:** no `.item()`, `.cpu()`, `.tolist()`, `.numpy()` or `synchronize()` on any
forward path in the patch set. (`r4d_mxfp4_moe.py:110` `.tolist()` is hot-set setup at
repack time, not per step.)

`k4/smoke_imports.py` now runs its tensor checks **inside `torch.inference_mode()`** and
asserts `h.is_inference()` first — the old version ran outside it, which is exactly why
this reached an arm.


## Why #1 did not engage in the k4patch2 arm (2026-09-04) -- FIXED

**Root cause:** the fused path was in `forward_cuda`, which is never called. `CustomOp`
dispatch (`custom_op.py:174-207`) resolves `enabled = self._enforce_enable or
self.enabled()`; with inductor on, `custom_ops` defaults to `["none"]`, so `default_on()`
is False (`custom_op.py:295-311`) and `dispatch_forward` returns `forward_native` at line
191 -- `forward_cuda` (and `forward_hip`, which just delegates to it) is unreachable. The
QSA indexer calls `norm(tensor)` via `apply_qsa_rmsnorm` (`indexer_qsa.py:65-71`), i.e.
`__call__` -> the dispatched method, so it always landed on the unfused 10-kernel
decomposition. Same design intent as the `rms_norm=['native']` IR priority: with inductor
on, everything is routed to the decomposition on purpose.

Not the cause: `torch.compiler.is_compiling()` is False during graph capture (verified --
a captured replay shows the same count as eager), and the env default is "1" with no `-e`.

**Fix:** the fused attempt moved into `GemmaRMSNorm.forward_native` as `_try_fused`, still
gated on `not torch.compiler.is_compiling()` so a compiled region still traces the fusible
decomposition for inductor. `forward_cuda` is back to the stock one-line delegate.

**Verified through the real dispatch** (`k4/gemma_dispatch.py`, indexer dims H=128):

| | kernels / call, `m(x)` | in HIP-graph replay |
|---|---|---|
| VLLM_GEMMA_NORM_FUSED=1 | **1** | **1** |
| VLLM_GEMMA_NORM_FUSED=0 | 10 | 10 |

`dispatched method: forward_native`, `enabled(): False` in both -- i.e. the fix engages on
the path production actually takes. Gate=0 is bit-identical to the stock fp32 formula
(max abs 0.0); gate=1 differs on 29.3% of elements, max rel 7.8e-3, mean rel 1.6e-3.

**Trap:** the old `k4/gemma_verify.py` called `m.forward_cuda` / `m.forward_native`
directly and so "proved" a patch the dispatcher never reached. Any check of a CustomOp
must call the module. Note too that once `forward_native` is patched it is no longer a
control -- compare against an explicit fp32 reference, not `forward_native`.

## Measured attribution of the k4patch2 arm (`k4/prof_diff.py`)

One median decode step, tp0, `prof_base_h15` vs `prof_k4patch2`: 3007 -> 2959, **-48**.

| kernel | delta | patch |
|---|---|---|
| `__amd_rocclr_copyBuffer` | 87 -> 39 (**-48**) | **#6, working exactly as predicted** |
| `elementwise_kernel_manual_unroll` | 282 -> 258 (-24) | #4a |
| `vectorized_gather_kernel` | 12 -> 24 (+12) | #4a |
| `index_elementwise_kernel` | 27 -> 39 (+12) | #4a |
| (none) | 0 | #1 -- did not engage |

- **#6 works.** `Memcpy DtoD` is unchanged at 39 in both traces, which is why it looked
  inert: on ROCm a device-to-device `copy_` is executed by the `__amd_rocclr_copyBuffer`
  **kernel** and is not recorded as a memcpy event. Count the kernel, not the memcpy.
- **#4a is a wash**: -24 copy kernels, +24 indexing kernels, net 0, and it costs +33 MB of
  VRAM. Recommend dropping it -- VRAM is the decode lever on this box.

## `VLLM_GEMMA_NORM_FUSED=2` -- fp32-exact fused norm (2026-09-04, now the default)

Mode 1 reaches the fused kernel by rounding `(1 + w)` to bf16, which is a real numerics
change (1 bf16 ulp on ~24% of output elements) and moved the generated text in the
k4patch3 arm. Mode 2 satisfies the kernel's `weight.dtype == x.dtype` guard from the
other side instead: cast `x` up to fp32, run the fused kernel with the precomputed fp32
`(1 + w)`, cast the result back. That is the same arithmetic the stock decomposition
does -- normalize in fp32, multiply by fp32 `(1 + w)`, round once at the end.

Modes (`patches/model_executor/layers/layernorm.py`, `GemmaRMSNorm._try_fused`):

| `VLLM_GEMMA_NORM_FUSED` | path | kernels/call |
|---|---|---|
| 0 | stock `ir.ops.rms_norm` decomposition | 10 |
| 1 | fused kernel, `(1 + w)` rounded to bf16 | 1 |
| 2 (default) | cast up, fused kernel in fp32, cast back | 3 |

Mode 2 only takes the no-residual path; `fused_add_rms_norm` writes the residual in
place and it has to stay bf16, so a residual call falls through to the stock path. The
QSA indexer never passes a residual.

**Measured** (`k4/gemma_mode2.py` via `k4/run_gemma_mode2.sh`, in-container, real
dispatch through `nn.Module.__call__`, all 26 `*.indexer.{q,k}_layernorm.weight`
tensors from `/mnt/llm-storage/q38fn-heretic2-mxfp4-fp8`, x scales 0.1/1/10, 4/32/256
tokens, 2,915,328 output elements per mode, reference = the stock fp32 formula written
out explicitly):

| mode | kernels eager | kernels in HIP-graph replay | max abs | max rel | elements differing | max ulp |
|---|---|---|---|---|---|---|
| 0 | 10 | 10 | 0 | 0 | 0 / 2,915,328 | 0 |
| 1 | 1 | 1 | 3.125e-2 | 7.812e-3 | 689,413 / 2,915,328 (23.7%) | 1 |
| 2 | 3 | 3 | 7.812e-3 | 7.752e-3 | **6 / 2,915,328** | 1 |

Mode 2's six differing elements are single bf16 ulps from the fused kernel's variance
reduction order, not from the weight -- floor for any kernel that is not a bit-for-bit
replay of `torch.mean`. Mode 0 is bit-identical, which also confirms the reference.

Cost vs mode 1: +2 kernels per norm site. The k4patch3 arm showed 32
`vllm::rms_norm_kernel` per step, so mode 2 should land near 3216 kernels/step instead
of mode 1's 3152 (baseline 3490) -- roughly -274 rather than -338, with the text
unchanged.

## #8 draft-only W4 lm_head -- `VLLM_DRAFT_W4_LMHEAD=1` (2026-09-04, default 0)

`patches/models/qwen4_exp/amd/mtp.py` (new mount). Adds `Qwen4ExpMTP._w4_draft_logits`
and routes `compute_logits` through it when the env gate is on. Depends on K3's
`vllm/model_executor/kernels/draft_w4_lmhead.py` (`pack_w4a16`, `gemm_w4a16`);
the import is guarded, so an arm that mounts this file without K3's module logs a
warning and keeps the bf16 head.

### Where the draft logits are computed

- Draft model class `Qwen4ExpMTP.compute_logits` (mtp.py) -- distinct from the target's
  `Qwen4ExpForCausalLM.compute_logits` (model.py:789). Only the speculator calls it.
- Callers: `v1/worker/gpu/spec_decode/speculator.py:329` (`_greedy_sample_draft`) and
  `:343` (`sample_draft` with a logits cache). `draft_sample_method` defaults to
  `"greedy"` and the launcher does not set it, so `draft_logits is None` and the greedy
  path runs: full logits, then `logits.argmax(-1)`.
- 4 calls per decode step: `_generate_draft` at draft position 0, then
  `_multi_step_decode` steps 1..3 (`num_speculative_tokens=4`).
- Draft decode steps replay a FULL CUDA graph
  (`autoregressive/speculator.py:134`, PIECEWISE is refused for draft decodes), so the
  W4 GEMM has to be graph-capturable and the packing has to happen before capture.

### The lm_head module is SHARED with the target

`load_eagle_model` (`v1/worker/gpu/spec_decode/eagle/utils.py:83-90`) deletes the draft's
own `lm_head` and assigns the target's, because `_should_share` returns True whenever the
draft does not set `has_own_lm_head` -- and `Qwen4ExpMTP` does not. So:

- the draft's `self.lm_head` **is** the target's `ParallelLMHead` (same weight tensor);
- the packed W4 copy is a side buffer on the MTP wrapper (`self._draft_w4_head`), never
  a mutation of `lm_head.weight`, so the target's logits path is bit-identical;
- packing cannot happen in `__init__` or `load_weights` -- both run before the swap. It
  is deferred to the first `compute_logits`, which lands in the eager profile run
  (`model_runner.py:788` calls `speculator.propose(dummy_run=True)` at line 858, before
  `speculator.capture()` at line 946). A `torch.cuda.is_current_stream_capturing()` guard
  warns and stays on bf16 if that warmup order ever changes, rather than allocating into
  a graph's private pool.

The gather and trim after the GEMM are the stock `LogitsProcessor` ones
(`_gather_logits` + `[..., :org_vocab_size]`), so the draft's argmax sees the same shape
and dtype as before. If the logits processor is ever not a plain projection
(`scale != 1`, `soft_cap`, an fp32 `head_dtype`, `logits_as_input`) the W4 path disables
itself with a warning instead of silently dropping the operation.

### Measured cost of the bf16 draft head (`prof_base_h15`, tp0, `k4/kern_top.py`)

The lm_head GEMM is `wvSplitK_hf_sml_<__hip_bfloat16, ...>`, and it does **not** appear in
the `generation` annotation at all -- all 5 per step run in the ~8.9 ms gap between
annotations:

| variant | per step | median dur | what |
|---|---|---|---|
| `wvSplitK_hf_sml_<bf16, 32, 2, 16, 8, 2, 1>` | 4.03 | 994 us | the 4 DRAFT heads |
| `wvSplitK_hf_sml_<bf16, 32, 4, 16, 8, 2, 5>` | 0.97 | 998 us | the 1 TARGET head |

124160 x 2560 bf16 = 635.6 MB per rank; 635.6 MB / 994 us = 639 GB/s, i.e. ~99% of the
R9700's 644.6 GB/s. The head is purely weight-bandwidth-bound, so 4-bit weights should
scale it almost linearly: ~250 us per call, **~3.0 ms/step saved** on the 4 draft calls.

Everything else in the gap is small: `clav_ag_push_kernel` 13 x 20.4 us (the vocab-
parallel all-gather, ~0.5 MB/row), `r4d_ar_oneshot` 12 x 9.6 us, `reduce_kernel`
12 x 3.5 us (the argmax). Gap total 8.88 ms, of which 7.14 ms is kernel-busy and ~5.0 ms
is lm_head.

### Measurement trap this exposes

`k4/prof_diff.py` used to count only the `generation` annotation, which is the model
forward. That window is 46.4 ms of a 55.3 ms step and contains **none** of the lm_head
work. `prof_diff.py` now defaults to the full step (annotation start -> next annotation
start); `--annotated` reproduces the old window. Re-measured with the wide window,
prof_base_h15 -> prof_k4patch2 is 3443 -> 3393 kernels, 58.9 -> 53.7 ms.

### No pruned-vocab draft head exists in the fork

`grep` for `dynmtp` / `drafthead` / `draft_head` finds nothing; the only `radiance` hits
are `_radiance_stop` (an early-exit controller in `v1/spec_decode/llm_base_proposer.py`)
and a comment in `model_executor/kernels/r4d_lib.py`. The draft does compute all 248,320
vocab entries when only the argmax is used.

**But the fork already has the machinery to skip the all-gather**:
`LogitsProcessor.get_top_tokens` (`logits_processor.py:189`) does a vocab-parallel local
argmax and gathers only (value, index) pairs, and the speculator uses it whenever
`speculative_config.use_local_argmax_reduction` is true (`speculator.py:327`). It is
off by default, and `Qwen4ExpMTP` did not implement `get_top_tokens`, so it could not be
turned on for this model at all. **Now implemented -- see #9 below**, with the W4 head
folded inside it rather than stacked on it (it calls `_apply_head` directly, so it would
otherwise BYPASS the W4 `compute_logits` override). Worth ~0.14 ms/step on this trace (the
~100 us of all-gather plus ~40 us of argmax), not the GEMM.

### Wired to K3's real kernel (2026-09-04)

`patches/model_executor/kernels/draft_w4_lmhead.py` (K3's file, added to `MOUNTS.txt`,
line 15) is now the module the guarded import finds. K3 measured it in a HIP graph at
N=124160 K=2560: 261-307 us/call against 1006-1059 us for the bf16 head, 3.8x at draft M,
so the four draft calls a step go ~4.0 ms -> ~1.05 ms. It is +161 MB of VRAM per rank,
not a saving: the target still holds the bf16 lm_head and the draft shares it.

Guards added on this side, all sticky-disable with a warning rather than raising:

- `available()` -- false when libr4d has no `r4d_gemm_w4a16_nt_m64` entry point.
- lm_head shard must be 2-D with `K % 128 == 0` (the kernel is group-128 only).
  124160 x 2560 per rank qualifies.
- per call, `hidden_states` must be 2-D with `shape[-1] == packed.k`; anything else
  falls through to bf16 for that call instead of tripping the kernel's assert.

Quality: 9.99% relative Frobenius error on the weight (the 4-bit grid, not the layout).
That can only lower the MTP acceptance rate; the target's logits path is untouched, so it
cannot change emitted text. Compare arms on ms/step AND on `spec_decode_num_accepted` --
a W4 arm that speeds up the step but drops acceptance can be a net loss.

**Activation range.** K3's kernel casts the bf16 hidden state to f16 and deliberately does
not clamp, so a draft hidden state past 65504 becomes inf (costs an acceptance, never a
wrong target token). Unverified on real states -- see the dump below.

### Measured in production (team lead, arm `w4head` = LRU combo + gate on)

It engaged (`draft W4 lm_head packed: (124160, 2560) -> 4-bit`) and
`r4d_gemm_w4a16` x4 ran at 255 us against 4 x ~1 ms before; full step 41.7 -> 38.9 ms in
the profile. ab3, three prompt types:

| | ms/step | accept | net tok/s |
|---|---|---|---|
| LRU combo | 35.0 / 35.2 / 34.3 | 0.464 / 0.724 / 0.547 | 81.3 / 110.7 / 93.0 |
| + `VLLM_DRAFT_W4_LMHEAD=1` | 31.9 / 32.1 / 31.1 | 0.421 / 0.706 / 0.530 | 84.5 / 119.4 / 100.3 |

So the acceptance cost is real (-3 to -9% relative) but smaller than the step saving:
**net +4 to +8% tok/s**. Adopted in `$HOME/launch_q38fn_lru.sh`. This is why the arm
has to be read on both numbers -- on ms/step alone the win would have looked like 9%.

### `VLLM_DRAFT_HS_DUMP` -- capture real draft hidden states (default off)

`VLLM_DRAFT_HS_DUMP=/path/prefix.npz` (with `VLLM_DRAFT_HS_DUMP_N`, default 200) makes
`compute_logits` keep a rolling window of the last N rows of its input and rewrite the npz
every N rows, logging `max|x|` each time (the f16-range check above). Key `hidden_states`,
float32.

**It must be run with `--enforce-eager`.** `AutoregressiveSpeculator.capture` records
"model forward + compute_logits + sample" as ONE full graph
(`v1/worker/gpu/spec_decode/autoregressive/speculator.py:154-158`), so this python does
not re-run on replay and would only ever see the warmup rows. The helper also returns
early while `torch.cuda.is_current_stream_capturing()`. Rolling (rather than first-N)
collection is what keeps the profile run's dummy rows out of the file: they fall out of
the window as soon as real traffic arrives.

## `MOUNTS_COMBO3.txt` -- the candidate arm plus #2 and #7 (2026-09-04)

`launch_q38fn_lru.sh` defaults to `MOUNTS_COMBO2.txt`, which does NOT mount the four #2/#7
files, so `VLLM_GDN_STRIDED_QKV=1` / `VLLM_FUSED_SHARED_GATE=1` are inert there no matter
what the env says -- that is why those two have never been measured live. `MOUNTS_COMBO3.txt`
is COMBO2 plus exactly those four (`qwen_gdn_linear_attn.py`, `fused_sigmoid_gating.py`,
`fused_gate_mul.py`, `qwen2_moe.py`); COMBO2 is untouched. To measure them:

    MOUNTS_FILE=$REPO/patches/MOUNTS_COMBO3.txt \
      $HOME/launch_q38fn_lru.sh 15.0     # with -e VLLM_GDN_STRIDED_QKV=1 \
                                              #      -e VLLM_FUSED_SHARED_GATE=1

Both are bit-identical by construction (no numerics change, only fewer launches), so
unlike #8 this arm should move ms/step with the acceptance rate flat -- an accept-rate
change here would mean something is wrong, not that a trade was made.

## #9 `get_top_tokens`: vocab-parallel draft argmax (2026-09-04)

`patches/models/qwen4_exp/amd/mtp.py`. Adds `Qwen4ExpMTP.get_top_tokens(hidden_states)`,
which the fork's speculator calls INSTEAD of `compute_logits` whenever
`speculative_config.use_local_argmax_reduction` is true
(`v1/worker/gpu/spec_decode/speculator.py:327`, `v1/spec_decode/llm_base_proposer.py:442`).
Until now the flag could not be turned on for this model at all: the speculator raises if
the draft class has no `get_top_tokens`, and `Qwen4ExpMTP` had none.

Instead of all-gathering 248,320 logit columns and taking a global argmax, each rank
argmaxes its own 124,160-column shard and only (value, index) pairs are gathered:
O(batch * 2 * tp_size) instead of O(batch * vocab_size). Worth ~0.14 ms/step on the
`prof_base_h15` trace (~100 us of all-gather plus ~40 us of argmax), plus the launches.

**No env gate, and that is deliberate**: the method is inert unless the speculative config
sets `use_local_argmax_reduction`, so its mere presence changes nothing. Turn it on in the
launcher's `--speculative-config` JSON.

**The W4 head is folded IN, not stacked.** `get_top_tokens` bypasses `compute_logits`
entirely, so a W4 override there would have been silently skipped -- the arm would have
looked like the W4 win had evaporated. `_w4_draft_logits` is therefore split: the new
`_w4_local_logits` does guards + pack + GEMM and returns SHARD-LOCAL logits;
`_w4_draft_logits` adds the stock gather and vocab trim on top; `get_top_tokens` skips
both and runs the (value, index) reduction on the same local logits. With the W4 head off
or refused, it delegates to the stock `LogitsProcessor.get_top_tokens` unchanged.

The reduction is a copy of the stock one (`logits_processor.py:189-239`) with `_apply_head`
replaced by the W4 GEMM, minus its `soft_cap` and `scale` branches, which are unreachable
here because `_w4_local_logits` refuses both outright. It keeps the shard's
`num_org_vocab_padding` mask (padded slots to -inf) and the `org_vocab_start_index` offset,
so a draft token can never be a padding slot or a shard-local id.

### The W4 verdict is now unanimous across TP ranks (K3's catch)

Every W4 guard (`available()`, the lm_head shape, the non-plain LogitsProcessor refusal,
the missing mount) is evaluated **per rank**. Under `compute_logits` a one-rank failure was
merely a mixed-precision all-gather; under `get_top_tokens` it is silently WRONG -- one
shard would contribute W4 values and the other bf16 values, and the reduction compares
those values directly to pick the winning shard, so a ~10%-magnitude difference on one side
biases which shard wins the token.

So the decision moved into `_pack_w4_head_once`, called once at warmup: each rank votes,
`tensor_model_parallel_all_reduce` sums the votes, and anything short of unanimous disables
the W4 head on every rank. **No guard may return early above that all-reduce** -- that is
why the guards now set a `reason` string and fall through instead of returning. A guard
that returned early on one rank would deadlock the collective rather than disagree with it.
It costs one `.item()` host sync, once, eagerly, before capture.

### Two things about the reduction that are NOT evidence (K3)

- **The padding mask is dead code today.** `vocab_size` is 248320 and the checkpoint's
  `lm_head.weight` has exactly 248320 rows, so `pad_vocab_size(248320, 64)` is a no-op:
  padded == org, 124160 per shard, `num_org_vocab_padding == 0`. The stock
  `[..., :org_vocab_size]` trim was already a no-op too. Keep the mask -- it is correct if
  the vocab ever changes -- but a passing arm is NOT evidence that the masking works.
  Same for K3's side: 124160 % 16 == 0, so his `n_pad == n` and no slice happens either.
- **Tie-breaking differs from the all-gather path.** The W4 output is bf16 (8 mantissa
  bits), so exact ties between the two shards' top values are not rare. `argmax` over the
  concatenated tensor broke ties by lowest index; the (value, index) reduction breaks them
  by comparison order. That is a property of the local-argmax path, not of W4 -- but if a
  token ever differs between the two paths, look here before suspecting the kernel.

The global index is `local + shard.org_vocab_start_index`, taken from the shard's own row
count rather than `org_vocab_size // tp_size`; identical today, different failure mode if
the vocab ever stops dividing evenly.

**The two levers are independent in one direction only**: `use_local_argmax_reduction`
without W4 works (delegates to stock), W4 without the flag works (goes through
`compute_logits`), both together work. But W4's numerics reach the draft either way, so the
acceptance-rate reading above still applies to a combined arm.

## #2 GDN strided qkv -- `VLLM_GDN_STRIDED_QKV=1` (2026-09-04, default 0)

Two mounts:
`patches/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py` and
`patches/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`.

**The hitlist's premise was wrong.** `fused_sigmoid_gating_delta_rule_update` does *not*
take q/k/v strides: its wrapper calls `q.contiguous()` / `k.contiguous()` /
`v.contiguous()` (fused_sigmoid_gating.py:249-251) and the kernel addresses them with
hardcoded contiguous arithmetic, `p_q = q + (bos * H + i_h) * K + o_k`, advancing by
`H * K` per token. Handing it strided views would only have moved the copies inside.

So the kernel now takes the token stride as a parameter:
`p_q = q + bos * stride_q_tok + i_h * K + o_k`, advancing by `stride_q_tok` (same for k
and v). A contiguous tensor passes `H * K` / `HV * V` and addresses byte-for-byte as
before. The wrapper derives the stride with `_token_stride()` and falls back to
`.contiguous()` for any layout the kernel cannot walk (non-unit last stride, a
(heads, dim) row that is not packed, or a batch that is not exactly T tokens). The module
exports `SUPPORTS_STRIDED_QKV = True` as a capability probe.

On the GDN side, `rearrange_mixed_qkv_strided` returns the same three
`(1, T, heads, dim)` tensors as `as_strided` views into the packed qkv -- no copies at
all -- and `_rearrange_for_fused` picks between it and the stock repack. Only the two
**decode** call sites use it (qwen_gdn_linear_attn.py:1380 spec-decode and :1469 non-spec
decode, both feeding the fused kernel). The prefill site at :1434 feeds
`chunk_gated_delta_rule`, a different op, and is untouched.

The GDN patch probes `SUPPORTS_STRIDED_QKV` before using the views: with the stock FLA
kernel mounted the views would still be *correct* (the wrapper copies them) but would buy
nothing, and a silent no-op is worse than a warning.

**Measured** (`k4/test_gdn_strided.py` via `k4/run_gdn_test.sh`, production per-rank
geometry: 8 key heads x 128 + 8 x 128 + 24 value heads x 128 = [T, 5120], 4 requests x 4
MTP tokens, bf16, `use_qk_l2norm_in_kernel=True`, `inplace_final_state=True`):

- output **bit-identical**, `ssm_state` **bit-identical** (`torch.equal`, not a tolerance)
- 6 -> 2 GPU launches per GDN layer, i.e. **-4 per layer, -144 kernels/step** at 36 GDN
  layers -- exactly the hitlist estimate. (The remaining 2 are the fused kernel and the
  test's own state clone.)

Degraded-mount behaviour verified separately (`k4/degraded_gdn.py`): with only the GDN
file mounted, the probe returns False, logs a warning, and the stock repack runs.

## #7 fused shared-expert gate -- `VLLM_FUSED_SHARED_GATE=1` (2026-09-04, default 0)

`patches/model_executor/layers/fused_gate_mul.py` (new file) and
`patches/model_executor/models/qwen2_moe.py`.

`Qwen2MoeMLP.forward` ends in `out = F.sigmoid(self.expert_gate(x)[0]) * out` -- two aten
kernels that inductor cannot fuse because the shared expert runs on an aux stream. This
is the model's shared expert: `Qwen3NextMLP` **is** `Qwen2MoeMLP`
(qwen3_next.py:51 imports it under that alias), and `Qwen4ExpSparseMoeBlock` extends
`Qwen3NextSparseMoeBlock`, which builds it with `expert_gate=self.shared_expert_gate`.

One Triton kernel replaces both. It reproduces aten's rounding exactly -- sigmoid in
fp32, rounded to the gate's dtype, then the multiply -- so the result is bit-identical,
and it writes a fresh tensor rather than mutating `out` in place, because `out` is a
`down_proj` result that may be an all-reduce output buffer. `supported()` gates it to the
`[T, 1] x [T, H]` shape it is written for; anything else takes the stock path.

**Measured** (`k4/test_gate_mul.py`, hidden_size 2560, T in {1, 4, 20, 64, 2048}, gate
scales 0.02 / 1.0 / 30.0 so the sigmoid saturates in both directions):

- **bit-identical** to `F.sigmoid(g) * out` on every shape and scale (`torch.equal`)
- 2 -> 1 launches, i.e. **-48 kernels/step** at 48 MoE layers -- the hitlist estimate
- `supported()` refuses a non-`[T, 1]` gate and a dtype mismatch

### Shared caveat for both

Each is a Triton kernel whose first launch JIT-compiles. The engine's eager warmup runs
before graph capture, so compilation lands there; if the warmup order ever changes, the
compile would happen during capture. Neither kernel autotunes (fixed `num_warps`, no
`@triton.autotune`), so there is no device sync to trip over, but it is worth knowing.
Note also that the strided FLA kernel compiles a fresh specialization because the strides
are `tl.constexpr` -- one extra compile at startup, not per step.


## #11 fused silu+quant -- `VLLM_FUSED_SILU_QUANT=1` (2026-09-04, default 0)

`model_executor/layers/fused_silu_mul_quant.py` (new), wired into `_act_quant` in
both hotcold MoE modules.

### What it replaces

Both r4d apply paths did the same three statements after GEMM1:

```python
ic2 = torch.empty(mtk, N1 // 2, dtype=hidden_states.dtype, device=...)
self.activation(activation, ic2, ic1.view(-1, N1))          # torch.ops._C.silu_and_mul
q2, s2 = ops.scaled_fp8_quant(ic2, use_per_token_if_dynamic=True)
```

`ic2` is never read again after the quant, so the bf16 tensor exists only to be
handed from one kernel to the next. One Triton kernel now does silu_and_mul and
the per-token fp8 quant in a single pass and does not write `ic2` at all
(`write_act=False`; the bf16 output is still available for tests).

### Counts (measured, `prof_w4head` rank 0, 84 decode steps)

`k4/kattrib.py` counts kernels in a kineto trace; the per-step divisor is 84,
from `_get_num_sampled_and_rejected_kernel` (84) and confirmed by
`r4d_gemm_w4a16` (336 = 4/step) and `lru_gather_k` (4032 = 48/step).

| kernel | count | /step | what it is |
|---|---|---|---|
| `vllm::act_and_mul_kernel<BFloat16,...>` | 4368 | **52.00** | the routed MoE activation -- this patch removes it |
| `vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided` | 16128 | 192.00 | 52 of these are the `ic2` quant this patch absorbs |
| `vllm::moe::moe_sum_vec_dynamic_kernel` | 4368 | 52.00 | not foldable, see below |
| `triton_poi_fused_mul_silu_slice_0` | 4368 | 52.00 | the shared expert's SiluAndMul under inductor -- NOT folded, see below |
| `per_token_group_quant_8bit_kernel` | 8736 | 104.00 | the fp8 block-quant attention path, outside the MoE block (K2's lane) |
| `_hc_silu_kernel` | 9156 | 109.00 | not the MoE activation |

52 MoE blocks per step = 48 layers + 4 MTP draft layers (`mtp_num_hidden_layers=1`),
which is why the MoE counts are 52 and the LRU counts are 48.

Net: **-52 launches/step**, plus the removed `act_and_mul` GPU time (76 us/step)
and the `mtk x 320` bf16 write and read-back.

### Bit-identical, proven on device

`k4/probe_silu_quant.py`, run in the image on GPU 1, asserts `torch.equal` on the
activation, the fp32 scale and the raw fp8 payload against
`torch.ops._C.silu_and_mul` + `ops.scaled_fp8_quant` -- exact on all 30
rows x {320,640} x {unit, 1e-3, 60x} cases and on the all-zero row (scale lands
on vllm's floor, 1/(448*512) = 4.35965e-06). Three of those cases are now in
`k4/smoke_imports.py`, so a future edit that drifts the rounding fails the smoke.

Getting there needed three details from the reference kernels, each of which
changes the answer on its own:
- `act_and_mul_kernel` rounds silu to the storage dtype *before* the multiply
  (`c10::BFloat16` round-trip), then rounds the product on store.
- `dynamic_per_token_scaled_fp8_quant_kernel` floors the scale at
  `1/(FP8_MAX*512)`, so an all-zero row does not divide by zero.
- it *divides* by the scale; it does not multiply by a reciprocal.

`fp8_dtype()` on gfx1201 is `torch.float8_e4m3fn` (max 448), not the MI300 fnuz.

### Guards

- `VLLM_FUSED_SILU_QUANT` defaults to 0; the old two-launch path is untouched.
- The kernel module is imported through a guarded accessor, so an unmounted
  `fused_silu_mul_quant.py` warns once and keeps the stock path -- it must not
  drop the whole r4d MoE to Triton emulation.
- Only `MoEActivation.SILU` with `clamp_limit is None` takes the fused path;
  anything else (gelu, swigluoai, a clamped silu) falls through.
- `supported()` requires 2-D bf16/fp16, an even last dim, unit stride on the last
  dim, and a half-row that fits one Triton block (<= 8192). One program owns a
  whole row, so a wider row would be silently truncated; it is refused instead.
- Both hotcold modules got the identical change so the LRU and non-LRU paths
  cannot drift.
- `_apply_split` row coverage (confirmed by K1, 2026-09-04): `ic1` is `torch.empty`
  and the hot/cold GEMMs fill disjoint row sets, but every (token,k) slot is in
  exactly one of `map_hot`/`map_cold` and the LRU manager preserves that, so all
  rows are written before the quant and the per-row amax sees the same data
  `scaled_fp8_quant(ic2)` saw. A future "skipped rows" mode would break this.
- Side streams are safe: `_gemm_split` joins with `cur.wait_stream(side)` before
  `_act_quant`, so the fused kernel reads `ic1` on the current stream exactly as
  `silu_and_mul` did (`VLLM_R4D_HOT_SIDE_STREAM` is measured-dead anyway).

### Two things deliberately NOT folded

**`moe_sum` (52/step).** It reduces the `[M, top_k, H]` GEMM2 output. Folding it
into the GEMM epilogue means editing libr4d's closed grouped-GEMM kernel, which
we do not have. Agreed with the lead: no.

**The shared expert's silu (`triton_poi_fused_mul_silu_slice_0`, 52/step).** The
same fusion applies -- its `down_proj` quant could be handed forward through
`r4dhip._a8_slot` -- but it would NOT be bit-identical, and the probe says why:

```
shared-expert silu: inductor==eager False, mine==eager True, mine==inductor False
```

Inductor's fused `silu(x[..., :d]) * x[..., d:]` skips the intermediate round to
bf16 that both eager and the CUDA kernel do. Any hand-written kernel that matches
the documented semantics therefore differs from what the shared expert computes
today. It is a further -52/step for a real (small) numerics change, so it is the
lead's call, not a free win. Not implemented.

**Lead's decision (2026-09-04): NO for now.** The kernel stays available but off --
`silu_mul_quant(..., write_act=True)` already returns the bf16 activation alongside
`(q, s)`, so wiring the shared expert later is a call-site change only. Revisit only
if a numerics change is on the table for other reasons.

### Recipe

`MOUNTS_COMBO4.txt` = `MOUNTS_COMBO3.txt` + the new kernel file:

```
MOUNTS_FILE=$REPO/patches/MOUNTS_COMBO4.txt \
EXTRA_DOCKER_ARGS="-e VLLM_GDN_STRIDED_QKV=1 -e VLLM_FUSED_SHARED_GATE=1 -e VLLM_FUSED_SILU_QUANT=1" \
  $HOME/launch_q38fn_lru.sh 15.0
```

The launcher already mounts `hotcold/r4d_mxfp4_moe_lru.py` by default, so no
extra mount is needed for the wiring itself. Being bit-identical, this arm must
move ms/step with `spec_decode_num_accepted` FLAT; if acceptance moves, the
rounding drifted and the smoke assertion is the place to look.


## #12 QSA rope gather folded into the mrope kernel -- `VLLM_QSA_ROPE_GATHER=1` (2026-09-04, default 0)

`model_executor/layers/rotary_embedding/mrope.py` (kernel + wrapper),
`models/qwen4_exp/amd/indexer_qsa.py` (call site).

### What it replaces

`apply_qsa_rope` gathers the rope rows for this step and hands the result to the
Triton kernel:

```python
cos_cache, sin_cache = cos_sin_halves(rotary_emb, tensor)
cos, sin = cos_cache[positions], sin_cache[positions]   # 2 index kernels + 2 buffers
tensor, _ = triton_mrope(..., cos, sin, ...)
```

The kernel already computes a per-token row base (`t_cos = cos + pid * half_rd`).
With `positions=` it takes that base from `positions` [3, num_tokens] instead, so
`cos`/`sin` can be the whole caches and the two gathers disappear. `#4a` had turned
one gather plus two `.contiguous()` copies into two gathers; this removes what was
left of them.

### Counts (measured, `prof_w4head` rank 0, 84 decode steps)

| kernel | count | /step | what it is |
|---|---|---|---|
| `index_elementwise_kernel<..., OpaqueType<2>>` | 2718 | **32.36** | exactly 16 `apply_qsa_rope` calls/step (12 QSA layers + 4 MTP drafts) x 2 gathers |

Each one is 2.83 us of GPU time for a `[3, T, 32]` bf16 gather at T~5 -- launch
latency, not bandwidth. **-32 launches/step, -92 us/step**, and two fewer
allocations per call inside the graph pool.

### Bit-identical, proven on device

`k4/probe_rope_gather.py`, on GPU 1 in the image: 43 cases at the real QSA geometry
(head_dim 128, rotary_dim 64, mrope_section [11, 11, 10]) across T in
{1, 5, 17, 64, 2048}, n_qh in {4, 1}, both `is_neox_style` and both
`mrope_interleaved`, plus a transposed (non-contiguous) `positions` and an int32
one. Every case `torch.equal` on both q and k, and a control asserts the kernel
actually rotated (`changed=True`). The kernel loads the same bytes it used to load;
no arithmetic changed. Two of those cases are in `k4/smoke_imports.py`.

### Guards

- `VLLM_QSA_ROPE_GATHER` defaults to 0; without it the call site is the old two-gather path.
- `positions=` defaults to `None` in `triton_mrope`, so every existing caller
  (including `MRotaryEmbedding.forward_cuda`) is bit-for-bit unchanged.
- The indexer never hard-depends on the new keyword: `_rope_gather_supported()`
  inspects `triton_mrope`'s signature once and warns, keeping the stock path, if an
  unpatched `mrope.py` is mounted.
- Only the `positions.ndim == 2` (mrope) branch folds; the 1-D branch feeds
  `apply_rotary_emb`, which needs materialised cos/sin.
- `normalize_compressed_keys` hands over a transposed `first_rope_positions`, so the
  wrapper calls `.contiguous()` on `positions` -- the kernel's pointer arithmetic
  assumes row-major [3, num_tokens]. Covered by the transposed probe cases.

### Recipe

No new mount: both files are already in `MOUNTS_COMBO4.txt`.

```
EXTRA_DOCKER_ARGS="... -e VLLM_QSA_ROPE_GATHER=1"
```

Bit-identical, so ms/step must fall with `spec_decode_num_accepted` FLAT.
