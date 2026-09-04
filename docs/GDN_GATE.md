# GDN gate — CLAV_GDN=0 on Qwen4Exp (gfx1201)

Investigated read-only against `local/q38fn-rocm10:try1` (no GPU used). The env var that
actually gates the native GDN path is **`CLAV_GDN`** (legacy alias `NO_AMD_GDN_HIP`, inverted
sense) — this is a real, currently-live vLLM `general_plugin`, installed as a separate wheel
outside the `fork_vllm` tree, so grepping only `fork_vllm` (as `models/qwen4_exp/config.py`
does) undersells how much is here.

## 1. What the HIP path is, and where it's registered

Package `/usr/local/lib/python3.12/dist-packages/gdn_hip/` (pip dist-info: `gdn_hip 0.1.0`),
registered as a vLLM `vllm.general_plugins` entry point
`gdn_hip = gdn_hip.register:register` (confirmed via
`importlib.metadata.entry_points(group="vllm.general_plugins")` inside the image — it lists
`gdn_hip`, `clav_attn`, `rfi_hip`, `radiance_dynmtp`, plus the two upstream LoRA resolvers and
`quark_online_quant`).

- **`register.py:54-124`**: `register()` runs in every worker process at plugin-load time
  (`load_general_plugins()`, before `init_device()`). Gate logic (register.py:67-69):
  ```python
  if (os.environ.get("CLAV_GDN", "1") != "1"
          or os.environ.get("NO_AMD_GDN_HIP", "0") == "1"
          or os.environ.get("DISABLE_ALL_CLAV", "0") == "1"):
      ...  # logs "[clav_gdn] disabled by CLAV_GDN=0 ... using the fla-Triton GDN path."
      return False
  ```
  On by default (`CLAV_GDN` defaults to `"1"`) for gfx12x; `Qwen4ExpConfig.__init__` forces
  `CLAV_GDN=0` into `os.environ` before the engine/workers spawn
  (`config.py:207-229`, `_force_qwen4_exp_env`/`_QWEN4_EXP_REQUIRED_ENV`), and children inherit
  it — so every worker's `register()` call takes the disabled branch and the plugin never even
  probes gfx12x or imports its compiled extension.
- If not disabled, `register()` checks `current_platform.is_rocm() and on_gfx12x()`
  (register.py:87-95), then imports `gdn_hip.vllm_oot`, which runs
  `@PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")` on `QwenGdnHipAttention`
  (`vllm_oot.py:67-68`) — vLLM's `PluggableLayer.__new__` looks the in-tree class name up in
  `op_registry_oot` and instantiates this subclass instead, zero edits to vLLM source
  (`vllm_oot.py:1-6`).
- **Kernels**: `gdn_hip/op.py` loads `gdn_hip_C*.so` (1 MB compiled extension,
  `/usr/local/lib/python3.12/dist-packages/gdn_hip/gdn_hip_C.cpython-312-x86_64-linux-gnu.so`)
  and registers 4 custom ops: `gdn_hip::{gdn_prefill_r, gdn_prefill_r2, gdn_decode_r,
  gdn_verify_r}` (op.py:22-53). Symbol names in the .so (`nm`/`strings`) show the actual device
  kernels underneath: `gdn_r_h_kernel`, `gdn_r_h_ds_kernel`, `gdn_r_uw_kernel`, `gdn_r_o_kernel`,
  `gdn_r_prep_kernel`, `gdn_r_prep2_kernel`, `gdn_decode_r_kernel`, templated over
  `{fp16,bf16,fp32}` and a chunk constant (`Li32`). This is a **different, newer** package from
  the `r4d_gdn_*` kernels mentioned in `r4d_lib.py`/`libr4d` — those live in `/app/r4dhip/r4d.so`
  and are exercised by an older, separate wrapper (`radiance_gdn.py`, see §2); `gdn_hip_C.so` is
  self-contained and does not appear to call into `r4d.so`.
- `forward_core.py` (§`forward_core_gdn_hip`) dispatches per batch shape onto one of the 4 ops
  directly from **raw pre-conv `mixed_qkv`** — conv, split, prep and the recurrence are fused
  into a single kernel call each for decode/verify, and prep is folded into `prefill_r2` (no
  separate conv kernel, no rearrange/cat — see `op.py:1-25` docstring). This is what to look for
  in a kernel-name profile: **`gdn_decode_r_kernel`** for decode, **`gdn_r_h_kernel` /
  `gdn_r_h_ds_kernel`** (chunked recurrence, DS = "dim-state" conv layout) and **`gdn_r_prep_kernel`
  / `gdn_r_prep2_kernel`** for prefill.

`CLAV_GDN` is a coarse **plugin-registration** kill switch (checked once, at process start,
before any per-layer/per-request logic runs). It is *not* the same axis as the
per-layer/per-batch fallback described next.

## 2. Kernel geometry vs. Qwen4Exp's GDN shape — dims match; layout may not

`gdn_hip/vllm_oot.py:56-58`:
```python
def _dims_supported(layer) -> bool:
    return (getattr(layer, "head_k_dim", 0) == 128
            and getattr(layer, "head_v_dim", 0) == 128)
```
This is the **only** shape gate, checked per-layer, per-forward, independent of `CLAV_GDN`
(vllm_oot.py:97-104): if it fails, that layer silently falls back to the fla-Triton
`super()._forward_core(...)` with a `logger.warning_once`. The comment at `op.py:1-16` says the
"r-family" (the only compiled kernels since a 2026-08-08 deprecation) covers **head dims 128/128
exactly**; older "wmma/chunked/recurrent/conv" kernels for other dims were retired.

Qwen4Exp's `linear_attn` config: `linear_key_head_dim=128`, `linear_value_head_dim=128` —
**this matches the kernel envelope exactly.** So the documented, per-layer safety net (dims
gate → automatic Triton fallback, no crash) would already have caught a head-dim mismatch on its
own; forcing the whole plugin off at the `CLAV_GDN` level is a stronger, coarser measure than
that gate alone would require. That's a real signal that whatever crashes is either (a) something
the per-layer dims check doesn't cover, or (b) something in `register()`'s own import/registration
path (before any per-layer logic runs at all).

One asymmetry worth flagging but **not confirmed as the cause**: Qwen4Exp's GDN has
`linear_num_key_heads=16` vs `linear_num_value_heads=48` (a 3x ratio), and
`fork_vllm/models/qwen4_exp/amd/model.py:208-211` instantiates
`QwenGatedDeltaNetAttention(..., gqa_interleaved_layout=False)` — the **Qwen3.5-style
"non-interleaved"** in_proj weight layout (`in_proj_b`/`in_proj_a` as two separate weights,
`output_sizes=[num_v_heads]*2`), as opposed to Qwen3-Next's interleaved
`[b_g0,a_g0,b_g1,a_g1,...]` single fused weight (`qwen_gdn_linear_attn.py:568-582`). The 3x K:V
head ratio itself is handled generically by the base Python layer regardless of interleaving
(`fix_query_key_value_ordering`, `qwen_gdn_linear_attn.py:614-643`, using
`num_v_heads // num_k_heads`), and mixed_qkv is fully split/reshaped in Python before it would
reach `gdn_hip`'s ops — so this is not an obvious crash vector, just a genuinely different weight
memory layout than whatever checkpoints (presumably Qwen3-Next, interleaved) the `gdn_hip_C`
prep/decode kernels were primarily validated against. There is a documented TP≥2 correctness
special-case for exactly this non-interleaved layout
(`maybe_disable_tp`, `qwen_gdn_linear_attn.py:585-600`, citing
`github.com/vllm-project/vllm/issues/35924`) — but it is gated `current_platform.is_cuda()`,
so it does **not** engage on this ROCm target; ruled out as-is, but it shows the non-interleaved
layout has a known history of TP-related edge cases elsewhere.

## 3. Why it crashes — no commit message found; one strong piece of first-party evidence

No commit history is available for this reasoning: `fork_vllm` on `big` is not a git checkout,
and the codeberg `vllm-radiance` clone (shallow, 4 commits) has no `qwen4`/`Qwen4Exp`-specific
commit and predates the `gdn_hip` package split (its `radiance_gdn.py` wraps `r4d_gdn_*`
directly, a different/older code path — see §1). `config.py`'s own comment (line 200-206) only
says the combination "does not merely serve slower, it goes down" without naming the mechanism.

The strongest first-party evidence found is inside `gdn_hip_C.so` itself (`strings`):
```
[gdn_hip] FATAL: gdn_r state out of ssm-cache range: max|S|=%g%s (head=%d, limit=%g). The state diverged; aborting.
```
This is a **hard, intentional process abort** (not a Python exception, not a segfault) guarding
against the recurrent SSM state's magnitude exceeding a sanity bound — i.e. a **numerics
divergence detector**, not a shape/dimension check (those are all in Python, in
`_dims_supported`, and fail soft). `register.py`'s own HISTORY comment (lines 16-23) separately
references "the production NaN trap" as the reason the fp16 ssm-cache dtype stays rejected —
i.e. there is a known prior incident class of GDN state numerics going bad in production on this
kernel family, independent of Qwen4Exp.

## 4. What the fla-Triton path does instead (for profiling)

`vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.QwenGatedDeltaNetAttention` base
`_forward_core` (not read in full here, but architecturally): a short-conv1d over `mixed_qkv`,
then Triton chunked kernels from `third_party/flash_linear_attention/ops/chunk.py` (prefill —
chunked gated-delta-rule scan, typically split into intra-/inter-chunk kernels plus a gated
RMSNorm) and `ops/fused_recurrent.py` (decode — a fused recurrent-update kernel per step). These
are the "~29 fla-Triton chunk kernels" referenced in `vllm_oot.py:76-83`'s warmup-skip comment,
and are what a profile shows in place of `gdn_decode_r_kernel`/`gdn_r_h_kernel` when
`CLAV_GDN=0` is in effect. Being Triton/JIT, first-call latency includes autotune/compile, unlike
the AOT `gdn_hip_C` kernels.

## 5. Assessment: is `CLAV_GDN=1` for Qwen4Exp shape-unsafe, numerics-risky, or plausibly safe?

**Not a shape mismatch** — `_dims_supported` (128/128) matches Qwen4Exp's GDN config exactly, and
that gate already fails soft (per-layer Triton fallback, no crash) if it didn't match. **Most
likely a numerics risk**, evidenced by the `[gdn_hip] FATAL: ... state diverged; aborting.` guard
compiled into the kernel and the package's own history of an SSM-state NaN issue in production.
This can't be stated as *proven* for Qwen4Exp specifically — no commit message or issue ties the
two together, and the non-interleaved GDN weight layout (§2) is a plausible-but-unconfirmed
secondary contributor (a layout the r-family kernels may not have been built/tested against, on
top of any pure numerics issue). Recommend not flipping `CLAV_GDN=1` in production without first
reproducing the crash on an isolated CPU-adjacent/short run, and — if pursued — instrumenting
`gdn_hip`'s state-check FATAL path (or an intercepted `LD_PRELOAD`/patched abort) to capture
`max|S|`/head/limit at the point of divergence rather than losing the process outright.

## Files referenced
- `/usr/local/lib/python3.12/dist-packages/gdn_hip/{register.py,vllm_oot.py,forward_core.py,op.py}`
  (in-image; not present in the `fork_vllm` host copy since it's a separate installed package)
- `$REPO/fork_vllm/models/qwen4_exp/config.py:200-229`
- `$REPO/fork_vllm/models/qwen4_exp/amd/model.py:206-212`
- `$REPO/fork_vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:356-380,560-650`
- `$REPO/fork_vllm/model_executor/kernels/r4d_lib.py` (the older r4d-based GDN path, for contrast)
- `$REPO/k2/vllm-radiance/radiance_gdn.py`, `patch_gdn_wmma.py`, `patch_gdn_metadata.py`
  (shallow clone of `codeberg.org/StillDeadcode/vllm-radiance`, predecessor code, kept for reference)

## Uncertainty flags
- No commit/issue text directly explains the Qwen4Exp crash; §3's numerics-divergence read is
  inference from the compiled FATAL string plus the package's stated NaN history, not a
  confirmed root cause.
- The non-interleaved-layout hypothesis in §2 is explicitly flagged unconfirmed and the one
  concrete related code path found (`maybe_disable_tp`) is CUDA-only and does not apply here.
