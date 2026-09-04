# FP8HIP knobs — Fp8HipBlockScaledMMKernel (gfx1201)

Investigated read-only against `local/q38fn-rocm10:try1` (`/app/fp8hip/libfp8hip_gemm.so`,
61304 bytes, built with ROCm 7.2.3 clang, 27-Aug). No source tree for `fp8hip/csrc` was found
on `big` or in either target image (only referenced by comment) — the env-var and dispatch
semantics below come from disassembling the .so with `objdump`/`nm`/`strings` (no GPU used).

## 1. Python wrapper and call site

`/app/vllm/vllm/model_executor/kernels/linear/scaled_mm/fp8hip.py` (host copy:
`$REPO/fork_vllm/model_executor/kernels/linear/scaled_mm/fp8hip.py`).

- `_lib()` (line 33) loads `$FP8HIP_LIB` (default `/app/fp8hip/libfp8hip_gemm.so`) via `ctypes.CDLL`
  and binds one symbol: `fp8hip_gemm_w8a8_launch`, `restype=c_int`,
  `argtypes = [c_void_p]*5 + [c_int]*3 + [c_void_p]` (fp8hip.py:34-39).
- `_fp8hip_block_scaled_mm_func` (fp8hip.py:52-71) calls it as
  `launch(qx_ptr, weight_shuf_ptr, x_scale_ptr, w_scale_ptr, y_ptr, M, N, K, stream_ptr)`,
  `y = empty(M, N, bf16)`. Non-zero `rc` raises `RuntimeError` in Python.
- `Fp8HipBlockScaledMMKernel.can_implement` (fp8hip.py:126-139) gates entry: weight scale
  group must be `(128,128)`, `K % 128 == 0`, `N % 16 == 0`, `out_dtype == bf16`. Weights are
  pre-shuffled once at load into WMMA fragment order (`shuffle_weight_gfx1201`, fp8hip.py:44-51).
- Registered above the Triton `RDNA4Fp8BlockScaledMMKernel` in the ROCm FP8-block kernel
  priority list; `is_supported()` requires gfx1200/1201 + the .so present, and honors
  `VLLM_DISABLE_FP8HIP=1` / `DISABLE_ALL_CLAV=1` to fall back to Triton (fp8hip.py:99-116).

Only `fp8hip_gemm_w8a8_launch` is called from Python. `fp8hip_gemm_w8a8_fallback` (a second,
more general kernel) and the three `FP8HIP_*` env knobs are consumed **entirely inside the
.so**, not from Python — confirmed below.

## 2. Decode shapes (Qwen3.8-Flash-Next / `q38fn-heretic2-mxfp4-fp8`, TP=2, 2x R9700 gfx1201)

From `/mnt/llm-storage/q38fn-heretic2-mxfp4-fp8/config.json` (`text_config`): `hidden_size=2560`,
`num_attention_heads=24`, `head_dim=256`, `num_key_value_heads=2`; GDN
(`linear_attn`): `linear_num_key_heads=16`, `linear_key_head_dim=128`,
`linear_num_value_heads=48`, `linear_value_head_dim=128`. MoE experts (`num_experts=512`,
`moe_intermediate_size=640`) are mxfp4, a different kernel (`r4dhip`/`r4d.so`) — out of scope.
The fp8 block-scaled linears are the dense attention + GDN projections and (likely)
`shared_expert`/`lm_head`.

Per-rank (TP=2) `(M, N, K)` for the fp8hip GEMM, decode `M` = batch size (task scope: `M ≤ 16`):

| linear | K (in) | N (out, per rank) | N % 128 |
|---|---|---|---|
| qkv_proj (fused) | 2560 | (24·256 + 2·256 + 2·256)/2 = 3584 | 0 |
| o_proj | 3072 (24·256/2) | 2560 (full, all-reduced) | 0 |
| linear_attn in_proj (qkvz+ba) | 2560 | 16384/2 = 8192 | 0 |
| linear_attn out_proj | 3072 (48·128/2) | 2560 (full) | 0 |

(`in_proj` width per k-head-group = `head_k_dim*2 + (head_v_dim*2)*num_v_heads/num_k_heads`
= `128*2 + 256*3` = 1024, ×16 k-heads = 16384 before TP; see
`qwen_gdn_linear_attn.py:623-629`.)

**Every N above is a multiple of 128.** Given the fallback rule in §3, this model's fp8hip
calls should always hit the tiled WMMA kernels, never `fp8hip_gemm_w8a8_fallback`, regardless
of env settings (barring a future non-128-aligned TP split).

## 3. `FP8HIP_DBL` / `FP8HIP_GEOM` / `FP8HIP_GROUP_M` — read inside the .so

All three are read **once per process**, in `fp8hip_gemm_w8a8_launch` itself, via a shared
helper at file offset `0xec60`: `v = getenv(name); return v ? strtol(v, NULL, 10) : default;`
(`nm -D` confirms `getenv@GLIBC_2.2.5` / `strtol@GLIBC_2.2.5` are undefined symbols pulled in
only by this function). Each is then cached in a C++ function-local `static`, guarded by
`__cxa_guard_acquire/release` (classic lazy-init) — **so a value change requires a fresh
process**, not just a fresh call. **All three are plain base-10 integers** (`strtol(..., base
10)`) — none use an `"AxB"`-style string format despite `GEOM`'s name suggesting a geometry
string.

Call sites and defaults (objdump `--start-address=0xe880`, function
`fp8hip_gemm_w8a8_launch`; string addresses confirmed via `objdump -s -j .rodata
--start-address=0xea0 --stop-address=0x1030`):

| env var | string @ | default (unset) | semantics (from the disasm) |
|---|---|---|---|
| `FP8HIP_GEOM` | `0xf51` | `-1` (auto) | Selects the tile-geometry/template-family index: `1` = small-M family (`fp8hip_gemm_w8a8_tiled<1,2,4,...>`), `2` = large-M family (`<2,2,4,...>`). If unset (`-1`), the class is chosen from `M`: `M<97 → 1`, `M≥97 → 2` (`cmp r10d,0x61` at `e91f`). If set to `0` (or the class otherwise resolves to 0), the launcher takes the **fallback kernel** path instead (see below). |
| `FP8HIP_DBL` | `0xef3` | `-1` (auto) | Boolean-ish override for the second (independent) template bool axis — the four compiled instantiations are `<Li1,Li2,Li4,Lb0,Lb0>`, `<1,2,4,0,1>`, `<2,2,4,1,0>`, `<2,2,4,1,1>`, i.e. GEOM-class and this flag are orthogonal. If unset, the axis defaults from `M`: `M<189 → one variant, M≥189 → the other` (`cmp r10d,0xbd` at `e944`). If explicitly set to a value `≥0`, that (nonzero-vs-zero) directly overrides the axis regardless of `M`. Exact physical meaning of the two variants (e.g. which is genuinely "double-buffered") is not recoverable from disassembly alone — worth an A/B to learn empirically which raw value is faster for decode `M`. |
| `FP8HIP_GROUP_M` | `0x101d` | `0` (auto→16) | Raw integer fed straight into the kernel launch as a supergrouping/L2-locality tile-group size (classic CUTLASS/Triton "group_m" trick over the `ceil(N/128)`-wide tile grid). If `>0`, used as-is; otherwise the launcher defaults to `16` (`mov r10d,0x10; test r12d,r12d; cmovg r10d,r12d` at `e9c8-e9d1`). |

**Hard gate (before any env is read):** if `K % 128 != 0` or `N % 16 != 0`, `launch()` returns
`-1` immediately (`e8a3-e8b9`) — this can't actually fire in practice since Python's
`can_implement` already enforces it.

**Fallback vs. tiled dispatch** (`e928-e939`): the tiled WMMA path (the 4 kernel
instantiations, selected by `FP8HIP_GEOM`-class and the `FP8HIP_DBL`-axis) is taken only when
`(GEOM-class != 0) AND (N % 128 == 0)`. Otherwise `fp8hip_gemm_w8a8_fallback` runs instead,
with a `ceil(N/64) x ceil(M/32)` grid (`ea19-ea51`) — i.e. **`FP8HIP_GEOM=0` is a de facto "use
the fallback kernel" switch**, and any GEMM with `N` not a multiple of 128 is silently routed
there regardless of env settings. Per §2, that never happens for this model's shapes.

## 4. Sweep plan (decode, `M ≤ 16`, ≤ 10 combos)

Given `M ≤ 16` is far below both auto-heuristic boundaries (`97` for GEOM-class, `189` for the
DBL axis), the *default* (all unset) already resolves to GEOM-class `1` and the "small-M" DBL
variant — so most of the sweep value is in (a) confirming the default is actually fastest and
(b) exploring `GROUP_M`, which the M-heuristics don't touch at all.

1. **baseline** — all three unset (production default).
2. `VLLM_DISABLE_FP8HIP=1` — control: Triton competitor, bounds the fp8hip win/loss.
3. `FP8HIP_GEOM=1` — explicit; should be a no-op vs (1), sanity-checks the M<97 default.
4. `FP8HIP_GEOM=2` — force the "prefill" family on decode-shaped `M`; tests whether the `M=97`
   cutoff is well-placed or too conservative for this model's N's.
5. `FP8HIP_DBL=0`
6. `FP8HIP_DBL=1` — (5) vs (6) vs (1) identifies which raw value the default resolves to for
   `M<189`, and whether the heuristic already picked the faster one.
7. `FP8HIP_GROUP_M=1` — no supergrouping.
8. `FP8HIP_GROUP_M=8`
9. `FP8HIP_GROUP_M=32` — larger than the default 16, worth trying since per-rank `N` up to 8192
   (64 tiles of 128) leaves room for bigger groups.
10. best-of(5/6) × best-of(7-9) combined.

Each combo needs a fresh server process (values are cached for the process lifetime via the
`__cxa_guard`-protected static locals — a mid-run env change would not take effect).

## Uncertainty flags

- No `fp8hip/csrc` source was found on `big` or in either image; all of §3 is reverse-engineered
  from x86-64 disassembly of the host-side launcher. The device-side WMMA kernel bodies
  (GCN/RDNA assembly) were not disassembled — only the host dispatch logic that selects among
  them.
- Which physical `FP8HIP_DBL` value (`0` vs nonzero) is literally "double-buffered" vs. the
  other tuning axis is not determinable from the host-side disassembly (that lives in the
  device kernel body); treat the two settings as exploratory rather than semantically labeled.
