# k3 follow-up: can the 1.33 GB/step of bf16 hyper-connection weights be quantized?

Environment: R9700 (gfx1201), `local/q38fn-rocm10:try1`, every GPU run under
`flock $REPO/gpu.lock`, one GPU (`cuda:0` of `HIP_VISIBLE_DEVICES=1,2`).
Cold-pool timing as in RESULTS.md (a >192 MB pool of distinct weight copies, one
graphed call per copy, so the number is DRAM bandwidth and not the 64 MB MALL).

## Verdict

**Kernel side: yes — but not with the kernel the fork would pick.** `wvSplitK_int4_g`
is 1.9-2.1x faster than bf16 `wvSplitK` at n<=3 and *illegal* at n=4 and n=5 for the
K=10240 shapes, where production actually lives (MTP-4 verify = 5 rows). At n=5 the
W4 path as currently wired is a **30% regression** (4693 vs 3618 us/step), because the
gate in `rdna_hybrid_w4a16.py` sends it to the Triton kernel at 32 us/call. The kernel
that does work is `r4d_gemm_w4a16_nt_m64` in `r4d.so`: **5.79 us flat for n=1..5** on
the 336x10240 shape, 2.5-3.0x the bf16 kernel, no LDS cliff. It cannot take the
K=320 up-projection (K must divide by 128).

**Quality side: no, I would not trust W4 here.** int4 group-128 asymmetric costs
**13-16% relative output error per tensor**; even group-32 costs 9-11%. These weights
produce the gate that mixes the four residual streams, ~100 times over 48 layers.
fp8 costs **2.5-2.6%** and, crucially, per-**tensor** fp8 costs the same as per-channel
(2.65% vs 2.63%) — so the shipped per-tensor `wvSplitKQ` W8A8 kernel is not a quality
compromise on the weight side. int8 per-tensor is *not* an option (6.8-10.6%).

**Recommendation: fp8, not int4.** It is 1.7-1.9x on the same shapes (2148 vs 3515
us/step at n=4), it needs no new packer, and it is ~4x cheaper in error. The one gap
is that `wvSplitKQ` refuses n=5, so an n=5 path has to be found before it can ship
against an MTP-4 config.

**Free win, independent of any quantization:** `r4d_gemm_bf16_nt_m64` takes these
shapes with a *plain* [N,K] bf16 weight — no repacking, correctness verified here —
and is flat at 14.4 us where `wvSplitK` degrades to 17.2 us at n=5. That is
**356 us/step (-9.8%) for a dispatch change and nothing else.**

---

## 1. Kernel side

### 1.1 What is legal

| kernel | needs | 336x10240 | 320x10240 | 10240x320 | 512x2560 |
|---|---|---|---|---|---|
| `wvSplitK_int4_g` | K%16, K%group, **K*n <= 39321** | n<=3 only | n<=3 only | all n (group<=64) | all n |
| `r4d_gemm_w4a16_nt_m64` | M<=64, N%16, **K%(SK*128)** | OK | OK | **illegal** (K=320) | OK |
| `r4d_gemm_bf16_nt_m64` | M<=64, K%(SK*16) | OK | OK | OK | OK |
| `wvSplitKQ` (fp8 W8A8) | K%16, **n<=4** | n<=4 | n<=4 | n<=4 | n<=4 |

Two hard limits decide everything:

- **`wvSplitK_int4_g` stages the whole activation in LDS** and checks
  `K*N <= get_lds_size_int4()/2*1.2 = 39321` (`skinny_gemms_int4.cu:681`). At K=10240
  that is n<=3. There is no `_big_` variant (the bf16 kernel has one). The Python
  gate in `rdna_hybrid_w4a16.py` (`K*M <= 32768`) is even tighter and silently routes
  n>=4 to Triton instead of raising.
- **r4d's W4 group is fixed at 128** (`r4d_gemm_w4a16_nt_m64_group()`), and K must
  divide by `SK*128`. K=320 fails for every SK, so `input_mix_weight_up` has no r4d
  W4 kernel. (`r4d_gemm_mxfp4a8_nt_m64` needs only K%32 and would accept
  10240x320 — it is the kernel the MoE already runs at K=320 — but it wants fp8
  activations and I did not benchmark it.)
- There is **no W8A8 GEMM in r4d at all** (the 8-bit-ish entries are `w4a8` and
  `mxfp4a8`, both 4-bit weights). The only 8-bit skinny GEMM in the tree is
  vLLM's `wvSplitKQ`, and it is per-tensor-scale fp8 on *both* operands.

### 1.2 Cold per-call us, 336x10240 (the 98-call shape)

| n | bf16 wvSplitK cu32 | bf16 r4d | int4_g128 cu32 | triton w4a16 | fp8 wvSplitKQ | r4d w4a16 |
|---|---|---|---|---|---|---|
| 1 | 14.70 | 14.36 | 7.29 | 32.24 | 8.74 | **5.79** |
| 2 | 14.96 | 14.35 | 9.03 | 32.31 | 8.93 | **5.77** |
| 3 | 15.72 | 14.39 | 9.58 | 31.75 | 9.36 | **5.80** |
| 4 | 16.86 | 14.42 | *illegal* | 31.67 | 9.74 | **5.78** |
| 5 | 17.18 | 14.45 | *illegal* | 32.03 | *n>4 unsupported* | **5.79** |

Bandwidth tells the same story: bf16 `wvSplitK` runs at 400-480 GB/s, `wvSplitK_int4_g`
only reaches 190-270 GB/s (so 4x fewer bytes buys ~2x, not ~4x), and r4d w4a16 reaches
316 GB/s. Nothing here is near the ~640 GB/s DRAM roofline except the bf16 kernels.

### 1.3 Projected decode-step cost (249 calls: 98 + 2 + 100 + 49)

Per-call best-in-family x call count, us/step:

| family | n=1 | n=3 | n=4 | n=5 |
|---|---|---|---|---|
| bf16 `wvSplitK` cu32 (today) | 3470 | 3646 | 3515 | **3618** |
| bf16 `r4d_gemm_bf16_nt_m64` | 3224 | 3241 | 3253 | **3262** |
| W4, vLLM int4 path as wired | 1609 | 2095 | 4528 | **4693** |
| W4, best kernel per shape | 1450 | 1729 | 1895 | **2011** |
| fp8 W8A8 `wvSplitKQ` | 1947 | 2055 | 2148 | *n/a* |

At the production point (n=5, MTP-4):

- doing nothing but switching bf16 dispatch to r4d: **-356 us/step (-9.8%)**
- fp8 W8A8, if an n=5 path exists: **~-1.4 ms/step (-39%)**, extrapolating from n=4
- W4 with r4d for K=10240/2560 and vLLM int4-g64 for the up: **-1.61 ms/step (-44%)**
- W4 as the fork would actually dispatch it: **+1.08 ms/step (+30%), a regression**

A decode step at gb14 is ~21 ms (47.6 tok/s), so -1.4 ms is roughly **+7%** end-to-end,
plus a second-order win: the HC weights are `disable_tp=True`, i.e. **replicated**, so
fp8 frees ~0.67 GB and W4 ~0.98 GB *per GPU* — which only converts into throughput if
`--cpu-offload-gb` is lowered by the same amount afterwards.

### 1.4 Correctness

Every `wvSplitK_int4_g` number above was checked against a dequantized fp32 reference:
rel err 2.3e-3 - 2.7e-3 (bf16 accumulation noise, same order as the bf16 kernel's
1.6e-3). Triton path 3.2e-3 - 3.7e-3. `r4d_gemm_bf16_nt_m64` 1.6e-3 against `F.linear`.

**`r4d_gemm_w4a16_nt_m64` timings are on random bytes.** Its weight must be
pre-permuted into WMMA fragment order by `radiance_w4.py`, which is **not in
`$REPO/libr4d/`**. Byte counts, sizes and access pattern are exact so the
timing is real, but no value was verified. The layout is fully specified in the
kernel's header comment (`idx = lane%16, k = 8*(e>>2)+4*(lane>>4)+(e&3)`; scale in the
low half of a dword, f16 of `-(1024+zero)` in the high half), so a packer is ~50 lines
and self-checkable against a dequantized reference.

---

## 2. Quality side

All 298 real tensors (100 `input_mix_weight_down`, 100 `..._up`, 98
`block_inject_weight`) from `/mnt/llm-storage/q38fn-heretic2-mxfp4-fp8`, quantized on
CPU, error measured as `||Wq x - W x|| / ||W x||` with 256 random activations. For the
K=10240 inputs `x` is scaled per channel by that module's own `hc_norm.weight`, which
is exactly the scale the layer sees (the input is `grouped_gemma_rmsnorm(...)`); for
the K=320 input to the up-projection there is no cheap proxy and `x` is iid.

Mean over tensors (min-max in `hc_quant2.log`):

| scheme | `..._down` (320x10240) | `..._up` (10240x320) | `block_inject` (4x10240) |
|---|---|---|---|
| int4 g128 asym | 0.1343 | *K not divisible* | 0.1615 |
| int4 g64 asym | 0.1158 | 0.1094 | 0.1374 |
| int4 g32 asym | 0.0980 | 0.0913 | 0.1143 |
| int4 g128 sym (uint4b8) | 0.1666 | — | 0.2035 |
| **int8 per-channel** | **0.0207** | **0.0113** | **0.0360** |
| int8 per-tensor | 0.0678 | 0.1063 | 0.0545 |
| **fp8 e4m3 per-channel** | **0.0263** | **0.0254** | **0.0262** |
| **fp8 e4m3 per-tensor** | **0.0265** | **0.0265** | 0.0262 |
| fp8 block 128x128 | 0.0262 | 0.0263 | 0.0254 |

Weight distribution (mean over tensors):

| kind | kurtosis | rms | absmax | max/rms per 128-group (mean / p99) | frac \|w\|>4rms |
|---|---|---|---|---|---|
| `input_mix_weight_down` | 10.2 | 0.052 | 1.39 | 3.43 / 6.23 | 0.39% |
| `input_mix_weight_up` | **30.9** | 0.090 | 4.29 | 3.40 / 6.09 | 0.61% |
| `block_inject_weight` | **26.0** | 0.027 | 0.44 | 3.88 / 7.52 | 0.71% |

Reading of these numbers:

- The int4 error is **not** an outlier problem that a smaller group fixes — the
  per-group dynamic range is a benign 3.4, and the 13% is simply the uniform-grid noise
  floor of 16 levels over +-3.4 sigma (step/sqrt(12) ~= 0.13 rms). Going g128 -> g32
  quadruples the scale traffic and only buys 13% -> 10%. **4 bits is structurally
  short here**, and no group size rescues it.
- Kurtosis 26-31 on `_up` and `block_inject` is why **per-tensor int8 collapses**
  (6.8-10.6%) while per-tensor **fp8 does not** (2.6%): fp8's error is relative
  precision (3 mantissa bits), so it is scale-invariant and indifferent to outliers.
  This is the single most useful result in this section, because per-tensor is exactly
  what the shipped `wvSplitKQ` kernel takes.
- Caveat that bounds all of the above: with iid `x` this metric is essentially
  `||dW||_F / ||W||_F`. Real activations are correlated and the true impact can be
  either side of it. And the W8A8 kernel also quantizes the **activation** to
  per-tensor fp8, which is additional error I did not measure.

**Would I trust W4 here? No.** These are not a wide FFN where 4-bit noise averages out
over thousands of accumulations: `down` is a rank-320 projection whose output goes
through silu and a second projection to produce a 4-way gate over the residual streams,
applied twice per layer for 48 layers. A 13% perturbation of that gate is a change to
how the model routes its residual, ~100 times in series. W8 (fp8) at 2.6% is the
defensible choice, and even that should be gated on a PPL + needle check, not on this
metric.

---

## 3. Integration side

How these linears are built (`vllm/models/qwen4_exp/amd/hyperconnection.py:88-130`):

```
pad_size = (-(lora_rank + hc_count)) % 16          # = 12
use_combine=True : MergedColumnParallelLinear(10240, [320, 4, 12],
                       bias=False, quant_config=None, disable_tp=True)   -> M=336
use_combine=False: ReplicatedLinear(10240, 320, quant_config=None)       -> M=320
always           : ReplicatedLinear(320, 10240, quant_config=None)       -> up
```

Findings:

1. **`quant_config=None` is hardcoded** in all three constructors, and `GatedResidual`
   is built from a `HyperConnectionConfig` that carries no quant_config at all
   (`model.py:262,266,433`). So removing the `ignore` patterns from the checkpoint
   would do **nothing** on its own — the modules would still build an
   `UnquantizedLinearMethod`. Plumbing a quant_config into `HyperConnectionConfig`
   and through to the three constructors is the first change.
2. **The `ignore` list has three patterns to drop**: `re:.*hyper_connection.*`,
   `re:.*input_mix.*`, `re:.*block_inject.*` (checkpoint
   `quantization_config`, format `mxfp4-pack-quantized`, 19 entries).
3. **The merged-336 padding survives quantization cleanly.** `packed_modules_mapping`
   already names the third shard `_input_mix_padding` (`model.py:600`), and
   `_QWEN4_EXP_IGNORED_MISSING_SUFFIXES` already tolerates missing `_weight_scale` /
   `_input_scale`, so a quantized method will create 12 rows of weight+scale that no
   checkpoint tensor fills; the forward `split()` discards those columns, so garbage
   there is harmless *provided* the loader's missing-parameter check keeps tolerating
   them. In fact the padding **helps**: it is what makes M=336 a multiple of 16, which
   is exactly r4d's N constraint.
4. **Checkpoint surgery is the same shape as `fp8_surgery.py`** that produced this
   checkpoint: for fp8 it is one `weight_scale` (per-tensor) or `weight_scale_inv`
   (block) per tensor, and dropping the three ignore patterns. For W4 it is
   `weight_packed` + `weight_scale` + `weight_zero_point` in
   compressed-tensors layout, plus `config_groups` entry.
5. **Two shape hazards for a W4 checkpoint**: `input_mix_weight_up` has K=320, so
   `group_size=128` is illegal (`K % group_size` check in `can_implement`) — that
   tensor must be group 64 or 32, i.e. the checkpoint needs *two* group sizes, which
   compressed-tensors expresses as two `config_groups`. And `block_inject_weight` is
   4x10240; after merging it is rows 320..323 of a 336-row tensor, so it is only ever
   quantized as part of the merged tensor, never on its own.
6. **The n>=4 cliff has to be solved in the kernel**, not the checkpoint. Options, in
   order of cost: (a) call `r4d_gemm_w4a16_nt_m64` (needs the packer written); (b) add
   a `_big_` variant to `skinny_gemms_int4.cu` copying the bf16 kernel's K-splitting;
   (c) cheap hack — split the K=10240 GEMM into two K=5120 calls and add
   (5120*5 = 25600 fits LDS), at the cost of one extra ~4 us launch per call.

### Effort estimate (no sources were modified)

| step | effort |
|---|---|
| fp8 (recommended): extend `fp8_surgery.py` to the 3 HC patterns, drop the 3 ignore entries | 0.5 d |
| plumb `quant_config` into `HyperConnectionConfig` + `GatedResidual` | 0.5 d |
| verify the `_input_mix_padding` shard loads with a quant method (likely 1-2 loader tweaks) | 0.5-1 d |
| find/build an n=5 fp8 path (`wvSplitKQ` caps at 4; measure `torch._scaled_mm` first) | 0.5-1 d |
| PPL + needle validation at 256K | 0.5 d |
| **fp8 total** | **~2.5-3.5 d** |
| W4 instead: + write and verify the r4d fragment packer | +1-2 d |
| W4: + two-group-size compressed-tensors checkpoint and dual dispatch | +1 d |
| W4: quality risk | **not recommended at any effort** |

---

## Artifacts

`bench_int4.py`, `bench_r4d.py`, `bench_fp8.py`, `hc_quant.py`, `hc_names.py`;
results `int4_down.json` `int4_hc_up.json` `int4_moe_router.json`
`int4_hc_down_plain.json` `r4d_down.json` `r4d_hc_up.json` `r4d_moe_router.json`
`r4d_hc_down_plain.json` `fp8_results.json` `hc_quant.json` `hc_quant2.json`,
logs `hc_quant.log` `hc_quant2.log`.

Note: the `q38fn-mxfp4` container exited (137) partway through this session. I did not
stop it and did not restart it.
