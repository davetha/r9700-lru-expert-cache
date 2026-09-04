# VRAM census: q38fn-heretic2-mxfp4-fp8, TP=2, `--cpu-offload-gb 40 --cpu-offload-params experts`

Method: safetensors headers only (name/dtype/shape from the little-endian JSON header of each
`.safetensors` shard — no tensor data read, no GPU, no model load). Host copy is `root:600` so the
header script ran inside a CPU-only `docker run --entrypoint python3` against `local/q38fn-rocm10:try1`
with `/mnt/llm-storage` and `$REPO/k2` bind-mounted read-only; no model was constructed, no
GPU device was touched. Script: `$REPO/k2/census_headers.py`, output
`$REPO/k2/tensor_headers.json` (152,389 tensors). Classifier:
`$REPO/k2/census_compute.py`.

Config: `/mnt/llm-storage/q38fn-heretic2-mxfp4-fp8/config.json` — `architectures:
["Qwen4ExpForConditionalGeneration"]` (vision tower **is** loaded — the launcher passes
`--limit-mm-per-prompt.image 8`), 48 main layers (`full_attention_interval=4` → layers
3,7,11,...,47 are QSA full-attention, the other 36 are GDN `linear_attn`), `mtp_num_hidden_layers=1`,
`hc_count=4`, 512 routed experts/layer, `moe_intermediate_size=640`.

## 1. Where the 6.04 GiB actually is

| category | tensors | GiB/rank | resident? |
|---|---:|---:|---|
| PLE ngram shards (`*.ple.*`) | 138 | 47.75 | **host** (`VLLM_PLE_CPU_OFFLOAD=1`) |
| main-layer routed experts (`layers.N.mlp.experts.*`, MXFP4) | 147,456 | 29.88 | **host** — fits inside the 40 GiB budget whole, see §2 |
| hyper-connection (all `*hyper_connection*`, main+MTP) | 398 | 1.230 | **VRAM** |
| **MTP routed experts (`mtp.layers.0.mlp.experts.*`, FP8)** | 3,072 | **1.172** | **VRAM — never touched by the offloader, see §3** |
| linear_attn (GDN in_proj/out_proj/conv1d/A_log/dt_bias/norm) | 432 | 0.977 | VRAM |
| lm_head | 1 | 0.592 | VRAM |
| embed_tokens | 1 | 0.592 | VRAM |
| vision tower (`model.visual.*`) | 333 | 0.453 | VRAM |
| self_attn fp8 (q/k/v/o + scales, the 12 QSA layers) | 96 | 0.278 | VRAM |
| MoE router gates (`mlp.gate`, main+MTP) | 49 | 0.120 | VRAM |
| shared experts (main+MTP, incl. `shared_expert_gate`) | 340 | 0.063 | VRAM |
| indexer (`self_attn.indexer.*`, main+MTP) | 45 | 0.085 | VRAM |
| MTP non-expert module (fc_embedding/fc_hidden, self_attn, hc, norms) | 29 | 0.104 | VRAM |
| self_attn q/k/v norms | 24 | ~0 | VRAM |
| **sum of everything above except PLE + main experts** | | **5.5738** | VRAM |

Reported: **6.04 GiB**. Header-only weight sum: **5.5738 GiB**. Gap: **0.466 GiB**.

Known non-weight allocations that land inside the same "Model loading" memory delta (they're
constructed alongside the layers, not lazily at first request):
- r4d/clav all-reduce scratch (from your startup-log numbers): slot 40960 KiB + scratch 184320 KiB
  + clav_ag band 8192 KiB + scratch 32768 KiB + r4d ar max 49152 KiB = **0.301 GiB**
- `Qwen4ExpModel._mtp_hidden_buffer` (`models/qwen4_exp/amd/model.py:453`, allocated when
  `speculative_config.method=="mtp"`): `[max_num_batched_tokens, hc_count*hidden_size]` bf16 =
  2048 × 10240 × 2B = **0.039 GiB**

5.5738 + 0.301 + 0.039 = **5.914 GiB**. Remaining gap to 6.04 GiB: **~0.13 GiB (2%)** — plausibly
HIP/PyTorch caching-allocator rounding (allocations aren't byte-tight, they round to allocator
segment size) plus assorted small buffers (K/Q layernorm caches, position-embedding tables in the
vision tower, etc.) not individually itemized. Not chasing further; the two named buffer sources
account for the bulk of the gap and the weight table is exact from the checkpoint header.

TP=2 split rule applied per tensor (checked against the actual parallel-linear classes in
`models/qwen4_exp/amd/model.py`, `models/qwen4_exp/amd/mtp.py`,
`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `model_executor/models/qwen3_vl.py`,
`model_executor/models/qwen3_next.py`):
- **halved** (col or row parallel): self_attn q/k/v/o (`QKVParallelLinear`/`RowParallelLinear`),
  linear_attn in_proj_qkvz/in_proj_ba/conv1d/out_proj/A_log/dt_bias (all TP-sharded —
  `qwen_gdn_linear_attn.py:398,411,422,447,450,476`), shared_expert gate_up/down (`Qwen3NextMLP`),
  vision-tower attn.qkv/proj + mlp.linear_fc1/fc2 (`qwen3_vl.py:412,421`, unless
  `is_vit_use_data_parallel()` — **flagged uncertain**, see below), embed_tokens/lm_head
  (`VocabParallelEmbedding`/`ParallelLMHead`), mtp.fc_embedding/fc_hidden
  (`ColumnParallelLinear`, `mtp.py:212,220`), routed/MTP experts (FusedMoE splits the
  `moe_intermediate_size` dim).
- **replicated** (full size both ranks): hyper-connection (`disable_tp=True`,
  `hyperconnection.py:105`), MoE router `gate` and `shared_expert_gate` (`ReplicatedLinear`,
  `qwen3_next.py:124,132`), indexer `index_qk_proj` (`ReplicatedLinear`, `indexer_qsa.py:110`),
  q_norm/k_norm/indexer layernorms, vision `patch_embed.proj` (`Conv3dLayer`) and `pos_embed`
  (plain `nn.Embedding`).

**Uncertainty flagged:** vision-tower ColumnParallel/RowParallel linears carry
`disable_tp=use_data_parallel` (`qwen3_vl.py:411,421`). If `is_vit_use_data_parallel()` is true for
this launch (vLLM's common default for ViT — each rank runs the encoder on its own images rather
than splitting weights), the vision tower is **replicated**, not split, and its true VRAM cost is
0.906 GiB/rank, not 0.453. That alone would close most of the remaining 0.13 GiB gap and then some;
I did not find the resolved value of `is_vit_use_data_parallel()` for this launch config in the
time available — treat the 0.453 GiB vision-tower line as a lower bound.

## 2. Confirms the main-layer experts are all going to host

29.88 GiB/rank of main-layer MXFP4 experts fit entirely inside the 40 GiB `--cpu-offload-gb` budget,
so `UVAOffloader._maybe_offload_to_cpu` (`model_executor/offloader/uva.py:64`) never hits the budget
ceiling while walking the 48 decoder layers — every `experts`-segment-matching parameter gets moved,
regardless of layer order. Nothing partially offloaded here; this category is cleanly 0 GiB on GPU.

## 3. MTP experts: in VRAM, and *structurally* can't be reached by `--cpu-offload-params experts`

This is the standout finding. `--cpu-offload-params experts` never sees the MTP expert weights,
because the offloader is only ever invoked from `make_layers()`:

```
# model_executor/models/utils.py:889-897
from vllm.model_executor.offloader import get_offloader
...
modules = (
    [PPMissingLayer() for _ in range(start_layer)]
    + get_offloader().wrap_modules(
        layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
    )
    + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
)
```

`Qwen4ExpModel.__init__` (`models/qwen4_exp/amd/model.py:415`) builds the 48 main layers through
`make_layers()`, so those pass through `get_offloader().wrap_modules()` and are eligible for
`cpu_offload_params` matching. `Qwen4ExpMultiTokenPredictor.__init__`
(`models/qwen4_exp/amd/mtp.py:222`) builds its one layer as a **plain `nn.ModuleList`**:

```python
self.layers = nn.ModuleList(
    Qwen4ExpDecoderLayer(
        draft_vllm_config,
        layer_type="full_attention",
        prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
    )
    for idx in range(self.num_mtp_layers)
)
```

No `get_offloader().wrap_modules()` call anywhere in `mtp.py`. So MTP's 512 experts (converted from
MXFP4 to FP8 block[128,128] by your `fp8_surgery.py` — `G1 = re.compile(r'mtp\..*\.mlp\.experts\..*')`)
are **always resident on GPU**, independent of `cpu_offload_gb`/`cpu_offload_params` values, currently
1.172 GiB/rank (2.34 GiB total across TP=2).

Fix if you want this freed: after `self.layers` is built in `Qwen4ExpMultiTokenPredictor.__init__`,
call `get_offloader().wrap_modules(iter(self.layers))` and reassign, the same pattern `make_layers()`
uses. Budget headroom exists — main experts use only 29.88 of the 40 GiB ceiling, so the extra 1.17
GiB/rank has room without raising `--cpu-offload-gb`. Since MTP only runs once per accepted draft
token (not once per layer per token like the main stack), the added PCIe cost per forward pass should
be small relative to the 15%/GB-freed payoff you're chasing, but I have not measured it — this is an
inferred low-risk claim, not a benchmarked one.

## 4. Non-weight VRAM consumers found in code (not part of the 6.04 GiB weight census)

- **GDN/mamba state cache** — `MambaStateShapeCalculator.gated_delta_net_state_shape`
  (`model_executor/layers/mamba/mamba_utils.py:258`), called from
  `get_gdn_mamba_state_shape_from_config` (`models/qwen4_exp/amd/model.py:709`). Per rank, per
  cache block, per GDN layer: `temporal_state_shape = (num_v_heads/tp=24, head_v_dim=128,
  head_k_dim=128)` + a conv_state of `(conv_dim/tp, conv_kernel-1+num_spec)` where
  `conv_dim = head_k_dim*num_k_heads*2 + head_v_dim*num_v_heads = 128*16*2+128*48=10240` (matches
  the `linear_attn.conv1d.weight` shape `[10240,1,4]` in the checkpoint). 36 GDN layers × 24 × 128 ×
  128 × 4B(fp32) = 56.6 MB/seq/rank temporal + ~2.8 MB/seq/rank conv ≈ **0.055 GiB/seq/rank**. At the
  production `--max-num-seqs 4` that's **~0.22 GiB/rank total**. Dtype comes from
  `MambaStateDtypeCalculator.gated_delta_net_state_dtype(model_dtype, cache_config.mamba_cache_dtype,
  cache_config.mamba_ssm_cache_dtype)` (`model.py:699`) — the launcher sets neither
  `--mamba-cache-dtype` nor `--mamba-ssm-cache-dtype`, so this resolves through vLLM's own "auto"
  default; I did not trace that default to a concrete dtype in the time available, so the fp32
  assumption above is the upper bound, not confirmed. This is allocated during KV-cache profiling
  (after weight load), not part of "Model loading took 6.04 GiB".
- **QSA/indexer buffers** — `_QSAStateCache.__init__` (`models/qwen4_exp/common/qsa_cache.py:571`
  area): `token_to_req_buffer`/`slot_mapping_buffer`/`logical_positions_buffer` sized
  `max_tokens` (int32/int64, a few hundred KB total) plus `k_work_metadata_buffer` sized
  `max_k_work × 2` int32 when enabled. All small (order of a few MB), not itemized further.
- **r4d/clav all-reduce buffers** — sizes are from your own startup log (cited in §1): slot
  40960 KiB scratch 184320 KiB, clav_ag band 8192 KiB scratch 32768 KiB, r4d ar max 49152 KiB.
  Source not located in this pass (likely native-side allocation inside the `.so`, not visible in
  the ctypes wrapper) — treat the numbers as given.
- **Spec-decode MTP buffer** — `Qwen4ExpModel._mtp_hidden_buffer` (§1), 0.039 GiB/rank, fixed size
  (`max_num_batched_tokens`-driven, not per-seq).

## 5. Ranked levers for freeing VRAM

| # | Change | GiB/rank saved | Effort / risk |
|---|---|---:|---|
| 1 | **Route MTP's `nn.ModuleList` through `get_offloader().wrap_modules()`** (§3) so `--cpu-offload-params experts` reaches it | **1.17** | Small code change (`mtp.py`, ~5 lines); budget headroom already exists (29.88/40 GiB used); MTP forward runs far less often than the main stack, so PCIe cost should be low — not yet benchmarked |
| 2 | **Add `visual` to `--cpu-offload-params`** (flag-only, if `.visual.` survives as a literal parameter-name segment and the current traffic mix has few/no images — confirmed by your own `expert_hist.py` corpus: code/prose/json/math, no vision task) | **0.45** (0.91 if the vision tower turns out to be replicated, see §1 uncertainty) | Zero code change, just `--cpu-offload-params experts,visual`; costs one PCIe round-trip only on requests that actually attach an image |
| 3 | **FP8 embed_tokens + lm_head** — extend `fp8_surgery.py`'s conversion set; these are the two largest single BF16 tensors still resident | ~0.59 | Checkpoint regeneration (CPU-only, same recipe pattern as attention); accuracy risk is real — these are read on *every* token including rare/adversarial vocab, which is plausibly why the reference recipe's `ignore` list left them alone; needs a coherence check before shipping, same caliber as your Q6_K/probe traps |
| 4 | **FP8 hyper-connection linears** — currently full-size BF16 on *both* ranks (`disable_tp=True`), so this is the one weight category TP doesn't already discount | ~0.6 | Extend the fp8_surgery classifier to hyper-connection module linears (all shapes are multiples of 128, e.g. `[10240,320]`); touches every token at every one of 48+1 layers, so this is the highest-traffic quantization target on the list — needs real quality validation, not just a shape check |
| 5 | **GDN state dtype / `--max-num-seqs`** | ~0.03–0.11/rank (fp32→bf16, or lower `max_num_seqs`) | Smallest win on this list and the riskiest: your own `GDN_GATE.md`/gdn_hip history flags a "production NaN trap" and a state-divergence FATAL abort tied to numeric precision of exactly this cache — don't touch the dtype without re-running whatever gate caught that before |

Recommend #1 and #2 first: both are low-risk (no requantization, no new numerics), sum to
**1.62 GiB/rank / 3.24 GiB total**, and by your own ~15%-per-GB-freed rule that's roughly a
**24% cut in cold PCIe traffic** if that multiplier holds this far up the curve (rough
extrapolation, not measured).
