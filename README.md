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


## Benchmark log

Every A/B arm of the campaign, in chronological order. All runs: 2x R9700 (gfx1201) TP=2,
`local/q38fn-rocm10:try1` image, greedy, MTP-4 unless stated, `max-model-len 262144`,
`max-num-seqs 4` unless stated, single stream, `bench/ab3.py` best-of-3 (256 / 800 / 600
output tokens for the prose / JSON / code prompts). ms/step is wall time divided by the
`vllm:spec_decode_num_drafts_total` delta, i.e. one target-verify forward plus its 4 draft
passes; it is the metric to compare kernel changes on, because any numerics change alters the
greedy text and therefore the MTP acceptance rate and tok/s. `accept` is accepted / drafted
tokens. Prefill is a 12.5K-token prompt with `max_tokens 1` (the probe was added mid-campaign,
so early arms have no prefill figure). Greedy output on this box is not reproducible across
server restarts (see caveats), so differences of a few percent between arms are noise.

| arm | configuration | prose | JSON | code | ms/step (prose/JSON/code) | accept (prose/JSON/code) | prefill tok/s |
|---|---|---|---|---|---|---|---|
| `r10_base_h15` | baseline: static hot set 15 GB, MTP-4, NBT 2048 (ROCm 10 image) | 60.2 | 68.6 | 89.1 | 46.26/55.28/41.83 | 0.454/0.7/0.686 |  |
| `r10_skinny0_h15` | baseline + VLLM_ROCM_USE_SKINNY_GEMM=0 (hipBLASLt for bf16 skinny GEMMs) | 60.1 | 67.1 | 84.4 | 46.79/58.18/42.84 | 0.453/0.726/0.655 |  |
| `mtp2` | baseline, MTP-2 | 62.3 | 63.7 | 79.5 | 36.72/42.7/33.55 | 0.638/0.859/0.831 |  |
| `mtp3` | baseline, MTP-3 | 60.4 | 68.0 | 85.7 | 40.35/50.68/38.06 | 0.476/0.816/0.755 |  |
| `vram2_h16` | baseline, hot 16 GB, NSEQ 2, NBT 1024 | 62.2 | 74.3 | 90.5 | 44.75/54.13/41.16 | 0.446/0.754/0.685 |  |
| `best1_h17_nbt512` | baseline, hot 17 GB, NSEQ 2, NBT 512 | 64.5 | 77.9 | 93.9 | 43.12/51.63/39.71 | 0.446/0.754/0.685 | 1942.7 |
| `k4patch3` | baseline + K4 kernel-count patches (GemmaRMSNorm mode 1) | 62.1 | 72.1 | 89.2 | 45.3/55.73/41.5 | 0.453/0.754/0.679 | 3108.2 |
| `k4k3` | baseline + K4 patches + K3 r4d bf16 dispatch | 62.5 | 68.8 | 82.8 | 45.01/55.34/38.33 | 0.453/0.701/0.544 | 3114.6 |
| `lru0ctl` | LRU build mounted, VLLM_R4D_LRU=0 (control, restart #1) | 58.5 | 66.1 | 76.6 | 44.62/54.78/37.83 | 0.401/0.656/0.475 | 3052.3 |
| `lru0ctl_b` | LRU build mounted, VLLM_R4D_LRU=0 (control, restart #2) | 60.8 | 68.7 | 81.3 | 44.3/54.89/38.03 | 0.432/0.692/0.522 | 3066.3 |
| `lru_noins` | LRU=1 with VLLM_R4D_LRU_MAX_INSERTS=0 (machinery on, no inserts) | 61.1 | 70.5 | 81.4 | 45.09/55.88/38.4 | 0.441/0.735/0.531 | 3091.8 |
| `lru1` | LRU expert cache ON, hot 15 GB, NBT 2048 | 76.7 | 112.6 | 94.1 | 35.14/35.17/34.1 | 0.429/0.744/0.551 | 3161.0 |
| `lru_h17_nbt512` | LRU ON, hot 17 GB, NSEQ 2, NBT 512 | 86.2 | 113.3 | 94.9 | 33.36/34.61/33.47 | 0.472/0.729/0.542 | 2531.5 |
| `combo1` | LRU + K4 + K3 + UVA offload of embed_tokens/visual | 81.3 | 110.7 | 93.0 | 34.97/35.24/34.33 | 0.464/0.724/0.547 | 3392.6 |
| `w4head` | combo1 + W4 draft-only lm_head | 84.5 | 119.4 | 100.3 | 31.89/32.06/31.14 | 0.421/0.706/0.53 | 3195.0 |
| `c2_mtp3` | w4head stack, MTP-3 | 94.0 | 115.7 | 93.1 | 29.27/29.79/29.7 | 0.588/0.816/0.588 | 3166.6 |
| `c2_mtp6` | w4head stack, MTP-6 | 75.7 | 115.3 | 101.7 | 37.58/38.77/38.57 | 0.313/0.579/0.489 | 3470.3 |
| `c4` | + fused LRU bookkeeping kernel, GDN strided qkv, fused shared gate (c4) | 89.4 | 122.2 | 105.8 | 30.14/31.17/29.86 | 0.421/0.704/0.538 | 3190.9 |
| `c5_localargmax` | c4 + use_local_argmax_reduction (draft argmax) | 88.0 | 129.2 | 101.8 | 31.28/31.44/30.23 | 0.438/0.769/0.519 | 3171.5 |
| `c6_siluquant` | c4 + fused silu+fp8-quant | 88.7 | 123.0 | 105.7 | 30.37/30.98/30.37 | 0.421/0.704/0.552 | 3169.9 |
| `c7` | c6 + QSA rope gather fold (c7) | 93.2 | 125.9 | 107.4 | 30.19/30.27/29.55 | 0.462/0.704/0.545 | 3219.8 |
| `c8_v2` | c7 on librlu_v2 (rewritten victim selection) (c8) | 91.8 | 124.1 | 106.2 | 29.99/30.83/29.43 | 0.438/0.707/0.53 | 3174.1 |
| `t1_nbt4096` | c8, NBT 4096 | 96.2 | 122.8 | 103.4 | 30.24/30.86/29.61 | 0.477/0.7/0.517 | 3444.9 |
| `t3_h16` | c8, hot 16 GB | 91.7 | 122.0 | 109.4 | 28.77/30.64/28.56 | 0.41/0.686/0.533 | 3268.1 |
| `t4_fp8triton` | c8 + VLLM_DISABLE_FP8HIP=1 (Triton fp8 fallback) | 99.9 | 123.9 | 113.1 | 29.47/30.31/31.38 | 0.483/0.69/0.638 | 3262.8 |
| `t5_h16_nbt4096` | c8, hot 16 GB, NBT 4096 | 89.3 | 126.7 | 107.3 | 29.86/30.21/29.43 | 0.414/0.711/0.541 | 3462.3 |
| `t7b` | c8 + K3 dispatcher fall-through (census arm) | 89.1 | 131.3 | 114.7 | 30.88/30.77/31.51 | 0.438/0.763/0.654 | 3337.8 |
| `t8b` | c8, NBT 4096, shared_expert_gate GEMV on r4d (launcher default) | 91.4 | 130.8 | 118.7 | 29.48/29.56/30.44 | 0.424/0.719/0.658 | 3538.7 |
| `t9` | t8b + experimental fp8 skinny kernel (VLLM_HC_FP8SK=1) | 90.5 | 129.2 | 116.3 | 29.16/29.34/29.82 | 0.407/0.698/0.621 | 3513.2 |


### Per-step kernel breakdown, baseline vs final

Median full decode step (target forward + 4 MTP drafts + sampler) from the torch profiler,
rank 0, top 12 kernels by time. The profiler adds ~2 us per launch, so these step times are
~15-20% above the unprofiled `ms/step` in the table above; the proportions are what matter.

**baseline** (`prof_base_h15`): full decode step 55.0 ms under the profiler, 3490 kernel launches, 40.5 ms kernel-busy

| kernel | launches/step | ms/step |
|---|---|---|
| `void r4d_gemm_moe_mxfp4a8_nt_b16_kernel` | 192 | 19.28 |
| `void wvSplitK_hf_sml_` | 248 | 7.55 |
| `void fp8hip_gemm_w8a8_tiled` | 96 | 2.93 |
| `void wvSplitK_hf_big_` | 100 | 1.47 |
| `void r4d_ar_oneshot_2rank_exact_kernel` | 109 | 1.22 |
| `Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_M` | 52 | 1.04 |
| `void r4d_gemm_mxfp4a8_nt_m64_kernel` | 96 | 0.66 |
| `void at::native::elementwise_kernel_manual_unr` | 331 | 0.58 |
| `__amd_rocclr_batchMemOp` | 1 | 0.54 |
| `void at::native::vectorized_elementwise_kernel` | 339 | 0.43 |
| `void vllm::moe::topkGating` | 52 | 0.39 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 36 | 0.35 |

**final (t8b)** (`prof_t8b_gate_r4d`): full decode step 35.2 ms under the profiler, 2823 kernel launches, 22.9 ms kernel-busy

| kernel | launches/step | ms/step |
|---|---|---|
| `void r4d_gemm_moe_mxfp4a8_nt_b16_kernel` | 192 | 5.06 |
| `void r4d_gemm_bf16_nt_m64_kernel` | 362 | 4.50 |
| `void fp8hip_gemm_w8a8_tiled` | 96 | 2.84 |
| `lru_gather_k` | 48 | 1.41 |
| `void r4d_ar_oneshot_2rank_exact_kernel` | 109 | 1.12 |
| `void r4d_gemm_w4a16_nt_m64_kernel` | 4 | 1.02 |
| `__amd_rocclr_batchMemOp` | 1 | 0.80 |
| `void r4d_gemm_mxfp4a8_nt_m64_kernel` | 96 | 0.67 |
| `void wvSplitK_hf_sml_` | 39 | 0.55 |
| `void vllm::moe::topkGating` | 52 | 0.40 |
| `fused_sigmoid_gating_delta_rule_update_kernel` | 36 | 0.35 |
| `fused_moe_kernel` | 8 | 0.33 |


The MoE grouped GEMM line is the LRU cache: 19.3 ms of cold-expert PCIe reads at 28.4 GB/s
became 5.1 ms of resident GEMM plus 1.4 ms of `lru_gather_k` misses. `wvSplitK` (the
hyper-connection bf16 GEMMs) moved to `r4d_gemm_bf16_nt_m64`; the `Cijk` hipBLASLt line
(the 2560 -> 1 shared-expert gate) is gone; the four `r4d_gemm_w4a16_nt_m64` launches are
the W4 draft lm_head replacing four ~1 ms bf16 GEMMs that ran outside the annotated forward.

### Concurrency

`bench/concurrent_bench.py`, B simultaneous streams of 500 tokens on different prompts,
aggregate tok/s and steady-state ms/step (engine steps = drafts delta / running requests):

| streams | static hot set | LRU only (`lru1` stack) | full stack (`c8`) |
|---|---|---|---|
| B=1 | 76.6 (37.9 ms/step) | 94.0 (33.7) | 105.7 (29.9) |
| B=2 | 88.6 (58.8) | 114.9 (45.5) | — |
| B=4 | 126.6 (97.1) | 151.8 (79.1) | 165.7 (72.6) |

### Long-context correctness

`bench/needle.py`: a 3-field needle (name / part number / ticket id) at 10 / 50 / 90 % depth
of 32K, 128K and 200K-line documents, prompts of 36,442 / 148,889 / 256,389 tokens.
**9/9 PASS** on the LRU stack (`lru1`) and again on the full stack (`combo1`); the same 9/9
held on the static-hot-set baseline.

### Numerics probes

`bench/lpprobe.py` (teacher-forced prompt logprobs on fixed 260-800-token texts) and
`bench/genprobe.py` (greedy decode logprobs), mean |dlogprob| per token:

| pair | prose | code | json |
|---|---|---|---|
| two LRU=0 servers, identical config (restart noise floor) | 0.064 | 0.020 | 0.016 |
| LRU=0 vs LRU=1 with `MAX_INSERTS=0` (cache machinery on, no inserts) | 0.000 | 0.010 | 0.015 |
| LRU=0 vs LRU=1 | 0.13 | 0.018 | 0.028 |

The LRU's excess over the restart floor is a uniform scale factor across the distribution
(partition churn: the same rows are computed by the resident and the fallback call on
different steps), not a tail — a wrong expert would show as a heavy tail. The kernel-level
tests (`tests/lru/test_numerics_prod.py`, `test_slots_prod.py`) are bit-exact.

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
| `VLLM_GEMMA_NORM_FUSED` | `2` | dispatch the fused `rms_norm` in eager regions. `2` casts to fp32 around the fused kernel, matching the stock decomposition up to reduction order, for **-224 kernels/step**; it is the file default and was on in every arm from `combo1` onward. `1` is the original bf16-weight variant, -288/step but it **perturbs ~30% of elements**; `0` is the stock 10-kernel path. See `docs/PATCHES.md` #1. |
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

## Negative results

Things that were built, measured, and did not earn a place in the default configuration.
They are here because the measurement is worth more than the outcome, and because the next
person should not spend a day rediscovering them.

* **An open fp8 skinny GEMM is not the lever.** `hcq_gemm_fp8blk_nt_m16` reimplements the
  closed `fp8hip_gemm_w8a8_tiled` for M<=16 — same operands byte for byte, including the
  pre-shuffled weight, so it needs no second copy in VRAM. It is correct and graph-safe, and
  it is **1.01x** on the production shape mix in isolation. In the running server (arm `t9`)
  it is a dead heat: 2.81 ms/step across 96 calls against fp8hip's 2.81 ms/step across 96.
  The reason is that these layers are already memory-bound — fp8hip runs within 4-9% of a
  pure-read control kernel, i.e. at 91-96% of what the access pattern can do at all. Shipped
  behind `VLLM_HC_FP8SK` (default `0`) in `kernels/experimental/fp8skinny/`, whose README is
  mostly about the benchmarking method: a nominal-spec roofline said "47% of peak" and was
  measuring nothing.
* **The one-kernel Gemma norm was rejected on one element.** Mode 3 folded mode 2's
  cast / fused / cast into a single Triton kernel, -64 launches/step. The gate for shipping
  was bit-identity with mode 2; it missed by **1 element in 81.8M**. `docs/PATCHES.md` P-A.
* **The mode-2 shared-gate fold was superseded.** `VLLM_FUSED_SHARED_GATE=2` would have saved
  another 52 launches/step, but K3's dispatch change routes the same GEMV to r4d for a larger
  win, and the fold showed no divergence in 8.2M elements without being provable. Keep it
  off. `docs/PATCHES.md` #7m2.
* **`VLLM_SKINNY_CU_COUNT` is not a lever.** 19-point CuCount curves, including
  non-powers-of-two, put the projected saving at **0 us/step**; setting it far above the real
  WGP count makes things worse, not better. `docs/RESULTS.md`.
* **The `FP8HIP_*` env knobs do nothing reachable.** Disassembly of the closed
  `libfp8hip_gemm.so` shows all three are consumed inside a code path the vLLM wrapper never
  enters. `docs/FP8HIP_KNOBS.md`.
* **Side-stream hot/cold overlap halved throughput.** Overlapping the cold gather on a second
  stream inside the HIP graph was much worse than serialising it — cross-stream waits are
  expensive here. `VLLM_R4D_HOT_SIDE_STREAM` is measured-dead; arm `side1` measured
  32.7 / 39.7 / 90.7.
* **No prefetch predictor beats LRU residency.** Scored against the LRU's own miss stream —
  64,095 real misses — a last-step predictor covers 0.0% at every budget, frequency 0.4-6.4%,
  co-occurrence 2.0-16.2% at 22-45x wasted bytes. 66.7% of missed experts never appear again
  in the generation, so the misses are compulsory and there is nothing to predict.
  `docs/K1_PROGRESS.md` Task I(c).
* **Nothing beat a plain UVA gather for the cold transfer.** Staged pinned copies, wider
  slabs and the alternatives in `docs/COLD_TRANSFER.md` all land at or below the 28.4 GB/s the
  simple path already gets — which is itself a PCIe Gen4 root-port limit, not a software one.

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
