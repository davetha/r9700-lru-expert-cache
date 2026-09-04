# Review: device-side LRU expert cache, and what is left on the table

Read-only review, 2026-09-04. Scope: `k1/LOCALITY.md`, `k1/lru/**`, `patches/hotcold/r4d_mxfp4_moe_lru.py`,
`k1/lru/FUSE.md`, `k1/COLD_TRANSFER.md`, `k4/KERNEL_HITLIST.md`, `patches/PATCHES.md`, `k3/HC_QUANT.md`,
`k2/VRAM_CENSUS.md`, plus `k1/PROGRESS.md` and the `arms_*.log` / `ab3_*.json` / `k3/*.json` measurement
record. Nothing was run; no GPU work; only this file written.

Step budget I am reasoning against — the `w4head` arm, which is the 84.5 / 119.4 / 100.3 result:

```
step 31.9 ms (annotation 32.1) | kernel busy 19.7 ms | NOT BUSY 12.4 ms (39%)
  4.22  192x  r4d_gemm_moe_mxfp4a8_nt_b16   (was 21.06 pre-LRU)
  3.15  292x  r4d_gemm_bf16_nt_m64          (hyper-connection)
  2.84   96x  fp8hip_gemm_w8a8_tiled        (GDN + QSA linears)
  2.39   48x  lru_gather_k                  (the misses, over PCIe)
  1.11   48x  Cijk_Ailk_Bljk_BBS_...        (hipBLASLt; see L3)
  0.87   97x  r4d_ar_oneshot_2rank_exact
  0.67   96x  r4d_gemm_mxfp4a8_nt_m64       (shared expert)
  0.40  210x  elementwise_kernel_manual_unroll
 ~4.1         everything else
```

The LRU did what LOCALITY said it would: the MoE grouped GEMM fell 21.06 -> 4.22 ms and the PCIe cost
reappeared as 2.39 ms of `lru_gather_k`, i.e. ~85 MB/step at the 28.4 GB/s ceiling, matching the
trace-replay prediction (85.6 MB/step) to within noise. The `lru_noins` arm (`MAX_INSERTS=0`:
61.1 / 70.5 / 81.4, indistinguishable from `lru0ctl`) is a proper ablation and rules out the
machinery-not-the-policy explanation.

---

# 1. Correctness

Rated **confirmed** (I read the code and the mechanism is there), **likely** (strong reading, one
check away), **speculative** (a hazard class, not a demonstrated defect).

## C1 — CONFIRMED. The numerics tests structurally cannot detect an unwritten output row, and the `-1` coverage claim is false

`_apply_split` allocates the intermediates uninitialised:

* `r4d_mxfp4_moe_lru.py:770` `ic1 = torch.empty(mtk, N1, ...)`
* `r4d_mxfp4_moe_lru.py:780` `ic3 = torch.empty(mtk, H, ...)`
* `r4d_mxfp4_moe_lru.py:784` `ops.moe_sum(ic3.view(M, top_k, H), output)` — sums **all** `top_k` rows,
  with no expert map, so every row is read.

Correctness therefore rests on the invariant "hot ∪ cold covers `[0, mtk)` exactly once". That invariant
is real and the fused kernel checks it, but the *numerics* tests cannot fail if it breaks:
`k1/lru/test_numerics_lru.py:183,188` allocate `y_split` and `y_ref` with `torch.zeros`, and the all-UVA
reference has the identical hole — an unwritten row is zero in both and compares equal. Production uses
`torch.empty`, where the same row is whatever the caching allocator last left there.

Second, `PROGRESS.md` Task F states that `test_numerics_prod.py` covered "the padded `-1` rows". It did
not: the only routing generator is `test_numerics_lru.py:143`,
`topk_ids = torch.randint(0, E, ...)` — never negative, in any suite. `test_lru.py`'s `-1` suite exercises
the *manager*'s routed-marking, not the aligns or the GEMM.

How reachable is an unwritten row today? Only via `topk_ids[i] < 0` or `>= E`. `_apply_split` is entered
only when `expert_map is None` (`:729`), there is no EP, and `fused_topk` fills every slot — so I believe
the answer is "not reachable in the current config". But it is an unguarded, untested invariant sitting
in front of `torch.empty`.

**Fix, ~20 lines:** a debug gate that fills `ic1`/`ic3` with NaN before the split and asserts no NaN
survives `moe_sum`; and one `test_numerics_prod` case with `-1` punched into `topk_ids`. Correct the
PROGRESS.md claim either way.

## C2 — CONFIRMED, and it is *clean*: the fused kernel's invalid-id handling matches vLLM exactly

Worth recording because it is the obvious place for the fused kernel to have drifted, and it has not.
`lru_fused_k` skips out-of-range ids at `r4d_lru.hip:470` (`if (e < 0 || e >= E) continue;`) in the
counting pass and at `:497` in the placement pass. vLLM's own kernel does the same in
`get_local_expert_id` (`csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:88-95`:
`if (expert_id >= num_experts || expert_id < 0) return <invalid>;` *then* `expert_map[expert_id]`) —
so it never indexes the map with a negative id either, and the fused kernel is a faithful drop-in on
padded rows. (Read from `$HOME/R9V/vendor/vllm/`, not the image's copy; the semantics are stable
upstream but if you want it airtight, diff the two.)

Likewise `_align_sizes` (`r4d_mxfp4_moe_lru.py:281-287`) reproduces the wrapper's allocation formula
(`moe_align_block_size.py:74-80`) exactly, including the `topk_ids.numel() < num_experts` clamp, and
`pad_sorted_ids` is False in both. `cum_h[S] <= L` holds for the same reason. No mismatch.

## C3 — LIKELY. TP-rank divergence is a throughput bug, not a correctness bug — and the docs oversell it; there is no check either way

`k1/lru/README.md` frames identical cache evolution across ranks as a requirement of the design. It is
not a correctness requirement. `test_numerics_lru` and `test_slots_prod.py` check (d) establish that the
r4d grouped GEMM is **partition-invariant**: (resident call + fallback call) is bit-identical to one
all-UVA call, whatever the partition. So a rank whose table diverged still computes a bit-identical
partial sum; it just misses more. The all-reduce sees the same numbers.

The real gap is the opposite one: **nothing checks that the ranks agree**, so a divergence would be
silent and would show up only as unexplained gather time. Given the cache is 257 int32s per layer, a
debug-gated checksum of `slot_expert` folded into an existing collective every N steps would make the
property observable for ~nothing. I would add it before trusting the design under concurrency (C7),
where the batch composition — not just the routing — starts to matter.

## C4 — LIKELY. The fused kernel has never run in a server, and its one untested exit is the one that cannot be reached from production

`FUSE.md` §5 risk 1 is the right risk and it was taken seriously: the manager's two early `return`s became
branches (`r4d_lru.hip:376` `bool ins = (cnt <= max_distinct)`, `:393` `ins = (nins > 0)`), and
`test_fused.py` forces both a read-through step (`max_distinct=0`) and a zero-miss step. Good.

The third exit was not converted to a *tested* path: the victim-search `break` at `r4d_lru.hip:420-424`
(`best == LLONG_MAX`, nothing evictable). Reading it, it is correct — `s_nins` is lowered to `r`,
`n_miss` is written at `:446`, and the LDS state is flushed back at `:442-445`. But it is unreachable
under the production gate (`cnt <= max_distinct = S/2` guarantees at least S/2 unrouted slots), so no
production run will ever exercise it, and no test forces it. It is the only path that can leave
`n_miss` inconsistent with the `miss[]` rows the gather will read. Forcing it costs one test case
(`max_distinct = E`, `S` small, route every resident expert).

Also: `VLLM_R4D_LRU_FUSE=1` has never been in an arm. `librlu_fused.so` is only in `bench_fused.py` and
`test_fused.py`. See M4 — the failure mode of a stale `sorted_ids` is *plausible text*, not a crash, so
that arm needs the needle test and `spec_decode_num_accepted`, not just tok/s.

## C5 — LIKELY (hygiene). `_lru_step_fused` ignores FUSE.md's own preallocation requirement, and the reason it survives is not the reason given

`r4d_mxfp4_moe_lru.py:655-660` allocates six `torch.empty` per layer per forward. `FUSE.md` §5 risk 2
says these "need to be preallocated and persistent", explicitly per layer. The docstring at `:649-650`
justifies `torch.empty` with "the kernel writes every element, sentinels included" — which is about
*initialisation*, not about *lifetime*, and is not why it is safe. It is safe because capture-time
allocations come from the graph's private pool and stream order keeps layer L's consumers ahead of
layer L+1's producer — the same reason vLLM's own `moe_align_block_size` wrapper gets away with it.
That is fine, but it costs six allocator round-trips per layer on every non-captured call (warm-up,
`--enforce-eager`, any uncaptured shape), and it leaves the file disagreeing with its own design note.
Preallocating per layer is 48 × ~7 KB and removes the ambiguity.

## C6 — CONFIRMED. The fused-launch saving is over-counted by 8%

`FUSE.md` §2 prices the win at `4 × 52` nodes on "48 + 4×1 MTP MoE layer invocations". The MTP layer
never enters `_apply_split`: `VRAM_CENSUS.md` §3 shows `Qwen4ExpMultiTokenPredictor.__init__` builds a
plain `nn.ModuleList` that `get_offloader().wrap_modules()` never touches, LOCALITY §6 retracts on the
same grounds, and `KERNEL_HITLIST.md` §8 measures the draft MoE taking the Triton `fused_moe_kernel`
path. The production trace settles it: `moe_align_block_size` is **192/step at per = 4.00 over the 48
MoE layers**, not 208.

So the real reduction is `4 × 48 = 192` nodes, the pure-dispatch ceiling is 0.46 ms not 0.50, and the
0.478 ms measured by `bench_fused.py` is itself on a synthetic 52-layer graph — expect **~0.44 ms/step**
in production, 1.4% at B=1. Nothing changes about whether to do it; it does change the arithmetic if
anyone stacks these estimates.

## C7 — SPECULATIVE, and the largest blind spot. Chunked prefill mixed with decode turns the cache off for that step, and nothing has ever measured it

The read-through gate (`r4d_lru.hip:116` and `:376`) keys on the distinct expert count of the **whole
batch**. A step that mixes a prefill chunk with running decodes trips it, so:

* the decode rows in that step get no admission (correct, but the cache stops adapting), and
* `mtk > VLLM_R4D_LRU_FUSE_MAX` (2048) also drops out of the fused path.

Every number published so far is blind to this. LOCALITY §5 says the capture is single-stream
("`max_num_seqs 4`, but one request at a time"); E.4's read-through frequency (0.12% of (layer,step) at
B=4) was computed on *merged decode* traces that contain no prefill chunk at all; every `ab3` arm is
serial. `k1/concurrent_bench.py` was written for exactly this and has never been run (E.1 says so
deliberately, to avoid polluting an in-flight control).

This is not a defect — it is the single largest unquantified gap between the measured 84/119/100 and
what a loaded server would see.

## C8 — SPECULATIVE (a bug family that already fired once). Unchecked stamp sign

`(s_st[i] << SLOT_BITS) | (long long)i` at `r4d_lru.hip:156` and `:416` is well-defined only for
non-negative stamps. `_build_lru` (`:602-613`) is careful about this and the Task G note explains why.
But the invariant lives only in a comment, and the *same* mistake already shipped once in
`slot_alloc.KernelLRU` (Task G: seeded negative, evicted the hottest profile expert first, 1.3-2.9%
worse than zero stamps, and went unnoticed through the whole MAX_INSERTS study). One
`assert slot_stamp.min() >= 0` in `_build_lru` closes the family for free.

Related, no action needed: overflow needs 2^43 steps; `S > SLOT_MASK` and `S > MAXS` are both guarded in
the C entry points (`:531`, `:549`).

## C9 — SPECULATIVE. Two unasserted layout assumptions on the C boundary

`_r4d_lru_manage`/`_r4d_lru_fused` pass `miss.data_ptr()` and the kernel indexes `miss[2*j]` /
`miss[2*j+1]`, which is correct only for a contiguous row-major `(cap, 2)`. `torch.full` produces that,
so it holds — but the gather wrapper *does* validate its slab sizes (`:257-259`) while the manager
validates nothing. Same for `topk_ids.reshape(-1)`: correct because `fused_topk` allocates contiguous
int32 (FUSE.md §1 verified the dtype), but nothing asserts it. Cheap asserts, ctypes has no type safety.

## C10 — Not a defect; a coupling worth writing down

`_gemm_split`'s `VLLM_R4D_HOT_SIDE_STREAM=1` branch (`:685-696`) runs the hot GEMM on a side stream and
the cold GEMM on the main one, both writing the same `ic1`/`ic3` via `_into`. That is safe *only*
because hot and cold `sorted_ids` partition `[0, mtk)`. Before the LRU that partition was a load-time
constant; now it is recomputed on the GPU every step. The safety of a concurrent-write pattern is now
data-dependent on the manager. The side-stream arm is not in the current best stack — if it comes back
(it is a natural pairing with L5), the partition check belongs in the same test as the numerics check.

## Things I checked and found sound

* Nothing routed this step can be evicted — the victim scan skips `routed[ex]` (`:155`, `:415`), which
  also covers a slot filled earlier in the same call, so "evicted an expert a later pass needs" cannot
  happen. And there is no later pass: `_apply_split` runs the manager once per layer per forward, and
  the four MTP draft iterations do not touch r4d at all (C6).
* Global writes to `slot_stamp` at `:126` / `:386` are separated from the LDS mirror load by
  `__syncthreads()`, which fences global as well as shared. No race.
* `n_miss[0] = 0` is written before every early exit, so a read-through or zero-miss step leaves the
  gather with an empty grid rather than a stale miss list.
* CUDA-graph batch padding polluting the routed set: **refuted by the trace.** If decode batches were
  padded to a captured bucket, LOCALITY's modal step width would be 8, not the 5 rows it measures.
* LDS budget for `lru_fused_k` is ~37 KB at E=512/S=1024 (FUSE.md's "~20 KB" undercounts `s_tab`,
  `cum_*` and `s_bkt`), still inside 64 KB. Irrelevant for a one-workgroup kernel, but the note is wrong.

---

# 2. Levers not yet exploited, ranked by expected ms/step

## L1 — Attribute the 12.4 ms of non-busy time. Up to ~4.7 ms (15%). Costs one CPU-minute.

This is now the biggest single bucket in the step, larger than any kernel. `step0_nodecost.py` priced a
graph node at **2.40 us of pure dispatch** (not the 3.72 us median gap, which overstates the marginal
cost by 26%). At ~3200 nodes that is 7.7 ms — the irreducible replay floor. Measured non-busy is
12.4 ms. **~4.7 ms is unaccounted for and nobody has looked.**

Candidates, in order of prior:

1. **TP skew at the 97 `r4d_ar_oneshot` sync points.** The all-reduce kernel is only 0.87 ms of busy
   time, so any imbalance between ranks materialises as gap on whichever rank arrives first. Two ranks,
   separate root ports, 97 barriers per step: 48 us of average skew is 4.7 ms.
2. **The draft-loop / graph-boundary region.** `KERNEL_HITLIST.md` §8 measured an 8.88 ms
   inter-annotation region of which 7.14 ms was kernel-busy, ~5.0 ms of it the bf16 draft lm_head. The
   W4 draft head (on in this arm) removed ~3 ms of that *busy* time. It did not remove the region — so
   several ms of what used to be busy may now be idle wall.
3. Genuinely per-node, i.e. my 2.40 us extrapolates badly to a 3200-node graph.

Each answer points somewhere different: (1) → balance work / reduce collective count; (2) → attack the
speculator's graph structure; (3) → L2 and L9 are the only levers and everything else is capped.
`prof_gaps.py` already exists; its 120 us threshold is far too high for this (the gaps are ~2-15 us).
See M1.

## L2 — Turn on the k4 patches that are already built. ~-1.24 ms (3.9%). No new code.

Built, unit-verified, sitting behind default-off gates, and **none of them are in the `w4head` arm's
`-e` list**:

| patch | nodes | ms at 2.40 us | numerics |
|---|---:|---:|---|
| `VLLM_GEMMA_NORM_FUSED=2` (fp32-exact fused norm) | -274 | -0.66 | 6 of 2.9 M elements, 1 bf16 ulp |
| `VLLM_GDN_STRIDED_QKV=1` | -144 | -0.35 | **bit-identical** (`torch.equal`, output and `ssm_state`) |
| `VLLM_FUSED_SHARED_GATE=1` | -48 | -0.12 | **bit-identical** |
| `VLLM_MOE_OUTPUT_ALIAS` (default on; verify it fired) | -48 | -0.12 | copy removal |

This is the cheapest millisecond on the board: it needs an arm, not a kernel. Two cautions. (a) Verify
each fires — `#1` silently did not engage in `k4patch2` because `forward_cuda` is unreachable when
inductor is on, and `#6` looked inert because ROCm executes a device-to-device `copy_` as the
`__amd_rocclr_copyBuffer` *kernel*, not as a memcpy event. Count the kernel, not the memcpy. (b) The
norm patch needs a greedy-divergence gate **with a base-vs-base control** — this server's greedy output
is not reproducible across restarts, so an LRU=1-vs-LRU=0-style text difference proves nothing on its
own (Task F).

## L3 — The 48 hipBLASLt `Cijk_` GEMMs at 1.11 ms. ~-0.75 ms (2.3%), and the replacement is already measured.

`KERNEL_HITLIST.md` §4 attributes these, by Python stack, to `layers/utils.py rocm_unquantized_gemm_impl`
— the MoE router gate, N=512 K=2560, one per layer, **~23 us each**. K3 has already benchmarked
`r4d_gemm_bf16_nt_m64` on exactly that shape: **7.43 us flat at m = 1..5, rel err 1.7e-3**
(`k3/r4d_moe_router.json`), and `patches/utils.py:152-155` names "the router" among the four shapes it
swept, with SK=4 as the measured optimum.

But the count is unchanged at 48 across arms (1.02 ms unpatched, 1.11 ms patched), while the 292
hyper-connection calls *did* move (`wvSplitK_hf_sml_` 195 + `_big_` 97 = 292 → `r4d_gemm_bf16_nt_m64`
292). So the router is falling through to `F.linear` (`patches/utils.py:403`). The gates to look at are
`:382` (`if use_skinny:`), `:386` (`if m > 8 and 0 < n <= 5:`) and `_R4D_BF16_MIN_N=3` at `:157`.

Either the router does not satisfy `use_skinny` and never reaches `_r4d_bf16_skinny`, or the 48 calls are
a different site the base-arm attribution mislabelled. **Both are one measurement away** (M3), and if it
is the router the fix is a shape/gate entry, not a kernel. 3x on a 1.11 ms bucket is the best
effort-to-millisecond ratio on this list after L2.

## L4 — Raise `--max-num-batched-tokens`. Prefill ~1.7x, no decode cost. Measured evidence in hand.

Each prefill chunk re-pulls essentially the whole *non-resident* expert set of every layer over PCIe, so
total prefill PCIe scales with the **number of chunks**, not the number of tokens. The record already
contains the experiment, run in the wrong direction:

* `best1_h17_nbt512` — prefill **1943 tok/s** (6.44 s for 12518 tokens)
* NBT-2048 arms (`combo1`, `w4head`) — prefill **3393 / 3195 tok/s** (3.69 / 3.92 s)

Nobody has tried NBT above 2048. 2048 → 4096 should approach halving prefill PCIe. Cost is activation
VRAM plus `_mtp_hidden_buffer`, which is `max_num_batched_tokens × hc_count × hidden` bf16 =
0.039 → 0.078 GiB/rank (`VRAM_CENSUS.md` §1). VRAM is the binding constraint (`vram4_h18` and
`vram4_h17_3` both died on KV-cache sizing), so this trades against L7. One arm, prefill number only.

Corollary worth recording: **NBT=512 as a VRAM lever is a trap.** It buys hot-expert slots and costs
1.65x on prefill.

## L5 — Hide `lru_gather` behind compute. Ceiling -2.2 ms; realistic ~-1.0 ms.

`lru_gather_k` is 2.39 ms/step of pure PCIe with ~zero compute, sitting on the critical path immediately
in front of the MoE GEMM. The link is idle for the other ~29 ms of the step, so the ceiling on
overlapping it is the full 2.2 ms.

**Cross-layer prefetch does not work, and LOCALITY already contains the proof.** The lead's suggested
predictor — reuse the previous token's routing at layer L+1 — prefetches precisely the experts that are
*already resident*, because a 95.4%-hit LRU is exactly "keep what recent tokens routed". The residual
4.6% is by definition the set the previous tokens did **not** route. There is no cheap predictor here;
the Belady gap (2.60% vs 4.68%) is the whole clairvoyance premium and it is 0.7 ms/step, not the 2.2.
Do not spend time on a predictor.

**What works without any prediction is a three-way split inside the layer.** Bucket this step's rows
into (i) experts resident *before* the manager ran, (ii) experts inserted *this* step, (iii) still cold.
Issue `lru_gather` on a side stream; run GEMM(i) concurrently on the main stream; then GEMM(ii)+(iii).
No prediction, no numerics change (the GEMM is partition-invariant — C3). The fused kernel already makes
one counting pass and can emit a third bucket set for LDS, not launches; the only new cost is one extra
`_into` call per GEMM per layer = +96 nodes = **+0.23 ms**. Hideable work is GEMM(i) ≈ 22 us against a
~50 us gather, so expect **~1.0 ms net**, not 2.2. Prerequisite: L-fused must ship first (M4). Note it
competes with `VLLM_R4D_HOT_SIDE_STREAM` for the same stream, and see C10.

## L6 — Hyper-connection in fp8. ~-0.8 to -1.4 ms (3-4%) *and* 0.67 GB/rank.

3.15 ms/step across 292 calls, reading 1.33 GB/step of bf16 weights that are **replicated on both ranks**
(`disable_tp=True`, `hyperconnection.py:105`) — the one weight class TP does not discount, which is why
it is simultaneously the biggest remaining compute bucket after the MoE GEMM *and* VRAM lever #4.

The quality work is done and the answer is unambiguous: fp8 e4m3 **per-tensor** costs 2.65% relative
output error and is indistinguishable from per-channel (2.63%), while int4 costs 13-16% at *any* group
size (uniform-grid noise floor at 16 levels, not an outlier problem, so no group size rescues it) and
per-tensor int8 collapses to 6.8-10.6% on these kurtosis-26-31 tensors. Per-tensor is exactly what the
shipped `wvSplitKQ` W8A8 kernel takes.

The blocker is named and narrow: `wvSplitKQ` refuses n>4 and MTP-4 runs the target at n=5. Two unlocks
HC_QUANT did not cost out:

* **4+1 split** — two `wvSplitKQ` calls per site. +249 launches/step = +0.60 ms of dispatch against
  -1.4 ms of GEMM: net **-0.8 ms**. Ugly but it needs no new kernel and no checkpoint.
* **`torch._scaled_mm` at n=5**, which HC_QUANT itself suggests measuring first and nobody has.

Either way, measure the kernel before anyone builds a checkpoint (M7). Also note the MXFP4 escape hatch
is closed: `r4d_gemm_mxfp4a8_nt_m64` would accept both HC shapes (K%32), but 4 bits is 13-16% on a gate
that mixes the residual streams ~100 times in series. Do not.

## L7 — More slots. -0.78 ms for +2.1 GB/rank — and the GB does not exist yet.

`slot_alloc.py` E.3 prices 12207 → 14000 slots at 3.24 → 2.46 ms/step. But `vram4_h17_3` and `vram4_h18`
both died on KV-cache sizing, so this is gated behind freeing VRAM: L6 (0.67 GB), plus `VRAM_CENSUS`
lever #1 (route the MTP `nn.ModuleList` through `get_offloader().wrap_modules()`, **1.17 GB/rank**,
~5 lines, still not done — `VLLM_UVA_OFFLOAD_EMBED`/`_VISUAL` are already on in the best arm).

Two planning rules to retire, because both were sized against the pre-LRU traffic and are now wrong by
an order of magnitude:

* **"+1 GB removes ~15% of remaining cold traffic"** — that curve describes the *static* policy.
  Post-LRU the marginal GB is worth **0.4-0.8 ms/step** (E.3), and the miss curve is flat above ~12 GB
  (LOCALITY §3).
* **Pinning all 512 experts' E8M0 scales resident** (COLD_TRANSFER §3) — 6.0% of cold bytes was 1.4 ms
  when cold traffic was 431.9 MB/step. Against today's 85.6 MB/step it is **0.18 ms for 1.96 GB/rank**.
  Dead.

## L8 — Re-measure MTP depth. Unknown, plausibly +2-4%. The trade moved under you.

The MTP-2/3/4 sweep (`arms_mtp.log`) was run **before** the LRU, when the target forward carried 21 ms of
PCIe-bound MoE GEMM and every extra draft row widened it. Now the resident GEMM is bandwidth-bound in
the *weights*, which do not scale with rows, so a 5th draft position is nearly free in the target forward
and costs only the ~1 ms of one more draft iteration. JSON, at accept 0.706 and 3.83 tok/step, is the
shape that would gain most; `mtp5` died in the old sweep and deserves one retry on the current stack.

Counter-force to measure alongside: more rows ⇒ more distinct experts per layer per step ⇒ higher miss
rate (LOCALITY §5's B-scaling is the right model, 31.4 → 42.2 distinct from B=1 to B=2). Report
`lru_gather` ms as well as tok/s, or you will read a wash as a win.

## L9 — Structurally fewer launches. The only remaining 5+ ms lever, and a large project.

7.7 ms of the step is the 2.40 us/node floor, and every lever above nibbles 50-300 nodes at a time. The
only step change is fusing whole sub-blocks. The clearest target after the MoE bookkeeping is the
hyper-connection: **486 kernels/step for 3.08 ms of work**, 5 launches per HC module × 96 modules, and
`hc.py` already owns three of those kernels (`_hc_silu`, `_hc_gate_mix`, `_hc_combine_norm`). Folding
them plus the two `wvSplitK`/r4d GEMMs into one persistent kernel per module is ~-380 nodes = -0.9 ms
of dispatch on top of L6's arithmetic saving. ROCm has no device-side graph launch equivalent to
CUDA 12.4's, so per-layer megakernels are the only shape this can take. Flag, do not start yet.

## L10 — Ruled out, so nobody re-derives them

* **Stage-then-compute for cold experts**: +3.3 to +9.4% (COLD_TRANSFER §2). The closed kernel already
  hides all of its compute behind the PCIe read (`uva us == stage us` to 0.3%).
* **A cleverer eviction policy**: Belady 2.60% vs LRU 4.68% against static 23.31% — LRU already takes 86%
  of the achievable win. The entire remaining policy headroom is ~0.7 ms/step and needs clairvoyance.
* **Pinning half the slots to the profile** (`hybrid`): 24% worse than pure LRU. **LFU**: +4.06 pp
  across context switches. **Equal slots per layer**: worse than the water-fill at every budget above
  6000 (E.2). Keep MAX_INSERTS=64 and THRESH=0.5 (E.4: zero steps exceeded 64 misses at any B).
* **Faster host link**: EPYC 74F3 is Gen4-only; 28.4 GB/s is a platform constant on this box.

---

# 3. What I would measure, in priority order

| # | Measurement | Cost | Decision it enables |
|---|---|---|---|
| **M1** | Gap attribution on `prof_w4head`. `prof_gaps.py` with the `>120 us` threshold dropped to ~15 us, plus a full gap histogram, plus the same run on `…tp1…` aligned against `…tp0…`. | one CPU-minute, no GPU | Where the ~4.7 ms of non-node gap lives. TP skew → balance work / cut collectives. Draft-loop boundary → attack the speculator. Neither → L2/L9 are the only levers and the step is capped near 24 ms. **Do this first; it re-ranks everything below.** |
| **M2** | One arm: `w4head` + `VLLM_GEMMA_NORM_FUSED=2 VLLM_GDN_STRIDED_QKV=1 VLLM_FUSED_SHARED_GATE=1`, re-censused with `k4/graph_census.py`, plus a greedy-divergence gate with a base-vs-base control. | 1 arm | Whether the 2.40 us/node model holds at scale (expect ~3200 → ~2690 nodes, -1.2 ms). If it does not, L9 is not worth starting. |
| **M3** | Attribute the 48 `Cijk_Ailk_Bljk_BBS_*` calls in `prof_w4head` to a Python stack (`k4/align.py` + `attrib.py`), or add one `logger.warning_once` at `patches/utils.py:403`. | minutes | L3: is 0.75 ms available for a gate change, or is the hitlist's router attribution stale? |
| **M4** | `VLLM_R4D_LRU_FUSE=1 R4D_LRU_LIB=…/librlu_fused.so` as an arm, judged on ms/step **and** the 256K needle test **and** `spec_decode_num_accepted`. | 1 arm | Whether the ~0.44 ms (C6) is real end-to-end. Prerequisite for L5. A stale-`sorted_ids` defect produces plausible text, not a crash — do not judge this arm on tok/s alone. |
| **M5** | `k1/concurrent_bench.py` at B=1/2/4 on the current best arm, logging read-through-gate frequency and `n_miss`/step per layer. | 1 arm + harness | C7 and LOCALITY §5's only synthetic result. Whether 84/119/100 survives real concurrency, and how often chunked prefill switches the cache off. This is the largest gap between the measured number and a production claim. |
| **M6** | NBT sweep 2048 / 3072 / 4096, prefill throughput only, watching KV-cache sizing at startup. | 1-2 arms | L4, and the VRAM budget that L7 competes for. |
| **M7** | `torch._scaled_mm` and a 4+1 `wvSplitKQ` split at the four HC shapes at n=5, cold-pool timed as in `k3/bench_fp8.py`. | GPU window, no server | L6. Settles whether the 1.4 ms is reachable *before* anyone regenerates a checkpoint. |
| **M8** | MTP-5 (and 6) on the current stack, reporting tok/step, accept, `lru_gather` ms, and distinct experts/layer. | 1-2 arms | L8. The pre-LRU sweep is stale. |
| **M9** | Add the C1 NaN-poison check and a `-1` case to `test_numerics_prod.py`; add the C4 forced-`break` case to `test_fused.py`; add the C3 cross-rank `slot_expert` checksum behind a debug env. | half a day | Closes the three places where a real defect would currently be invisible. Cheap insurance before M5 changes the batch shapes the cache has ever seen. |

Two standing cautions that apply to every arm above:

* **Everything about this cache is scored on 12 prompts, single-stream.** LOCALITY, `slot_alloc`, the
  MAX_INSERTS verdict and the stamp warm-start all trace back to `routes_rank0.npz`. M5 is what converts
  them from "true of this trace" to "true of this server".
* **Greedy output on this server is not reproducible across restarts** (Task F, confirmed at the server
  level). Any arm judged on text — L2's norm patch, L6's fp8 — needs a base-vs-base control and a
  divergence gate with an actual power calculation, not a 128-token eyeball.
