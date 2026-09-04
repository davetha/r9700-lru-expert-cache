# Fusing the MoE routing bookkeeping into one kernel

Design note before writing any code. Target: `_apply_split` in
`patches/hotcold/r4d_mxfp4_moe_lru.py`.

## 1. What is actually launched today

Per MoE layer, per forward, the *bookkeeping* (everything that is not GEMM/activation/quant):

| # | call | kernels | why that many |
|---|------|---------|----------------|
| 1 | `r4d_lru_manage` | 1 | one workgroup |
| 2 | `r4d_lru_gather` | 1 | grid (chunks, lanes) |
| 3 | `moe_align_block_size(topk_ids, 16, E, table, ignore_invalid=True)` | **2** | see below |
| 4 | `moe_align_block_size(topk_ids, 16, E, map_cold, ignore_invalid=True)` | **2** | " |
| 5 | `topk_weights.to(float32).reshape(-1).contiguous()` | **0** | already fp32 — see below |
| | | **6** | |

The 2-per-align is not obvious and is worth writing down. In
`csrc/libtorch_stable/moe/moe_align_sum_kernels.cu` the single-kernel "small batch expert"
path is gated on `(topk_ids.numel() < 1024) && (num_experts <= 64)`. We have `num_experts =
E = 512`, so **we never take it** — we always take the general path, which is
`align_kernel<<<2, 1024>>>` (block 0 counts + scans + writes `expert_ids`; block 1 fills
`sorted_ids` with the sentinel) followed by a separate
`count_and_sort_expert_tokens_kernel<<<(1, ceil(mtk/256))>>>` that scatters tokens with
`atomicAdd` on a global cumsum buffer. It also does a `new_empty` for that cumsum buffer on
every call.

**Both dtype casts are no-ops (checked, 2026-09-04).** This layer runs TP-without-EP, so the
modular kernel's prepare/finalize is `prepare_finalize/no_dp_ep.py`, whose
`topk_indices_dtype()` returns `None` (lines 48-49 and 108-109). With `indices_type=None`,
`router/fused_topk_router.py:fused_topk` (and `fused_topk_bias` in the bias router)
allocates `topk_ids` as **int32** and `topk_weights` as **float32**. So:

- `_lru_step`'s `topk_ids.reshape(-1).to(torch.int32)` returns the same tensor — there is
  **no hidden 8th launch**, nothing to remove;
- `topk_weights.to(torch.float32)` also returns `self`, `reshape(-1)` on a contiguous tensor
  is a view and `.contiguous()` is then a no-op, so item 5 costs **zero** kernels.

Today's bookkeeping is therefore **6 launches, not 7, and fusion removes 4, not 5.** (Static
reading of the installed vLLM; a one-line dtype log on the next server run would nail it.)

## 2. What fusion buys — measured, not estimated

6 -> 2 (fused bookkeeping + gather; the gather stays separate, see §4). **4 launches saved per
MoE layer.** Layer count is from the checkpoint config, not assumed: `num_hidden_layers = 48`
and `mtp.num_hidden_layers = 1`, so at MTP-4 a decode step runs 48 + 4x1 = **52** MoE layer
invocations: **208 launches of the ~3250/step**.

### Step 0 result (`step0_nodecost.py`, 2026-09-04, R9700, lock held)

The 3.72 us figure everything was being multiplied by is a *median inter-kernel gap*, not a
*marginal cost*: it says nothing about what replay time does when a node is removed. So
measure that instead. Capture

    52 x [ lru_manage, lru_gather, k x (moe_align(table), moe_align(map_cold), tw cast) ]
    + 2886 filler nodes (15.6 us elementwise kernels, interleaved)

for k = 0..3 and time replay. Everything except the k copies is identical; k=0 is the fused
ideal, k=1 is today. Node counts 2990 / 3250 / 3510 / 3770 bracket the real step. Round-robin
over k so drift hits all arms equally; spread within an arm was 0.2%.

| k | nodes | us/replay | delta | us/node |
|---|-------|-----------|-------|---------|
| 0 | 2990 | 45609 | — | — |
| 1 | 3250 | 46376 | 767 | 2.950 |
| 2 | 3510 | 47143 | 1534 | 2.951 |
| 3 | 3770 | 47914 | 2305 | 2.955 |

Dead linear: **2.95 us per node**, not 3.72. A control arm with the same node *count* but
one-element kernels in place of the real bookkeeping gives **2.40 us/node**, so of the 2.95,
**2.40 us is launch overhead and 0.55 us is the removed kernels' own execution** — and the
fused kernel still has to do that work somewhere, so only the 2.40 is truly recovered.

**Bottom line: 4 x 52 = 208 nodes x 2.95 us = 0.61 ms/step gross, of which 0.50 ms is pure
dispatch.** Expect **~0.5 ms/step** net after the fused kernel absorbs the align work — call
it **1.5% at B=1** (33.7 ms/step), 1.1% at B=2, 0.6% at B=4 (79.1 ms/step).

That is above the 0.4 ms/step bar I set for myself below, so it is worth doing — but it is a
1.5% decode win at batch 1 and less as batch grows, and it should be planned as that. The
same measurement prices *every* launch anywhere in the step at 2.4 us, which is the more
useful number: the step's ~3250 nodes carry ~7.8 ms of dispatch at 33.7 ms/step (23%),
matching the earlier profiler estimate. Any bulk launch reduction is worth the same per node.

### The original estimate, for the record
At the 3.72 us median gap and a 7-launch count the estimate was 5 x 52 x 3.72 = 0.97 ms/step.
Two independent corrections pull it down to ~0.5: the launch count is 6 not 7 (the fp32 cast
was never a kernel), and the marginal node cost is 2.4-2.95 us, not 3.72.

### The alternative that was on the table
Spending the effort on the GemmaRMSNorm guard instead (~320 kernels/step, i.e. ~0.77 ms of
dispatch at 2.4 us/node) — **already done by K4** (mode 2 fp32-exact fused path,
`patches/model_executor/layers/layernorm.py`, 3490 -> ~3200 kernels/step), so it is not an
alternative any more.

## 3. The fused kernel

One workgroup per layer, `NT = 256` (as `lru_manage_k` today), single entry, **single exit**.

**Signature** (extends `r4d_lru_manage`): existing manager args, plus `BLOCK` and, for each
of hot and cold, `sorted_ids`, `expert_ids`, `npad`. No `tw` output — §1 showed
`topk_weights` already arrives as contiguous fp32, so the caller passes it straight through.

**Phases**, `__syncthreads()` between each:

0. *manager*, byte-for-byte the current `lru_manage_k` body: `step++`, clear/mark `routed`,
   `blkSum` distinct, read-through gate, refresh stamps for hits, `blkExScan` miss
   enumeration in ascending expert id, `nins` rounds of `blkMin` over `(stamp<<20)|slot`
   skipping routed experts, update `table`/`map_cold`/`slot_expert`/`slot_stamp`/`miss`/`n_miss`.
1. *fill*: `sorted_hot[0..Lh) = mtk`, `sorted_cold[0..Lc) = mtk` (the sentinel vLLM uses is
   `numel`, i.e. `mtk` — the GEMM skips rows `>= mtk`), `expert_ids_* = -1` over the full
   `max_num_m_blocks`.
2. *count*, one pass over `i in [0, mtk)`, reading the table the manager just updated:
   `e = topk_ids[i]`; skip if out of range; `s = table[e]`; `s >= 0 ? cnt_hot[s]++ : cnt_cold[e]++`
   (LDS atomics). The two subsets are complementary by construction — that is exactly the
   invariant `test_slots_prod.py` check (c) already enforces — so one pass feeds both.
3. *scan*: exclusive scan over `E` of `ceil(cnt/BLOCK)*BLOCK`, separately for hot and cold.
   `npad_hot = cum_hot[E]`, `npad_cold = cum_cold[E]`.
4. *expert_ids*: for each bucket `j` with a nonzero count, `expert_ids[b] = j` for
   `b in [cum[j]/BLOCK, cum[j+1]/BLOCK)`. Note the hot side's "expert id" is the **slot
   index**, which is what `_gemm_split` needs to index `h["w1"]`.
5. *placement*: `sorted[rank(i)] = i`.

**LDS**: `s_se[1024]` 4 KB + `s_st[1024]` 8 KB + `cnt/cum` for hot and cold ~8 KB (cnt and cum
can alias) ~= 20 KB of the 64 KB budget.

**Placement order.** vLLM's is nondeterministic (global `atomicAdd` cursors), and order within
a block is numerically irrelevant here — `sorted_ids` holds the token index `i`, which is also
the output row, so no accumulation order changes. But determinism is cheap at decode and keeps
the "both TP ranks compute the identical thing" property the manager was built for, so:
`mtk <= 1024` -> deterministic rank (thread `t` owns token `t`, counts matching experts among
`j < t`; O(mtk) per thread, ~200 iterations at decode); `mtk > 1024` -> LDS atomic cursors.

## 4. What NOT to fuse

- **The gather stays a separate launch.** It wants grid `(chunks, lanes)` = many workgroups
  saturating PCIe; folding it into a one-workgroup kernel would cost far more in copy
  bandwidth than the one launch it saves.
- **Above `mtk > FUSE_MAX` (default 2048), fall back to today's four calls.** One workgroup
  filling 2 x 28160 int32 of sentinel at prefill (`mtk = 20480`) is ~225 KB of stores from a
  single CU. Prefill is not dispatch-bound; decode is. This mirrors the read-through gate the
  manager already has.

## 5. Risks, in order

1. **The manager has two early `return`s** (`cnt > max_distinct` read-through at r4d_lru.hip:116,
   `nins == 0` at :133) and a `break` when nothing is evictable. In the fused kernel every one
   of those must become a branch to phase 1, not a return. A missed one leaves `sorted_ids` /
   `expert_ids` / `npad` **holding the previous graph replay's values** — silently wrong output,
   no crash, and it would only show up on prefill or on a step with no misses. This is the
   defect most likely to ship. Single exit point, and the test must cover a read-through step
   and a zero-miss step explicitly.
2. **Buffer lifetime under HIP graph capture.** `sorted_ids`/`expert_ids`/`npad`/`tw` are
   `torch.empty` per call today. The fused kernel needs them preallocated and persistent.
   Allocate **per layer** (48 x ~7 KB at decode sizes), not one shared set, to avoid a
   cross-layer aliasing hazard that stream order would hide until it did not.
3. ~~`topk_ids` dtype~~ — resolved in §1: int32 already, and `topk_weights` is already fp32,
   so the fused kernel's `tw` output is a straight copy (or can be dropped entirely, since
   the existing tensor is already what the GEMM wants — one fewer output to get wrong).

## 6. Validation

Same standard as `test_slots_prod.py`, extended:

- **vs vLLM's own op, per block**: run both on the same `topk_ids`/`table`/`map_cold` and
  compare `npad` exactly, `expert_ids` exactly, and `sorted_ids` **as a set per 16-token
  block** (vLLM is not stable within a block). Padding entries must be exactly `mtk`.
- **complementarity**: every `(token, k)` slot appears in exactly one of hot/cold, and the two
  `sorted_ids` together cover `[0, mtk)` exactly once.
- **read-through and zero-miss steps** forced explicitly (risk 1).
- **end-to-end**: the existing split-vs-all-UVA bit-for-bit check with the fused kernel
  substituted for the manager + both aligns, on real routing from `routes_rank0.npz`.
- **negative control**: as with `FAULT=` in `test_slots_prod.py`, perturb one output
  (`npad` off by one block, one `expert_ids` entry) and confirm the test fails. A validation
  that has never failed has not been shown to work.


---

# Built and measured (2026-09-04)

`lru_fused_k` in `k1/lru/r4d_lru.hip`, exported as `r4d_lru_fused`, built into
**`k1/lru/librlu_fused.so`** (a separate .so on purpose: `librlu.so` is mmapped by whatever
server is running, and overwriting it under a live process is not worth the risk. The two
existing kernels are byte-for-byte unchanged, so the new .so is a superset).

Python side, in the regenerated `patches/hotcold/r4d_mxfp4_moe_lru.py`:
`VLLM_R4D_LRU_FUSE=1` plus `R4D_LRU_LIB=/w/build/kernels/librlu.so`. Default is off, and with
`VLLM_R4D_LRU_FUSE=1` against a .so that lacks the symbol the loader **raises** rather than
quietly falling back. `VLLM_R4D_LRU_FUSE_MAX` (default 2048) sends wider steps down the old
path; prefill is not dispatch-bound and one workgroup filling 2 x 28160 int32 is not free.

## Result

| arm | nodes | us/step | |
|-----|-------|---------|---|
| A today (manage + gather + 2 aligns) | 3198 | 46073 | |
| B fused (fused + gather) | 2990 | 45595 | **-478 us** |

**0.478 ms/step**, i.e. 9.2 us per layer for 4 launches removed, against a pure-dispatch
ceiling of 0.499 ms. 1.4% at B=1 (33.7 ms/step), 0.6% at B=4.

The first version returned only **0.249 ms** — the fused kernel's own execution ate half the
saving. The cost was barriers, not work: two `blkExScan` calls (the manager's Hillis-Steele
scan) are 8 rounds x 2 `__syncthreads()` each = 32 barriers, ~7 us/layer. Replaced with one
chunked two-level scan carrying both sides at once (256 partials -> 8 chunks of 32 -> 8
totals) = **3 barriers**, and the deterministic placement now reads its keys from LDS instead
of doing mk*mk global loads. That recovered the other 0.23 ms. The manager's own helpers were
left alone: they are on the live LRU path team-lead is benchmarking.

## Validation (`k1/lru/test_fused.py`, all PASS)

Every check replays the reference from the *same starting state*, since the fused kernel
rewrites the table it then reads.

- manager state (table/map_cold/slot_expert/slot_stamp/step/miss/n_miss) bit-identical to
  `r4d_lru_manage`, over 60 steps of real captured routing plus 5 synthetic geometries;
- vs **vllm's own op**: `npad` and `expert_ids` exactly, `sorted_ids` as a multiset **per
  expert**. Not per block, which is what §6 originally said and is wrong: vllm scatters with
  a global atomicAdd cursor, so when an expert spans several blocks the split of its tokens
  across them is not stable either. Per-block comparison passed at mk=50 (<= 1 block per
  expert) and failed at mk=400 for that reason alone;
- hot and cold `sorted_ids` partition [0, mk) exactly once;
- forced read-through (`max_distinct=0`) and forced zero-miss steps — the two paths where an
  early return would leave the previous replay's outputs standing;
- inside a captured HIP graph: 12 replays with changing routing match 12 eager calls in both
  outputs and cache state, and a zero-insert replay after an insert step is not stale;
- geometries mk = 10, 50, 200 (deterministic placement) and 400, 2050, 20480 (cursors);
- **end-to-end**: `FUSE=1 python3 test_numerics_prod.py` — the fused outputs drive the real
  r4d MXFP4 GEMM and the split result stays bit-identical to the all-UVA reference at
  production geometry, decode and prefill;
- negative control: perturbing `npad`, one `expert_ids` entry, or one `sorted_ids` entry each
  make the comparator fail.
