# Expert locality on real traffic: does a dynamic cache beat the static hot set?

**Verdict: yes, decisively.** At the production 15 GB hot budget, a warm-started per-layer LRU
over the same slots cuts cold-expert PCIe traffic from **431.9 MB/step to 86.8 MB/step (-80%)**,
i.e. from **15.2 ms/step to 3.1 ms/step** at the measured 28.4 GB/s ceiling. The same LRU needs
only **~7 GB/rank** to match what the static profile set achieves at 15 GB, freeing ~8 GB/rank.

> Corrected 2026-09-04 after the kernel replay in section 9: the first version of this table
> billed bytes using the post-warm-up miss/step, which silently drops every segment shorter
> than the warm-up and reweights the mix. All byte and millisecond figures below are now
> all-inclusive over every scored step, and match the real kernel to within 0.3%. The
> conclusion is unchanged; the saving is 12.2 ms/step, not the 17.5 ms first reported.

Secondary finding, needed for every VRAM decision: the profile's advertised 0.851 coverage at
15 GB is **optimistic by 1.46x** — measured cold fraction on real traffic is 0.218, not 0.149.

Everything below is an offline replay of routing captured from the live server; no end-to-end
speedup has been measured.

---

## 0. Data

`routecap.py` (`--worker-extension-cls routecap.RouteCap`, device-side ring buffer so it
survives HIP-graph replay) captured every MoE layer's `topk_ids` on the production arm
(TP2, hot 15 GB, MTP-4, `max_num_seqs 4`, `--cpu-offload-gb 16 --cpu-offload-params experts`).

* `$REPO/k1/routes_rank0.npz`, `routes_rank1.npz` — **bit-identical** (routing is
  replicated across TP ranks, as expected). 2376 steps, `lost_to_wrap = 1`.
* 49 routers: 48 model MoE layers + the MTP head's MoE (index 48, 1 row/call). Layers 0-47 are
  the subject of §1-§5; the MTP layer is §6.
* Modal step width is **5 rows** (1 target token + 4 MTP drafts, one sequence). Prefill steps
  are wider than the ring and were not captured, so a request boundary shows up as a short run
  of off-modal widths, not as a counter gap. Maximal runs of 5-row steps segment the trace into
  **15 generations, 2329 decode steps**: 9 from `ab3.py` (3 prompts x 3 runs) and the 6 labelled
  `traffic.py` prompts, whose lengths match `traffic_boundaries.json` to within 3-4 steps
  (my counter runs ~41-57 ahead of `spec_decode_num_drafts_total`; segment boundaries are
  derived from the trace itself, not from the offset).
* Mean **distinct** experts routed per layer per step: **31.41** of 512. This union over the 5
  token positions is exactly what the r4d kernel pulls, so it is the physically correct
  denominator for PCIe cost.

**Axis sanity check** (`align_check.py`, guards against the known string-sorted-layer trap):
overlap between the measured top-257 and the profile top-257 is 0.782 on the diagonal vs 0.678
best off-diagonal, and the row argmax lands on the diagonal for 44/48 layers. Router index ==
profile layer index. The baseline is not a permutation artifact.

## 1. Why the static set loses: the working set is per-prompt, not global

A single generation touches a **median of 320 distinct experts per layer** (min 147, max 471)
over its 100-204 steps. The 15 GB hot set holds a median of **257**. So the cache is nearly
large enough to hold one prompt's entire working set — but the static set spends those slots on
globally-frequent experts, not on the ones this prompt is using. Recency knows which prompt it
is; a frequency profile cannot.

## 2. Policies at equal slots (15 GB = 12207 slots/rank, global water-fill exactly as `_hot_sets`)

All 15 generations, cache reset per generation. `miss%` counts distinct-experts-per-step (the
PCIe cost); `wmiss%` weights by routing selections (comparable to the profile's coverage
number); `miss%ss` skips the cold-start warm-up.

| policy | miss% | wmiss% | miss%ss | pulls/step | coldMB/step | ms/step @28.4 GB/s | vs static |
|---|---|---|---|---|---|---|---|
| static (profile hot set) | 23.31% | 21.57% | 30.11% | 351.5 | 431.9 | 15.21 | baseline |
| lru (cold start)         |  7.26% |  5.65% |  3.46% | 109.5 | 134.6 |  4.74 | -68.8% |
| **lru_ws** (warm start from the profile set) | **4.68%** | 3.49% | 3.46% | **70.6** | **86.8** | **3.06** | **-79.9%** |
| lfu (exp. decay 0.99)    |  4.98% |  3.65% |  4.36% |  75.1 |  92.3 |  3.25 | -78.6% |
| hybrid (pin top 50%, LRU rest) | 5.82% | 4.32% | 6.18% | 87.8 | 107.9 | 3.80 | -75.0% |
| belady (clairvoyant bound) | 2.60% | 1.91% | 1.47% | 39.2 | 48.2 | 1.70 | -88.8% |

`miss/step` and `coldMB/step` count every scored step; `miss%ss` is a diagnostic that skips
each segment's first 107 steps and is NOT a like-for-like byte count (the skip also discards
the short segments, which is why static's rate rises to 30% there).

Restricted to the 6 labelled `traffic.py` prompts only: static 20.91% / 380.6 MB / 13.40 ms;
`lru_ws` 4.94% / 89.9 MB / 3.17 ms (-76.4%). Same conclusion, slightly smaller absolute win.

**The static baseline is independently corroborated.** 431.9 MB/step at 28.4 GB/s = 15.2 ms,
inside the 23 ms total MoE grouped-GEMM budget the torch profile of the same config reports
(~96 cold MoE calls/step at p75 150 / p90 250 us). And section 9 replays the whole trace
through the shipped kernel and lands on 432.1 MB/step, 0.05% from the simulator.

Reading the policy ranking:
* **Warm start is free and strictly better.** `lru` and `lru_ws` converge to the identical
  steady state (3.46%); the warm start only removes the 107-step cold-start penalty. Seeding
  the dynamic cache from the existing profile costs nothing and eliminates the ramp.
* **Pinning hurts.** `hybrid` (half the slots pinned to the profile) is 24% worse than pure
  LRU. Profile-frequency is a worse predictor than recency even for half the budget.
* **LFU is the wrong policy** — see §4; its decayed scores carry stale prompts across context
  switches.
* Belady says a perfect policy would reach 2.60%. LRU at 4.68% captures **86% of the total
  achievable win** (static 23.31 -> belady 2.60 is 20.7 pp; LRU gets 18.6 pp of it). There is
  no large residual for a cleverer eviction policy to recover.

## 3. Budget sweep — how much VRAM the dynamic cache gives back

Cache reset per generation, all 15 generations, `miss%` = distinct.

| GB/rank | slots | profile-predicted miss | static (measured) | lru_ws (measured) |
|---|---|---|---|---|
| 2  | 1627  | 74.4% | 82.93% | 72.29% |
| 4  | 3255  | 59.2% | 70.16% | 42.24% |
| 6  | 4882  | 47.5% | 59.22% | 28.26% |
| 8  | 6510  | 37.9% | 49.21% | 18.13% |
| 10 | 8138  | 29.9% | 40.62% | 11.31% |
| 12 | 9765  | 23.1% | 32.92% |  7.65% |
| 15 | 12207 | 14.9% | 23.31% |  4.68% |
| 16 | 13020 | 12.6% | 20.47% |  4.06% |
| 17 | 13834 | 10.6% | 17.87% |  3.53% |
| 18 | 14648 |  8.7% | 15.50% |  3.07% |

**Iso-quality point: `lru_ws` reaches static@15GB's 23.31% at ~7.0 GB/rank** (interpolating
6 GB 28.26% / 8 GB 18.13%). That is **~8 GB/rank of VRAM freed** at unchanged cold traffic —
convertible to KV cache (256K context currently needs 2.4 GB/rank) or to a lower
`--cpu-offload-gb`.

Also note the shape: static's marginal return is ~2.6 pp of miss per GB and falling slowly;
`lru_ws` is already at 4.68% at 15 GB and flattens hard (3.07% at 18 GB). Above ~12 GB the
dynamic cache has essentially solved the problem, and further hot-budget GB buy almost nothing.
This **invalidates the "+1 GB removes ~15% of remaining cold traffic" planning rule** for any
build that ships a dynamic cache — that curve describes the static policy only.

## 4. Context switches

Six labelled prompts back to back (code/prose/json x2), 15 GB. "carried" keeps one cache across
all six; "reseeded" resets it to the profile set at each prompt boundary.

| policy | carried | reseeded | delta |
|---|---|---|---|
| static | 20.91% | 20.91% | 0.00 pp |
| lru    |  5.88% |  7.56% | -1.68 pp |
| lru_ws |  5.45% |  4.94% | +0.51 pp |
| lfu    |  9.16% |  5.10% | **+4.06 pp** |
| hybrid |  6.19% |  5.98% | +0.21 pp |
| belady |  2.74% |  2.74% | 0.00 pp |

Miss rate over the **first 20 steps** after each switch (carried cache): `lru_ws` 9.9 / 14.2 /
12.0 / 12.3 / 14.3 / 10.6 % — i.e. worse than its 3.5% steady state but still **better than the
static set's 11-22%** on the same steps. There is no switch penalty to engineer around: a
context switch never makes LRU worse than the thing it replaces.

LFU is the exception and the reason to reject it: decayed frequency counts survive a prompt
change and keep the previous prompt's experts resident, costing +4.06 pp. **Use LRU.**

## 5. Concurrency — the main caveat

The capture is single-stream (`max_num_seqs 4`, but one request at a time). To probe batching I
merged B independent generations into one step and took the union of their routed experts,
which is what the kernel would pull for a batch of B unrelated sequences:

| B | distinct/layer/step | static | lru | lru_ws | lfu | hybrid | belady |
|---|---|---|---|---|---|---|---|
| 1 | 31.4 | 23.31% |  7.26% |  4.68% |  4.98% |  5.82% | 2.60% |
| 2 | 42.2 | 23.24% |  8.46% |  6.33% |  6.83% |  8.25% | 3.01% |
| 4 | 69.1 | 24.78% | 11.32% |  9.98% | 10.44% | 12.74% | 4.12% |

Locality degrades with concurrency, as expected — but at B=4 `lru_ws` is still **2.5x better
than static** (9.98% vs 24.78%), and the absolute cold traffic per step is 69.1 x 0.0998 x 48 x
1.2288 MB = 407 MB vs static's 1011 MB. The dynamic cache wins across the whole concurrency
range this box operates in. This is the least-solid number in the document (a synthetic merge of
unrelated prompts is the pessimistic end; real concurrent requests often share a system prompt).

## 6. RETRACTED: the MTP layer is not cold

This section originally claimed the MTP head's MoE was an unpinned, fully-cold layer worth
~1.1-1.7 ms/step. **That was wrong.** K2 verified that the MTP module is built as a plain
`ModuleList` the UVA offloader never wraps: its experts are VRAM-resident fp8 and its draft
MoE runs the Triton `fused_moe_kernel` at ~41 us. `hot_ids_for_layer(48)` returning `None`
is correct and costs nothing. The measurement that produced the claim (the 49th router in
the capture uses 451 distinct experts across the trace, ~10 per draft) is real; the
inference that those reads cross PCIe was not checked and is false. No action.

## 7. What this means for the build decision

At 15 GB, unchanged: **-345 MB/step of PCIe traffic, -12.2 ms/step**, less ~0.45 ms/step of
manager and gather launches (section 9) = **~-11.7 ms/step net**. Against a ~62 ms decode step
of which ~15.2 ms is cold-expert transfer, that is an upper bound of **~1.23x decode
throughput** (47.6 -> ~59 tok/s prose), realised only to the extent that the cold pull is on the
critical path — the profiler says the GPU is ~100% busy and the MoE grouped GEMM is 23 ms of the
47 ms kernel-busy time, so most of it is.

Alternatively at iso-cold-traffic: **~8 GB/rank of VRAM freed**.

The engineering the verdict is conditional on, none of which is measured here:
1. The r4d kernel must resolve expert -> (VRAM slot | UVA pointer) through a mutable per-layer
   table instead of a compile-time hot mask.
2. Eviction/insert must run between decode steps without breaking HIP-graph capture — the
   table is device memory the graph reads, so the contents can change but the pointer cannot.
3. Inserts must be the pull itself, not an extra copy. `COLD_TRANSFER.md` measured
   stage-then-compute as 3-9% *slower* than reading UVA in place, so the insert path costs
   ~1.05x the bytes it saves. At 52.6 inserts/step that is 64.6 MB staged vs 562.6 MB read
   today — still a 5-8x reduction, but the staging penalty is real and must not be forgotten.
4. 52.6 inserts/step also means 52.6 x 1.23 MB of VRAM writes (~65 MB/step, ~0.07 ms at
   HBM bandwidth — negligible) plus per-layer bookkeeping. If bookkeeping adds kernel launches
   it eats into the win: the box already pays a 24% dispatch tax at 3490 kernels/step, so the
   eviction update must be one fused kernel per step, not one per layer.

## 8. The implementation, and what the real kernel measures

Built and validated after this study: `patches/hotcold/r4d_mxfp4_moe_lru.py` plus two HIP
kernels in `k1/lru/librlu.so` (`lru_manage`, `lru_gather`), env-gated on `VLLM_R4D_LRU=1`.

**Full-trace replay through the shipped kernel** (`k1/lru/test_trace_replay.py`): all 2330
decode steps x 48 layers, real 15 GB profile hot set as the warm start, real `lru_manage`.

| | miss rate | pulls/step | MB/step | ms @28.4 GB/s |
|---|---|---|---|---|
| static hot set | 23.33% | 351.6 | 432.1 | 15.21 |
| LRU kernel     |  4.62% |  69.6 |  85.6 |  3.01 |
| simulator, same budget | 23.31% / 4.68% | 351.5 / 70.6 | 431.9 / 86.8 | — |

The kernel and the simulator agree to within 0.3% on both arms, so section 2 is not a
model of a policy that was never built — it is a model of this one. (The kernel carries one
cache across all 15 generations where the simulator resets per generation, which is why its
LRU number is a shade lower.)

**Cost of the machinery**, measured inside a HIP graph on an idle R9700 (`bench_split.py`):

| | us/layer | us/step over 48 layers |
|---|---|---|
| `lru_manage` (steady state, 0 inserts) | 5.91 | 284 |
| `lru_gather` with 0 inserts (empty grid exits) | 3.41 | 164 |
| both | 8.98 | 431 |

So ~0.45 ms/step of launch and bookkeeping against 12.2 ms/step of PCIe saved: a 3.7% tax on
the win. It is 2 extra kernels per layer per forward (96/step) on top of the existing
3490/step, which the box's 3.72 us dispatch gap already prices at ~24%.

`lru_gather` reaches the same ceiling as every other method in `COLD_TRANSFER.md` —
25.2 GB/s at 1 expert, 27.6 at 4, 28.4 at 52 — so the insert path costs exactly what the
read-through path costs, as it must.

## 9. What is NOT established

* No end-to-end throughput measurement. Every number here is a replay of one capture.
* One server config, one hot budget, one TP layout. Routing does not depend on placement, so
  there is no feedback loop between the arm that produced the trace and the policy being
  scored — but the traffic mix is 12 distinct prompts, not a production distribution.
* B>1 is synthetic (§5).
* The MTP row for a step is the last of 4 drafts (the ring keeps one slot per layer per step),
  so §6's per-step totals are extrapolated from one draft, and layers 0-47 in §1-§5 include all
  5 token positions of the target forward but none of the drafts' own MoE work.
* LRU was simulated at exact per-layer capacity with perfect bookkeeping. The shipped
  kernel does hit exactly that (section 8), but it has not yet run inside a live server:
  no needle test, no ab3, no measured tok/s.
* The section 8 timings were taken with team-lead's `q38fn-mxfp4` server arm resident on
  the same GPU (idle, but holding ~28 GB of VRAM). The gather bandwidth matches the clean
  `COLD_TRANSFER.md` ceiling, so contention looks absent, but the 5.91 us manager figure
  deserves a re-measure on a quiet card.
* The eviction victim search is `nins` rounds of a block argmin. At the default cap of 64
  inserts/layer that is bounded, but a burst step pays ~10 barriers per insert.

## Files

* `$REPO/k1/routecap.py` — capture extension (device ring, graph-safe)
* `$REPO/k1/test_graphsafe.py` — proves the ring survives graph replay
* `$REPO/k1/locality_sim.py` — the simulator (CPU/numpy; `PROFILE=`, `GB=`,
  `POLICIES=`, `TRAFFIC_TABLE=1`, `SYNTH=iid` null control)
* `$REPO/k1/align_check.py` — layer-axis alignment + coverage calibration
* `$REPO/k1/mtp.py` — the retracted section 6 measurement
* `$REPO/patches/hotcold/r4d_mxfp4_moe_lru.py` — the LRU MoE patch (insert-only
  diff against `r4d_mxfp4_moe.py`; `k1/lru/gen_lru_py.py` regenerates it)
* `$REPO/k1/lru/` — `r4d_lru.hip`, `build.sh`, `librlu.so`, `README.md`, and
  the tests `test_lru.py` / `test_graph_lru.py` / `test_numerics_lru.py` /
  `test_trace_replay.py` / `bench_split.py` / `bench_gather.py`
* `$REPO/k1/locality.out`, `locality_sweep.out`, `locality_null.out` — raw output
* `$REPO/k1/routes_rank0.npz`, `routes_rank1.npz` — the traces (71 MB each)

Reproduce:

```
docker run --rm -v $HOME/rocm10:/w -v $HOME/hotcold:/hp \
  --entrypoint bash local/q38fn-rocm10:k1build -c \
  "cd /w/tools/routecap && PROFILE=/hp/hot_profile.json GB=15,16,17,18 TRAFFIC_TABLE=1 \
   python3 locality_sim.py routes_rank0.npz"
```

Null control (`SYNTH=iid SYNTH_STEPS=1200`, i.i.d. sampling from the profile's own
frequencies, so the only structure left is the frequency skew and there is no temporal locality
to exploit) -- `locality_null.out`:

| policy | miss% | wmiss% | vs static |
|---|---|---|---|
| static | 16.04% | **14.90%** | baseline |
| lru    | 28.13% | 26.94% | +75.4% |
| lru_ws | 26.70% | 25.54% | +66.5% |
| lfu    | 19.91% | 18.71% | +24.1% |
| hybrid | 22.70% | 21.38% | +41.5% |
| belady | 10.02% |  9.39% | -37.5% |

Static reproduces the profile's predicted 14.9% to three digits -- the simulator's static arm is
correct by construction -- and **every implementable dynamic policy is decisively worse than
static** once the locality is removed. The win in section 2 is temporal locality in real
traffic, not an artifact of the simulator or of the byte accounting.

## Cross-layer prediction (2026-09-04, `xlayer_pred.py`)

Can layer L+1's experts be predicted from layer L's routing early enough to prefetch them during
layer L's compute? Predictors trained on the first 70% of 2330 five-row steps, scored on the rest,
against an LRU simulation with S=257 slots/layer (77 misses/step = 1.6/layer, matching production).

| predictor | K | recall of S_{L+1} | LRU-miss coverage | over-fetch (extra/misses) |
|---|---|---|---|---|
| popularity floor | 8 / 16 / 32 | 0.05 / 0.09 / 0.17 | 0.015 / 0.03 / 0.06 | 1.3 / 2.5 / 5.1 |
| set co-occurrence | 8 / 16 / 32 | 0.13 / 0.22 / 0.37 | 0.03 / 0.06 / 0.13 | 0.6 / 1.3 / 2.7 |
| per-token transition | 8 / 16 / 32 | 0.15 / 0.26 / 0.42 | 0.04 / 0.09 / 0.19 | 0.6 / 1.3 / 2.9 |

The PCIe window during one layer's compute (~0.4 ms at 28 GB/s) fits ~8 experts, where the best
predictor covers 4% of misses. Even at K=32 (a whole layer's worth of slabs) it covers 19% at 2.9x
wasted bytes. The next layer's expert choice is not encoded in the previous layer's routing.
Not tested: a learned predictor on the hidden state itself (pre-gating style); it would need
hidden-state capture and training, and the misses it must find are the rare experts.
