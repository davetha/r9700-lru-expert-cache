# Device-side LRU expert cache

Makes the contents of the existing hot-expert VRAM buffers mutable instead of fixed at load
time. Same slots, same bytes, same GEMM — the only thing that changes is which expert is in
which slot, decided on the GPU between decode steps. Rationale and the numbers that justify
it: `../LOCALITY.md`.

## Files

`r4d_lru.hip` is the **only** source. One `build.sh` run produces all three entry points
(`r4d_lru_manage`, `r4d_lru_gather`, `r4d_lru_fused`); the build fails loudly if any of them
is missing from the result.

| | |
|---|---|
| `r4d_lru.hip` | the kernels — manager, gather, and the fused manager+align |
| `build.sh` | hipcc build (run it in `local/q38fn-rocm10:k1build`) |
| **`librlu_v2.so`** | **the current artifact** — gfx1201, loads in `local/q38fn-rocm10:try1`, and what `launch_q38fn_lru.sh` points at |
| `librlu.so`, `librlu_fused.so` | superseded builds, kept only because running servers still have them mapped. Do **not** rebuild either in place. They are older than `r4d_lru.hip` and must not be shipped. |
| `r4d_lru.hip.pre-fuse`, `r4d_lru.hip.pre-victim` | historical snapshots of the source, for A/B against the old serial victim loop. Not built by anything. |
| `../../patches/hotcold/r4d_mxfp4_moe_lru.py` | the MoE patch (insert-only diff vs `r4d_mxfp4_moe.py`) — the single live copy; there is deliberately no duplicate in this directory |
| `gen_lru_py.py` | regenerates that file from the base one |
| `FUSE.md` | the fused kernel: what it merges, why, and its measured cost |

## Env knobs

| variable | default | meaning |
|---|---|---|
| `VLLM_R4D_LRU` | `0` | `1` enables the cache. `0` is byte-for-byte today's static behaviour. |
| `R4D_LRU_LIB` | — | path to the `.so`. Required when the cache is on. Use `librlu_v2.so`. |
| `VLLM_R4D_LRU_THRESH` | `0.5` | a step routing more than this fraction of a layer's slots does no inserts and reads through instead (keeps prefill chunks and wide batches from thrashing the cache) |
| `VLLM_R4D_LRU_MAX_INSERTS` | `64` | hard cap on inserts per layer per forward; also the miss-buffer size |
| `VLLM_R4D_LRU_CHUNKS` | `8` | gather grid.x (slab split) |
| `VLLM_R4D_LRU_LANES` | `16` | gather grid.y (concurrent experts) |
| `VLLM_R4D_LRU_FUSE` | `0` | `1` runs one kernel for the manager **and** both `moe_align_block_size` outputs: 6 launches per layer instead of 8. See `FUSE.md`. |
| `VLLM_R4D_LRU_FUSE_MAX` | `2048` | above this `mtk` the fused path is skipped and the split path runs (prefill) |

`VLLM_R4D_HOT_PROFILE` and `VLLM_R4D_HOT_GB` still choose the slot budget and the warm
start, exactly as before.

Startup lines to look for:

```
r4d LRU expert cache: ON (lib ..., thresh 0.50, max_inserts 64, grid 8x16)
r4d LRU: layer 0 -> 257 slots warm-started from the profile hot set, read-through above 128 distinct experts/step
```

## Design

Per layer, allocated in `_build_hot` and never reallocated (so every pointer a HIP graph
captures stays valid):

* `table[E] int32` — expert -> slot, or -1. **This is the existing `map_hot` tensor**, now
  mutable, so `moe_align_block_size` needs no change.
* `map_cold[E] int32` — expert -> itself when not resident, else -1. The existing tensor.
* `slot_expert[S] int32`, `slot_stamp[S] int64`, `step[1] int64`, `routed[E] uint8`,
  `miss[cap,2] int32`, `n_miss[1] int32`. ~10 KB/layer at E=512, S=257.

Per forward, ahead of the two `moe_align_block_size` calls:

1. **`lru_manage`** (one workgroup). Marks the distinct experts routed this step; if there
   are more than `S*THRESH` of them it returns having changed nothing. Otherwise it
   refreshes the stamp of every resident routed expert, enumerates the misses **in expert-id
   order** via a two-level scan, and for each one takes the slot with the smallest
   `(stamp, slot index)` **among slots whose expert is not routed this step** — so nothing
   needed by the step in flight, including a slot filled a moment ago, is ever evicted. It
   rewrites `table`/`map_cold` and emits the `(expert, slot)` miss list.
2. **`lru_gather`**. Grid `(CHUNKS, LANES)`; blocks past `n_miss` exit. Copies the six
   per-expert buffers (`w1`, `w2` and their `ws_t`/`wref` scale rows) from the UVA host
   tensors into the new slot, 16 bytes per lane.

Then the unchanged flow: `moe_align(topk_ids, table)` for the resident call and
`moe_align(topk_ids, map_cold)` for the fallback call. Both calls are kept, so a
read-through step still works and prefill is unaffected.

`lru_fused` (`VLLM_R4D_LRU_FUSE=1`) does step 1 and both `moe_align` outputs in one launch.
Same state, same results; `FUSE.md` has the details.

### Victim selection

The original manager picked victims one at a time: `nins` rounds of a block argmin over all
`S` slots, ~10 `__syncthreads()` each. Correct, but it cost ~1 us per insert and — worse —
fell off a cliff past 13 inserts in one step (see Known limits).

The current kernel picks them all at once, on this equivalence: the serial loop's output is
exactly the `nins` smallest victim keys of the **initially** evictable set, in ascending
order. Installing an expert into a slot makes that slot non-evictable and changes no other
slot's key, so the set only shrinks by the slots already chosen. So each evictable slot can
just rank itself — `rank = #{j : key[j] < key[i]}` against the mirrored key array — and any
slot with `rank < nins` writes itself to `s_vic[rank]`. One barrier, then every victim is
installed in parallel (victims are distinct and their experts are distinct, and an evicted
expert is never routed while the inserted one is, so no two threads touch the same entry).

Keys are `(stamp << SLOT_BITS) | slot`, which is a total order, so both TP ranks still make
identical choices. Non-evictable slots park at `LLONG_MAX`.

The ranking pass is flat in `nins` but costs ~3.7 us at `S=257` regardless (256 threads each
reading `S` LDS words), so it is a loss for the small insert counts that dominate. The kernel
is therefore a **hybrid**: `nins <= NSER` (4) takes the old serial argmin, above that it
ranks. That is the whole `nins` range in one shape, with no cliff:

| inserts | 0 | 1 | 2 | 4 | 8 | 13 | 14 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| us/call, new | 6.97 | 8.61 | 9.21 | 10.57 | 11.14 | 11.12 | 11.05 | 11.92 | 12.97 |
| us/call, old | 6.97 | 8.67 | 9.50 | 11.39 | 15.9 | 18.6 | **43.4** | 86 | 225 |

**Determinism.** Both TP ranks see identical routing, so the caches must stay identical.
Nothing here uses atomics or any other run-to-run ordering: misses are enumerated by expert
id and victims by a total order on `(stamp, slot)`.

**Graph safety.** No host sync, no allocation, no branch on host state. Only the *contents*
of the state tensors change between replays, never a pointer, so a captured graph keeps
adapting. Verified in `test_graph_lru.py`.

## Validation

Run everything under the GPU lock, one card:

```
flock -w 3600 <repo>/gpu.lock docker run --rm --ipc host --group-add video \
  --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v "$REPO_ROOT":/w \
  -v "$PROFILE_DIR":/hp --entrypoint bash local/q38fn-rocm10:k1build -c \
  'cd /w/k1/lru && python3 test_lru.py && python3 test_numerics_lru.py &&
   NLAYER=4 python3 test_graph_lru.py && python3 test_trace_replay.py &&
   python3 test_fused.py && python3 test_victim_equiv.py'
```

| test | what it proves | result |
|---|---|---|
| `test_lru.py` | 6 suites, 1230 steps (churn, production shapes, insert cap, read-through gate, `-1` padding, the real captured trace) checked against an independent numpy LRU after **every** step: table, map_cold, slot_expert, slot_stamp, step counter, miss list, plus the complementarity and bijection invariants. Then 40 steps of real gathers byte-compared against the UVA source. | ALL CHECKS PASSED |
| `test_numerics_lru.py` | the real r4d grouped GEMM: (resident call + fallback call) vs one all-UVA call, 4 shapes incl. `down`, 13.3 inserts/step of churn | bit-identical every step |
| `test_graph_lru.py` | captures 4 layers x (manage + gather) into a HIP graph, replays 20x with changing routing | step counter advances 20, cache mutates, invariants hold, gathered bytes bit-identical |
| `test_trace_replay.py` | the whole 2330-step production trace x 48 layers through the real kernel with the real 15 GB hot set | static 23.33% / 432.1 MB/step -> LRU 4.62% / 85.6 MB/step (-80.2%), within 0.3% of the simulator |
| `test_fused.py` | the fused kernel against manager + two `moe_align_block_size` calls | identical state and identical align outputs |
| `test_victim_equiv.py` | the new victim selection against the old serial loop (`librlu_fused.so`), 19 cases x both kernels: production B=1/B=4, at and past the old cliff, zero misses, both hybrid branches, capped by `max_inserts`, empty slots, stamp ties, evict-all, `S=1024`, `maxi=1`/`1024`. Full state compared exactly, plus a 3-perturbation negative control. | VICTIM EQUIVALENCE PASSED |
| `bench_split.py`, `bench_gather.py`, `bench_fused.py`, `bench_victim.py` | cost | manager 5.91 us/layer, empty gather 3.41, both 8.98 (= 431 us/step over 48 layers); gather 25.2 GB/s at 1 expert, 28.4 at 52; fused 0.466 ms/step |

Timing benchmarks are only valid while holding `<repo>/gpu.lock`, and eager
launches are CPU-bound (~47 us) and hide the kernel entirely — `bench_victim.py` captures a
HIP graph of `REP` iterations and takes the min, and any new benchmark here must do the same.

## Known limits

* `S <= 1024` (the manager mirrors slot state in LDS). Real layers are 182-344 slots.
* Inserts beyond `MAX_INSERTS` in one step are simply left cold that step, chosen by lowest
  expert id. Correct, mildly biased, invisible at the measured 1.5 inserts/layer/step.
* **The 14-insert cliff in the old serial victim loop was never root-caused — it was routed
  around.** In `r4d_lru.hip.pre-victim` / `librlu_fused.so`, cost per call jumps from 18.6 us
  at 13 inserts to 43.4 at 14, and keeps climbing (225 us at 64). It is not a smooth
  per-insert cost: it is a step. Measured facts, none of which explained it: the threshold is
  `k=14` for `S=257` and `S=320`, `k=16` for `S=200`, and absent at `S=129` up to 16; it does
  not move with `E` (2048 and 4096 both cliff at 14); and it is not a kernel-duration
  threshold. The current kernel's ranking path is flat across the whole range, so the cliff
  cannot be reached — but if anyone reinstates a serial victim loop, or raises `NSER` above
  ~13, it comes back. The old loop is preserved in the two snapshots above for A/B.
* At the production insert rate (~1.5/layer/step at B=1, 5-13 at B=4) the new kernel and the
  old one are within noise; the win is entirely in the tail. Live A/B (arm c8) measured
  B=1 91.8/124.1/106.2 tok/s — same as the previous arm, as expected — and B=4 165.7 agg.
