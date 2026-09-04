# Device-side LRU expert cache

Makes the contents of the existing hot-expert VRAM buffers mutable instead of fixed at load
time. Same slots, same bytes, same GEMM — the only thing that changes is which expert is in
which slot, decided on the GPU between decode steps. Rationale and the numbers that justify
it: `../LOCALITY.md`.

## Files

| | |
|---|---|
| `r4d_lru.hip` | the two kernels |
| `build.sh` | hipcc build (run it in `local/q38fn-rocm10:k1build`) |
| `librlu.so` | built artifact, gfx1201, loads in `local/q38fn-rocm10:try1` |
| `../../patches/hotcold/r4d_mxfp4_moe_lru.py` | the MoE patch (insert-only diff vs `r4d_mxfp4_moe.py`) |
| `gen_lru_py.py` | regenerates that file from the base one |

## Env knobs

| variable | default | meaning |
|---|---|---|
| `VLLM_R4D_LRU` | `0` | `1` enables the cache. `0` is byte-for-byte today's static behaviour. |
| `R4D_LRU_LIB` | — | path to `librlu.so`. Required when the cache is on. |
| `VLLM_R4D_LRU_THRESH` | `0.5` | a step routing more than this fraction of a layer's slots does no inserts and reads through instead (keeps prefill chunks and wide batches from thrashing the cache) |
| `VLLM_R4D_LRU_MAX_INSERTS` | `64` | hard cap on inserts per layer per forward; also the miss-buffer size |
| `VLLM_R4D_LRU_CHUNKS` | `8` | gather grid.x (slab split) |
| `VLLM_R4D_LRU_LANES` | `16` | gather grid.y (concurrent experts) |

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

**Determinism.** Both TP ranks see identical routing, so the caches must stay identical.
Nothing here uses atomics or any other run-to-run ordering: misses are enumerated by expert
id and victims by a total order on `(stamp, slot)`.

**Graph safety.** No host sync, no allocation, no branch on host state. Only the *contents*
of the state tensors change between replays, never a pointer, so a captured graph keeps
adapting. Verified in `test_graph_lru.py`.

## Validation

Run everything under the GPU lock, one card:

```
flock -w 3600 $REPO/gpu.lock docker run --rm --ipc host --group-add video \
  --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v $REPO_ROOT:/w \
  -v $REPO_ROOT/profiles:/hp --entrypoint bash local/q38fn-rocm10:k1build -c \
  'cd /w/tests/lru && python3 test_lru.py && python3 test_numerics_lru.py &&
   NLAYER=4 python3 test_graph_lru.py && python3 test_trace_replay.py'
```

| test | what it proves | result |
|---|---|---|
| `test_lru.py` | 6 suites, 1230 steps (churn, production shapes, insert cap, read-through gate, `-1` padding, the real captured trace) checked against an independent numpy LRU after **every** step: table, map_cold, slot_expert, slot_stamp, step counter, miss list, plus the complementarity and bijection invariants. Then 40 steps of real gathers byte-compared against the UVA source. | ALL CHECKS PASSED |
| `test_numerics_lru.py` | the real r4d grouped GEMM: (resident call + fallback call) vs one all-UVA call, 4 shapes incl. `down`, 13.3 inserts/step of churn | bit-identical every step |
| `test_graph_lru.py` | captures 4 layers x (manage + gather) into a HIP graph, replays 20x with changing routing | step counter advances 20, cache mutates, invariants hold, gathered bytes bit-identical |
| `test_trace_replay.py` | the whole 2330-step production trace x 48 layers through the real kernel with the real 15 GB hot set | static 23.33% / 432.1 MB/step -> LRU 4.62% / 85.6 MB/step (-80.2%), within 0.3% of the simulator |
| `bench_split.py`, `bench_gather.py` | cost | manager 5.91 us/layer, empty gather 3.41, both 8.98 (= 431 us/step over 48 layers); gather 25.2 GB/s at 1 expert, 28.4 at 52 |

## Known limits

* `S <= 1024` (the manager mirrors slot state in LDS). Real layers are 182-344 slots.
* Inserts beyond `MAX_INSERTS` in one step are simply left cold that step, chosen by lowest
  expert id. Correct, mildly biased, invisible at the measured 1.5 inserts/layer/step.
* The victim search is `nins` rounds of a block argmin, ~10 barriers each. Bounded by
  `MAX_INSERTS`; a burst step pays for it.
* Not measured in a live server: no needle test, no ab3, no tok/s.
