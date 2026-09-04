# Documents

These were written as working notes during the session that produced the code, and they refer
to the tree that produced it, not to this repository's layout. Mapping:

| in the documents | in this repository |
|---|---|
| `k1/lru/r4d_lru.hip`, `k1/lru/build.sh` | `kernels/lru/` |
| `k1/lru/test_*.py`, `bench_*.py` | `tests/lru/` |
| `k1/routecap.py`, `locality_sim.py`, `slot_alloc.py`, `align_check.py` | `tools/routecap/` |
| `k1/moe_ref_harness.py`, `cold_transfer.py`, `cmp_dense.py` | `tests/k1/` |
| `k2/*.py`, `k3/*.py`, `k4/*.py` | `tests/k2/`, `tests/k3/`, `tests/k4/` |
| `k3/draft_w4_lmhead.py`, `patches/model_executor/layers/fused_*.py` | `kernels/` |
| `patches/<file>.py` (a modified copy) | `patches/<file>.py.diff` (a unified diff) |
| `patches/hotcold/r4d_mxfp4_moe_lru.py` | `patches/…/experts/r4d_mxfp4_moe.py.lru.diff` |
| `patches/hotcold/r4d_mxfp4_moe.py` | `patches/…/experts/r4d_mxfp4_moe.py.static.diff` |
| `patches/utils.py` | `patches/model_executor/layers/utils.py.diff` |
| `fork_vllm/<rel>` | fetched from the image by `patches/apply_patches.sh` |
| `MOUNTS_COMBO4.txt` | generated as `build/MOUNTS.txt` |
| `$REPO/`, `$HOME/` | this checkout / your home directory |
| `/w` inside a container | this checkout, bind-mounted at `/w` |

Data artifacts the documents cite — `routes_rank*.npz`, `aligned.pkl`, `tensor_headers.json`,
`prof_*/`, the `ab3_*.json` arm results — are not in the repository. Regenerate them with
`tools/routecap/routecap.py`, `tests/k2/census_headers.py` and `bench/ab3.py`.

| document | what it is |
|---|---|
| `LOCALITY.md` | the routing-locality analysis that predicted the LRU win, and the simulator results |
| `K1_PROGRESS.md` | K1's full working log: the LRU design, the victim-selection rewrite and its equivalence proof (Task I(a)), what each MTP draft row costs in cold traffic (I(b)), and why no prefetch predictor beats LRU residency (I(c)) |
| `FUSE.md` | folding the manager and both `moe_align_block_size` calls into one kernel |
| `COLD_TRANSFER.md` | measuring the PCIe/UVA cold path: what the gather costs and where its ceiling is |
| `PATCHES.md` | every patch in `patches/`: what it changes, its env gate, kernels saved, evidence |
| `KERNEL_HITLIST.md` | the per-step kernel census the patches were chosen from |
| `REVIEW.md` | a read-only review of the LRU work: correctness findings, what is left on the table |
| `HC_QUANT.md` | quantization choices for the hot/cold split (K3) |
| `RESULTS.md` | K3's skinny-GEMM and cold-path benchmark results |
| (see `kernels/experimental/fp8skinny/README.md`) | the open fp8 skinny GEMM: why it is a non-win, and the benchmarking method that established it |
| `VRAM_CENSUS.md` | where the 32 GB per card actually goes |
| `FP8HIP_KNOBS.md` | the closed `libfp8hip_gemm.so` tuning knobs found by probing |
| `GDN_GATE.md` | the gated delta-net attention path and its dispatch conditions |

**One correction.** `REVIEW.md` was written minutes before the `c4` arm ran. Its finding C4
says `VLLM_R4D_LRU_FUSE=1` "has never been in an arm" — that is no longer true; `c4`, `c5` and
`c6` all ran with the fused kernel and it is the default in `launch/launch_q38fn.sh`. The rest
of the review stands, including C1 (the numerics tests cannot detect an unwritten output row)
and C4's untested victim-search exit, both of which are still open.
