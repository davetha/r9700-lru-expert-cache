# Tests

Everything here runs inside the container. Two images are used:

* `local/q38fn-rocm10:try1` — the runtime image (`docker/Dockerfile`). Enough to `dlopen` a
  prebuilt `.so`; no compiler.
* `local/q38fn-rocm10:build` — the same plus g++ (`docker/Dockerfile.build`). Needed to build
  the kernels, and used for the kernel tests because some of them compile.

Mount this checkout at `/w`. The tests expect:

| path in container | is |
|---|---|
| `/w` | this checkout |
| `/w/build/kernels/librlu.so` | built by `kernels/lru/build.sh` |
| `/w/profiles/hot_profile.json` | shipped |
| `/w/artifacts/routes_rank0.npz` | **not shipped** — capture it with `tools/routecap/routecap.py` |

Override any of them with the env var each script documents in its docstring
(`R4D_LRU_LIB`, `TRACE`, `PROFILE`, `GATHER_SO`, ...).

## The GPU lock

The tests take a whole GPU. Serialize them against anything else on the box with a lock file
in the checkout:

```bash
flock -w 3600 gpu.lock docker run --rm --ipc host --group-add video \
  --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 \
  -v "$PWD:/w" --entrypoint bash local/q38fn-rocm10:build -c 'cd /w/tests/lru && python3 test_lru.py'
```

`HIP_VISIBLE_DEVICES=1` is one card; the kernel tests are single-GPU. Note that `docker rm -f`
returns before VRAM is released — wait for `mem_info_vram_used` to drop before the next run.

## What each test needs

### `tests/lru/` — the LRU expert cache

| file | GPU | needs | what it proves |
|---|---|---|---|
| `test_lru.py` | yes | `librlu.so` | 6 suites, 1230 steps, checked against an independent numpy LRU after **every** step: table, `map_cold`, `slot_expert`, `slot_stamp`, step counter, miss list, complementarity and bijection. Then 40 steps of real gathers byte-compared against the UVA source. |
| `test_numerics_lru.py` | yes | `librlu.so` | the real r4d grouped GEMM: (resident call + fallback call) vs one all-UVA call, 4 shapes, 13.3 inserts/step of churn. Expect bit-identical every step. |
| `test_numerics_prod.py` | yes | `librlu.so` | the same at production shapes |
| `test_slots_prod.py` | yes | `librlu.so`, a routing capture | production slot counts against a real trace |
| `test_graph_lru.py` | yes | `librlu.so` | captures 4 layers x (manage + gather) into a HIP graph, replays 20x with changing routing. `NLAYER=4` keeps it quick. |
| `test_trace_replay.py` | yes | `librlu.so`, a routing capture | the full 2330-step trace x 48 layers through the real kernel: static 23.3% / 432 MB/step -> LRU 4.6% / 86 MB/step |
| `test_fused.py` | yes | `librlu.so` | `lru_fused_k` against the two-kernel path, including the read-through and zero-miss exits |
| `test_victim_equiv.py` | yes | **two** libraries | compares the v2 batched victim ranking against the older serial argmin, byte for byte over `table` / `map_cold` / `slot_expert` / `slot_stamp` / `miss` / `n_miss`, for both `r4d_lru_manage` and `r4d_lru_fused`, across 19 cases plus a 3-perturbation negative control. Build both sources first (see below) and pass `OLD_LIB` / `NEW_LIB`. |
| `bench_split.py`, `bench_gather.py`, `bench_fused.py`, `bench_victim.py` | yes | `librlu.so` | cost. Manager 5.91 us/layer, empty gather 3.41, both 8.98; gather 25.2 GB/s at 1 expert, 28.4 at 52. |
| `step0_nodecost.py` | yes | `librlu.so` | the cost of a step that inserts nothing |

To build the older kernel that `test_victim_equiv.py` compares against:

```bash
docker run --rm -v "$PWD:/repo" --entrypoint bash local/q38fn-rocm10:build -c \
  'mkdir -p /repo/build/kernels; SDK=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core; /opt/rocm/bin/hipcc -O3 \
   -std=c++17 -fPIC --offload-arch=gfx1201 --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode \
   -shared /repo/kernels/lru/r4d_lru_pre_victim.hip -o /repo/build/kernels/librlu_old.so'
# then, inside the test container:
OLD_LIB=/w/build/kernels/librlu_old.so NEW_LIB=/w/build/kernels/librlu.so python3 test_victim_equiv.py
```

### `tests/k1/` — the MoE ABI and the cold path

`moe_ref_harness.py` (reference implementation of the split against the real r4d GEMM),
`cold_transfer.py` (PCIe gather bandwidth; wants `cold_gather.so`, build it the same way as
`librlu.so` from `kernels/lru/cold_gather.hip`), `cmp_dense.py` (open libr4d vs the image's
closed `r4d.so` — needs a libr4d build, see the README), `test_edge_dense.py`,
`test_graphsafe.py`, `probe_peak.py`, `probe_trace.py`. All GPU.

### `tests/k2/` — VRAM census and the closed-kernel gates

`smoke_k2.py` (GPU) checks the UVA-offload and GDN gates still import and dispatch.
`census_headers.py` writes `tensor_headers.json` from a checkpoint (no GPU, but wants the
weights); `census_compute.py` reduces it (no GPU).

### `tests/k3/` — skinny GEMM, quantization, the W4 draft head

`verify_r4d.py`, `w4_verify.py`, `w4_agree.py`, `w4_confirm.py`, `w4_quant_err.py` are
correctness; `bench_skinny.py`, `bench_cold.py`, `bench_prod.py`, `bench_r4d.py`,
`bench_fp8.py`, `bench_int4.py`, `w4_bench.py` are timing. All GPU, all single-card.
`shapes.py` / `shapes2.py` / `block.py` are shape helpers the others import.

The `*_fp8sk` / `*_roof` / `test_layout` / `smoke_patch` harnesses belong to the experimental
fp8 skinny GEMM (`kernels/experimental/fp8skinny/`). They read two libraries by env var:
`FP8SK_LIB` (default `/w/build/kernels/libhcqfp8sk.so`, built by that directory's `build.sh`)
and `FP8HIP_LIB` (default `/app/fp8hip/libfp8hip_gemm.so`, the closed kernel inside the fork
image — not shipped, and only reachable from within a container built on that image).
`test_layout.py` is the exception: it proves the weight-shuffle layout on CPU and needs
neither a GPU nor either library.

### `tests/k4/` — the kernel-count patches

`smoke_imports.py` is the one to run first: it imports every patched module with the gates on
and off and fails loud on an ImportError, which is the failure that once produced a silent r4d
fallback. `test_gdn_strided.py` and `test_gate_mul.py` check the two bit-identical fusions
against the stock path. `gemma_*.py` are the GemmaRMSNorm numerics study behind
`VLLM_GEMMA_NORM_FUSED` — `gemma_real.py` and `gemma_numerics.py` are what produced the
perturbation table in `docs/PATCHES.md`. `probe_silu_quant.py` and `rope_kernels.py` are kernel
counts. `smoke.sh`, `smoke_degraded.sh`, `run_gdn_test.sh`, `run_gemma_mode2.sh` are the
container wrappers; they take `REPO_ROOT`.

## Not a test

`bench/` drives a running server over HTTP (`http://127.0.0.1:8057` by default, `BASE` to
override) and is not part of this suite.
