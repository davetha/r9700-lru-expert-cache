# Prebuilt kernel binaries

Our own kernels, compiled for **gfx1201 (R9700)** with the ROCm 10.0.0 hipcc from the
`local/q38fn-rocm10:build` image (HIP 7.15, AMD clang 23). They load in the runtime image
(`local/q38fn-rocm10:try1`) and in the ROCm 7.2.3 fork image (`libamdhip64.so.7` either way).
Use them if you don't want to compile; `kernels/*/build.sh` reproduces them from source.

| file | source | exports | sha256 (see SHA256SUMS) |
|---|---|---|---|
| `librlu.so` | `kernels/lru/r4d_lru.hip` | `r4d_lru_manage`, `r4d_lru_gather`, `r4d_lru_fused` (v2 victim selection) | `b8d0014b…` |
| `libhcqfp8sk.so` | `kernels/experimental/fp8skinny/hcq_fp8skinny.hip` | `hcq_gemm_fp8blk_nt_m16`, `_max_m`, `hcq_stream_probe` (experimental, not used by the launcher) | `da58ba71…` |
| `cold_gather.so` | `kernels/lru/cold_gather.hip` | `cold_gather` (transfer-study kernel, tests only) | `0d81ae03…` |

Install: `./prebuilt/install.sh` copies them to `build/kernels/`, which is where
`launch/launch_q38fn.sh` looks (`LRU_LIB=/build/kernels/librlu.so` inside the container).

Verify: `sha256sum -c prebuilt/SHA256SUMS`.

Not included, deliberately: tcclaviger's closed `r4d.so` (the MoE grouped GEMM, QSA
sparse attention, all-reduce) and `libfp8hip_gemm.so`. They are not ours to redistribute;
you get them by pulling `tcclaviger/vllm:DevQwenNextFlash`, which is what the Dockerfile
does.
