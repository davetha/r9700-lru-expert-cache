# Hot-path profile of the closed MoE GEMM, the all-reduce and the bf16 GEMV (2026-09-04)

Question: with the open libr4d sources in hand, is there decode time left in the kernels?
Method: per-call cost in a HIP graph against a READ-ONLY control that touches the same bytes
(`moe_read_probe.hip`, `k3/hcq_stream_probe`), pools > 64 MB Infinity Cache, real 5-row
routings replayed from `k1/routes_rank0.npz` (31.1 distinct experts/layer, S=257 slots, 6 layer pools).
GPU 1 (R9700), server stopped, under the GPU lock. Scripts: `moe_hot_probe.py`, `bench_bf16_ctl.py`.

## Closed `r4d_gemm_moe_mxfp4a8_nt_b16` (no source) — resident/hot call, production cfg

| call | bytes/call | GEMM us | read control us | GEMM % of ceiling | gap x 48/step |
|---|---|---|---|---|---|
| gate_up N=640 K=2560 cfg 4/8/2 | 27.1 MB | 59.7 (454 GB/s) | 48.2 (561 GB/s) | 81% | 0.55 ms |
| down N=2560 K=320 cfg 8/2/2 | 13.5 MB | 39.4 (344 GB/s) | 26.3 (515 GB/s) | 67% | 0.63 ms |

Standalone mean of the two = 49.5 us; the c7 production trace shows 49.9 us mean for the
resident calls (the cold fallback call is 1.5 us). The probe reproduces production.
**Ceiling of any rewrite or cfg change: ~1.2 ms/step of ~30 (4%).** `down` is the worse one:
K=320 with SK=2 gives every wave only 10 k-steps (2.5 KB of weight) before an LDS reduction
and epilogue, so fixed per-wave cost dominates. Both launch 8,000 waves/call (SQ_WAVES),
50 blocks of which ~31 are live.
EM capacity is NOT a factor: vLLM's moe_align clamps the sorted buffer to numel*16 = 800 at
50 rows, so the E*15 padding never applies at decode (the EMCAP=512 vs S arms are identical).

## Open `r4d_gemm_bf16_nt_m64` (source in libr4d) — production HC shapes, M=5, 512 MB pool

| shape | GEMM us | read control us | GEMM/read |
|---|---|---|---|
| hc_up 10240x320 | 14.35 | 12.75 | 1.13x |
| hc_down_merged 336x10240 | 13.94 | 13.28 | 1.05x |
| hc_down_plain 320x10240 | 13.44 | 12.75 | 1.05x |
| moe_router 512x2560 | 7.07 | 6.50 | 1.09x |

At the ceiling. Best case from a rewrite ~0.2 ms/step. Dead.

## Open `r4d_ar_oneshot_2rank_exact` (source in libr4d) — from the c7 traces, no bench needed

Pairing the i-th all-reduce kernel on rank 0 with the i-th on rank 1 (`arskew.py`):
min(dur) over the pair = the kernel's own cost: p50 3.2 us, mean 4.0 us = **0.44 ms/step**;
rank 0 observes 10.4 us mean = 1.13 ms/step. The other ~0.7 ms/step is the faster rank
spinning for the slower one (|start skew| p50 7.8 us, mean 18.8 us). Kernel tuning caps at
~0.1 ms/step; the lever is rank skew (the two ranks gather different cold-expert byte counts).

## Hardware counters on gfx1201 (rocprofv3, ROCm 10 pip SDK)

`--pmc` aborts with `aqlprofile API table load failed` because the SDK ships only
`libhsa-amd-aqlprofile64.so.1` and the runtime dlopens the unversioned name; fix:
`ln -sf $SDK/lib/libhsa-amd-aqlprofile64.so.1 /usr/lib/libhsa-amd-aqlprofile64.so` and
`LD_LIBRARY_PATH=$SDK/lib` (`run_counters.sh`). After that only GRBM_GUI_ACTIVE, SQ_WAVES and
SQ_BUSY_CYCLES read non-zero; FETCH_SIZE, GL2C_*, TA_*, VALU, SQ_WAVE_CYCLES, occupancy are all 0
on this part in this build. Counters cannot separate memory stalls from occupancy here;
the read-control method is the usable tool.

## Between-step host gap (added later the same day)

`hosttl.py`-style timeline of the c7 trace: after the draft graph's last kernel, the worker thread
runs ~1.4 ms of serial Python before the next `hipGraphLaunch`: sampled-token D2H copies (0.15),
`prepare_inputs_embeds` -> eager `embed_input_ids` through the multimodal path (0.44, with a
dynamo region and an eager TP all-reduce), `prepare_inputs` (rope positions, ngram context; 0.33),
PLE-offload `prepare_forward` (0.2). `hipGraphLaunch` costs 1.25 ms of CPU but overlaps the GPU.
The draft passes are one graph region (397 kernels) with no host gaps. Async scheduling was a wash
because this is worker-side input prep, not scheduler time. PLE tables: 47.75 GiB fp8, CPU-resident.

Measured fixes (ms/step, paired): fp8 target lm_head -1.02 (adopted), text-only mode -0.49
(off: disables images), skip empty cold MoE call -0.26 (off by choice). Details in the README.
