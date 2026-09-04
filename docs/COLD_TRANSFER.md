# Cold-expert transfer study (R9700 pair on `big`)

**Verdict: there is nothing to win here, and stage-then-compute would lose 3-9%.**
Host->device transfer on this machine tops out at **28.4 GB/s per GPU**, and the closed MoE
kernel reading pinned host memory over UVA already runs at **26.7-28.3 GB/s** — i.e. at
94-100% of the hardware ceiling, with its compute fully hidden behind the transfer. No copy
API, no batched API, and no hand-written gather kernel goes faster, because the limit is the
PCIe link, not the method. The ~55-60 GB/s the task assumed is **not reachable on this host**.

Scripts: `$REPO/k1/cold_transfer.py`, `$REPO/k1/cold_gather.hip`
(-> `cold_gather.so`), `$REPO/k1/probe_peak.py`. Raw output:
`$REPO/k1/cold_transfer.out`. All runs held `$REPO/gpu.lock`.

---

## 0. Why 28.4 and not 57: the root port is Gen4

Each R9700 has an on-card PCIe switch. The card-to-switch link is Gen5 x16 (32 GT/s), which is
what `amd-smi`/`lspci` on the endpoint reports — but the switch's **uplink to the EPYC root
port negotiates 16 GT/s x16 and is flagged `(downgraded)`**:

```
0000:83:00.0  LnkSta: Speed 32GT/s, Width x16              Navi48 (R9700 #1)
0000:82:00.0  LnkSta: Speed 32GT/s, Width x16              on-card switch, downstream port
0000:81:00.0  LnkSta: Speed 16GT/s (downgraded), Width x16 on-card switch, UPSTREAM port
0000:80:01.1  LnkSta: Speed 16GT/s, Width x16              EPYC root port
  ... identical chain for c6:00.0 / c5:00.0 / c4:00.0 / c0:03.1 (R9700 #2)
```

Board is **H12SSL-NT with an EPYC 74F3 (Milan)** — Zen 3 EPYC is PCIe **Gen4 only**. 28.4 GB/s
is textbook Gen4 x16 payload throughput. Reading the endpoint's link status alone is
misleading; the ceiling is set two hops up. A Gen5 host would roughly double this path.

`probe_peak.py`, 256 MiB contiguous copy, both GPUs, 1/2/4/8 streams:

| direction | streams=1 | 2 | 4 | 8 |
|---|---|---|---|---|
| H2D | 28.4 | 28.4 | 28.3 | 28.3 |
| D2H | 27.7 | 26.8 | 27.2 | 27.4 |

Identical on `cuda:0` and `cuda:1`. Splitting across streams buys nothing, so it is not a
single-DMA-queue limit either.

---

## 1. Transfer methods, pinned host -> VRAM, `nsel` expert `wq` slabs

`contig` = one `hipMemcpyAsync` of `nsel` *contiguous* slabs (not a gather — a bandwidth
reference only). `memcpy_n` = `nsel` back-to-back `hipMemcpyAsync`. `batch` =
`hipMemcpyBatchAsync` (present in HIP 7.15; `attrs` unsupported, pass NULL). `gather` = the
custom HIP kernel in `cold_gather.hip`: `grid=(chunks, nsel)`, 256 threads, 16-byte
(`u32x4`) grid-stride loads straight off the UVA host pointer. Best-of-3 windows of 20 runs,
timed with events recorded **on the work stream**.

### gate_up (N=640 K=2560, slab 819200 B)

| nsel | method | solo us | solo GB/s | dual us | dual GB/s/GPU | aggregate GB/s |
|---|---|---|---|---|---|---|
| 10 | contig | 294.4 | 27.8 | 332.8 | 24.6 | 49.2 |
| 10 | memcpy_n | 344.8 | 23.8 | 344.9 | 23.7 | 47.5 |
| 10 | batch | 297.5 | 27.5 | 298.5 | 27.4 | 54.9 |
| 10 | **gather** | **291.9** | **28.1** | 303.3 | 27.0 | 54.0 |
| 25 | contig | 726.8 | 28.2 | 727.5 | 28.1 | 56.3 |
| 25 | memcpy_n | 847.4 | 24.2 | 863.5 | 23.7 | 47.4 |
| 25 | batch | 730.0 | 28.1 | 731.0 | 28.0 | 56.0 |
| 25 | **gather** | **723.7** | **28.3** | 724.1 | 28.3 | 56.6 |
| 50 | contig | 1448.3 | 28.3 | 1448.9 | 28.3 | 56.5 |
| 50 | memcpy_n | 1696.1 | 24.2 | 1726.0 | 23.7 | 47.5 |
| 50 | batch | 1452.0 | 28.2 | 1452.6 | 28.2 | 56.4 |
| 50 | **gather** | **1443.1** | **28.4** | 1443.5 | 28.4 | 56.7 |
| 80 | contig | 2314.2 | 28.3 | 2314.6 | 28.3 | 56.6 |
| 80 | memcpy_n | 2719.0 | 24.1 | 2758.7 | 23.8 | 47.5 |
| 80 | batch | 2317.8 | 28.3 | 2318.4 | 28.3 | 56.5 |
| 80 | **gather** | **2306.7** | **28.4** | 2307.1 | 28.4 | 56.8 |

### down (N=2560 K=320, slab 409600 B)

| nsel | method | solo us | solo GB/s | dual us | dual GB/s/GPU | aggregate GB/s |
|---|---|---|---|---|---|---|
| 10 | contig | 150.2 | 27.3 | 198.6 | 20.6 | 41.2 |
| 10 | memcpy_n | 197.0 | 20.8 | 201.0 | 20.4 | 40.7 |
| 10 | batch | 153.1 | 26.7 | 154.2 | 26.6 | 53.1 |
| 10 | **gather** | **147.9** | **27.7** | 148.4 | 27.6 | 55.2 |
| 25 | contig | 366.2 | 28.0 | 366.8 | 27.9 | 55.8 |
| 25 | memcpy_n | 487.6 | 21.0 | 502.3 | 20.4 | 40.8 |
| 25 | batch | 369.7 | 27.7 | 370.3 | 27.7 | 55.3 |
| 25 | **gather** | **363.8** | **28.2** | 364.2 | 28.1 | 56.2 |
| 50 | contig | 727.0 | 28.2 | 727.3 | 28.2 | 56.3 |
| 50 | memcpy_n | 975.6 | 21.0 | 1004.1 | 20.4 | 40.8 |
| 50 | batch | 731.5 | 28.0 | 731.2 | 28.0 | 56.0 |
| 50 | **gather** | **723.5** | **28.3** | 724.1 | 28.3 | 56.6 |
| 80 | contig | 1159.8 | 28.3 | 1160.1 | 28.2 | 56.5 |
| 80 | memcpy_n | 1560.7 | 21.0 | 1609.9 | 20.4 | 40.7 |
| 80 | batch | 1163.3 | 28.2 | 1164.0 | 28.2 | 56.3 |
| 80 | **gather** | **1155.2** | **28.4** | 1155.5 | 28.4 | 56.7 |

**Readings.**
- The custom gather kernel is the best method at every size, but only by 0.4-1% over `batch`
  and over a *contiguous* copy of the same bytes. Scattered vs contiguous costs nothing: the
  gather is bandwidth-bound, not address-bound.
- `hipMemcpyAsync` per expert (what the obvious implementation would do) is the **worst**:
  23.8-24.2 GB/s for gate_up and 20.8-21.0 GB/s for down, i.e. 15-26% off the ceiling. The
  gap is per-copy submit overhead (~1.4 us each), and it is worse for `down` because the
  slab is half the size, so the same overhead amortizes over half the bytes. If any staging
  is ever written, it must be a batched call or a kernel, never a loop of copies.
- `hipMemcpyBatchAsync` exists in HIP 7.15 and works
  (`dsts, srcs, sizes, count, NULL, NULL, 0, &failIdx, stream`); it lands within 0.5% of the
  custom kernel and is the zero-maintenance choice.
- **Both GPUs pulling at once is free**: 28.3-28.4 GB/s *each*, 56.8 GB/s aggregate, with the
  dual number equal to the solo number to within noise for every method at nsel >= 25. There
  is **no root-complex contention** — the two cards hang off different root ports on
  different IO dies. (The small dips at nsel=10 are launch skew between the two devices over
  a ~150-300 us window, not contention.)
- Gather kernel geometry barely matters: chunks in {4..64} x {plain, nontemporal} all land at
  28.4 GB/s. 8 workgroups of 256 threads per slab is already enough memory-level parallelism
  to saturate a Gen4 link. `chunks=64, threads=256, plain loads` was used for section 2.
- **(b), the torch `index_select`-on-a-UVA-device-view variant, was not run as such**:
  PyTorch offers no Python-level way to build a CUDA tensor aliasing an arbitrary device
  pointer (no `from_blob` binding), so it needs a C extension either way. `cold_gather.hip`
  *is* that gather, with explicit control of vector width and wave count, and it is already
  at the ceiling — `index_select` could not beat it.

---

## 2. End to end: hot vs UVA vs staged, same expert set

Closed kernel `r4d_gemm_moe_mxfp4a8_nt_b16` from `/app/r4dhip/r4d.so`, E=512, top_k=10,
M = ceil(nsel/10) tokens so exactly `nsel` distinct experts are routed.
`hot` = all weights in VRAM. `uva` = `wq`, `ws`, `wref` all in pinned host memory (what
production does today). `staged` = gather all three into compact VRAM buffers with the
expert ids remapped to 0..nsel-1, then run the kernel on them.

| case | nsel | M | hot us | uva us | stage us | stage+kernel us | vs uva | uva GB/s |
|---|---|---|---|---|---|---|---|---|
| gate_up | 10 | 1 | 18.2 | 317.7 | 318.2 | 332.3 | **+4.6%** | 27.4 |
| gate_up | 25 | 3 | 30.0 | 777.0 | 776.9 | 808.7 | **+4.1%** | 28.0 |
| gate_up | 50 | 5 | 46.1 | 1543.7 | 1545.5 | 1594.0 | **+3.3%** | 28.2 |
| gate_up | 80 | 8 | 81.9 | 2464.2 | 2467.8 | 2545.2 | **+3.3%** | 28.3 |
| down | 10 | 1 | 12.3 | 163.7 | 165.8 | 179.2 | **+9.4%** | 26.7 |
| down | 25 | 3 | 20.5 | 395.5 | 396.3 | 416.8 | **+5.4%** | 27.7 |
| down | 50 | 5 | 36.6 | 778.2 | 781.4 | 818.8 | **+5.2%** | 28.1 |
| down | 80 | 8 | 56.0 | 1240.1 | 1242.6 | 1303.7 | **+5.1%** | 28.2 |

Cold bytes per expert = `wq` + `ws` + `wref` = 819200+51200+640 = 871040 B (gate_up),
409600+25600+2560 = 437760 B (down). "uva GB/s" is `nsel * slab / uva us`.
The staged output was verified **bit-identical** to the UVA output in every row.

**The decisive line is `uva us` vs `stage us`: they are equal to within 0.3%.** The closed
kernel spends exactly as long reading host memory as a pure transfer of the same bytes
takes. Its compute (`hot us`, 12-82 us) is *entirely* hidden behind the PCIe read. Staging
first therefore serializes what is currently overlapped: you pay the transfer, and then the
kernel's compute on top, which is the +3.3% to +9.4% seen above. The penalty is largest
where the compute-to-bytes ratio is highest (`down`, small nsel).

---

## 3. Projected effect on production

Production decode step: 37 ms busy, MoE kernel 23 ms of it, 96 cold calls (48 layers x 2
GEMMs), cold-call p75 149 us / p90 253 us.

- 96 calls x ~150-250 us = **14.4-24.0 ms**, which is the 23 ms measured. That whole 23 ms is
  PCIe transfer at the link ceiling — at 28.4 GB/s it corresponds to ~650 MB of cold expert
  weight pulled across per step (consistent with ~10 cold experts per layer at
  0.871+0.438 = 1.31 MB per expert per layer).
- **Switching to stage-then-compute: -0.5 to -2.2 ms per step (a loss, not a saving).**
  3.3-9.4% added to 14.4-24.0 ms. Do not do it.
- Prefetching the next layer's cold experts during the current layer's compute cannot help
  either: routing is data-dependent per layer (you do not know layer L+1's expert set until
  L+1's router runs), and within a layer both GEMMs share one routing but the link is already
  saturated by the first of them. There is no idle link time to fill inside the MoE region.
- The only levers left are **fewer cold bytes** and **a faster link**, and time scales
  linearly with bytes at this ceiling:
  - residency/hot-set work is worth ~0.23 ms of step time per 1% of cold bytes removed;
  - keeping the E8M0 scales (`ws` + `wref`) permanently resident removes 6.0% of the cold
    bytes -> ~1.4 ms per step, but it is not free: all 512 experts' scales for all 48 layers
    is 40.9 MB/layer = **1.96 GB per GPU**, which would have to come out of the same VRAM
    budget the hot experts compete for. Worth costing against simply raising residency;
  - a Gen5 host (EPYC 9004+ / any board whose root port does 32 GT/s) would roughly halve
    the 23 ms. On this H12SSL/Milan box that ceiling is fixed.

---

## 4. What was not established

- Whether the on-card switch's `(downgraded)` Gen4 uplink is a firmware/BIOS setting or a
  hard platform limit was not investigated — EPYC Milan is Gen4-only, so a Gen5 negotiation
  should not be possible regardless, but the "downgraded" flag was not chased further.
- Peer-to-peer (GPU<->GPU) bandwidth was not measured; only host<->device.
- All numbers are with the GPUs otherwise idle under the lock. The vLLM server container
  `q38fn-mxfp4` was running but not serving during these runs.
