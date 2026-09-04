"""How fast can N cold expert slabs get from pinned host RAM into VRAM on the R9700s?

Production (hot/cold MoE residency patch) has the CLOSED kernel dereference cold expert
weights straight out of pinned host memory over PCIe (UVA). That path measures ~26-30 GB/s
in moe_ref_harness even though card2/card3 sit on PCIe 5.0 x16 (32 GT/s x16 ~= 55-60 GB/s
usable). This script asks whether STAGING the cold slabs into VRAM first and then running
the kernel on VRAM is cheaper than letting the kernel read host memory.

Section 1 -- raw transfer of the wq slabs (the dominant bytes: 819200 B per gate_up expert,
409600 B per down expert):
    contig      one hipMemcpyAsync of nsel contiguous slabs   (PCIe upper bound, not a gather)
    memcpy_n    nsel hipMemcpyAsync calls, one stream
    batch       hipMemcpyBatchAsync (HIP 7.15), one call
    gather      custom HIP kernel, 16-byte loads off a UVA host pointer (cold_gather.hip)
  each solo, and with BOTH GPUs pulling at once (what TP=2 does every step).

Section 2 -- end to end for the same expert set:
    hot         closed kernel, weights in VRAM              (lower bound)
    uva         closed kernel reading host memory           (what production does today)
    staged      best transfer of wq+ws+wref into VRAM, then the closed kernel

Env: CASES=gate_up,down  NSEL=10,25,50,80  REPS=20  SWEEP=1  DUAL=1  SEC2=1
Run under the GPU lock (see GPU_LOCK_PROTOCOL.md).
"""
import ctypes
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_ref_harness import BLOCK, GROUP, load, moe_align_block_size_torch, pick_cfg  # noqa: E402

CASES = {"gate_up": (640, 2560), "down": (2560, 320)}
WANT = os.environ.get("CASES", "gate_up,down").split(",")
NSEL = [int(v) for v in os.environ.get("NSEL", "10,25,50,80").split(",")]
REPS = int(os.environ.get("REPS", "20"))
E = int(os.environ.get("E", "512"))
TOPK = int(os.environ.get("TOPK", "10"))
SEED = int(os.environ.get("SEED", "0"))
GATHER_SO = os.environ.get("GATHER_SO", "/w/build/kernels/cold_gather.so")
R4D = os.environ.get("R4D_LIB", "/app/r4dhip/r4d.so")

c_vp, c_sz, c_i, c_l = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_long

hip = ctypes.CDLL("libamdhip64.so.7")
hip.hipMemcpyAsync.argtypes = [c_vp, c_vp, c_sz, c_i, c_vp]
hip.hipMemcpyAsync.restype = c_i
hip.hipMemcpyBatchAsync.argtypes = [ctypes.POINTER(c_vp), ctypes.POINTER(c_vp),
                                    ctypes.POINTER(c_sz), c_sz, c_vp, c_vp, c_sz,
                                    ctypes.POINTER(c_sz), c_vp]
hip.hipMemcpyBatchAsync.restype = c_i
hip.hipHostGetDevicePointer.argtypes = [ctypes.POINTER(c_vp), c_vp, ctypes.c_uint]
hip.hipHostGetDevicePointer.restype = c_i
H2D = 1

gth = ctypes.CDLL(GATHER_SO)
gth.cold_gather.argtypes = [c_vp, c_vp, c_vp, c_i, c_l, c_i, c_i, c_i, c_vp]
gth.cold_gather.restype = c_i


def ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} -> hip rc={rc}")


def host_uva(nbytes, dev):
    """Pinned host buffer + the device pointer that GPU `dev` must use to reach it."""
    with torch.cuda.device(dev):
        h = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        h.random_(0, 256)
        p = c_vp()
        ck(hip.hipHostGetDevicePointer(ctypes.byref(p), c_vp(h.data_ptr()), 0),
           "hipHostGetDevicePointer")
    return h, p.value


# ------------------------------------------------------------------ transfer methods
def mk_contig(dst, hptr, slab, nsel, stream, **_):
    n = slab * nsel

    def go():
        ck(hip.hipMemcpyAsync(c_vp(dst), c_vp(hptr), c_sz(n), H2D, c_vp(stream)), "memcpy")
    return go


def mk_memcpy_n(dst, hptr, slab, nsel, stream, ids=None, **_):
    args = [(c_vp(dst + i * slab), c_vp(hptr + int(e) * slab)) for i, e in enumerate(ids)]
    n = c_sz(slab)
    st = c_vp(stream)
    f = hip.hipMemcpyAsync

    def go():
        for d, s in args:
            ck(f(d, s, n, H2D, st), "memcpy_n")
    return go


def mk_batch(dst, hptr, slab, nsel, stream, ids=None, **_):
    A = c_vp * nsel
    S = c_sz * nsel
    dsts = A(*[dst + i * slab for i in range(nsel)])
    srcs = A(*[hptr + int(e) * slab for e in ids])
    sizes = S(*([slab] * nsel))
    fail = c_sz(0)
    cnt = c_sz(nsel)
    st = c_vp(stream)

    def go():
        ck(hip.hipMemcpyBatchAsync(dsts, srcs, sizes, cnt, None, None, c_sz(0),
                                   ctypes.byref(fail), st), "batch")
    return go


def mk_gather(dst, hptr, slab, nsel, stream, ids_dev=None, chunks=8, threads=256, nt=0, **_):
    def go():
        ck(gth.cold_gather(c_vp(dst), c_vp(hptr), c_vp(ids_dev), nsel, slab,
                           chunks, threads, nt, c_vp(stream)), "gather")
    return go


def timed(go, dev, reps, stream):
    """Per-device elapsed time for `reps` back-to-back runs, in microseconds per run.

    The events MUST be recorded on the same stream the work is enqueued on -- recording on
    torch's default stream measures an empty stream and reports terabytes per second.
    """
    with torch.cuda.device(dev):
        for _ in range(3):
            go()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        best = 1e9
        for _ in range(3):
            a.record(stream)
            for _ in range(reps):
                go()
            b.record(stream)
            torch.cuda.synchronize()
            best = min(best, a.elapsed_time(b) * 1e3 / reps)
        return best


# ------------------------------------------------------------------ section 1
def section1(devs):
    torch.manual_seed(SEED)
    print("=" * 100)
    print("SECTION 1 -- pinned host -> VRAM transfer of nsel expert wq slabs")
    print("=" * 100)
    best_cfg = {}
    for name in WANT:
        N, K = CASES[name]
        slab = N * (K // 2)
        print(f"\n--- {name}: N={N} K={K}  slab={slab} B ({slab/2**20:.3f} MiB), E={E} "
              f"(host buffer {E*slab/2**20:.0f} MiB per GPU)")
        bufs = {}
        for d in devs:
            h, hp = host_uva(E * slab, d)
            with torch.cuda.device(d):
                stage = torch.empty(max(NSEL) * slab, dtype=torch.uint8, device=f"cuda:{d}")
                st = torch.cuda.Stream(device=d)
            bufs[d] = (h, hp, stage, st)

        # one sweep of the custom kernel's launch geometry, on the largest nsel
        if os.environ.get("SWEEP", "1") == "1":
            d = devs[0]
            h, hp, stage, st = bufs[d]
            ns = max(NSEL)
            ids = torch.randperm(E)[:ns]
            ids_d = ids.to(torch.int32).to(f"cuda:{d}")
            rows = []
            for nt in (0, 1):
                for ch in (1, 2, 4, 8, 16, 32, 64):
                    go = mk_gather(stage.data_ptr(), hp, slab, ns, st.cuda_stream,
                                   ids_dev=ids_d.data_ptr(), chunks=ch, nt=nt)
                    us = timed(go, d, REPS, st)
                    rows.append((ns * slab / us / 1e3, ch, nt, us))
            rows.sort(reverse=True)
            print("    gather sweep (nsel=%d): " % ns + "  ".join(
                f"ch={c}{'/nt' if n else ''}:{g:.1f}" for g, c, n, _ in rows[:6]))
            best_cfg[name] = (rows[0][1], rows[0][2])
        else:
            best_cfg[name] = (8, 0)
        ch, nt = best_cfg[name]
        print(f"    gather config used: chunks={ch} threads=256 nontemporal={nt}")

        print("    %-6s %-11s %10s %10s | %10s %10s %10s"
              % ("nsel", "method", "us", "GB/s", "us(dual)", "GB/s(dual)", "GB/s(2gpu)"))
        for ns in NSEL:
            ids = torch.randperm(E)[:ns]
            byts = ns * slab
            builders = [("contig", mk_contig), ("memcpy_n", mk_memcpy_n),
                        ("batch", mk_batch), ("gather", mk_gather)]
            for mname, mk in builders:
                gos = {}
                for d in devs:
                    h, hp, stage, st = bufs[d]
                    idd = ids.to(torch.int32).to(f"cuda:{d}")
                    bufs[d] = (h, hp, stage, st)
                    gos[d] = (mk(stage.data_ptr(), hp, slab, ns, st.cuda_stream,
                                 ids=ids.tolist(), ids_dev=idd.data_ptr(),
                                 chunks=ch, nt=nt), idd)
                solo = timed(gos[devs[0]][0], devs[0], REPS, bufs[devs[0]][3])
                du = dun = float("nan")
                if len(devs) > 1:
                    du = dual(gos, {d: bufs[d][3] for d in devs}, devs, REPS)
                    dun = byts / du / 1e3
                print("    %-6d %-11s %10.1f %10.1f | %10.1f %10.1f %10.1f"
                      % (ns, mname, solo, byts / solo / 1e3, du, dun, 2 * dun))
        for d in devs:
            del bufs[d]
        torch.cuda.empty_cache()
    return best_cfg


def dual(gos, streams, devs, reps):
    """Both GPUs pulling at once; report the slower device's own elapsed time."""
    evs = {}
    for d in devs:
        with torch.cuda.device(d):
            for _ in range(3):
                gos[d][0]()
    for d in devs:
        torch.cuda.synchronize(d)
    for d in devs:
        with torch.cuda.device(d):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(streams[d])
            for _ in range(reps):
                gos[d][0]()
            b.record(streams[d])
            evs[d] = (a, b)
    for d in devs:
        torch.cuda.synchronize(d)
    return max(evs[d][0].elapsed_time(evs[d][1]) for d in devs) * 1e3 / reps


# ------------------------------------------------------------------ section 2
def build_moe(N, K, nsel, dev, seed):
    """Weights for E experts + routing that touches exactly `nsel` distinct experts."""
    g = torch.Generator(device=f"cuda:{dev}").manual_seed(seed)
    dv = f"cuda:{dev}"
    wq = torch.randint(0, 256, (E, N, K // 2), generator=g, dtype=torch.uint8, device=dv)
    ws = torch.randint(127, 136, (E, K // GROUP, N), generator=g, dtype=torch.uint8, device=dv)
    wref = torch.full((E, N), 135, dtype=torch.uint8, device=dv)
    sel = torch.randperm(E, generator=g, device=dv)[:nsel]
    M = max(1, -(-nsel // TOPK))
    flat = sel[torch.randint(0, nsel, (M * TOPK,), generator=g, device=dv)]
    flat[:nsel] = sel                                  # guarantee exactly nsel distinct
    topk_ids = flat.to(torch.int32).reshape(M, TOPK)
    si, ei, npad = moe_align_block_size_torch(topk_ids, BLOCK, E)
    qx = (torch.randn(M, K, generator=g, device=dv) * 8).to(torch.float8_e4m3fn)
    xs = torch.full((M,), 0.01, device=dv)
    tw = torch.rand(M * TOPK, generator=g, device=dv)
    return dict(wq=wq, ws=ws, wref=wref, sel=sel, M=M, mtk=M * TOPK, topk_ids=topk_ids,
                si=si, ei=ei, npad=npad, qx=qx, xs=xs, tw=tw)


def moe_go(lib, cse, N, K, cfg, wq_p, ws_p, wref_p, ei, out, stream):
    wv, sk, npw = cfg
    f = lib.r4d_gemm_moe_mxfp4a8_nt_b16
    args = (cse["qx"].data_ptr(), cse["xs"].data_ptr(), wq_p, ws_p, wref_p,
            out.data_ptr(), cse["si"].data_ptr(), ei.data_ptr(), cse["npad"].data_ptr(),
            0, cse["si"].numel(), cse["mtk"], TOPK, K, N, wv, sk, npw, stream)

    def go():
        f(*args)
    return go


def section2(devs, best_cfg):
    d = devs[0]
    dv = f"cuda:{d}"
    lib = load(R4D)
    print("\n" + "=" * 100)
    print("SECTION 2 -- closed kernel: weights hot in VRAM vs read over UVA vs staged first")
    print("=" * 100)
    print("    %-8s %-5s %-4s %10s %10s %10s %10s %10s"
          % ("case", "nsel", "M", "hot us", "uva us", "stage us", "sk+ker us", "vs uva"))
    out_rows = []
    for name in WANT:
        N, K = CASES[name]
        cfg = pick_cfg(N, K)
        slabs = [("wq", N * (K // 2)), ("ws", (K // GROUP) * N), ("wref", N)]
        tot_slab = sum(s for _, s in slabs)
        ch, nt = best_cfg.get(name, (8, 0))
        for ns in NSEL:
            cse = build_moe(N, K, ns, d, SEED)
            with torch.cuda.device(d):
                st = torch.cuda.Stream(device=d)
                out = torch.zeros(cse["mtk"], N, dtype=torch.bfloat16, device=dv)
                hot = moe_go(lib, cse, N, K, cfg, cse["wq"].data_ptr(), cse["ws"].data_ptr(),
                             cse["wref"].data_ptr(), cse["ei"], out, st.cuda_stream)
                t_hot = timed(hot, d, REPS, st)

                # cold: all three weight tensors live in pinned host memory (production)
                hbufs, hptrs = [], []
                for t in (cse["wq"], cse["ws"], cse["wref"]):
                    h = torch.empty(t.numel(), dtype=torch.uint8, pin_memory=True)
                    h.copy_(t.reshape(-1).cpu())
                    p = c_vp()
                    ck(hip.hipHostGetDevicePointer(ctypes.byref(p), c_vp(h.data_ptr()), 0), "uva")
                    hbufs.append(h)
                    hptrs.append(p.value)
                uva = moe_go(lib, cse, N, K, cfg, hptrs[0], hptrs[1], hptrs[2],
                             cse["ei"], out, st.cuda_stream)
                t_uva = timed(uva, d, REPS, st)

                # staged: compact VRAM copies of just the nsel touched experts,
                # expert ids remapped to 0..nsel-1 so the kernel's e*stride still lands right.
                comp = torch.full((E,), -1, dtype=torch.int32, device=dv)
                comp[cse["sel"].long()] = torch.arange(ns, dtype=torch.int32, device=dv)
                ei2 = torch.where(cse["ei"] >= 0, comp[cse["ei"].long().clamp(min=0)], cse["ei"])
                ids_d = cse["sel"].to(torch.int32).contiguous()
                stg = [torch.empty(ns * s, dtype=torch.uint8, device=dv) for _, s in slabs]
                gs = [mk_gather(stg[i].data_ptr(), hptrs[i], slabs[i][1], ns,
                                st.cuda_stream, ids_dev=ids_d.data_ptr(), chunks=ch, nt=nt)
                      for i in range(3)]

                def stage():
                    for g_ in gs:
                        g_()
                t_stage = timed(stage, d, REPS, st)
                ker = moe_go(lib, cse, N, K, cfg, stg[0].data_ptr(), stg[1].data_ptr(),
                             stg[2].data_ptr(), ei2, out, st.cuda_stream)

                def both():
                    stage()
                    ker()
                t_both = timed(both, d, REPS, st)

                # correctness: staged path must equal the UVA path bit for bit
                o1 = torch.zeros_like(out)
                o2 = torch.zeros_like(out)
                moe_go(lib, cse, N, K, cfg, hptrs[0], hptrs[1], hptrs[2], cse["ei"], o1,
                       st.cuda_stream)()
                moe_go(lib, cse, N, K, cfg, stg[0].data_ptr(), stg[1].data_ptr(),
                       stg[2].data_ptr(), ei2, o2, st.cuda_stream)()
                torch.cuda.synchronize()
                same = torch.equal(o1.view(torch.int16), o2.view(torch.int16))
            print("    %-8s %-5d %-4d %10.1f %10.1f %10.1f %10.1f %+9.1f%s"
                  % (name, ns, cse["M"], t_hot, t_uva, t_stage, t_both,
                     100 * (t_both - t_uva) / t_uva, "" if same else "  !!MISMATCH"))
            out_rows.append((name, ns, cse["M"], t_hot, t_uva, t_stage, t_both, same,
                             ns * tot_slab))
            del cse, hbufs, stg, o1, o2, out
            torch.cuda.empty_cache()
    return out_rows


def main():
    devs = [0, 1] if (os.environ.get("DUAL", "1") == "1"
                      and torch.cuda.device_count() > 1) else [0]
    for d in devs:
        torch.zeros(1, device=f"cuda:{d}")
        p = torch.cuda.get_device_properties(d)
        print(f"cuda:{d} = {p.name} {p.total_memory/2**30:.1f} GiB")
    bc = section1(devs)
    if os.environ.get("SEC2", "1") == "1":
        section2(devs, bc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
