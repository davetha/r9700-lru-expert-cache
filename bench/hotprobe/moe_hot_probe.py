#!/usr/bin/env python3
"""Hot-path cost of the closed r4d_gemm_moe_mxfp4a8_nt_b16 vs a read-only control on the
SAME bytes, replaying real 5-row decode routings (k1/routes_rank0.npz) against LRU-sized slot
pools (S slots/layer, L layer pools so nothing is Infinity-Cache resident).

  MODE=graph   HIP-graph timing (default)        MODE=eager   plain calls for rocprofv3
  S=257 L=6 NCALL=96 EMCAP=512|S  CASES=gate_up,down  CHUNKS=2,4,8,16  ONLY=gemm|read
"""
import ctypes, json, os, statistics, sys, time
import numpy as np, torch
sys.path.insert(0, "/w/k1")
from moe_ref_harness import moe_align_block_size_torch, pick_cfg, GROUP, BLOCK

S = int(os.environ.get("S", "257")); L = int(os.environ.get("L", "6"))
NCALL = int(os.environ.get("NCALL", "96")); MODE = os.environ.get("MODE", "graph")
EMCAP = os.environ.get("EMCAP", "512"); CASES = os.environ.get("CASES", "gate_up,down").split(",")
CHUNKS = [int(c) for c in os.environ.get("CHUNKS", "2,4,8,16").split(",")]
ONLY = os.environ.get("ONLY", "")
TOPK, ROWS = 10, 5
SHAPES = {"gate_up": (640, 2560), "down": (2560, 320)}

lib = ctypes.CDLL("/app/r4dhip/r4d.so", mode=os.RTLD_NOW | os.RTLD_DEEPBIND)
lib.r4d_gemm_moe_mxfp4a8_nt_b16.restype = None
lib.r4d_gemm_moe_mxfp4a8_nt_b16.argtypes = [ctypes.c_long] * 10 + [ctypes.c_int] * 8 + [ctypes.c_long]
pr = ctypes.CDLL("/w/k1/hotprobe/libmoeprobe.so")
pr.moe_read_probe.restype = ctypes.c_int
pr.moe_read_probe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_long,
                              ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
torch.cuda.set_device(0)
g = torch.Generator(device="cuda").manual_seed(0)
sink = torch.zeros(4, dtype=torch.int32, device="cuda")

z = np.load("/w/k1/routes_rank0.npz"); ids = z["ids"].reshape(z["ids"].shape[0], -1, 32, TOPK)
live = (ids[:, 0, :, 0] >= 0).sum(1); five = np.where(live == ROWS)[0]

def stream(): return torch.cuda.current_stream().cuda_stream

def build(case):
    N, K = SHAPES[case]
    pools = []
    for _ in range(L):
        wq = torch.randint(0, 256, (S, N, K // 2), generator=g, dtype=torch.uint8, device="cuda")
        ws = torch.randint(118, 128, (S, K // GROUP, N), generator=g, dtype=torch.uint8, device="cuda")
        wref = torch.full((S, N), 130, dtype=torch.uint8, device="cuda")
        pools.append((wq, ws, wref))
    calls = []
    ealign = 512 if EMCAP == "512" else S
    for i in range(NCALL):
        step, layer = five[i % len(five)], i % 48
        r = torch.from_numpy(ids[step, layer, :ROWS].astype(np.int64)).cuda()   # [5,10] experts
        assert (r >= 0).all()
        uniq, inv = torch.unique(r, return_inverse=True)
        slots = torch.randperm(S, generator=g, device="cuda")[: uniq.numel()]
        topk_slots = slots[inv].to(torch.int32)                                 # [5,10] slots
        sorted_ids, expert_ids, npad = moe_align_block_size_torch(topk_slots, BLOCK, ealign)
        mtk = ROWS * TOPK
        top_k = TOPK if case == "gate_up" else 1
        arows = ROWS if case == "gate_up" else mtk
        qx = (torch.randn(arows, K, generator=g, device="cuda") * 0.3).to(torch.float8_e4m3fn)
        xs = torch.rand(arows, generator=g, device="cuda") * 0.01 + 0.001
        tw = torch.rand(mtk, generator=g, device="cuda")
        c = torch.zeros(mtk, N, dtype=torch.bfloat16, device="cuda")
        calls.append(dict(pool=i % L, sorted_ids=sorted_ids, expert_ids=expert_ids, npad=npad,
                          qx=qx, xs=xs, tw=tw, c=c, mtk=mtk, top_k=top_k, em=sorted_ids.numel(),
                          slots=slots.to(torch.int32).contiguous(), ndist=uniq.numel()))
    return N, K, pools, calls

def gemm_fn(N, K, pools, cl, cfg):
    wq, ws, wref = pools[cl["pool"]]; wv, sk, npw = cfg
    a = (cl["qx"].data_ptr(), cl["xs"].data_ptr(), wq.data_ptr(), ws.data_ptr(), wref.data_ptr(),
         cl["c"].data_ptr(), cl["sorted_ids"].data_ptr(), cl["expert_ids"].data_ptr(),
         cl["npad"].data_ptr(), cl["tw"].data_ptr() if cl["top_k"] == 1 else 0,
         cl["em"], cl["mtk"], cl["top_k"], K, N, wv, sk, npw)
    return lambda: lib.r4d_gemm_moe_mxfp4a8_nt_b16(*a, stream())

def read_fn(N, K, pools, cl, chunks):
    wq, ws, _ = pools[cl["pool"]]
    n = cl["ndist"]; idx = cl["slots"].data_ptr()
    def go():
        rc = pr.moe_read_probe(wq.data_ptr(), idx, n, N * K // 2, chunks, 256, sink.data_ptr(), stream())
        rc |= pr.moe_read_probe(ws.data_ptr(), idx, n, (K // GROUP) * N, max(1, chunks // 4), 256, sink.data_ptr(), stream())
        assert rc == 0
    return go

def time_graph(fns, rounds=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:4]: f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for f in fns: f()
    torch.cuda.synchronize(); gr.replay(); torch.cuda.synchronize()
    v = []
    for _ in range(rounds):
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record(); gr.replay(); t1.record(); torch.cuda.synchronize()
        v.append(t0.elapsed_time(t1) * 1e3 / len(fns))
    del gr
    return statistics.median(v)

res = {}
for case in CASES:
    N, K, pools, calls = build(case)
    cfg = pick_cfg(N, K)
    ndist = statistics.mean(c["ndist"] for c in calls)
    slab = N * K // 2 + (K // GROUP) * N
    mb = ndist * slab / 1e6
    print(f"\n=== {case} N={N} K={K} cfg={cfg} S={S} L={L} EM={calls[0]['em']} "
          f"distinct/call {ndist:.1f} -> {mb:.2f} MB/call", flush=True)
    r = dict(N=N, K=K, cfg=cfg, S=S, L=L, em=calls[0]["em"], ndist=ndist, mb_per_call=mb)
    if MODE == "graph":
        if ONLY in ("", "gemm"):
            us = time_graph([gemm_fn(N, K, pools, c, cfg) for c in calls])
            r["gemm_us"] = us; print(f"   GEMM      {us:7.2f} us/call  {mb/us*1e3:5.0f} GB/s", flush=True)
        if ONLY in ("", "read"):
            best = None
            for ch in CHUNKS:
                us = time_graph([read_fn(N, K, pools, c, ch) for c in calls])
                print(f"   READ ch={ch:2d} {us:7.2f} us/call  {mb/us*1e3:5.0f} GB/s", flush=True)
                if best is None or us < best[0]: best = (us, ch)
            r["read_us"], r["read_chunks"] = best
        if "gemm_us" in r and "read_us" in r:
            r["gemm_over_read"] = r["gemm_us"] / r["read_us"]
            print(f"   GEMM / READ = {r['gemm_over_read']:.2f}x  (GEMM at {100/r['gemm_over_read']:.0f}% of the read control)")
    else:   # eager, for rocprofv3: a handful of distinct calls, each repeated
        ch = int(os.environ.get("EAGER_CHUNKS", "8")); nrep = int(os.environ.get("NREP", "5"))
        fg = [gemm_fn(N, K, pools, c, cfg) for c in calls[:8]]
        fr = [read_fn(N, K, pools, c, ch) for c in calls[:8]]
        for f in fg + fr: f()
        torch.cuda.synchronize()
        for _ in range(nrep):
            for f in fg: f()
            torch.cuda.synchronize()
            for f in fr: f()
            torch.cuda.synchronize()
        print(f"   eager: {nrep}x8 GEMM + {nrep}x8 READ(ch={ch}) done")
    res[case] = r
    del pools, calls; torch.cuda.empty_cache()
tag = os.environ.get("TAG", f"{MODE}_S{S}_EM{EMCAP}")
json.dump(res, open(f"/w/k1/hotprobe/res_{tag}.json", "w"), indent=1)
print("DONE", tag)
