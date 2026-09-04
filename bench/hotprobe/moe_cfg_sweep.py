#!/usr/bin/env python3
"""(WV, SK, NPW) sweep of the closed r4d_gemm_moe_mxfp4a8_nt_b16 on the LRU hot path, same
setup as moe_hot_probe.py (real 5-row routes, S=257 slots, 6 layer pools, HIP graph).
Also reports max |diff| of each cfg's output vs the production cfg on one call (reduction
order changes with SK, so bitwise equality is not expected)."""
import ctypes, json, os, statistics, sys
import numpy as np, torch
sys.path.insert(0, "/w/k1")
from moe_ref_harness import moe_align_block_size_torch, pick_cfg, GROUP, BLOCK

S = int(os.environ.get("S", "257")); L = int(os.environ.get("L", "6"))
NCALL = int(os.environ.get("NCALL", "96")); CASES = os.environ.get("CASES", "gate_up,down").split(",")
TOPK = 10; ROWS = int(os.environ.get("ROWS", "5"))
CFGLIST = [tuple(int(v) for v in c.split(",")) for c in os.environ.get("CFGLIST", "").split(";") if c]
SHAPES = {"gate_up": (640, 2560), "down": (2560, 320)}
lib = ctypes.CDLL("/app/r4dhip/r4d.so", mode=os.RTLD_NOW | os.RTLD_DEEPBIND)
lib.r4d_gemm_moe_mxfp4a8_nt_b16.restype = None
lib.r4d_gemm_moe_mxfp4a8_nt_b16.argtypes = [ctypes.c_long] * 10 + [ctypes.c_int] * 8 + [ctypes.c_long]
torch.cuda.set_device(0)
g = torch.Generator(device="cuda").manual_seed(0)
z = np.load("/w/k1/routes_rank0.npz"); ids = z["ids"].reshape(z["ids"].shape[0], -1, 32, TOPK)
live = (ids[:, 0, :, 0] >= 0).sum(1); five = np.where(live == 5)[0]   # always 5-row steps; ROWS>5 concatenates them
def stream(): return torch.cuda.current_stream().cuda_stream

def build(case):
    N, K = SHAPES[case]; pools = []
    for _ in range(L):
        wq = torch.randint(0, 256, (S, N, K // 2), generator=g, dtype=torch.uint8, device="cuda")
        ws = torch.randint(118, 128, (S, K // GROUP, N), generator=g, dtype=torch.uint8, device="cuda")
        wref = torch.full((S, N), 130, dtype=torch.uint8, device="cuda")
        pools.append((wq, ws, wref))
    calls = []
    for i in range(NCALL):
        step, layer = five[i % len(five)], i % 48
        chunks = [ids[five[(i + j) % len(five)], layer, :5] for j in range((ROWS + 4) // 5)]
        r = torch.from_numpy(np.concatenate(chunks)[:ROWS].astype(np.int64)).cuda()
        uniq, inv = torch.unique(r, return_inverse=True)
        slots = torch.randperm(S, generator=g, device="cuda")[torch.arange(uniq.numel(), device="cuda") % S]
        sorted_ids, expert_ids, npad = moe_align_block_size_torch(slots[inv].to(torch.int32), BLOCK, 512)
        mtk = ROWS * TOPK; top_k = TOPK if case == "gate_up" else 1; arows = ROWS if case == "gate_up" else mtk
        qx = (torch.randn(arows, K, generator=g, device="cuda") * 0.3).to(torch.float8_e4m3fn)
        xs = torch.rand(arows, generator=g, device="cuda") * 0.01 + 0.001
        tw = torch.rand(mtk, generator=g, device="cuda")
        calls.append(dict(pool=i % L, sorted_ids=sorted_ids, expert_ids=expert_ids, npad=npad, qx=qx, xs=xs,
                          tw=tw, c=torch.zeros(mtk, N, dtype=torch.bfloat16, device="cuda"), mtk=mtk,
                          top_k=top_k, em=sorted_ids.numel(), ndist=uniq.numel()))
    return N, K, pools, calls

def gemm_fn(N, K, pools, cl, cfg):
    wq, ws, wref = pools[cl["pool"]]; wv, sk, npw = cfg
    a = (cl["qx"].data_ptr(), cl["xs"].data_ptr(), wq.data_ptr(), ws.data_ptr(), wref.data_ptr(), cl["c"].data_ptr(),
         cl["sorted_ids"].data_ptr(), cl["expert_ids"].data_ptr(), cl["npad"].data_ptr(),
         cl["tw"].data_ptr() if cl["top_k"] == 1 else 0, cl["em"], cl["mtk"], cl["top_k"], K, N, wv, sk, npw)
    return lambda: lib.r4d_gemm_moe_mxfp4a8_nt_b16(*a, stream())

def time_graph(fns, rounds=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:4]: f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for f in fns: f()
    torch.cuda.synchronize(); gr.replay(); torch.cuda.synchronize(); v = []
    for _ in range(rounds):
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record(); gr.replay(); t1.record(); torch.cuda.synchronize(); v.append(t0.elapsed_time(t1) * 1e3 / len(fns))
    del gr; return statistics.median(v)

def cfgs(N, K):
    out = []
    for sk in (1, 2, 4, 8):
        if K % (sk * GROUP): continue
        for wv in (1, 2, 4, 8, 16, 32):
            if wv * sk * 32 > 1024: continue
            for npw in (1, 2, 4, 8):
                if wv * npw * sk * 1024 > 65536: continue
                if wv * npw > N // 16: continue
                out.append((wv, sk, npw))
    return out

res = {}
for case in CASES:
    N, K, pools, calls = build(case)
    prod = pick_cfg(N, K); mb = statistics.mean(c["ndist"] for c in calls) * (N * K // 2 + (K // GROUP) * N) / 1e6
    # reference output with the production cfg on call 0
    gemm_fn(N, K, pools, calls[0], prod)(); torch.cuda.synchronize(); ref = calls[0]["c"].float().clone()
    rows = []
    print(f"\n=== {case} N={N} K={K} prod cfg={prod} {mb:.2f} MB/call  ({len(cfgs(N,K))} cfgs)", flush=True)
    valid = set(cfgs(N, K))
    for cfg in [c for c in CFGLIST if c in valid] if CFGLIST else cfgs(N, K):
        try:
            us = time_graph([gemm_fn(N, K, pools, c, cfg) for c in calls])
        except Exception as e:
            print(f"   cfg {cfg}: FAILED {type(e).__name__}"); torch.cuda.synchronize(); continue
        calls[0]["c"].zero_(); gemm_fn(N, K, pools, calls[0], cfg)(); torch.cuda.synchronize()
        d = (calls[0]["c"].float() - ref).abs().max().item() / ref.abs().max().item()
        rows.append((us, cfg, d))
        print(f"   cfg wv={cfg[0]:2d} sk={cfg[1]} npw={cfg[2]}  {us:7.2f} us  {mb/us*1e3:4.0f} GB/s  maxrel_vs_prod {d:.1e}{'  <- prod' if cfg == prod else ''}", flush=True)
    rows.sort()
    pu = next((r[0] for r in rows if r[1] == prod), float("nan"))
    print(f"   BEST {rows[0][1]} {rows[0][0]:.2f} us vs prod {pu:.2f} us -> {pu-rows[0][0]:+.2f} us/call, x48 = {(pu-rows[0][0])*48/1e3:.3f} ms/step")
    res[case] = dict(prod=prod, prod_us=pu, best=rows[0][1], best_us=rows[0][0], all=[(r[1], r[0], r[2]) for r in rows])
    del pools, calls; torch.cuda.empty_cache()
json.dump(res, open(f"/w/k1/hotprobe/res_cfg_sweep_rows{ROWS}.json", "w"), indent=1); print("DONE")
