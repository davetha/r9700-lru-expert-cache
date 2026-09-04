#!/usr/bin/env python3
"""bf16 skinny GEMM (r4d_gemm_bf16_nt_m64, best cfg from k3/r4d_cfg_sweep.json) vs the
read-only control on the same bytes, production HC shapes, M=5, 512 MB pool, in-graph."""
import ctypes, json, os, statistics, torch
DEV = "cuda:0"; POOL = 512 << 20; M = 5
lib = ctypes.CDLL("/app/r4dhip/r4d.so"); L, I = ctypes.c_long, ctypes.c_int
lib.r4d_gemm_bf16_nt_m64.argtypes = (L, L, L, I, I, I, I, I, I, L)
pk = ctypes.CDLL("/w/k3/libhcqfp8sk.so"); pk.hcq_stream_probe.restype = ctypes.c_int
pk.hcq_stream_probe.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
sweep = json.load(open("/w/k3/r4d_cfg_sweep.json"))
best = {}
for e in sweep:
    k = e["tag"]
    if k not in best or e["us_n5"] < best[k]["us_n5"]: best[k] = e
SHAPES = [("hc_up", 10240, 320), ("hc_down_merged", 336, 10240), ("hc_down_plain", 320, 10240), ("moe_router", 512, 2560)]
torch.cuda.set_device(0); st = lambda: torch.cuda.current_stream().cuda_stream
out = torch.zeros(4, device=DEV, dtype=torch.int32)
def tg(fns, rounds=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:3]: f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns: f()
    torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize(); v = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); v.append(a.elapsed_time(b) * 1e3 / len(fns))
    del g; return statistics.median(v)
res = {}
for tag, N, K in SHAPES:
    nb = N * K * 2; P = max(4, min(200, POOL // nb))
    pool = [(torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16) for _ in range(P)]
    A = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16); C = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    e = best.get(tag); wv, sk = (e["wv"], e["sk"]) if e else (4, 4)
    g_us = tg([(lambda W: (lambda: lib.r4d_gemm_bf16_nt_m64(A.data_ptr(), W.data_ptr(), C.data_ptr(), M, K, N, wv, sk, 1, st())))(W) for W in pool])
    rbest = None
    for wgs in (32, 64, 128, 256, 512, 1024, 2048):
        us = tg([(lambda W, w=wgs: (lambda: pk.hcq_stream_probe(ctypes.c_void_p(W.data_ptr()), nb, w, ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(st()))))(W) for W in pool])
        if rbest is None or us < rbest[0]: rbest = (us, wgs)
    print(f"{tag:15s} N={N:5d} K={K:5d} {nb/1e6:5.2f} MB x{P}  GEMM cfg({wv},{sk}) {g_us:6.2f} us {nb/g_us/1e3:4.0f} GB/s | READ {rbest[0]:6.2f} us {nb/rbest[0]/1e3:4.0f} GB/s @wgs{rbest[1]} | GEMM/READ {g_us/rbest[0]:.2f}x", flush=True)
    res[tag] = dict(N=N, K=K, gemm_us=g_us, read_us=rbest[0], read_wgs=rbest[1], ratio=g_us / rbest[0])
    del pool; torch.cuda.empty_cache()
json.dump(res, open("/w/k1/hotprobe/res_bf16_ctl.json", "w"), indent=1); print("DONE")
