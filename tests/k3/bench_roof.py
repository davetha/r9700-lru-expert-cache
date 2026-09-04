#!/usr/bin/env python3
"""The control the fp8 GEMM comparison needs: what a kernel that ONLY reads the same weight
pool, in the same b128 pattern, in the same graph, achieves. If fp8hip and hcq are both at
this number then neither has anything left to win and the production gap is contention, not
the kernel."""
import ctypes, json, os, statistics, sys, torch

sys.path.insert(0, "/app/vllm")
DEV = "cuda:0"
POOL_BYTES = int(os.environ.get("POOL_BYTES", 1 << 30))
SHAPES = [("in_proj_qkvz", 8192, 2560), ("qkv_proj", 6656, 2560), ("out_o_proj", 2560, 3072)]

lib = ctypes.CDLL(os.environ.get("FP8SK_LIB", "/w/build/kernels/libhcqfp8sk.so"))
lib.hcq_stream_probe.restype = ctypes.c_int
lib.hcq_stream_probe.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p]


def probe(w, nbytes, wgs, out):
    rc = lib.hcq_stream_probe(ctypes.c_void_p(w.data_ptr()), nbytes, wgs,
                              ctypes.c_void_p(out.data_ptr()),
                              ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"probe rc={rc}")


def time_graph_pool(mk, pool, rounds=10):
    fns = [mk(W) for W in pool]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:4]:
            f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    torch.cuda.synchronize()
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    vals = []
    for _ in range(rounds):
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record(); g.replay(); t1.record(); torch.cuda.synchronize()
        vals.append(t0.elapsed_time(t1) * 1e3 / len(fns))
    del g, fns
    return statistics.median(vals)


torch.cuda.set_device(0)
out = torch.zeros(4, device=DEV, dtype=torch.int32)
res = {}
for name, N, K in SHAPES:
    nb = N * K
    P = max(4, min(160, POOL_BYTES // nb))
    pool = [torch.randint(0, 255, (N, K), device=DEV, dtype=torch.uint8) for _ in range(P)]
    print(f"\n=== {name} N={N} K={K}  {nb/1e6:.2f} MB x{P} = {P*nb/1e9:.2f} GB", flush=True)
    best = None
    for wgs in (20, 32, 52, 64, 128, 256, 512, 1024, 2048):
        us = time_graph_pool(lambda W, g=wgs: (lambda: probe(W, nb, g, out)), pool)
        gbs = nb / us / 1e3
        print(f"   wgs={wgs:5d}  {us:7.2f} us  {gbs:4.0f} GB/s", flush=True)
        if best is None or us < best[0]:
            best = (us, wgs, gbs)
    res[name] = {"N": N, "K": K, "best_us": best[0], "best_wgs": best[1], "best_gbs": best[2]}
    print(f"   BEST {best[2]:.0f} GB/s at {best[1]} workgroups", flush=True)
    del pool
    torch.cuda.empty_cache()
json.dump(res, open("/w/artifacts/bench_roof.json", "w"), indent=1)
print("\nDONE")
