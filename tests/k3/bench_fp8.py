#!/usr/bin/env python3
"""W8A8 fp8 skinny GEMM (wvSplitKQ, per-TENSOR scales) on the hyper-connection shapes.
Same cold-pool method. This is the only 8-bit skinny path the fork ships."""
import json, os, statistics
import torch
import vllm._custom_ops as ops

DEV, DT = 'cuda:0', torch.bfloat16
POOL_BYTES = 192 << 20
ROUNDS = 10
FP8 = torch.float8_e4m3fn
SHAPES = [(336, 10240, 'hc_down_merged'), (320, 10240, 'hc_down_plain'),
          (10240, 320, 'hc_up'), (512, 2560, 'moe_router')]
res = []


def time_pool(call, ncopies):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call(0)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(ncopies):
            call(i)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    v = []
    for _ in range(ROUNDS):
        t0.record(); g.replay(); t1.record(); torch.cuda.synchronize()
        v.append(t0.elapsed_time(t1) * 1e3 / ncopies)
    del g
    return statistics.median(v)


for M, K, tag in SHAPES:
    print(f'\n=== {tag}  W[{M},{K}] ===', flush=True)
    W = (torch.randn(M, K, device=DEV) * 0.02)
    sb = (W.abs().max() / 448.0).float().reshape(1)
    Wq = (W / sb).to(FP8)
    Wdq = Wq.float() * sb
    nb = max(2, POOL_BYTES // (M * K))
    pool = [Wq.clone() for _ in range(nb)]
    for n in range(1, 6):
        x = torch.randn(n, K, device=DEV)
        sa = (x.abs().max() / 448.0).float().reshape(1)
        xq = (x / sa).to(FP8)
        ref = (xq.float() * sa) @ Wdq.t()
        for cu in (32, 64):
            try:
                o = ops.wvSplitKQ(Wq, xq, DT, sa, sb, cu, None)
                e = ((o.float() - ref).norm() / ref.norm()).item()
                us = time_pool(lambda i: ops.wvSplitKQ(pool[i], xq, DT, sa, sb, cu, None), nb)
                res.append(dict(tag=tag, M=M, K=K, n=n, method=f'fp8_wvSplitKQ_cu{cu}',
                                us=round(us, 3), GBs=round(M * K / us / 1e3, 1), err=e))
                print(f'  wvSplitKQ cu{cu} n={n}  {us:8.2f} us  {M*K/us/1e3:7.1f} GB/s  rel={e:.2e}', flush=True)
            except Exception as ex:
                print(f'  wvSplitKQ cu{cu} n={n}  FAIL {type(ex).__name__}: {str(ex).splitlines()[0][:100]}', flush=True)
                res.append(dict(tag=tag, M=M, K=K, n=n, method=f'fp8_wvSplitKQ_cu{cu}', us=None,
                                err=None, note=str(ex).splitlines()[0][:120]))
    del pool, W, Wq
    torch.cuda.empty_cache()
json.dump(res, open('/w/tests/k3/fp8_results.json', 'w'), indent=1)
print('\nDONE', flush=True)
