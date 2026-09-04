#!/usr/bin/env python3
"""Cold-cache microbenchmark of the W4A16 skinny GEMM (wvSplitK_int4_g) against the
bf16 wvSplitK on the exact hyper-connection shapes of q38fn, n=1..5.

Cold = a pool of distinct weight copies larger than the 64 MB Infinity Cache, one
graphed call per copy, so the measured bandwidth is DRAM and not MALL.
"""
import json, os, statistics, sys, time
import torch
import torch.nn.functional as F
import vllm._custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
    pack_int4_exllama_shuffle, triton_w4a16_skinny_fmt_gemm)

DEV = 'cuda:0'
DT = torch.bfloat16
POOL_BYTES = 192 << 20
ROUNDS = 10
OUT = os.environ.get('OUT', '/w/tests/k3/int4_results.json')

# (M=out_features, K=in_features, tag, calls per decode step)
SHAPES = [
    (336, 10240, 'hc_down_merged', 98),
    (320, 10240, 'hc_down_plain', 2),
    (10240, 320, 'hc_up', 100),
    (512, 2560, 'moe_router', 49),
]
res = []


def quant_asym(W, g):
    """AWQ-style asymmetric uint4, group g along K. Returns packed int8 [N,K/2], scales, zp, dequant."""
    N, K = W.shape
    x = W.float().view(N, K // g, g)
    mn, mx = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    s = ((mx - mn) / 15.0).clamp(min=1e-8)
    z = (-mn / s).round().clamp(0, 15)
    q = (x / s + z).round().clamp(0, 15)
    dq = ((q - z) * s).view(N, K).to(DT)
    packed = pack_int4_exllama_shuffle(q.view(N, K).to(torch.uint8)).contiguous().view(torch.int8)
    return (packed, s.squeeze(-1).to(DT).contiguous(), z.squeeze(-1).to(DT).contiguous(), dq)


def time_graph_pool(call, ncopies):
    """call(i) issues one kernel against pool copy i. Returns median us per call."""
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
    vals = []
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(ROUNDS):
        t0.record(); g.replay(); t1.record(); torch.cuda.synchronize()
        vals.append(t0.elapsed_time(t1) * 1e3 / ncopies)
    del g
    return statistics.median(vals)


def rec(shape_tag, M, K, n, method, us, gbytes, err=None, note=''):
    r = dict(tag=shape_tag, M=M, K=K, n=n, method=method,
             us=None if us is None else round(us, 3),
             GBs=None if us is None else round(gbytes / (us * 1e-6) / 1e9, 1),
             err=err, note=note)
    res.append(r)
    txt = 'FAIL' if us is None else '%8.2f us  %7.1f GB/s' % (us, r['GBs'])
    ee = '' if err is None else '  rel=%.2e' % err
    print('%-16s %6dx%-6d n=%d %-22s %s%s  %s' % (shape_tag, M, K, n, method, txt, ee, note), flush=True)


def bench_shape(M, K, tag):
    print(f'\n=== {tag}  W[{M},{K}] ===', flush=True)
    Wref = (torch.randn(M, K, device=DEV) * 0.02).to(DT)
    bf16_bytes = M * K * 2
    nbf = max(2, POOL_BYTES // bf16_bytes)
    Wpool = [Wref.clone() for _ in range(nbf)]
    xs = {n: torch.randn(n, K, device=DEV, dtype=DT) for n in range(1, 6)}
    fp32ref = {n: xs[n].float() @ Wref.float().t() for n in range(1, 6)}

    for n in range(1, 6):
        x = xs[n]
        rec(tag, M, K, n, 'bf16_F.linear', time_graph_pool(lambda i: F.linear(x, Wpool[i]), nbf), bf16_bytes)
        try:
            o = ops.wvSplitK(Wref, x, 32, None)
            e = ((o.float() - fp32ref[n]).norm() / fp32ref[n].norm()).item()
            rec(tag, M, K, n, 'bf16_wvSplitK_cu32',
                time_graph_pool(lambda i: ops.wvSplitK(Wpool[i], x, 32, None), nbf), bf16_bytes, e)
        except Exception as ex:
            rec(tag, M, K, n, 'bf16_wvSplitK_cu32', None, bf16_bytes, note=str(ex)[:90])
    del Wpool
    torch.cuda.empty_cache()

    for g in (128, 64):
        if K % g:
            print(f'  group {g}: K={K} not divisible — skipped', flush=True)
            continue
        packed, sc, zp, dq = quant_asym(Wref, g)
        i4_bytes = packed.numel() + sc.numel() * 2 + zp.numel() * 2
        ni4 = max(2, POOL_BYTES // i4_bytes)
        pp = [packed.clone() for _ in range(ni4)]
        ps = [sc.clone() for _ in range(ni4)]
        pz = [zp.clone() for _ in range(ni4)]
        dqref = {n: xs[n].float() @ dq.float().t() for n in range(1, 6)}
        for n in range(1, 6):
            x = xs[n]
            for cu in (32, 64, 128):
                try:
                    o = ops.wvSplitK_int4_g(packed, x, sc, cu, g, zp, None)
                    e = ((o.float() - dqref[n]).norm() / dqref[n].norm()).item()
                    us = time_graph_pool(
                        lambda i: ops.wvSplitK_int4_g(pp[i], x, ps[i], cu, g, pz[i], None), ni4)
                    rec(tag, M, K, n, f'int4_g{g}_cu{cu}', us, i4_bytes, e)
                except Exception as ex:
                    rec(tag, M, K, n, f'int4_g{g}_cu{cu}', None, i4_bytes, note=type(ex).__name__ + ': ' + str(ex).split('\n')[0][:110])
            # triton fallback (what production would run when the skinny path is illegal)
            try:
                bq = [p.view(torch.int32) for p in pp]
                o = triton_w4a16_skinny_fmt_gemm(a=x, b_q=bq[0], scales=sc, group_size=g, zp=zp)
                e = ((o.float() - dqref[n]).norm() / dqref[n].norm()).item()
                us = time_graph_pool(
                    lambda i: triton_w4a16_skinny_fmt_gemm(a=x, b_q=bq[i], scales=ps[i], group_size=g, zp=pz[i]), ni4)
                rec(tag, M, K, n, f'triton_w4a16_g{g}', us, i4_bytes, e)
            except Exception as ex:
                rec(tag, M, K, n, f'triton_w4a16_g{g}', None, i4_bytes, note=type(ex).__name__ + ': ' + str(ex).split('\n')[0][:110])
        del pp, ps, pz
        torch.cuda.empty_cache()
    del Wref
    torch.cuda.empty_cache()


def main():
    print(torch.cuda.get_device_name(0), torch.__version__, flush=True)
    only = os.environ.get('ONLY')
    for M, K, tag, _ in SHAPES:
        if only and only not in tag:
            continue
        bench_shape(M, K, tag)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT}  ({len(res)} rows)', flush=True)


main()
