"""Verify + time the r4d bf16 skinny-GEMM dispatch in the patched
model_executor/layers/utils.py, at the four production decode shapes.

Correctness is against F.linear on the same operands. Timing is cold: a pool of
distinct weight copies larger than the 64 MB Infinity Cache, one graphed call
per copy, so the weight read is a real HBM read the way it is in a decode step.
"""
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.utils as U

SHAPES = [
    (336, 10240, "hc_down_merged", 98),
    (320, 10240, "hc_down_plain", 2),
    (10240, 320, "hc_up", 100),
    (512, 2560, "moe_router", 49),
]
POOL_BYTES = 192 << 20
DEV = "cuda:0"
ARM = os.environ.get("ARM", "r4d")
OUT = os.environ.get("OUT", "/w/tests/k3/r4dpatch_%s.json" % ARM)


def time_graph_pool(call, ncopies, rounds=10):
    for _ in range(3):
        for i in range(ncopies):
            call(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(ncopies):
            call(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for i in range(ncopies):
            call(i)
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / ncopies)
    del g
    return statistics.median(ts)


def main():
    torch.manual_seed(0)
    rows = []
    print("arm=%s VLLM_HC_R4D_BF16=%s" % (ARM, os.environ.get("VLLM_HC_R4D_BF16", "1")))
    print("r4d fn: %r  max_m=%d" % (U._r4d_bf16_gemm_fn(), U._R4D_BF16_MAX_M))

    for N, K, tag, ncalls in SHAPES:
        wbytes = N * K * 2
        ncopies = max(2, min(64, (POOL_BYTES + wbytes - 1) // wbytes))
        pool = [torch.randn(N, K, device=DEV, dtype=torch.bfloat16) / 8 for _ in range(ncopies)]
        cfg = U._r4d_bf16_cfg(N, K)
        print("\n== %s  N=%d K=%d  copies=%d  cfg=%r ==" % (tag, N, K, ncopies, cfg))
        for m in range(1, 6):
            x = (torch.randn(m, K, device=DEV, dtype=torch.bfloat16) / 8).contiguous()
            W = pool[0]
            ref = F.linear(x.float(), W.float())
            got = U.rocm_unquantized_gemm_impl(x, W, None)
            err = ((got.float() - ref).norm() / ref.norm()).item()
            engaged = U._r4d_bf16_skinny(x, W, None) is not None
            us = time_graph_pool(lambda i: U.rocm_unquantized_gemm_impl(x, pool[i], None), ncopies)
            gbs = wbytes / (us * 1e-6) / 1e9
            rows.append(dict(tag=tag, N=N, K=K, m=m, arm=ARM, r4d_engaged=engaged,
                             us=round(us, 3), GBs=round(gbs, 1), rel_err=err,
                             cfg=cfg, ncalls=ncalls))
            print("  m=%d  r4d_engaged=%-5s  %7.3f us  %6.1f GB/s  rel_err=%.2e"
                  % (m, engaged, us, gbs, err))
            if err > 3e-2:
                sys.exit("FAIL: %s m=%d rel_err=%.3e" % (tag, m, err))
        del pool
        torch.cuda.empty_cache()

    # fallback contracts: bias and fp16 must never reach the r4d kernel
    W = torch.randn(336, 10240, device=DEV, dtype=torch.bfloat16) / 8
    x = torch.randn(5, 10240, device=DEV, dtype=torch.bfloat16) / 8
    b = torch.randn(336, device=DEV, dtype=torch.bfloat16)
    assert U._r4d_bf16_skinny(x, W, b) is None, "bias must fall back"
    got = U.rocm_unquantized_gemm_impl(x, W, b)
    ref = F.linear(x.float(), W.float(), b.float())
    e_bias = ((got.float() - ref).norm() / ref.norm()).item()
    Wh, xh = W.half(), x.half()
    assert U._r4d_bf16_skinny(xh, Wh, None) is None, "fp16 must fall back"
    got = U.rocm_unquantized_gemm_impl(xh, Wh, None)
    ref = F.linear(xh.float(), Wh.float())
    e_fp16 = ((got.float() - ref).norm() / ref.norm()).item()
    print("\nfallbacks: bias rel_err=%.2e  fp16 rel_err=%.2e" % (e_bias, e_fp16))
    assert e_bias < 3e-2 and e_fp16 < 3e-2

    # a K the kernel cannot split (K % 16 != 0) must produce no config
    assert U._r4d_bf16_cfg(64, 8 * 5) is None, "K%16!=0 must yield no config"
    print("legality: K%16!=0 -> no config, OK")

    json.dump(rows, open(OUT, "w"), indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
