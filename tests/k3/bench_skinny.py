#!/usr/bin/env python3
"""Microbenchmark vLLM's ROCm skinny bf16 GEMM (wvSplitK) vs alternatives on gfx1201.

Call convention (csrc/rocm/skinny_gemms.cu::wvSplitK):
    ops.wvSplitK(weight[M,K] bf16 contiguous, x[n,K] bf16 contiguous, cu_count, bias|None) -> [n, M]
where M = out_features, n = rows (tokens).  The kernel launches grid=(cu_count).

Two timings are reported per variant:
  eager_us : per-call wall time in eager mode (CUDA events around each call).
             For small shapes this is CPU-dispatch bound, NOT kernel time.
  graph_us : per-call time inside a captured HIP graph (REPS calls captured,
             replayed, total/REPS).  This is what vLLM actually pays in a
             cudagraph-captured decode step, and is the number that matters.
"""
import json
import os
import statistics
import torch

import vllm._custom_ops as ops  # registers torch.ops._rocm_C

DEV = "cuda:0"
DT = torch.bfloat16
NS = [1, 2, 3, 4, 5]
CUS = [16, 32, 48, 64, 80, 96, 128, 160, 192, 200, 256, 384, 512]

# (M=out_features, K=in_features, tag, calls_per_decode_step_at_TP1)
SHAPES = [
    (320,   10240, "hc_input_mix_down",        100),
    (160,   10240, "hc_input_mix_down /2M",    100),
    (320,    5120, "hc_input_mix_down /2K",    100),
    (10240,   320, "hc_input_mix_up",          100),
    (5120,    320, "hc_input_mix_up /2M",      100),
    (10240,   160, "hc_input_mix_up /2K",      100),
    (512,    2560, "mlp.gate router",           51),
    (256,    2560, "mlp.gate router /2M",       51),
    (48,     2560, "linear_attn.in_proj_a",     72),
    (24,     2560, "linear_attn.in_proj_a /2M", 72),
    (640,    2560, "indexer/shared_exp",        15),
    (320,    2560, "indexer/shared_exp /2M",    15),
    (2560,    640, "shared_exp.down_proj",       1),
    (2560,    320, "shared_exp.down /2K",        1),
    (2560,   2560, "fc_embedding/ple_value",     3),
    (1280,   2560, "fc_embedding /2M",           3),
    (10240,  2560, "ple.key_proj",               1),
    (5120,   2560, "ple.key_proj /2M",           1),
    (12288,  2560, "mtp.q_proj",                 1),
    (6144,   2560, "mtp.q_proj /2M",             1),
    (2560,   6144, "mtp.o_proj",                 1),
    (2560,   3072, "mtp.o_proj /2K",             1),
    (248320, 2560, "lm_head",                    1),
    (124160, 2560, "lm_head /2M (TP2)",          1),
]

HAVE_AITER_TRITON = False
HAVE_AITER_TGEMM = False
AITER_TRITON_ERR = ""
AITER_TGEMM_ERR = ""
try:
    from aiter.ops.triton.gemm_a16w16 import gemm_a16w16  # noqa: F401
    HAVE_AITER_TRITON = True
except Exception as e:
    AITER_TRITON_ERR = repr(e)[:200]
try:
    from aiter.tuned_gemm import tgemm  # noqa: F401
    HAVE_AITER_TGEMM = True
except Exception as e:
    AITER_TGEMM_ERR = repr(e)[:200]


def eager_iters(nbytes):
    return 200 if nbytes < (64 << 20) else 30


def graph_reps(nbytes):
    if nbytes > (256 << 20):
        return 4
    if nbytes > (32 << 20):
        return 16
    return 50


def time_eager(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(True) for _ in range(iters)]
    en = [torch.cuda.Event(True) for _ in range(iters)]
    for i in range(iters):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    per = sorted(st[i].elapsed_time(en[i]) * 1e3 for i in range(iters))
    return per[len(per) // 2]


def time_graph(fn, reps, rounds=10):
    """Capture `reps` back-to-back calls into a HIP graph; return per-call us."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    vals = []
    for _ in range(rounds):
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        t0.record()
        g.replay()
        t1.record()
        torch.cuda.synchronize()
        vals.append(t0.elapsed_time(t1) * 1e3 / reps)
    del g
    return statistics.median(vals)


def main():
    torch.cuda.set_device(0)
    p = torch.cuda.get_device_properties(0)
    meta = {
        "gpu": p.name,
        "gcn_arch": getattr(p, "gcnArchName", "?"),
        "multi_processor_count": p.multi_processor_count,
        "torch": torch.__version__,
        "aiter_triton": True if HAVE_AITER_TRITON else AITER_TRITON_ERR,
        "aiter_tgemm": True if HAVE_AITER_TGEMM else AITER_TGEMM_ERR,
    }
    print("META " + json.dumps(meta), flush=True)

    results = []
    roof = []
    shapes = SHAPES[int(os.environ.get("SKIP", 0)):
                    int(os.environ.get("NSHAPES", len(SHAPES)))]
    for (M, K, tag, ncalls) in shapes:
        W = torch.randn(M, K, device=DEV, dtype=DT).contiguous()
        nbytes = W.numel() * 2
        eit = eager_iters(nbytes)
        gr = graph_reps(nbytes)

        g_sum = time_graph(lambda: torch.sum(W), gr)
        roof.append(dict(tag=tag, M=M, K=K, MB=nbytes / 1e6,
                         sum_graph_us=g_sum, sum_GBs=nbytes / g_sum / 1e3))
        print("ROOF %-26s M=%-6d K=%-5d %8.2fMB  sum(in-graph)=%8.2fus -> %.0f GB/s"
              % (tag, M, K, nbytes / 1e6, g_sum, nbytes / g_sum / 1e3), flush=True)

        for n in NS:
            X = torch.randn(n, K, device=DEV, dtype=DT).contiguous()
            ref = torch.nn.functional.linear(X.float(), W.float())
            refmax = max(ref.abs().max().item(), 1e-9)

            def rec(variant, fn, cu=None):
                try:
                    out = fn()
                    torch.cuda.synchronize()
                except Exception as e:
                    results.append(dict(tag=tag, M=M, K=K, n=n, variant=variant,
                                        cu=cu, err=repr(e)[:160]))
                    print("  ERR %s n=%d %s cu=%s: %s" % (tag, n, variant, cu, repr(e)[:120]),
                          flush=True)
                    return
                if tuple(out.shape) != (n, M):
                    results.append(dict(tag=tag, M=M, K=K, n=n, variant=variant, cu=cu,
                                        err="bad shape %s" % (tuple(out.shape),)))
                    print("  BADSHAPE %s n=%d %s cu=%s -> %s" % (tag, n, variant, cu, tuple(out.shape)),
                          flush=True)
                    return
                err = (out.float() - ref).abs().max().item()
                eg = time_eager(fn, eit)
                try:
                    gg = time_graph(fn, gr)
                except Exception as e:
                    gg = float("nan")
                    print("  GRAPHERR %s n=%d %s cu=%s: %s" % (tag, n, variant, cu, repr(e)[:120]),
                          flush=True)
                results.append(dict(tag=tag, M=M, K=K, n=n, variant=variant, cu=cu,
                                    eager_us=eg, graph_us=gg, maxabs=err, rel=err / refmax,
                                    graph_GBs=nbytes / gg / 1e3))

            rec("F.linear", lambda: torch.nn.functional.linear(X, W))
            if M > 8:
                for cu in CUS:
                    rec("wvSplitK", (lambda c=cu: ops.wvSplitK(W, X, c, None)), cu)
            if n == 1 and M % 4 == 0 and K <= 8192:
                rec("LLMM1", lambda: ops.LLMM1(W, X, 4))
            if HAVE_AITER_TRITON:
                rec("aiter_a16w16", lambda: gemm_a16w16(X, W))
            if HAVE_AITER_TGEMM:
                rec("aiter_tgemm", lambda: tgemm.mm(X, W, None))

            row = [r for r in results
                   if r.get("tag") == tag and r.get("n") == n and "graph_us" in r
                   and r["graph_us"] == r["graph_us"]]
            fl = next((r["graph_us"] for r in row if r["variant"] == "F.linear"), float("nan"))
            wv = [r for r in row if r["variant"] == "wvSplitK"]
            wvb = min(wv, key=lambda r: r["graph_us"]) if wv else None
            wv32 = next((r for r in wv if r["cu"] == 32), None)
            best = min(row, key=lambda r: r["graph_us"])
            print("  %-26s n=%d | F.linear=%8.2f | wv@32=%8.2f | wv_best=%8.2f@cu%-4s | BEST=%s%s %.2fus"
                  % (tag, n, fl,
                     wv32["graph_us"] if wv32 else float("nan"),
                     wvb["graph_us"] if wvb else float("nan"),
                     str(wvb["cu"]) if wvb else "-",
                     best["variant"],
                     ("@" + str(best["cu"])) if best.get("cu") else "",
                     best["graph_us"]), flush=True)
            del X
        del W
        torch.cuda.empty_cache()

    out = dict(meta=meta, roofline=roof, results=results, shapes=SHAPES)
    dst = os.environ.get("OUT", "/w/tests/k3/bench_results.json")
    with open(dst, "w") as f:
        json.dump(out, f)
    print("WROTE " + dst, flush=True)


if __name__ == "__main__":
    main()
