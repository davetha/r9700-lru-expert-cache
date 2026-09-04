#!/usr/bin/env python3
"""Cold-cache (DRAM-resident) companion to bench_skinny.py, plus torch.profiler
kernel-time for the top candidates.

bench_skinny.py replays the same weight tensor back-to-back, so it stays in the
64MB Infinity Cache of Navi48 (roofline shows >1.3 TB/s, above the ~640 GB/s
DRAM ceiling).  Here each call in the captured graph reads a *different* copy of
the weight, with the pool sized > 2x MALL, so every read comes from DRAM -- which
is what a real decode step does (the MoE expert stream evicts everything).
"""
import json
import os
import statistics
import torch

import vllm._custom_ops as ops

DEV = "cuda:0"
DT = torch.bfloat16
POOL_BYTES = 192 << 20  # > 2x the 64MB Infinity Cache
CUS = [16, 32, 48, 64, 96, 128, 160, 256, 384]

SHAPES = [
    (320,   10240, "hc_input_mix_down",        100),
    (160,   10240, "hc_input_mix_down /2M",    100),
    (10240,   320, "hc_input_mix_up",          100),
    (5120,    320, "hc_input_mix_up /2M",      100),
    (512,    2560, "mlp.gate router",           51),
    (256,    2560, "mlp.gate router /2M",       51),
    (48,     2560, "linear_attn.in_proj_a",     72),
    (640,    2560, "indexer/shared_exp",        15),
    (2560,   2560, "fc_embedding/ple_value",     3),
    (10240,  2560, "ple.key_proj",               1),
    (12288,  2560, "mtp.q_proj",                 1),
    (6144,   2560, "mtp.q_proj /2M",             1),
]
NS = [1, 5]


def time_graph_pool(mk, pool, rounds=10):
    """mk(W) -> callable.  Capture one call per pool entry; return per-call us."""
    fns = [mk(W) for W in pool]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:4]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
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
        t0 = torch.cuda.Event(True)
        t1 = torch.cuda.Event(True)
        t0.record()
        g.replay()
        t1.record()
        torch.cuda.synchronize()
        vals.append(t0.elapsed_time(t1) * 1e3 / len(fns))
    del g, fns
    return statistics.median(vals)


def main():
    torch.cuda.set_device(0)
    p = torch.cuda.get_device_properties(0)
    print("META " + json.dumps({"gpu": p.name, "arch": p.gcnArchName,
                                "mpc": p.multi_processor_count}), flush=True)
    out = []
    for (M, K, tag, ncalls) in SHAPES:
        nbytes = M * K * 2
        P = max(4, min(256, POOL_BYTES // nbytes))
        pool = [torch.randn(M, K, device=DEV, dtype=DT).contiguous() for _ in range(P)]
        print("SHAPE %-26s M=%-6d K=%-5d %.2fMB  pool=%d (%.0fMB)"
              % (tag, M, K, nbytes / 1e6, P, P * nbytes / 1e6), flush=True)

        roof = time_graph_pool(lambda W: (lambda: torch.sum(W)), pool)
        print("  roof   sum        = %8.2fus  %6.0f GB/s" % (roof, nbytes / roof / 1e3),
              flush=True)
        out.append(dict(tag=tag, M=M, K=K, n=None, variant="sum(roof)", cu=None,
                        us=roof, GBs=nbytes / roof / 1e3, pool=P))

        for n in NS:
            X = torch.randn(n, K, device=DEV, dtype=DT).contiguous()
            rows = []

            def rec(variant, mk, cu=None):
                try:
                    us = time_graph_pool(mk, pool)
                except Exception as e:
                    print("  ERR %s n=%d cu=%s: %s" % (variant, n, cu, repr(e)[:120]),
                          flush=True)
                    return
                r = dict(tag=tag, M=M, K=K, n=n, variant=variant, cu=cu,
                         us=us, GBs=nbytes / us / 1e3, pool=P)
                out.append(r)
                rows.append(r)

            rec("F.linear", lambda W: (lambda: torch.nn.functional.linear(X, W)))
            if M > 8:
                for cu in CUS:
                    rec("wvSplitK",
                        (lambda W, c=cu: (lambda: ops.wvSplitK(W, X, c, None))), cu)
            for r in rows:
                print("  n=%d %-10s cu=%-4s %8.2fus  %6.0f GB/s"
                      % (n, r["variant"], r["cu"] if r["cu"] else "-", r["us"], r["GBs"]),
                      flush=True)
            fl = next(r["us"] for r in rows if r["variant"] == "F.linear")
            wv = [r for r in rows if r["variant"] == "wvSplitK"]
            if wv:
                b = min(wv, key=lambda r: r["us"])
                print("  => n=%d  F.linear=%.2f  wv@32=%.2f  wv_best=%.2f@cu%d  winner=%s"
                      % (n, fl,
                         next(r["us"] for r in wv if r["cu"] == 32),
                         b["us"], b["cu"],
                         "F.linear" if fl <= b["us"] else "wvSplitK@%d" % b["cu"]),
                      flush=True)
            del X
        del pool
        torch.cuda.empty_cache()

    # ---- torch.profiler kernel time (eager, no graph) for the top shapes ----
    print("\n=== torch.profiler eager kernel durations (n=5) ===", flush=True)
    prof_rows = []
    for (M, K, tag) in [(320, 10240, "hc_input_mix_down"),
                        (10240, 320, "hc_input_mix_up"),
                        (12288, 2560, "mtp.q_proj")]:
        W = torch.randn(M, K, device=DEV, dtype=DT).contiguous()
        X = torch.randn(5, K, device=DEV, dtype=DT).contiguous()
        cands = [("F.linear", lambda: torch.nn.functional.linear(X, W)),
                 ("wvSplitK@32", lambda: ops.wvSplitK(W, X, 32, None)),
                 ("wvSplitK@64", lambda: ops.wvSplitK(W, X, 64, None)),
                 ("wvSplitK@160", lambda: ops.wvSplitK(W, X, 160, None))]
        for name, fn in cands:
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]) as pr:
                for _ in range(50):
                    fn()
                torch.cuda.synchronize()
            ev = [e for e in pr.key_averages() if e.device_time_total > 0]
            tot = sum(e.device_time_total for e in ev) / 50.0
            top = sorted(ev, key=lambda e: -e.device_time_total)[:2]
            desc = "; ".join("%s %.2fus" % (e.key[:40], e.device_time_total / 50.0)
                             for e in top)
            prof_rows.append(dict(tag=tag, M=M, K=K, variant=name,
                                  kernel_us=tot, detail=desc))
            print("  %-20s %-13s kernel=%8.2fus  | %s" % (tag, name, tot, desc),
                  flush=True)
        del W, X
        torch.cuda.empty_cache()

    dst = os.environ.get("OUT", "/w/tests/k3/cold_results.json")
    with open(dst, "w") as f:
        json.dump(dict(cold=out, profiler=prof_rows), f)
    print("WROTE " + dst, flush=True)


if __name__ == "__main__":
    main()
