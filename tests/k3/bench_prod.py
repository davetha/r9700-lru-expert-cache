#!/usr/bin/env python3
"""Production-exact skinny-GEMM mix for Qwen3.8-Flash-Next decode on gfx1201.

Derived from vllm/models/qwen4_exp/amd/{model,hyperconnection}.py:
  per layer: attn_hyper_connection + mlp_hyper_connection, each a GatedResidual
  with use_combine=True -> MergedColumnParallelLinear(disable_tp=True) of
  [hc_lowrank=320, hc_count=4, pad=12] = M 336, K = hidden*hc_count = 10240,
  followed by ReplicatedLinear input_mix_weight_up [10240, 320].
  The two hyper_connection_mixer instances use use_combine=False -> M 320.
  Plus one MoE router [512, 2560] per layer.

  98 x (336,10240) + 2 x (320,10240) + 100 x (10240,320) + 49 x (512,2560) = 249
which is exactly the 249 skinny calls/step seen in the decode profile.
All of these are ReplicatedLinear / disable_tp=True, i.e. NOT TP-split.

Timed cold (each graphed call reads a different weight copy, pool > 2x the 64MB
Infinity Cache) because a decode step touches ~1.3GB of these weights and cannot
keep any of them resident.
"""
import json
import os
import statistics
import torch

import vllm._custom_ops as ops

DEV = "cuda:0"
DT = torch.bfloat16
POOL_BYTES = 192 << 20
CUS = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512]
NS = [1, 2, 3, 4, 5]

MIX = [
    (336,   10240, "hc_down_merged (use_combine)", 98),
    (320,   10240, "hc_down_plain  (mixer)",        2),
    (10240,   320, "hc_up",                       100),
    (512,    2560, "moe_router",                   49),
]


def time_graph_pool(mk, pool, rounds=12):
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
    res = []
    for (M, K, tag, cnt) in MIX:
        nbytes = M * K * 2
        P = max(4, min(200, POOL_BYTES // nbytes))
        pool = [torch.randn(M, K, device=DEV, dtype=DT).contiguous() for _ in range(P)]
        roofr = time_graph_pool(lambda W: (lambda: torch.sum(W)), pool)
        print("\nSHAPE %-30s M=%-6d K=%-5d %.2fMB x%d calls/step  pool=%d  "
              "cold stream read=%.2fus (%.0f GB/s)"
              % (tag, M, K, nbytes / 1e6, cnt, P, roofr, nbytes / roofr / 1e3), flush=True)
        for n in NS:
            X = torch.randn(n, K, device=DEV, dtype=DT).contiguous()
            ref = torch.nn.functional.linear(X.float(), W_ref(pool[0]))
            refmax = max(ref.abs().max().item(), 1e-9)

            def err_of(out):
                return (out.float() - ref).abs().max().item() / refmax

            fl = time_graph_pool(lambda W: (lambda: torch.nn.functional.linear(X, W)), pool)
            e_fl = err_of(torch.nn.functional.linear(X, pool[0]))
            res.append(dict(tag=tag, M=M, K=K, n=n, cnt=cnt, variant="F.linear",
                            cu=None, us=fl, rel=e_fl))
            line = []
            for cu in CUS:
                try:
                    o = ops.wvSplitK(pool[0], X, cu, None)
                    e = err_of(o)
                    us = time_graph_pool(
                        (lambda W, c=cu: (lambda: ops.wvSplitK(W, X, c, None))), pool)
                except Exception as ex:
                    print("  ERR cu=%d: %s" % (cu, repr(ex)[:120]), flush=True)
                    continue
                res.append(dict(tag=tag, M=M, K=K, n=n, cnt=cnt, variant="wvSplitK",
                                cu=cu, us=us, rel=e))
                line.append("%d:%.2f" % (cu, us))
            wv = [r for r in res if r["tag"] == tag and r["n"] == n
                  and r["variant"] == "wvSplitK"]
            b = min(wv, key=lambda r: r["us"])
            rels = sorted({round(r["rel"], 10) for r in wv})
            print("  n=%d F.linear=%6.2f (rel %.2e) | wv@32=%6.2f | best %6.2f@cu%d | "
                  "rel across cu: %s"
                  % (n, fl, e_fl,
                     next(r["us"] for r in wv if r["cu"] == 32), b["us"], b["cu"],
                     ("CONSTANT %.2e" % rels[0]) if len(rels) == 1 else "VARIES %s" % rels),
                  flush=True)
            print("     " + "  ".join(line), flush=True)
            del X
        del pool
        torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("PROJECTED DECODE-STEP COST OF THE 249 SKINNY CALLS (cold, us/step)")
    print("=" * 90)
    for n in NS:
        tot = {}
        for cu in CUS:
            s = 0.0
            ok = True
            for (M, K, tag, cnt) in MIX:
                r = [x for x in res if x["tag"] == tag and x["n"] == n
                     and x["variant"] == "wvSplitK" and x["cu"] == cu]
                if not r:
                    ok = False
                    break
                s += r[0]["us"] * cnt
            if ok:
                tot[cu] = s
        flt = sum(next(x["us"] for x in res if x["tag"] == tag and x["n"] == n
                       and x["variant"] == "F.linear") * cnt
                  for (M, K, tag, cnt) in MIX)
        per = sum(min(x["us"] for x in res if x["tag"] == tag and x["n"] == n
                      and x["variant"] == "wvSplitK") * cnt
                  for (M, K, tag, cnt) in MIX)
        best = min(tot, key=lambda c: tot[c])
        print("n=%d  cu32(current)=%7.1f  best_single=cu%-4d %7.1f (%+.1f%%)  "
              "per_shape_opt=%7.1f (%+.1f%%)  F.linear_all=%7.1f (%+.1f%%)"
              % (n, tot[32], best, tot[best], 100 * (tot[best] / tot[32] - 1),
                 per, 100 * (per / tot[32] - 1), flt, 100 * (flt / tot[32] - 1)))
        print("     full cu curve: " +
              "  ".join("%d:%.0f" % (c, tot[c]) for c in sorted(tot)))

    dst = os.environ.get("OUT", "/w/tests/k3/prod_results.json")
    with open(dst, "w") as f:
        json.dump(dict(mix=MIX, cus=CUS, results=res), f)
    print("WROTE " + dst, flush=True)


def W_ref(W):
    return W.float()


if __name__ == "__main__":
    main()
