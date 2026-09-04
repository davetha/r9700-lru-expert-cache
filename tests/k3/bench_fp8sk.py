#!/usr/bin/env python3
"""hcq_gemm_fp8blk_nt_m16 vs the closed fp8hip_gemm_w8a8_tiled at the four production
fp8 block-scaled shapes of Qwen3.8-Flash-Next on 2x R9700, TP2, per rank.

Both kernels read the SAME pre-shuffled weight tensor and the same scales, both write into a
pre-allocated bf16 output, and both are timed in-graph over a pool large enough that the weight
cannot sit in the 64 MB Infinity Cache -- which is the whole point: in production 96 distinct
weights totalling ~1.3 GB stream per decode step, so a single-weight loop measures cache.
"""
import ctypes, json, os, statistics, sys, torch

sys.path.insert(0, "/app/vllm")
from vllm.model_executor.kernels.linear.scaled_mm.fp8hip import shuffle_weight_gfx1201
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

DEV = "cuda:0"
FP8 = torch.float8_e4m3fn
POOL_BYTES = int(os.environ.get("POOL_BYTES", 1 << 30))
CALLS = {"in_proj_qkvz": 36, "qkv_proj": 12, "out_proj": 48}   # per decode step, per rank
# (name, N, K, calls/step)   N/K are PER RANK (TP2)
SHAPES = [
    ("in_proj_qkvz", 8192, 2560, 36),
    ("qkv_proj",     6656, 2560, 12),
    ("out_o_proj",   2560, 3072, 48),
]
NS = [int(v) for v in os.environ.get("FP8SK_NS", "1,2,4,5,8,16").split(",")]

_hcq = ctypes.CDLL(os.environ.get("FP8SK_LIB", "/w/build/kernels/libhcqfp8sk.so"))
_hcq.hcq_gemm_fp8blk_nt_m16.restype = ctypes.c_int
_hcq.hcq_gemm_fp8blk_nt_m16.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
_fp8hip = ctypes.CDLL(os.environ.get("FP8HIP_LIB", "/app/fp8hip/libfp8hip_gemm.so"))
_fp8hip.fp8hip_gemm_w8a8_launch.restype = ctypes.c_int
_fp8hip.fp8hip_gemm_w8a8_launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]


def hcq(qx, w, xs, ws, y, M, K, N, wv, sk, npw):
    rc = _hcq.hcq_gemm_fp8blk_nt_m16(
        ctypes.c_void_p(qx.data_ptr()), ctypes.c_void_p(xs.data_ptr()),
        ctypes.c_void_p(w.data_ptr()), ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_void_p(y.data_ptr()), M, K, N, wv, sk, npw,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"hcq rc={rc} M={M} K={K} N={N} WV={wv} SK={sk} NPW={npw}")


def blt(qx, w, xs, ws, y, M, K, N):
    rc = _fp8hip.fp8hip_gemm_w8a8_launch(
        ctypes.c_void_p(qx.data_ptr()), ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(xs.data_ptr()), ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_void_p(y.data_ptr()), M, N, K,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if rc != 0:
        raise RuntimeError(f"fp8hip rc={rc}")


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


def make_weight(N, K):
    """A realistic 128x128-block fp8 weight: quantise a bf16 normal so the scales vary."""
    wref = torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.02
    blk = wref.view(N // 128, 128, K // 128, 128)
    amax = blk.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    ws = (amax / 448.0).float().contiguous()                       # [N/128, K/128] fp32
    wq = (blk / ws[:, None, :, None]).clamp(-448, 448).to(FP8)
    wq = wq.view(N, K).contiguous()
    return wq, ws


def cfgs(K):
    nkb = K // 128
    out = []
    for wv in (1, 2, 4, 8):
        for sk in (1, 2, 4, 8):
            if nkb % sk or wv * sk * 32 > 1024:
                continue
            for npw in (1, 2, 4):
                if (wv * npw * sk * 256 + 16 * nkb) * 4 > 64 * 1024:
                    continue
                out.append((wv, sk, npw))
    return out


def main():
    torch.cuda.set_device(0)
    p = torch.cuda.get_device_properties(0)
    print("META " + json.dumps({"gpu": p.name, "arch": p.gcnArchName,
                                "cus": p.multi_processor_count}), flush=True)
    results = {}
    for name, N, K, calls in SHAPES:
        wbytes = N * K
        P = max(4, min(160, POOL_BYTES // wbytes))
        wq0, ws0 = make_weight(N, K)
        pool_w, pool_s = [], []
        for i in range(P):
            if i == 0:
                wq, ws = wq0, ws0
            else:
                wq, ws = make_weight(N, K)
            pool_w.append(shuffle_weight_gfx1201(wq).contiguous())
            pool_s.append(ws)
            if i:
                del wq
        torch.cuda.empty_cache()
        roof = time_graph_pool(lambda W: (lambda: torch.sum(W.view(torch.uint8))), pool_w)
        print(f"\n=== {name}  N={N} K={K}  weight {wbytes/1e6:.2f} MB  x{calls} calls/step  "
              f"pool={P} ({P*wbytes/1e9:.2f} GB)\n    pure stream read {roof:.2f} us "
              f"= {wbytes/roof/1e3:.0f} GB/s (roofline probe)", flush=True)

        rows = {}
        for M in NS:
            x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
            qx, xs = per_token_group_quant_fp8(x, group_size=128, dtype=FP8)
            qx = qx.contiguous(); xs = xs.contiguous()
            assert xs.shape == (M, K // 128) and xs.dtype == torch.float32, (xs.shape, xs.dtype)
            y = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)

            # fp32 reference on the unshuffled weight
            ref = ((qx.float() * xs.repeat_interleave(128, 1))
                   @ (wq0.float() * ws0.repeat_interleave(128, 0).repeat_interleave(128, 1)).T)
            refmax = max(ref.abs().max().item(), 1e-9)

            def rel(o):
                return (o.float() - ref).abs().max().item() / refmax

            blt(qx, pool_w[0], xs, pool_s[0], y, M, K, N)
            torch.cuda.synchronize()
            y_blt = y.clone()
            t_blt = time_graph_pool(
                lambda W, S=None: (lambda: blt(qx, W, xs, pool_s[0], y, M, K, N)), pool_w)

            best = None
            for wv, sk, npw in cfgs(K):
                y.zero_()
                hcq(qx, pool_w[0], xs, pool_s[0], y, M, K, N, wv, sk, npw)
                torch.cuda.synchronize()
                r = rel(y)
                if r > 5e-2:
                    print(f"    !! WV={wv} SK={sk} NPW={npw} rel={r:.3e} SKIPPED", flush=True)
                    continue
                t = time_graph_pool(
                    lambda W, a=wv, b=sk, c=npw:
                        (lambda: hcq(qx, W, xs, pool_s[0], y, M, K, N, a, b, c)), pool_w)
                if best is None or t < best[0]:
                    best = (t, wv, sk, npw, r, y.clone())
            t_hcq, wv, sk, npw, r_hcq, y_hcq = best
            row = {"M": M, "fp8hip_us": t_blt, "hcq_us": t_hcq, "cfg": [wv, sk, npw],
                   "fp8hip_rel": rel(y_blt), "hcq_rel": r_hcq,
                   "hcq_vs_fp8hip_maxabs": (y_hcq.float() - y_blt.float()).abs().max().item(),
                   "hcq_eq_fp8hip": bool(torch.equal(y_hcq, y_blt)),
                   "fp8hip_gbs": wbytes / t_blt / 1e3, "hcq_gbs": wbytes / t_hcq / 1e3}
            rows[M] = row
            print(f"  M={M:3d}  fp8hip {t_blt:7.2f}us ({row['fp8hip_gbs']:4.0f} GB/s)  "
                  f"hcq {t_hcq:7.2f}us ({row['hcq_gbs']:4.0f} GB/s)  "
                  f"{t_blt/t_hcq:4.2f}x  cfg WV={wv} SK={sk} NPW={npw}  "
                  f"rel blas={row['fp8hip_rel']:.2e} hcq={r_hcq:.2e}  "
                  f"|hcq-fp8hip|={row['hcq_vs_fp8hip_maxabs']:.3e}", flush=True)
            del x, qx, xs, y, ref, y_blt, y_hcq
            torch.cuda.empty_cache()
        results[name] = {"N": N, "K": K, "calls": calls, "pool": P, "roof_us": roof,
                         "rows": rows}
        del pool_w, pool_s, wq0, ws0
        torch.cuda.empty_cache()

    print("\n=== per-step projection (per rank, target forward) ===")
    for M in NS:
        tb = sum(results[n]["rows"][M]["fp8hip_us"] * results[n]["calls"] for n in results)
        th = sum(results[n]["rows"][M]["hcq_us"] * results[n]["calls"] for n in results)
        print(f"  M={M:3d}: fp8hip {tb/1e3:6.3f} ms/step -> hcq {th/1e3:6.3f} ms/step "
              f"({(th-tb)/1e3:+.3f} ms, {tb/th:.2f}x)")
    json.dump(results, open("/w/artifacts/bench_fp8sk.json", "w"), indent=1)
    print("\nDONE", flush=True)


main()
