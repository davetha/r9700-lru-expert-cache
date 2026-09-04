"""Microbench: draft w4a16 lm_head vs the current bf16 path, N=124160 K=2560, cold cache.

The weight is 636 MB bf16 / 169 MB w4 -- both far past the 64 MiB last level -- so one copy
is already a cold read; no pool is needed at this size.
"""
import json, os, time
import torch
import torch.nn.functional as F
import vllm._custom_ops as ops
import vllm.model_executor.layers.utils as U
import vllm.model_executor.kernels.draft_w4_lmhead as W

dev = "cuda:0"
N = int(os.environ.get("N", "124160"))
K = int(os.environ.get("K", "2560"))
NS = [int(v) for v in os.environ.get("NS", "1,2,4,5,20").split(",")]
torch.manual_seed(0)
res = {"N": N, "K": K, "ns": NS}
props = torch.cuda.get_device_properties(0)
print(f"{props.name} CUs={props.multi_processor_count} "
      f"free/total={[v/2**30 for v in torch.cuda.mem_get_info()]}", flush=True)


def timeit(fn, calls=5, reps=7, warm=3):
    """median us per call, measured inside a HIP graph."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / calls)
    ts.sort()
    del g
    return ts[len(ts) // 2]


# ---- weights -------------------------------------------------------------------------
w = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
torch.cuda.synchronize(); t0 = time.perf_counter()
packed = W.pack_w4a16(w)
torch.cuda.synchronize()
res["pack_seconds"] = time.perf_counter() - t0
res["packed_MB"] = packed.nbytes / 2**20
res["bf16_MB"] = w.numel() * 2 / 2**20
print(f"pack {res['pack_seconds']:.2f}s  w4 {res['packed_MB']:.1f} MB vs bf16 "
      f"{res['bf16_MB']:.1f} MB", flush=True)

# same quantizer, ExLlama shuffle layout, for wvSplitK_int4_g
q, scale, zero = W.quantize_w4(w)
gq = q.view(N, K // 8, 8).to(torch.int32)
exl = (gq[:, :, 0] | (gq[:, :, 2] << 4) | (gq[:, :, 4] << 8) | (gq[:, :, 6] << 12)
       | (gq[:, :, 1] << 16) | (gq[:, :, 3] << 20) | (gq[:, :, 5] << 24)
       | (gq[:, :, 7] << 28)).contiguous()
exl_s = scale.bfloat16().contiguous()
exl_z = zero.bfloat16().contiguous()
del q, scale, zero, gq
cu = torch.cuda.get_device_properties(0).multi_processor_count
LDS_OK = lambda m: K * m <= int(64 * 1024 / 2 * 1.2)

# numeric agreement of the two int4 paths against the same dequantized reference
ref_rows = 4096
wref = W.unpack_w4(packed, torch.float32)[:ref_rows]
x = torch.randn(4, K, device=dev).bfloat16()
y_r4d = W.gemm_w4a16(x, packed)[:, :ref_rows].float()
r = F.linear(x.float().half().float(), wref)
res["r4d_rel_err_vs_dequant"] = ((y_r4d - r).norm() / r.norm()).item()
if LDS_OK(4):
    y_wv = ops.wvSplitK_int4_g(exl[:ref_rows], x, exl_s[:ref_rows], cu, W.GROUP,
                               exl_z[:ref_rows], None).float()
    res["wvint4_rel_err_vs_dequant"] = ((y_wv - r).norm() / r.norm()).item()
del wref, y_r4d, r
torch.cuda.empty_cache()
print("rel err vs dequant ref:", {k: v for k, v in res.items() if "rel_err" in k}, flush=True)

# ---- sweep + bench -------------------------------------------------------------------
cfgs = [(wv, sk, npw, nt)
        for wv in (1, 2, 4, 8) for sk in (1, 2, 4, 5, 10, 20)
        for npw in (1, 4) for nt in (0, 1)
        if W._legal(K, packed.n_pad // 16, wv, sk, npw)]
print(f"{len(cfgs)} legal cfgs", flush=True)

rows = {}
for n in NS:
    x = torch.randn(n, K, device=dev).bfloat16()
    r = {}
    r["bf16_prod"] = timeit(lambda: U.rocm_unquantized_gemm_impl(x, w, None))
    r["bf16_F.linear"] = timeit(lambda: F.linear(x, w))
    if LDS_OK(n):
        r["wvSplitK_int4_g"] = timeit(
            lambda: ops.wvSplitK_int4_g(exl, x, exl_s, cu, W.GROUP, exl_z, None))
    sweep = {}
    for (wv, sk, npw, nt) in cfgs:
        try:
            sweep[f"{wv},{sk},{npw},{nt}"] = timeit(
                lambda: torch.ops.vllm.draft_w4_lmhead_gemm(
                    x, packed.wq, packed.wsz, packed.n, packed.k, wv, sk, npw, nt),
                calls=3, reps=5, warm=2)
        except RuntimeError as e:
            sweep[f"{wv},{sk},{npw},{nt}"] = f"ERR {e}"
    ok = {k: v for k, v in sweep.items() if isinstance(v, float)}
    best = min(ok, key=ok.get)
    r["r4d_w4a16_best"] = ok[best]
    r["r4d_w4a16_best_cfg"] = best
    r["r4d_w4a16_sweep"] = dict(sorted(ok.items(), key=lambda kv: kv[1])[:8])
    r["speedup_vs_bf16_prod"] = r["bf16_prod"] / ok[best]
    rows[n] = r
    print(f"n={n:3d} bf16_prod {r['bf16_prod']:8.1f}us  F.linear {r['bf16_F.linear']:8.1f}us"
          f"  wvint4 {r.get('wvSplitK_int4_g', float('nan')):8.1f}us"
          f"  r4d_w4 {ok[best]:8.1f}us ({best})  ->{r['speedup_vs_bf16_prod']:.2f}x", flush=True)

res["rows"] = rows
json.dump(res, open("/w/tests/k3/w4_bench.json", "w"), indent=1)
print("DONE", flush=True)
