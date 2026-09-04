#!/usr/bin/env python3
"""Correctness for hcq_gemm_fp8blk_nt_m16: every legal (WV,SK,NPW) at every production
shape and M, against an fp32 oracle and against the closed fp8hip kernel. Small buffers --
this needs ~200 MB, so it can run in a narrow window. Also checks the precondition codes
and HIP-graph replay."""
import ctypes, os, sys, torch

sys.path.insert(0, "/app/vllm")
from vllm.model_executor.kernels.linear.scaled_mm.fp8hip import shuffle_weight_gfx1201
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

DEV, FP8 = "cuda:0", torch.float8_e4m3fn
_hcq = ctypes.CDLL(os.environ.get("FP8SK_LIB", "/w/build/kernels/libhcqfp8sk.so"))
_hcq.hcq_gemm_fp8blk_nt_m16.restype = ctypes.c_int
_hcq.hcq_gemm_fp8blk_nt_m16.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
_blt = ctypes.CDLL(os.environ.get("FP8HIP_LIB", "/app/fp8hip/libfp8hip_gemm.so"))
_blt.fp8hip_gemm_w8a8_launch.restype = ctypes.c_int
_blt.fp8hip_gemm_w8a8_launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]


def hcq(qx, w, xs, ws, y, M, K, N, wv, sk, npw):
    return _hcq.hcq_gemm_fp8blk_nt_m16(
        ctypes.c_void_p(qx.data_ptr()), ctypes.c_void_p(xs.data_ptr()),
        ctypes.c_void_p(w.data_ptr()), ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_void_p(y.data_ptr()), M, K, N, wv, sk, npw,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))


def blt(qx, w, xs, ws, y, M, K, N):
    return _blt.fp8hip_gemm_w8a8_launch(
        ctypes.c_void_p(qx.data_ptr()), ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(xs.data_ptr()), ctypes.c_void_p(ws.data_ptr()),
        ctypes.c_void_p(y.data_ptr()), M, N, K,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))


def make_weight(N, K, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    wref = torch.randn(N, K, device=DEV, dtype=torch.float32, generator=g) * 0.02
    blk = wref.view(N // 128, 128, K // 128, 128)
    ws = (blk.abs().amax(dim=(1, 3)).clamp(min=1e-12) / 448.0).float().contiguous()
    wq = (blk / ws[:, None, :, None]).clamp(-448, 448).to(FP8).view(N, K).contiguous()
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


SHAPES = [(8192, 2560), (6656, 2560), (2560, 3072), (512, 2560), (256, 128)]
MS = [1, 2, 3, 4, 5, 8, 12, 16]
fails = []
torch.cuda.set_device(0)
print("cus", torch.cuda.get_device_properties(0).multi_processor_count, flush=True)

for N, K in SHAPES:
    wq, ws = make_weight(N, K, N * 7 + K)
    wsh = shuffle_weight_gfx1201(wq).contiguous()
    wdq = (wq.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1))
    worst = {}
    for M in MS:
        g = torch.Generator(device=DEV).manual_seed(M * 13 + N)
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, generator=g)
        qx, xs = per_token_group_quant_fp8(x, group_size=128, dtype=FP8)
        qx, xs = qx.contiguous(), xs.contiguous()
        assert xs.dtype == torch.float32 and xs.shape == (M, K // 128), (xs.dtype, xs.shape)
        ref = (qx.float() * xs.repeat_interleave(128, 1)) @ wdq.T
        rmax = max(ref.abs().max().item(), 1e-9)
        y = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
        y.zero_(); assert blt(qx, wsh, xs, ws, y, M, K, N) == 0
        torch.cuda.synchronize()
        y_blt = y.clone()
        r_blt = (y_blt.float() - ref).abs().max().item() / rmax
        for wv, sk, npw in cfgs(K):
            y.fill_(float("nan"))
            rc = hcq(qx, wsh, xs, ws, y, M, K, N, wv, sk, npw)
            torch.cuda.synchronize()
            if rc != 0:
                fails.append(f"N={N} K={K} M={M} cfg={wv},{sk},{npw} rc={rc}")
                continue
            r = (y.float() - ref).abs().max().item() / rmax
            d = (y.float() - y_blt.float()).abs().max().item() / rmax
            key = (wv, sk, npw)
            worst[key] = max(worst.get(key, 0.0), r)
            if not (r < 3.0 * max(r_blt, 1e-6) + 1e-4):
                fails.append(f"N={N} K={K} M={M} cfg={key} rel={r:.3e} vs fp8hip {r_blt:.3e}")
        print(f"  N={N:5d} K={K:5d} M={M:3d}  fp8hip rel {r_blt:.3e}   "
              f"hcq rel worst {max(worst.values()):.3e} over {len(worst)} cfgs   "
              f"|hcq-fp8hip|/max {d:.3e}", flush=True)
        del x, qx, xs, ref, y, y_blt
    del wq, ws, wsh, wdq
    torch.cuda.empty_cache()

# precondition codes: the guard in fp8hip_k3.py must reject exactly what the kernel rejects
N, K = 2560, 3072
wq, ws = make_weight(N, K, 1)
wsh = shuffle_weight_gfx1201(wq).contiguous()
qx = torch.zeros(16, K, device=DEV, dtype=FP8)
xs = torch.ones(16, K // 128, device=DEV, dtype=torch.float32)
y = torch.empty(16, N, device=DEV, dtype=torch.bfloat16)
for args, want in [((17, K, N, 2, 4, 1), -1), ((0, K, N, 2, 4, 1), -1),
                   ((16, K, N + 16, 2, 4, 1), -2), ((16, K, N, 2, 5, 1), -4),
                   ((16, K, N, 64, 4, 1), -5), ((16, K, N, 2, 4, 3), -6)]:
    rc = hcq(qx, wsh, xs, ws, y, *args)
    if rc != want:
        fails.append(f"precondition {args}: rc={rc} want {want}")
print("precondition codes checked", flush=True)

# graph replay must reproduce the eager result exactly
qx16 = torch.randn(5, K, device=DEV, dtype=torch.bfloat16)
qx16, xs16 = per_token_group_quant_fp8(qx16, group_size=128, dtype=FP8)
qx16, xs16 = qx16.contiguous(), xs16.contiguous()
y = torch.empty(5, N, device=DEV, dtype=torch.bfloat16)
assert hcq(qx16, wsh, xs16, ws, y, 5, K, N, 2, 4, 1) == 0
torch.cuda.synchronize()
y_eager = y.clone()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    hcq(qx16, wsh, xs16, ws, y, 5, K, N, 2, 4, 1)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
gph = torch.cuda.CUDAGraph()
y.zero_()
with torch.cuda.graph(gph):
    hcq(qx16, wsh, xs16, ws, y, 5, K, N, 2, 4, 1)
for _ in range(3):
    y.zero_(); gph.replay(); torch.cuda.synchronize()
    if not torch.equal(y, y_eager):
        fails.append("graph replay != eager")
print("graph replay checked", flush=True)

print("\n=== FAILURES ===" if fails else "\nALL OK")
for f in fails:
    print(" ", f)
sys.exit(1 if fails else 0)
