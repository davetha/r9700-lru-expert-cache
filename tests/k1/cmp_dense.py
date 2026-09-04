"""Shipped vs from-source r4d_gemm_mxfp4a8_nt_m64: bit-exactness and speed.

All three libraries are loaded into ONE process with ctypes (extern-C surface only, as
vllm's r4d_lib does), so the same device buffers feed every variant and the outputs can be
compared bit-for-bit instead of through two separate runs.

  R4D_SHIPPED  default /app/r4dhip/r4d.so
  R4D_OPEN     default /w/build/kernels/r4d_open.so
  R4D_OPEN_FAST default /w/build/kernels/r4d_open_fast.so   (-ffp-contract=fast, no -mcumode)
  R4D_OPEN_FMA default /w/build/kernels/r4d_open_fma.so      (-ffp-contract=fast AND -mcumode)
"""
import ctypes
import os
import sys
import time

import torch

sys.path.insert(0, "/w/build/libr4d")
from mxfp4_layout import permute_w  # noqa: E402

GROUP = 32
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

LIBS = [
    ("shipped", os.environ.get("R4D_SHIPPED", "/app/r4dhip/r4d.so")),
    ("open", os.environ.get("R4D_OPEN", "/w/build/kernels/r4d_open.so")),
    ("open_fast", os.environ.get("R4D_OPEN_FAST", "/w/build/kernels/r4d_open_fast.so")),
    ("open_fma", os.environ.get("R4D_OPEN_FMA", "/w/build/kernels/r4d_open_fma.so")),
]


def load(path):
    lib = ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_DEEPBIND)
    lib.r4d_gemm_mxfp4a8_nt_m64.restype = None
    lib.r4d_gemm_mxfp4a8_nt_m64.argtypes = (
        [ctypes.c_long] * 6 + [ctypes.c_int] * 7 + [ctypes.c_long])
    lib.r4d_gemm_mxfp4a8_nt_m64_max_m.restype = ctypes.c_int
    lib.r4d_gemm_mxfp4a8_nt_m64_group.restype = ctypes.c_int
    return lib


def pick_cfg(N, K):
    """Verbatim from vllm .../linear/mxfp4/r4dhip.py:pick_cfg (no env override)."""
    ntiles = N // 16
    sk = next(s for s in (8, 4, 2, 1) if K % (s * GROUP) == 0)
    npw = 2 if ntiles >= 2 else 1
    wv = max(1, min(32 // sk, 64 // (npw * sk), 8, ntiles))
    return wv, sk, npw


def make_case(M, N, K, seed=0, max_d=8):
    """Same construction as libr4d's test_mxfp4_gemm.make_case: every folded magnitude is
    exactly representable in e4m3, so the torch reference is exact for the weight and the
    only inexactness is fp32 accumulate order + bf16 output rounding."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb = K // GROUP
    ref = torch.randint(120, 136, (N,), generator=g, dtype=torch.int32)
    drop = torch.randint(0, max_d + 1, (nb, N), generator=g, dtype=torch.int32)
    e8m0 = (ref.unsqueeze(0) - drop).clamp(0, 254)
    ref = e8m0.max(dim=0).values
    codes = torch.randint(0, 16, (N, K), generator=g, dtype=torch.uint8)
    mag = E2M1[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) > 0, -1.0, 1.0)
    scale = torch.exp2((e8m0.float() - 127.0)).T
    Wf = ((mag * sign).reshape(N, nb, GROUP) * scale.unsqueeze(-1)).reshape(N, K)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    packed = permute_w(packed, N, K)
    a = (torch.randn(M, K, generator=g) * 0.4)
    af8 = a.to(torch.float8_e4m3fn)
    ascale = torch.full((M,), 0.7)
    ref_out = (af8.float() * ascale.unsqueeze(1)) @ Wf.T
    return (packed.cuda(), e8m0.to(torch.uint8).cuda(), ref.to(torch.uint8).cuda(),
            af8.cuda(), ascale.cuda(), ref_out.cuda())


def call_fn(lib, packed, e8m0, wref, af8, asc, c, M, K, N, wv, sk, mb, npw):
    st = torch.cuda.current_stream().cuda_stream
    a_p, s_p = af8.data_ptr(), asc.data_ptr()
    w_p, z_p, r_p, c_p = packed.data_ptr(), e8m0.data_ptr(), wref.data_ptr(), c.data_ptr()
    f = lib.r4d_gemm_mxfp4a8_nt_m64

    def go():
        f(a_p, s_p, w_p, z_p, r_p, c_p, M, K, N, wv, sk, mb, npw, st)
    return go


def bench(go, n=25):
    for _ in range(10):
        go()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(n):
        t = time.perf_counter()
        go()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best * 1e6


SHAPES = [
    # q38fn (hidden 2560, moe_intermediate 640, E=512, topk 10) TP2 per-rank shapes.
    ("moe.gate_up*", 640, 2560),
    ("moe.down*", 2560, 320),
    ("attn.q", 3072, 2560),
    ("attn.o", 2560, 3072),
    # libr4d bench_mxfp4_gemm.py's own shapes (Qwen3.8-27B TP2), for comparability with the
    # numbers published in the kernel's header comment.
    ("b:mlp.gate_up", 17408, 5120),
    ("b:mlp.down", 5120, 8704),
    ("b:gdn.in_qkv", 5120, 5120),
    ("b:attn.q", 6144, 5120),
]
if __name__ == "__main__":
    MS = [int(v) for v in os.environ.get("MS", "1,8,16,64").split(",")]

    libs = [(n, load(p)) for n, p in LIBS]
    print("libs:", [(n, p) for n, p in LIBS])
    for n, lib in libs:
        print(f"  {n}: max_m={lib.r4d_gemm_mxfp4a8_nt_m64_max_m()} "
              f"group={lib.r4d_gemm_mxfp4a8_nt_m64_group()}")
    print()
    print("%-16s %6s %6s %4s %-9s | %-46s | %s"
          % ("shape", "N", "K", "M", "cfg", "vs shipped (differing bf16 words / maxabs)",
             "us: " + " ".join(n for n, _ in libs)))

    for name, N, K in SHAPES:
        wv, sk, npw = pick_cfg(N, K)
        packed, e8m0, wref, af8, asc, exp = make_case(max(MS), N, K)
        for M in MS:
            a = af8[:M].contiguous()
            s = asc[:M].contiguous()
            mb = max(1, min(4, (M + 15) // 16))
            outs, times = {}, {}
            for n, lib in libs:
                c = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
                go = call_fn(lib, packed, e8m0, wref, a, s, c, M, K, N, wv, sk, mb, npw)
                go()
                torch.cuda.synchronize()
                outs[n] = c.clone()
                times[n] = bench(go)
            ship = outs["shipped"]
            cmp_txt = []
            for n, _ in libs[1:]:
                bits = (outs[n].view(torch.int16) != ship.view(torch.int16)).sum().item()
                md = (outs[n].float() - ship.float()).abs().max().item()
                cmp_txt.append(f"{n}:{bits}b/{md:.2e}")
            rel = ((ship.float() - exp[:M]).norm() / exp[:M].norm()).item()
            print("%-16s %6d %6d %4d %-9s | %-46s | %s  relerr(ship vs torch)=%.2e"
                  % (name, N, K, M, f"{wv}/{sk}/{mb}/{npw}", " ".join(cmp_txt),
                     " ".join("%7.1f" % times[n] for n, _ in libs), rel))
