#!/usr/bin/env python3
"""r4d.so skinny GEMM entry points on the q38fn hyper-connection shapes.

r4d_gemm_bf16_nt_m64 takes a PLAIN [N,K] bf16 weight, so it is checked for
correctness. r4d_gemm_w4a16_nt_m64 wants a weight pre-permuted into WMMA
fragment order by a packer (radiance_w4.py) that is not in this tree, so it is
run on random bytes: the timing is real (byte counts and access pattern are
exact), the VALUES are not checked. Cold pool as in bench_int4.py.
"""
import ctypes, json, os, statistics, sys
import torch
import torch.nn.functional as F

DEV, DT = 'cuda:0', torch.bfloat16
POOL_BYTES = 192 << 20
ROUNDS = 10
lib = ctypes.CDLL('/app/r4dhip/r4d.so')
L = ctypes.c_long
I = ctypes.c_int
lib.r4d_gemm_bf16_nt_m64.argtypes = (L, L, L, I, I, I, I, I, I, L)
lib.r4d_gemm_w4a16_nt_m64.argtypes = (L, L, L, L, I, I, I, I, I, I, I, I, L)
lib.r4d_gemm_w4a16_nt_m64_group.restype = I
GROUP = lib.r4d_gemm_w4a16_nt_m64_group()

SHAPES = [(336, 10240, 'hc_down_merged'), (320, 10240, 'hc_down_plain'),
          (10240, 320, 'hc_up'), (512, 2560, 'moe_router')]
res = []


def stream():
    return torch.cuda.current_stream().cuda_stream


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
    vals = []
    for _ in range(ROUNDS):
        t0.record(); g.replay(); t1.record(); torch.cuda.synchronize()
        vals.append(t0.elapsed_time(t1) * 1e3 / ncopies)
    del g
    return statistics.median(vals)


def cfgs_bf16(K):
    out = []
    for WV in (1, 2, 4, 8, 16):
        for SK in (1, 2, 4, 8, 16):
            if WV * SK * 32 > 1024 or WV * SK * 256 * 4 > 65536 or K % (SK * 16):
                continue
            out.append((WV, SK, 1))
    return out


def cfgs_w4(K):
    out = []
    for WV in (1, 2, 4, 8, 16):
        for SK in (1, 2, 4, 8, 16):
            for NPW in (1, 4):
                for NT in (0, 1):
                    ncols = WV * NPW
                    if WV * SK * 32 > 1024 or ncols * SK * 256 * 4 > 65536 or K % (SK * GROUP):
                        continue
                    out.append((WV, SK, 1, NPW, NT))
    return out


def run(N, K, tag):
    print(f'\n=== {tag}  W[{N},{K}]  (r4d naming: N={N} out, K={K} in) ===', flush=True)
    W = (torch.randn(N, K, device=DEV) * 0.02).to(DT)
    bfb = N * K * 2
    nbf = max(2, POOL_BYTES // bfb)
    Wp = [W.clone() for _ in range(nbf)]
    A = {m: torch.randn(m, K, device=DEV, dtype=DT) for m in range(1, 6)}
    C = {m: torch.empty(m, N, device=DEV, dtype=DT) for m in range(1, 6)}
    ref = {m: A[m].float() @ W.float().t() for m in range(1, 6)}

    # ---- bf16, correctness-checked, config sweep at M=5 then all M ----
    best = None
    for (WV, SK, MB) in cfgs_bf16(K):
        try:
            lib.r4d_gemm_bf16_nt_m64(A[5].data_ptr(), W.data_ptr(), C[5].data_ptr(),
                                     5, K, N, WV, SK, MB, stream())
            torch.cuda.synchronize()
            e = ((C[5].float() - ref[5]).norm() / ref[5].norm()).item()
            us = time_pool(lambda i: lib.r4d_gemm_bf16_nt_m64(
                A[5].data_ptr(), Wp[i].data_ptr(), C[5].data_ptr(), 5, K, N, WV, SK, MB, stream()), nbf)
            if e < 1e-2 and (best is None or us < best[0]):
                best = (us, WV, SK, MB, e)
        except Exception as ex:
            print(f'  bf16 WV{WV} SK{SK} MB{MB}: {str(ex)[:70]}', flush=True)
    if best:
        _, WV, SK, MB, _ = best
        print(f'  bf16 best config WV={WV} SK={SK} MB={MB}', flush=True)
        for m in range(1, 6):
            lib.r4d_gemm_bf16_nt_m64(A[m].data_ptr(), W.data_ptr(), C[m].data_ptr(),
                                     m, K, N, WV, SK, MB, stream())
            torch.cuda.synchronize()
            e = ((C[m].float() - ref[m]).norm() / ref[m].norm()).item()
            us = time_pool(lambda i: lib.r4d_gemm_bf16_nt_m64(
                A[m].data_ptr(), Wp[i].data_ptr(), C[m].data_ptr(), m, K, N, WV, SK, MB, stream()), nbf)
            res.append(dict(tag=tag, N=N, K=K, m=m, method=f'r4d_bf16_WV{WV}SK{SK}MB{MB}',
                            us=round(us, 3), GBs=round(bfb / us / 1e3, 1), err=e))
            print(f'  r4d_bf16   m={m}  {us:8.2f} us  {bfb/us/1e3:7.1f} GB/s  rel={e:.2e}', flush=True)
    del Wp; torch.cuda.empty_cache()

    # ---- w4a16, timing only ----
    if K % GROUP:
        print(f'  r4d w4a16: ILLEGAL, K={K} not divisible by group={GROUP}', flush=True)
    elif N % 16:
        print(f'  r4d w4a16: ILLEGAL, N={N} not a multiple of 16', flush=True)
    else:
        wq = torch.randint(0, 255, (N * K // 8,), dtype=torch.int32, device=DEV)
        sz = torch.randint(0, 1 << 30, (N * (K // GROUP),), dtype=torch.int32, device=DEV)
        wb = wq.numel() * 4 + sz.numel() * 4
        nw = max(2, POOL_BYTES // wb)
        qp = [wq.clone() for _ in range(nw)]
        sp = [sz.clone() for _ in range(nw)]
        bestw = None
        for (WV, SK, MB, NPW, NT) in cfgs_w4(K):
            try:
                lib.r4d_gemm_w4a16_nt_m64(A[5].data_ptr(), wq.data_ptr(), sz.data_ptr(),
                                          C[5].data_ptr(), 5, K, N, WV, SK, MB, NPW, NT, stream())
                torch.cuda.synchronize()
                us = time_pool(lambda i: lib.r4d_gemm_w4a16_nt_m64(
                    A[5].data_ptr(), qp[i].data_ptr(), sp[i].data_ptr(), C[5].data_ptr(),
                    5, K, N, WV, SK, MB, NPW, NT, stream()), nw)
                if bestw is None or us < bestw[0]:
                    bestw = (us, WV, SK, MB, NPW, NT)
            except Exception as ex:
                pass
        if bestw is None:
            print('  r4d w4a16: no legal config', flush=True)
        else:
            _, WV, SK, MB, NPW, NT = bestw
            print(f'  w4a16 best config WV={WV} SK={SK} MB={MB} NPW={NPW} NT={NT}  '
                  f'({len(cfgs_w4(K))} configs tried)', flush=True)
            for m in range(1, 6):
                us = time_pool(lambda i: lib.r4d_gemm_w4a16_nt_m64(
                    A[m].data_ptr(), qp[i].data_ptr(), sp[i].data_ptr(), C[m].data_ptr(),
                    m, K, N, WV, SK, MB, NPW, NT, stream()), nw)
                res.append(dict(tag=tag, N=N, K=K, m=m,
                                method=f'r4d_w4a16_WV{WV}SK{SK}MB{MB}NPW{NPW}NT{NT}',
                                us=round(us, 3), GBs=round(wb / us / 1e3, 1), err=None))
                print(f'  r4d_w4a16  m={m}  {us:8.2f} us  {wb/us/1e3:7.1f} GB/s  (values unchecked)', flush=True)
        del qp, sp; torch.cuda.empty_cache()
    del W; torch.cuda.empty_cache()


print(torch.cuda.get_device_name(0), 'r4d w4a16 group =', GROUP, flush=True)
only = os.environ.get('ONLY')
for N, K, tag in SHAPES:
    if only and only not in tag:
        continue
    run(N, K, tag)
json.dump(res, open(os.environ.get('OUT', '/w/tests/k3/r4d_results.json'), 'w'), indent=1)
print('\nDONE', len(res), 'rows', flush=True)
