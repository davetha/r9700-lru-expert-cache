"""Gather bandwidth vs grid, at the insert counts that actually occur (1-3 per LAYER per
step in steady state; 52 summed over 48 layers)."""
import ctypes, os
import numpy as np, torch
exec(open("/w/tests/lru/test_graph_lru.py").read().split("def main()")[0])
src_h, src_d = [], []
for b in SIZES:
    h, dp = uva(E * b); src_h.append(h); src_d.append(dp)
dst = [torch.zeros(S * b, dtype=torch.uint8, device=DEV) for b in SIZES]
miss = torch.zeros(MAXI, 2, dtype=torch.int32, device=DEV)
nm = torch.zeros(1, dtype=torch.int32, device=DEV)
def run(n, chunks, lanes, reps=20):
    nm.fill_(n)
    miss[:n, 0] = torch.arange(n, dtype=torch.int32, device=DEV)
    miss[:n, 1] = torch.arange(n, dtype=torch.int32, device=DEV)
    args = []
    for i in range(6):
        args += [ctypes.c_void_p(dst[i].data_ptr()), ctypes.c_void_p(src_d[i]),
                 ctypes.c_long(SIZES[i])]
    tail = [ctypes.c_void_p(miss.data_ptr()), ctypes.c_void_p(nm.data_ptr()),
            chunks, lanes, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)]
    for _ in range(3): lib.r4d_lru_gather(*args, *tail)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    st = torch.cuda.current_stream(); a.record(st)
    for _ in range(reps): lib.r4d_lru_gather(*args, *tail)
    b.record(st); torch.cuda.synchronize()
    us = a.elapsed_time(b) * 1000 / reps
    return us, n * PER / (us * 1e-6) / 1e9
print("per-expert payload %.3f MB (w1+w2+4 scale buffers)" % (PER / 1e6))
print("%-16s" % "grid" + "".join("%18s" % f"n={n}" for n in (1, 2, 4, 16, 52)))
for chunks, lanes in ((8, 16), (16, 16), (16, 32), (16, 64), (32, 32), (64, 8)):
    row = "  chunks=%2d lanes=%2d" % (chunks, lanes)
    for n in (1, 2, 4, 16, 52):
        us, gb = run(n, chunks, lanes)
        row += "  %7.1fus %5.1fGB/s" % (us, gb)
    print(row)
