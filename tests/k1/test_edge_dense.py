"""libr4d's own test_mxfp4_gemm.py shape/config grid, run through ctypes on every variant.

WHY NOT test_mxfp4_gemm.py ITSELF: in this image it reports "FAILURES: 0" while SKIPPING
every case -- `import r4d` (the pybind module) before torch touches the device leaves
torch.cuda.current_stream() raising "invalid argument to getCurrentStream", and the test
catches Exception per case and prints SKIP. Its exit status is 0 either way. This file runs
the same grid over the ctypes surface (no pybind import), which does not trip that.
"""
import os
import sys

import torch

sys.path.insert(0, "/w/tools/routecap")
from cmp_dense import LIBS, load, make_case  # noqa: E402

CASES = [(8, 512, 1024), (5, 512, 1024), (16, 1024, 2048), (64, 512, 1024),
         (1, 256, 512), (32, 5120, 3072), (8, 5120, 8704), (64, 17408, 5120),
         (8, 48, 1024), (5, 48, 5120), (64, 80, 1024)]
CFGS = [(4, 4, 1), (2, 4, 2), (8, 2, 1)]   # (WV, SK, NPW), as in the library's test

torch.zeros(1, device="cuda")
libs = [(n, load(p)) for n, p in LIBS]
st = torch.cuda.current_stream().cuda_stream
bad = 0
print("%-18s %-12s %-11s %s" % ("shape (M,N,K)", "WV/SK/MB/NPW", "relerr(ship)", "differing bf16 words vs shipped"))
for (M, N, K) in CASES:
    packed, e8m0, wref, af8, asc, exp = make_case(M, N, K)
    for (WV, SK, NPW) in CFGS:
        MB = max(1, min(4, (M + 15) // 16))
        outs = {}
        for n, lib in libs:
            c = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
            lib.r4d_gemm_mxfp4a8_nt_m64(
                af8.data_ptr(), asc.data_ptr(), packed.data_ptr(), e8m0.data_ptr(),
                wref.data_ptr(), c.data_ptr(), M, K, N, WV, SK, MB, NPW, st)
            torch.cuda.synchronize()
            outs[n] = c.clone()
        ship = outs["shipped"]
        rel = ((ship.float() - exp).norm() / exp.norm()).item()
        diffs = " ".join(
            "%s:%d" % (n, (outs[n].view(torch.int16) != ship.view(torch.int16)).sum().item())
            for n, _ in libs[1:])
        flag = ""
        if rel >= 2e-2 or any((outs[n].view(torch.int16) != ship.view(torch.int16)).any().item()
                              for n, _ in libs[1:]):
            flag = "   <-- FAIL"
            bad += 1
        print("%-18s %-12s %-11.3e %s%s"
              % (f"({M},{N},{K})", f"{WV}/{SK}/{MB}/{NPW}", rel, diffs, flag))
print("FAILURES:", bad)
sys.exit(1 if bad else 0)
