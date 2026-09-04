import os, sys, torch
sys.path.insert(0, "/app/vllm")
import vllm.model_executor.kernels.linear.scaled_mm.fp8hip as F
print("IMPORT-OK  ON=%s MAX_M=%d LIB=%s exists=%s" %
      (F._HCQ_ON, F._HCQ_MAX_M, F._HCQ_LIB_PATH, os.path.exists(F._HCQ_LIB_PATH)))
print("table:", F._HCQ_CFG_TABLE, "|", F._HCQ_TABLE_NOTE)
print("lib loads:", F._hcq_lib() is not None)


def mk(M, N, K, **bad):
    qx = torch.zeros(M, K, dtype=bad.get("qdt", torch.float8_e4m3fn))
    xs = torch.zeros(bad.get("xsshape", (M, K // 128)), dtype=bad.get("xsdt", torch.float32))
    ws = torch.zeros(N // 128, K // 128, dtype=bad.get("wsdt", torch.float32))
    return F._hcq_cfg(M, N, K, qx, xs, ws)


cases = [
    ("prod out_proj M=5", (5, 2560, 3072), {}, (2, 8, 1)),
    ("prod qkvz    M=5", (5, 8192, 2560), {}, (8, 2, 1)),
    ("prod qkv     M=16", (16, 6656, 2560), {}, (2, 4, 1)),
    ("M=17 over max", (17, 2560, 3072), {}, None),
    ("M=0", (0, 2560, 3072), {}, None),
    ("unknown shape", (5, 1024, 2560), {}, None),
    ("N%128", (5, 2560 + 16, 3072), {}, None),
    ("bad xs dtype", (5, 2560, 3072), {"xsdt": torch.bfloat16}, None),
    ("bad ws dtype", (5, 2560, 3072), {"wsdt": torch.bfloat16}, None),
    ("bad xs shape", (5, 2560, 3072), {"xsshape": (5, 24, 1)}, None),
    ("bad qx dtype", (5, 2560, 3072), {"qdt": torch.int8}, None),
]
bad = 0
for name, args, kw, want in cases:
    got = mk(*args, **kw)
    ok = got == want
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {name:22s} -> {got} (want {want})")
print("SMOKE", "OK" if not bad else "FAILED")
sys.exit(1 if bad else 0)
