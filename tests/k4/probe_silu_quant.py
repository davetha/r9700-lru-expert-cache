#!/usr/bin/env python3
"""Bit-exactness probe for the fused silu_and_mul + per-token fp8 quant kernel.

Compares against the exact pair it replaces in the r4d MoE
(torch.ops._C.silu_and_mul -> ops.scaled_fp8_quant) and, separately, reports
whether inductor's fused shared-expert silu keeps eager's intermediate rounding.
"""
import torch
import torch.nn.functional as F
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.model_executor.layers import fused_silu_mul_quant as K

dev = "cuda"
fp8 = current_platform.fp8_dtype()
print(f"fp8 dtype = {fp8}, max = {torch.finfo(fp8).max}")

torch.manual_seed(0)
bad = 0
# Decode-shaped rows (mtk = M*top_k) plus a couple of prefill-ish ones, over
# magnitude regimes that exercise the scale floor and the fp8 clamp.
for rows in (1, 10, 50, 80, 2048):
    for d in (320, 640):
        for scale_in, tag in ((1.0, "unit"), (1e-3, "tiny"), (60.0, "large")):
            x = (torch.randn(rows, 2 * d, device=dev, dtype=torch.bfloat16) * scale_in)
            assert K.supported(x), f"supported() rejected rows={rows} d={d}"
            ic2 = torch.empty(rows, d, device=dev, dtype=torch.bfloat16)
            torch.ops._C.silu_and_mul(ic2, x)
            q_ref, s_ref = ops.scaled_fp8_quant(ic2, use_per_token_if_dynamic=True)
            act, q, s = K.silu_mul_quant(x, fp8, write_act=True)
            ea = torch.equal(act, ic2)
            es = torch.equal(s, s_ref)
            eq = torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
            if not (ea and es and eq):
                bad += 1
                na = (act != ic2).sum().item()
                nq = (q.view(torch.uint8) != q_ref.view(torch.uint8)).sum().item()
                ns = (s != s_ref).sum().item()
                print(f"  MISMATCH rows={rows} d={d} {tag}: act {na}/{act.numel()} "
                      f"scale {ns}/{s.numel()} q {nq}/{q.numel()}")
            else:
                print(f"  exact  rows={rows:5d} d={d} {tag}")

# The zero row is a real case (a token routed to an expert with no mass):
# amax 0 must land on the scale floor, not a divide-by-zero.
z = torch.zeros(4, 640, device=dev, dtype=torch.bfloat16)
ic2 = torch.empty(4, 320, device=dev, dtype=torch.bfloat16)
torch.ops._C.silu_and_mul(ic2, z)
q_ref, s_ref = ops.scaled_fp8_quant(ic2, use_per_token_if_dynamic=True)
_, q, s = K.silu_mul_quant(z, fp8)
ok = torch.equal(s, s_ref) and torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
print(f"  zero row: {'exact' if ok else 'MISMATCH'}  scale={s.flatten()[0].item():.6g}")
bad += 0 if ok else 1

print(f"\nROUTED SITE: {'BIT-IDENTICAL' if bad == 0 else f'{bad} MISMATCHING CASES'}")

# --- shared-expert site: does inductor keep eager's intermediate rounding? ---
d = 320
x = torch.randn(64, 2 * d, device=dev, dtype=torch.bfloat16)
eager = F.silu(x[..., :d]) * x[..., d:]

def f(t):
    return F.silu(t[..., :d]) * t[..., d:]

comp = torch.compile(f, dynamic=False)(x)
_, _, _ = K.silu_mul_quant(x, fp8)          # warm the kernel
mine, _, _ = K.silu_mul_quant(x, fp8, write_act=True)
print(f"shared-expert silu: inductor==eager {torch.equal(comp, eager)}, "
      f"mine==eager {torch.equal(mine, eager)}, "
      f"mine==inductor {torch.equal(mine, comp)}")
