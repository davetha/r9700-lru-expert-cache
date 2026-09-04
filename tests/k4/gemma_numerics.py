"""GemmaRMSNorm: fp32-weight native decomposition (old) vs bf16-weight fused kernel (new).

Runs both real code paths through vllm.ir so the comparison includes whatever
provider actually dispatches on this box, not a hand-written reference.
"""
import torch
from vllm import ir
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

torch.manual_seed(0)
dev = "cuda"
EPS = 1e-6


def old_path(x, w):
    return ir.ops.rms_norm(x, w.float() + 1.0, EPS)


def new_path(x, w):
    return ir.ops.rms_norm(x, (w.float() + 1.0).to(x.dtype), EPS)


def fp64_ref(x, w):
    xd = x.double()
    v = xd.pow(2).mean(-1, keepdim=True)
    return xd * torch.rsqrt(v + EPS) * (w.double() + 1.0)


hdr = ("shape", "max|old-new|", "max rel", "mean rel", "rel(old,f64)",
       "rel(new,f64)", "bf16 differ%")
print("%18s %13s %10s %10s %13s %13s %12s" % hdr)
for M, H in ((1, 128), (4, 128), (16, 2560), (64, 2560), (512, 2560), (4096, 2560)):
    x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
    # real Gemma norm weights are small deltas around 0
    w = (torch.randn(H, device=dev) * 0.05).to(torch.bfloat16)
    o = old_path(x, w)
    n = new_path(x, w)
    differ = (o != n).float().mean().item() * 100
    o, n = o.float(), n.float()
    r = fp64_ref(x, w).float()
    d = (o - n).abs()
    rel = d / o.abs().clamp(min=1e-6)

    def relerr(a):
        return ((a - r).abs() / r.abs().clamp(min=1e-6)).mean().item()

    print("%18s %13.3e %10.3e %10.3e %13.3e %13.3e %12.2f"
          % (str((M, H)), d.max().item(), rel.max().item(), rel.mean().item(),
             relerr(o), relerr(n), differ))

# End-to-end through the module itself (exercises the memo + residual variant).
with set_current_vllm_config(VllmConfig()):
    m = GemmaRMSNorm(2560, eps=EPS).to(dev).to(torch.bfloat16)
with torch.no_grad():
    m.weight.copy_((torch.randn(2560, device=dev) * 0.05).to(torch.bfloat16))
x = torch.randn(512, 2560, dtype=torch.bfloat16, device=dev)
res = torch.randn(512, 2560, dtype=torch.bfloat16, device=dev)

a = m(x.clone()).float()
b = old_path(x, m.weight).float()
print("\nmodule vs fp32-weight path (no residual): max abs %.3e  max rel %.3e"
      % ((a - b).abs().max().item(),
         ((a - b).abs() / b.abs().clamp(min=1e-6)).max().item()))

x2, r2 = x.clone(), res.clone()
y2, ro2 = m(x2, r2)
x3, r3 = x.clone(), res.clone()
y3, ro3 = ir.ops.fused_add_rms_norm(x3, r3, m.weight.float() + 1.0, EPS)
print("module vs fp32-weight path (residual):    max abs %.3e  max rel %.3e  "
      "residual max abs %.3e"
      % ((y2.float() - y3.float()).abs().max().item(),
         ((y2.float() - y3.float()).abs()
          / y3.float().abs().clamp(min=1e-6)).max().item(),
         (ro2.float() - ro3.float()).abs().max().item()))
print("memo returns the same tensor twice:", m._weight_for(x) is m._weight_for(x))
