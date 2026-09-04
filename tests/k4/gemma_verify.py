"""Does the patched GemmaRMSNorm actually dispatch to a fused kernel in eager,
and how many GPU kernels does each path launch?"""
import os
import torch
from torch.profiler import ProfilerActivity, profile
from vllm import ir
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers import layernorm as L

EPS = 1e-6
H, M = 128, 8   # QSA indexer shape: [tokens*heads, index_head_dim]

print("GEMMA_NORM_FUSED       :", L.GEMMA_NORM_FUSED)
print("rms_norm priority      :", ir.ops.rms_norm.priority_providers()
      if hasattr(ir.ops.rms_norm, "priority_providers") else "n/a")
print("rms_norm impls         :", {k: v.supported for k, v in ir.ops.rms_norm.impls.items()})
print("fused impl picked      :", getattr(L._fused_norm_impl(ir.ops.rms_norm), "provider", None))
print("fused add impl picked  :",
      getattr(L._fused_norm_impl(ir.ops.fused_add_rms_norm), "provider", None))

with set_current_vllm_config(VllmConfig()):
    m = L.GemmaRMSNorm(H, eps=EPS).to("cuda").to(torch.bfloat16)
with torch.no_grad():
    m.weight.copy_((torch.randn(H, device="cuda") * 0.05).to(torch.bfloat16))

x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")


def count_kernels(fn, *a):
    fn(*a)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(*a)
        torch.cuda.synchronize()
    return sum(1 for e in p.events()
               if e.device_type == torch.autograd.DeviceType.CUDA and e.device_index >= 0)


old = lambda t: m.forward_native(t)
new = lambda t: m.forward_cuda(t)
n_old = count_kernels(old, x)
n_new = count_kernels(new, x)
print("\nGPU events / call, no residual : native %d -> fused %d" % (n_old, n_new))

o, n = old(x).float(), new(x).float()
rel = (o - n).abs() / o.abs().clamp(min=1e-6)
print("values: max abs %.3e  max rel %.3e  mean rel %.3e  bf16 differ %.1f%%"
      % ((o - n).abs().max().item(), rel.max().item(), rel.mean().item(),
         (old(x) != new(x)).float().mean().item() * 100))

# residual variant (in-place: clone per call)
r = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
def old_r(_):
    return m.forward_native(x.clone(), r.clone())
def new_r(_):
    return m.forward_cuda(x.clone(), r.clone())
print("\nGPU events / call, residual    : native %d -> fused %d"
      % (count_kernels(old_r, x), count_kernels(new_r, x)))
yo, ro = old_r(None)
yn, rn = new_r(None)
print("residual variant: out max rel %.3e   residual max abs %.3e"
      % (((yo.float() - yn.float()).abs()
          / yo.float().abs().clamp(min=1e-6)).max().item(),
         (ro.float() - rn.float()).abs().max().item()))

# env gate: with VLLM_GEMMA_NORM_FUSED=0 forward_cuda must be bit-identical to native
if os.environ.get("VLLM_GEMMA_NORM_FUSED") == "0":
    print("\ngate off -> forward_cuda bit-identical to forward_native:",
          torch.equal(m.forward_cuda(x), m.forward_native(x)))
