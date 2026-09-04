"""Does GemmaRMSNorm reach a fused kernel the way the QSA indexer actually calls it?

The indexer does `norm(tensor)` (apply_qsa_rmsnorm), i.e. nn.Module.__call__ -> the
CustomOp-dispatched method -- NOT forward_cuda. Counts GPU kernels per call for the
dispatched path, both eager and inside a HIP graph capture/replay.
"""
import os
import torch
from torch.profiler import ProfilerActivity, profile
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers import layernorm as L

EPS = 1e-6
H, M = 128, 32          # indexer_head_dim=128; 8 tokens x 4 index heads

_ctx = set_current_vllm_config(VllmConfig())
_ctx.__enter__()   # keep the config live: default_on()/enabled() read it lazily
m = L.GemmaRMSNorm(H, eps=EPS).to("cuda").to(torch.bfloat16)
with torch.no_grad():
    m.weight.copy_((torch.randn(H, device="cuda") * 0.05).to(torch.bfloat16))

print("VLLM_GEMMA_NORM_FUSED  :", os.environ.get("VLLM_GEMMA_NORM_FUSED", "(unset)"),
      "-> GEMMA_NORM_FUSED =", L.GEMMA_NORM_FUSED)
print("CustomOp.default_on()  :", CustomOp.default_on())
print("GemmaRMSNorm.enabled() :", L.GemmaRMSNorm.enabled())
disp = getattr(m, "_forward_method", None)
print("dispatched method      :", getattr(disp, "__name__", disp))


def kernels(fn, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in p.events()
               if e.device_type == torch.autograd.DeviceType.CUDA and e.device_index >= 0)


with torch.inference_mode():
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")

    n_call = kernels(lambda: m(x))                       # what the indexer really does
    n_native = kernels(lambda: m.forward_native(x))
    print("\nGPU kernels / call  m(x) [dispatched] : %d" % n_call)
    print("GPU kernels / call  forward_native     : %d" % n_native)

    # --- inside a HIP graph capture + replay, like the real decode step ---
    static = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            m(static)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = m(static)
    n_graph = kernels(lambda: g.replay())
    print("GPU kernels / replay of a captured m(x): %d" % n_graph)

    # forward_native is itself patched now, so it is NOT a control. Compute the stock
    # fp32 formula explicitly instead.
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
           * (m.weight.float() + 1.0)).to(x.dtype)
    got = m(x)
    d = (ref.float() - got.float()).abs()
    rel = d / ref.float().abs().clamp(min=1e-6)
    print("\nvs stock fp32 formula: max abs %.3e  max rel %.3e  mean rel %.3e  differ %.1f%%"
          % (d.max().item(), rel.max().item(), rel.mean().item(),
             (ref != got).float().mean().item() * 100))
