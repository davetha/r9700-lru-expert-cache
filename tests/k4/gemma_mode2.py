"""Mode-2 (fp32-exact) GemmaRMSNorm: kernel count through the real dispatch, and
numerics against the stock fp32 decomposition on the checkpoint's own indexer weights.
"""
import glob, json, os
import torch
from safetensors import safe_open
from torch.profiler import ProfilerActivity, profile
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers import layernorm as L

CKPT = "/models"
EPS = 1e-6
H = 128                       # indexer_head_dim

_ctx = set_current_vllm_config(VllmConfig())
_ctx.__enter__()
print("default VLLM_GEMMA_NORM_FUSED ->", L.GEMMA_NORM_FUSED,
      "(env:", os.environ.get("VLLM_GEMMA_NORM_FUSED", "unset") + ")")
print("CustomOp.default_on():", CustomOp.default_on(),
      " GemmaRMSNorm.enabled():", L.GemmaRMSNorm.enabled())

# ---- real indexer norm weights ----
idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
names = sorted(k for k in idx
               if k.endswith("layernorm.weight") and ".indexer." in k)
byfile = {}
for n in names:
    byfile.setdefault(idx[n], []).append(n)
W = {}
for f, ns in byfile.items():
    with safe_open(f"{CKPT}/{f}", framework="pt") as fh:
        for n in ns:
            W[n] = fh.get_tensor(n)
print(f"loaded {len(W)} indexer layernorm weights, dtype {next(iter(W.values())).dtype}")


def kernels(fn, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in p.events()
               if e.device_type == torch.autograd.DeviceType.CUDA and e.device_index >= 0)


def stock(x, w):
    """The unpatched GemmaRMSNorm arithmetic, written out."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
            * (w.float() + 1.0)).to(x.dtype)


m = L.GemmaRMSNorm(H, eps=EPS).to("cuda").to(torch.bfloat16)
print("dispatched method:", getattr(m, "_forward_method", None).__name__)

with torch.inference_mode():
    for mode in (0, 1, 2):
        L.GEMMA_NORM_FUSED = mode
        m._fused_weight = None
        # counts on the decode-shaped call: 8 tokens x 4 index heads
        x = torch.randn(8 * 4, H, dtype=torch.bfloat16, device="cuda")
        n_eager = kernels(lambda: m(x))
        static = x.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                m(static)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _o = m(static)
        n_graph = kernels(lambda: g.replay())

        # numerics over every real indexer weight, a few activation scales & token counts
        worst_abs = worst_rel = 0.0
        worst_ulp = 0
        nbad = ntot = 0
        for name, w in W.items():
            with torch.no_grad():
                m.weight.copy_(w.to("cuda", torch.bfloat16))
            m._fused_weight = None
            for scale in (0.1, 1.0, 10.0):
                for ntok in (4, 32, 256):
                    xx = (torch.randn(ntok, H, dtype=torch.bfloat16, device="cuda")
                          * scale)
                    ref = stock(xx, m.weight)
                    got = m(xx)
                    d = (ref.float() - got.float()).abs()
                    r = d / ref.float().abs().clamp(min=1e-6)
                    worst_abs = max(worst_abs, d.max().item())
                    worst_rel = max(worst_rel, r.max().item())
                    nbad += (ref != got).sum().item(); ntot += ref.numel()
                    if (ref != got).any():
                        # distance in bf16 ulps, via the raw 16-bit patterns
                        a = ref.view(torch.int16).to(torch.int32)
                        b = got.view(torch.int16).to(torch.int32)
                        worst_ulp = max(worst_ulp, (a - b).abs().max().item())
        print(f"mode {mode}: kernels eager {n_eager:2d}  graph-replay {n_graph:2d}  |"
              f" vs stock: max abs {worst_abs:.3e}  max rel {worst_rel:.3e}"
              f"  differing {nbad}/{ntot} elements, max {worst_ulp} bf16 ulp")
