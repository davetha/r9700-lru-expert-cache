"""Same old-vs-new GemmaRMSNorm comparison, but with the checkpoint's real
indexer q/k_layernorm weights instead of a synthetic prior."""
import glob, json, sys
import torch
from safetensors import safe_open
from vllm import ir

MODEL = "/models"
EPS = 1e-6
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
names = sorted(k for k in idx
               if "indexer" in k and "layernorm.weight" in k)[:6]
names += sorted(k for k in idx if k.endswith("self_attn.q_norm.weight"))[:3]
torch.manual_seed(0)

print("%-62s %8s %9s %10s %10s %11s"
      % ("tensor", "|w|max", "|1+w|min", "max rel", "mean rel", "bf16 differ%"))
for n in names:
    with safe_open(f"{MODEL}/{idx[n]}", "pt") as f:
        w = f.get_tensor(n).cuda()
    H = w.numel()
    x = torch.randn(512, H, dtype=torch.bfloat16, device="cuda")
    o = ir.ops.rms_norm(x, w.float() + 1.0, EPS)
    p = ir.ops.rms_norm(x, (w.float() + 1.0).to(x.dtype), EPS)
    differ = (o != p).float().mean().item() * 100
    o, p = o.float(), p.float()
    rel = (o - p).abs() / o.abs().clamp(min=1e-6)
    print("%-62s %8.4f %9.4f %10.3e %10.3e %11.2f"
          % (n.replace("model.language_model.", ""), w.abs().max().item(),
             (w.float() + 1.0).abs().min().item(),
             rel.max().item(), rel.mean().item(), differ))
