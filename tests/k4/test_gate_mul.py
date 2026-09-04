"""#7 fused shared-expert gate: bit-identity vs F.sigmoid(g) * out, and launches."""
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from vllm.model_executor.layers import fused_gate_mul as fgm

H = 2560          # hidden_size
dev, dt = "cuda", torch.bfloat16
torch.manual_seed(0)


def launches(fn, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in p.events()
               if e.device_type == torch.autograd.DeviceType.CUDA and e.device_index >= 0)


bad = 0
for T in (1, 4, 20, 64, 2048):
    for scale in (0.02, 1.0, 30.0):     # tiny, normal, saturating-sigmoid gates
        out = (torch.randn(T, H, dtype=dt, device=dev) * 2.0)
        g = (torch.randn(T, 1, dtype=dt, device=dev) * scale)
        assert fgm.supported(g, out), (T, scale)
        ref = F.sigmoid(g) * out
        got = fgm.sigmoid_gate_mul(g, out)
        if not torch.equal(ref, got):
            bad += 1
            d = (ref.float() - got.float()).abs()
            print(f"  MISMATCH T={T} scale={scale}: {int((ref != got).sum())} elems, max abs {d.max().item():.3e}")
print("bit-identical to F.sigmoid(g) * out on every shape/scale:", bad == 0)
assert bad == 0

out = torch.randn(20, H, dtype=dt, device=dev)
g = torch.randn(20, 1, dtype=dt, device=dev)
n_ref = launches(lambda: F.sigmoid(g) * out)
n_new = launches(lambda: fgm.sigmoid_gate_mul(g, out))
print(f"GPU launches: stock {n_ref}, fused {n_new}  (delta {n_new - n_ref})")
print(f"48 MoE layers/step -> {(n_new - n_ref) * 48:+d} kernels/step")

# refuses shapes it is not written for, rather than producing something wrong
assert not fgm.supported(torch.randn(4, 2, dtype=dt, device=dev), out)
assert not fgm.supported(g.float(), out)
print("supported() refuses a non-[T,1] gate and a dtype mismatch")
print("\nGATE MUL TEST OK")
