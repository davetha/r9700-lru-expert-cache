"""#2 GDN strided qkv: bit-identity + launch count vs the stock repack.

Production geometry per rank (TP=2): 8 key heads x 128, 24 value heads x 128,
so mixed_qkv is [T, 1024 + 1024 + 3072].
"""
import torch
from torch.profiler import ProfilerActivity, profile

from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating as fsg
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update as fused,
)

print("SUPPORTS_STRIDED_QKV =", getattr(fsg, "SUPPORTS_STRIDED_QKV", False))
assert getattr(fsg, "SUPPORTS_STRIDED_QKV", False), "patched FLA kernel not mounted"

H, HV, K, V = 8, 24, 128, 128
QD, KD, VD = H * K, H * K, HV * V
D = QD + KD + VD
N, TOK = 4, 4                      # 4 requests x 4 MTP tokens
T = N * TOK
dev, dt = "cuda", torch.bfloat16
torch.manual_seed(0)


def stock_rearrange(mixed):
    """rearrange_mixed_qkv verbatim: 3 reshape copies + one cat."""
    q, k, v = torch.split(mixed, [QD, KD, VD], dim=-1)
    f = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)], dim=0)
    s = mixed.shape[0]
    return (f[: s * QD].view(1, s, -1, K),
            f[s * QD : s * (QD + KD)].view(1, s, -1, K),
            f[s * (QD + KD):].view(1, s, -1, V))


def strided_rearrange(mixed):
    """rearrange_mixed_qkv_strided verbatim."""
    s = mixed.shape[0]
    row, col, base = mixed.stride(0), mixed.stride(1), mixed.storage_offset()
    out = []
    for off, dim, hd in ((0, QD, K), (QD, KD, K), (QD + KD, VD, V)):
        out.append(mixed.as_strided((1, s, dim // hd, hd),
                                    (s * row, row, hd * col, col), base + off * col))
    return tuple(out)


mixed = torch.randn(T, D, dtype=dt, device=dev) * 0.3
A_log = torch.randn(HV, dtype=torch.float32, device=dev)
dt_bias = torch.randn(HV, dtype=torch.float32, device=dev)
a = torch.randn(T, HV, dtype=dt, device=dev)
b = torch.randn(T, HV, dtype=dt, device=dev)
cu = torch.arange(0, T + 1, TOK, dtype=torch.int32, device=dev)
idx = torch.arange(1, N + 1, dtype=torch.int32, device=dev).unsqueeze(1).repeat(1, TOK)
state0 = torch.randn(N + 1, HV, V, K, dtype=torch.float32, device=dev) * 0.1


def run(rearr):
    q, k, v = rearr(mixed)
    st = state0.clone()
    o, fin = fused(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
                   initial_state=st, inplace_final_state=True, cu_seqlens=cu,
                   ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
    return o, fin


o_ref, s_ref = run(stock_rearrange)
o_new, s_new = run(strided_rearrange)
assert o_ref.shape == o_new.shape, (o_ref.shape, o_new.shape)
same_o = torch.equal(o_ref, o_new)
same_s = torch.equal(s_ref, s_new)
print(f"output {tuple(o_ref.shape)} bit-identical: {same_o}   ssm_state bit-identical: {same_s}")
if not (same_o and same_s):
    d = (o_ref.float() - o_new.float()).abs()
    print("  max abs", d.max().item())
assert same_o and same_s, "strided path changed the numerics"


def launches(fn, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in p.events()
               if e.device_type == torch.autograd.DeviceType.CUDA and e.device_index >= 0)


n_ref = launches(lambda: run(stock_rearrange))
n_new = launches(lambda: run(strided_rearrange))
print(f"GPU launches per GDN layer: stock {n_ref}, strided {n_new}  (delta {n_new - n_ref})")
print(f"36 GDN layers/step -> {(n_new - n_ref) * 36:+d} kernels/step")
print("\nGDN STRIDED TEST OK")
