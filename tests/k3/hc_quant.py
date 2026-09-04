#!/usr/bin/env python3
"""Quality cost of quantizing the Qwen3.8-Flash-Next hyper-connection weights.

For every real `input_mix_weight_down` / `input_mix_weight_up` / `block_inject_weight`
tensor in the checkpoint, quantize with the schemes a shipped kernel could consume and
report relative output error on activations whose per-channel scale matches what the
layer actually sees (hc_norm.weight for the K=10240 inputs). CPU only.
"""
import json, glob, struct, sys, time
import torch
from safetensors import safe_open

MODEL = '/models/q38fn-heretic2-mxfp4-fp8'
NX = 256
FP8MAX = 448.0
torch.set_num_threads(16)
torch.manual_seed(0)


def q_int_group(w, g, sym):
    """Asymmetric (sym=False, AWQ/uint4) or symmetric (sym=True, uint4b8) int4, group along K."""
    N, K = w.shape
    assert K % g == 0, (K, g)
    x = w.float().view(N, K // g, g)
    if sym:
        s = x.abs().amax(-1, keepdim=True) / 7.0
        s = s.clamp(min=1e-8)
        q = (x / s).round().clamp(-8, 7)
        return (q * s).view(N, K)
    mn = x.amin(-1, keepdim=True)
    mx = x.amax(-1, keepdim=True)
    s = ((mx - mn) / 15.0).clamp(min=1e-8)
    z = (-mn / s).round().clamp(0, 15)
    q = (x / s + z).round().clamp(0, 15)
    return ((q - z) * s).view(N, K)


def q_int8_chan(w):
    s = (w.float().abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-8)
    return (w.float() / s).round().clamp(-128, 127) * s


def q_fp8_chan(w):
    s = (w.float().abs().amax(1, keepdim=True) / FP8MAX).clamp(min=1e-12)
    return (w.float() / s).to(torch.float8_e4m3fn).float() * s


def q_fp8_block(w, B=128):
    N, K = w.shape
    pn, pk = (-N) % B, (-K) % B
    x = torch.nn.functional.pad(w.float(), (0, pk, 0, pn))
    Np, Kp = x.shape
    x = x.view(Np // B, B, Kp // B, B).permute(0, 2, 1, 3)
    s = (x.abs().amax((2, 3), keepdim=True) / FP8MAX).clamp(min=1e-12)
    q = (x / s).to(torch.float8_e4m3fn).float() * s
    return q.permute(0, 2, 1, 3).reshape(Np, Kp)[:N, :K]


def q_fp8_tensor(w):
    s = (w.float().abs().max() / FP8MAX).clamp(min=1e-12)
    return (w.float() / s).to(torch.float8_e4m3fn).float() * s


def q_int8_tensor(w):
    s = (w.float().abs().max() / 127.0).clamp(min=1e-8)
    return (w.float() / s).round().clamp(-128, 127) * s


def stats(w):
    x = w.float()
    r = x.flatten()
    k = ((r - r.mean()) ** 4).mean() / (r.var() ** 2 + 1e-30)
    # per (row, group-128 along K) dynamic range: max|w| / rms, the thing 4 bits has to cover
    N, K = x.shape
    g = 128 if K % 128 == 0 else 64
    v = x.view(N, K // g, g)
    dr = (v.abs().amax(-1) / (v.pow(2).mean(-1).sqrt() + 1e-30))
    return dict(kurtosis=k.item(), rms=r.pow(2).mean().sqrt().item(),
                absmax=r.abs().max().item(),
                dr_mean=dr.mean().item(), dr_p99=dr.flatten().kthvalue(int(0.99 * dr.numel())).values.item(),
                frac_gt_4rms=(r.abs() > 4 * r.pow(2).mean().sqrt()).float().mean().item())


def main():
    # activation scale proxy: hc_norm.weight of the same module (K=10240 inputs)
    norms, tensors = {}, {}
    for f in sorted(glob.glob(MODEL + '/*.safetensors')):
        with safe_open(f, 'pt') as sf:
            for k in sf.keys():
                if k.endswith('hc_norm.weight'):
                    norms[k[: -len('.hc_norm.weight')]] = sf.get_tensor(k).float()
                elif k.endswith(('input_mix_weight_down.weight', 'input_mix_weight_up.weight',
                                 'block_inject_weight.weight')):
                    tensors[k] = sf.get_tensor(k)
    print(f'loaded {len(tensors)} weight tensors, {len(norms)} hc_norms', flush=True)

    schemes = [
        ('int4_g128_asym', lambda w: q_int_group(w, 128, False), lambda K: K % 128 == 0),
        ('int4_g64_asym',  lambda w: q_int_group(w, 64, False),  lambda K: K % 64 == 0),
        ('int4_g32_asym',  lambda w: q_int_group(w, 32, False),  lambda K: K % 32 == 0),
        ('int4_g128_sym',  lambda w: q_int_group(w, 128, True),  lambda K: K % 128 == 0),
        ('int8_perchan',   q_int8_chan,                          lambda K: True),
        ('fp8_perchan',    q_fp8_chan,                           lambda K: True),
        ('fp8_pertensor',  q_fp8_tensor,                          lambda K: True),
        ('int8_pertensor', q_int8_tensor,                         lambda K: True),
        ('fp8_block128',   q_fp8_block,                          lambda K: True),
    ]

    agg = {}   # (kind, scheme) -> list of rel errors
    st = {}    # kind -> list of stat dicts
    t0 = time.time()
    for name, w in sorted(tensors.items()):
        kind = name.split('.')[-2]
        mod = name[: name.rfind('.' + kind)]
        W = w.float()
        N, K = W.shape
        if K == 10240 and mod in norms:
            x = torch.randn(NX, K) * norms[mod][None, :]
        else:
            x = torch.randn(NX, K)
        ref = x @ W.t()
        den = ref.norm()
        st.setdefault(kind, []).append(stats(W))
        for sname, fn, ok in schemes:
            if not ok(K):
                continue
            err = ((x @ fn(W).t()) - ref).norm() / den
            agg.setdefault((kind, sname), []).append(err.item())
    print(f'quantized in {time.time()-t0:.0f}s', flush=True)

    out = {'rel_err': {}, 'stats': {}}
    print('\n== relative output error  ||Wq x - W x|| / ||W x||  (mean over tensors, [min,max]) ==')
    print(f'{"tensor kind":28s} {"scheme":16s} {"n":>4s} {"mean":>9s} {"min":>9s} {"max":>9s}')
    for (kind, sname), v in sorted(agg.items()):
        t = torch.tensor(v)
        print(f'{kind:28s} {sname:16s} {len(v):4d} {t.mean():9.5f} {t.min():9.5f} {t.max():9.5f}')
        out['rel_err'][f'{kind}|{sname}'] = [t.mean().item(), t.min().item(), t.max().item(), len(v)]

    print('\n== weight distribution (mean over tensors) ==')
    print(f'{"tensor kind":28s} {"kurtosis":>10s} {"rms":>10s} {"absmax":>10s} {"dr_mean":>9s} {"dr_p99":>9s} {"frac>4rms":>10s}')
    for kind, rows in sorted(st.items()):
        m = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        print(f'{kind:28s} {m["kurtosis"]:10.2f} {m["rms"]:10.5f} {m["absmax"]:10.4f} '
              f'{m["dr_mean"]:9.2f} {m["dr_p99"]:9.2f} {m["frac_gt_4rms"]:10.6f}')
        out['stats'][kind] = m
    json.dump(out, open('/w/tests/k3/hc_quant2.json', 'w'), indent=1)
    print('\nDONE', flush=True)


main()
