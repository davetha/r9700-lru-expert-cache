"""Quantization error of the real lm_head under int4 asymmetric g128/g64/g32. CPU only."""
import json, struct, os
import numpy as np
import torch
import vllm.model_executor.kernels.draft_w4_lmhead as W

MODEL = "/m/q38fn-heretic2-mxfp4-fp8"
SHARD = f"{MODEL}/model-00001-of-00025.safetensors"
NRANK = int(os.environ.get("NRANK", "124160"))     # rank-0 vocab slice (TP2 of 248320)
NPROBE = int(os.environ.get("NPROBE", "200"))
CH = 8192

with open(SHARD, "rb") as f:
    hl = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hl))
info = hdr["lm_head.weight"]
base = 8 + hl
N, K = info["shape"]
off0, off1 = info["data_offsets"]
assert info["dtype"] == "BF16" and off1 - off0 == N * K * 2
mm = np.memmap(SHARD, dtype=np.uint16, mode="r", offset=base + off0, shape=(N, K))
print(f"lm_head {N}x{K} bf16; analysing rank-0 slice rows 0:{NRANK}", flush=True)


def bf16_rows(r0, r1):
    u = torch.from_numpy(np.ascontiguousarray(mm[r0:r1])).view(torch.bfloat16)
    return u.float()


def quant(w, g, clip):
    """The shipped quantizer, dequantized back -- exactly what the kernel will compute."""
    n, k = w.shape
    q, scale, zero = W.quantize_w4(w, g, clip)
    return ((q.float().view(n, k // g, g) - zero.float().unsqueeze(-1))
            * scale.float().unsqueeze(-1)).view(n, k)


torch.manual_seed(0)
H = torch.randn(NPROBE, K)                       # isotropic probe (NOT real hidden states)
H = H / H.norm(dim=1, keepdim=True) * (K ** 0.5)

ARMS = [("g128_minmax", 128, (1.0,)), ("g128_clip", 128, None),
        ("g64_clip", 64, None), ("g32_clip", 32, None)]
out = {"N": N, "K": K, "n_rank": NRANK, "n_probe": NPROBE, "groups": {}}
for tag, g, clip in ARMS:
    se = torch.zeros(()); sw = torch.zeros(()); mx = torch.zeros(())
    rowerr = []
    lg_ref = torch.zeros(NPROBE, NRANK)
    lg_q = torch.zeros(NPROBE, NRANK)
    for r0 in range(0, NRANK, CH):
        r1 = min(r0 + CH, NRANK)
        w = bf16_rows(r0, r1)
        d = quant(w, g, clip)
        e = d - w
        se += (e * e).sum(); sw += (w * w).sum(); mx = torch.maximum(mx, e.abs().max())
        rowerr.append((e.norm(dim=1) / w.norm(dim=1).clamp(min=1e-30)))
        lg_ref[:, r0:r1] = H @ w.T
        lg_q[:, r0:r1] = H @ d.T
        del w, d, e
    rowerr = torch.cat(rowerr)
    top1 = (lg_ref.argmax(1) == lg_q.argmax(1)).float().mean().item()
    t5r = lg_ref.topk(5, 1).indices
    t5q = lg_q.topk(5, 1).indices
    ov = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(t5r, t5q)) / (5 * NPROBE)
    dlog = (lg_q - lg_ref)
    bpr = 0.5 + 4.0 / g                          # bytes per weight (nibble + 4B scale/group)
    out["groups"][tag] = {
        "rel_fro_pct": (se.sqrt() / sw.sqrt()).item() * 100,
        "max_abs": mx.item(),
        "row_rel_pct": {p: torch.quantile(rowerr, q).item() * 100 for p, q in
                        (("p50", .5), ("p90", .9), ("p99", .99), ("max", 1.0))},
        "bytes_per_weight": bpr,
        "MB_per_rank": NRANK * K * bpr / 2**20,
        "logit_dlog_rms": dlog.std().item(),
        "logit_ref_std": lg_ref.std().item(),
        "logit_dlog_over_std_pct": (dlog.std() / lg_ref.std()).item() * 100,
        "isotropic_top1_agree": top1,
        "isotropic_top5_overlap": ov,
        "isotropic_top1_gap_mean": (lg_ref.topk(2, 1).values.diff(dim=1).abs().mean().item()),
    }
    print(tag, json.dumps(out["groups"][tag], indent=1), flush=True)

json.dump(out, open("/w/tests/k3/w4_quant_err.json", "w"), indent=1)
print("DONE", flush=True)
