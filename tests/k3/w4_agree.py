"""Top-1 agreement of the W4 draft lm_head vs the bf16 lm_head on REAL draft hidden states.

CPU only, no GPU window. Reads the full 248320-row lm_head from the checkpoint so the
argmax is the true full-vocab argmax (both TP shards), not a shard-local proxy.

  HS=/w/tests/k4/draft_hs.npz python3 /w/tests/k3/w4_agree.py
"""
import json, os, struct
import numpy as np
import torch
import vllm.model_executor.kernels.draft_w4_lmhead as W

MODEL = "/m/q38fn-heretic2-mxfp4-fp8"
SHARD = f"{MODEL}/model-00001-of-00025.safetensors"
HS = os.environ["HS"]
CH = 8192
GROUP = 128

d = np.load(HS)
key = "hidden_states" if "hidden_states" in d else list(d.keys())[0]
H32 = torch.from_numpy(np.ascontiguousarray(d[key])).float()
assert H32.dim() == 2, H32.shape
maxabs = H32.abs().max().item()
print(f"hidden states {tuple(H32.shape)} from {HS}[{key}], max|x|={maxabs:.1f} "
      f"({'OK' if maxabs < 65504 else 'OVERFLOWS f16'})", flush=True)

# What each path actually feeds the GEMM: bf16 x for the stock head, f16 x for r4d w4a16.
Hbf = H32.to(torch.bfloat16).float()
Hf16 = H32.to(torch.float16).float()
NPROBE, K = H32.shape

with open(SHARD, "rb") as f:
    hl = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hl))
info = hdr["lm_head.weight"]
N, KW = info["shape"]
assert KW == K, (KW, K)
off0, off1 = info["data_offsets"]
assert info["dtype"] == "BF16" and off1 - off0 == N * K * 2
mm = np.memmap(SHARD, dtype=np.uint16, mode="r", offset=8 + hl + off0, shape=(N, K))
print(f"lm_head {N}x{K} bf16 (full vocab, both TP shards)", flush=True)

lg_ref = torch.zeros(NPROBE, N)
lg_w4 = torch.zeros(NPROBE, N)      # shipped path: f16 activation x W4 weights
lg_w4b = torch.zeros(NPROBE, N)     # W4 weights, bf16 activation (isolates the f16 cast)
se = torch.zeros(()); sw = torch.zeros(())
for r0 in range(0, N, CH):
    r1 = min(r0 + CH, N)
    w = torch.from_numpy(np.ascontiguousarray(mm[r0:r1])).view(torch.bfloat16).float()
    n, k = w.shape
    q, scale, zero = W.quantize_w4(w, GROUP, None)
    dq = ((q.float().view(n, k // GROUP, GROUP) - zero.float().unsqueeze(-1))
          * scale.float().unsqueeze(-1)).view(n, k)
    e = dq - w
    se += (e * e).sum(); sw += (w * w).sum()
    lg_ref[:, r0:r1] = Hbf @ w.T
    lg_w4[:, r0:r1] = Hf16 @ dq.T
    lg_w4b[:, r0:r1] = Hbf @ dq.T
    del w, dq, e, q, scale, zero
    print(f"  rows {r1}/{N}", end="\r", flush=True)
print(flush=True)

a_ref = lg_ref.argmax(1)
res = {"hs": HS, "n_probe": NPROBE, "K": K, "N_vocab": N, "max_abs_x": maxabs,
       "weight_rel_fro_pct": (se.sqrt() / sw.sqrt()).item() * 100}
for tag, lg in (("w4_f16act", lg_w4), ("w4_bf16act", lg_w4b)):
    a = lg.argmax(1)
    t5r, t5 = lg_ref.topk(5, 1).indices, lg.topk(5, 1).indices
    ov = sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(t5r, t5)) / (5 * NPROBE)
    res[tag] = {
        "top1_agree": (a_ref == a).float().mean().item(),
        "top5_overlap": ov,
        "ref_top1_in_w4_top5": sum(int(a_ref[i].item() in set(t5[i].tolist()))
                                   for i in range(NPROBE)) / NPROBE,
        "dlogit_over_ref_std_pct": ((lg - lg_ref).std() / lg_ref.std()).item() * 100,
    }
res["ref_top1_gap_mean"] = lg_ref.topk(2, 1).values.diff(dim=1).abs().mean().item()
res["ref_logit_std"] = lg_ref.std().item()
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open("/w/tests/k3/w4_agree.json", "w"), indent=1)
print("DONE", flush=True)
