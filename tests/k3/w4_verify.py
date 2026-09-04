"""Correctness of draft_w4_lmhead against a dequantized reference. GPU, seconds."""
import json, os, sys
import torch
import torch.nn.functional as F
import vllm.model_executor.kernels.draft_w4_lmhead as W

dev = "cuda:0"
torch.manual_seed(0)
res = {"available": W.available(), "GROUP": W.GROUP, "MAX_M": W.MAX_M}
print("available", res["available"], "GROUP", W.GROUP, "MAX_M", W.MAX_M, flush=True)
assert res["available"], "kernel unavailable"

fail = 0

def check(tag, cond, extra=""):
    global fail
    print(f"  {'OK ' if cond else 'FAIL'} {tag} {extra}", flush=True)
    if not cond:
        fail += 1

# ---- 1. pack/unpack round-trip against quantize_w4 ------------------------------------
for (n, k) in [(256, 512), (1024, 2560), (48, 128), (7776, 2560)]:
    w = (torch.randn(n, k, device=dev) * 0.02).bfloat16()
    p = W.pack_w4a16(w)
    q, scale, zero = W.quantize_w4(w.float())
    ref = ((q.float().view(n, k // W.GROUP, W.GROUP) - zero.float().unsqueeze(-1))
           * scale.float().unsqueeze(-1)).view(n, k)
    got = W.unpack_w4(p, torch.float32)[:n]
    check(f"unpack==quantize_w4 N={n} K={k}",
          torch.equal(got, ref), f"maxdiff={(got-ref).abs().max().item():.3e}")
    rel = (ref - w.float()).norm() / w.float().norm()
    print(f"      quant rel-err {rel.item()*100:.3f}%", flush=True)

# ---- 2. kernel vs dequantized reference, every legal cfg ------------------------------
n, k = 1024, 2560
w = (torch.randn(n, k, device=dev) * 0.02).bfloat16()
p = W.pack_w4a16(w)
wref = W.unpack_w4(p, torch.float32)
worst = {}
for wv in (1, 2, 4, 8):
    for sk in (1, 2, 4, 5, 10, 20):
        for npw in (1, 4):
            for nt in (0, 1):
                if not W._legal(k, p.n_pad // 16, wv, sk, npw):
                    continue
                for m in (1, 2, 3, 5, 16, 20, 64):
                    x = torch.randn(m, k, device=dev).bfloat16()
                    y = torch.ops.vllm.draft_w4_lmhead_gemm(
                        x, p.wq, p.wsz, p.n, p.k, wv, sk, npw, nt)
                    r = F.linear(x.float().half().float(), wref)
                    e = ((y.float() - r).norm() / r.norm()).item()
                    key = (wv, sk, npw, nt)
                    worst[key] = max(worst.get(key, 0.0), e)
bad = {str(k_): v for k_, v in worst.items() if v > 5e-3}
check("kernel==dequant ref over all legal cfgs", not bad,
      f"n_cfg={len(worst)} worst={max(worst.values()):.3e} bad={bad}")
res["cfg_rel_err"] = {str(k_): v for k_, v in sorted(worst.items())}

# ---- 3. cfg picked by pick_cfg is legal and correct -----------------------------------
for m in (1, 2, 4, 5, 20, 64):
    x = torch.randn(m, k, device=dev).bfloat16()
    y = W.gemm_w4a16(x, p)
    r = F.linear(x.float().half().float(), wref)
    e = ((y.float() - r).norm() / r.norm()).item()
    check(f"gemm_w4a16 m={m} cfg={W.pick_cfg(m, p.n_pad, p.k)}", e < 5e-3, f"rel={e:.3e}")

# ---- 4. M > MAX_M fallback ------------------------------------------------------------
x = torch.randn(129, k, device=dev).bfloat16()
y = W.gemm_w4a16(x, p)
r = F.linear(x.float(), wref)
e = ((y.float() - r).norm() / r.norm()).item()
check("M=129 dequant fallback", e < 5e-3 and y.shape == (129, n), f"rel={e:.3e}")

# ---- 5. N not a multiple of 16 --------------------------------------------------------
n2 = 1000
w2 = (torch.randn(n2, k, device=dev) * 0.02).bfloat16()
p2 = W.pack_w4a16(w2)
x = torch.randn(5, k, device=dev).bfloat16()
y = W.gemm_w4a16(x, p2)
r = F.linear(x.float().half().float(), W.unpack_w4(p2, torch.float32)[:n2])
e = ((y.float() - r).norm() / r.norm()).item()
check(f"N={n2} padded to {p2.n_pad}", e < 5e-3 and y.shape == (5, n2), f"rel={e:.3e}")

# ---- 6. HIP-graph capture -------------------------------------------------------------
x = torch.randn(5, k, device=dev).bfloat16()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        W.gemm_w4a16(x, p)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    yg = W.gemm_w4a16(x, p)
x.copy_(torch.randn(5, k, device=dev).bfloat16())
g.replay()
torch.cuda.synchronize()
r = F.linear(x.float().half().float(), wref)
e = ((yg.float() - r).norm() / r.norm()).item()
check("HIP-graph capture + replay", e < 5e-3, f"rel={e:.3e}")

res["fail"] = fail
json.dump(res, open("/w/tests/k3/w4_verify.json", "w"), indent=1)
print("FAILURES", fail, flush=True)
sys.exit(1 if fail else 0)
