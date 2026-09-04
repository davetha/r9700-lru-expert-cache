"""Standalone reference + timing harness for r4d_gemm_moe_mxfp4a8_nt_b16.

Builds random E-expert MXFP4 weights, fp8 activations and top-k routing in EXACTLY the
layout vllm's R4dMxfp4MoEExperts hands the kernel (see MOE_ABI.md), calls the kernel through
ctypes, and checks it against a pure-torch dequantised reference. No vllm import: the
moe_align_block_size packing is reimplemented here (and cross-checked against vllm's own C op
when it is importable), so a replacement kernel can be validated with nothing but torch.

    R4D_LIB=/path/to/r4d.so     library under test          (default /app/r4dhip/r4d.so)
    CASES=gate_up,down          which shapes                (default both)
    MS=1,4,8,16                 token counts                (default 1,4,8,16)
    E=512 TOPK=10               expert count / top-k
    UVA=1                       also time the weight in pinned host memory (cold experts)
    SEED=0

Run it under the GPU lock:
    flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --ipc host --group-add video \
      --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v $REPO_ROOT:/w \
      --entrypoint bash local/q38fn-rocm10:try1 -c 'cd /w/tools/routecap && python3 moe_ref_harness.py'
"""
import ctypes
import os
import sys
import time

import torch

GROUP = 32          # K per E8M0 exponent, = r4d_gemm_moe_mxfp4a8_nt_b16_group()
BLOCK = 16          # tokens per aligned block,  = r4d_gemm_moe_mxfp4a8_nt_b16_block()
FP8_MAX = 448.0
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

LIB_PATH = os.environ.get("R4D_LIB", "/app/r4dhip/r4d.so")
SEED = int(os.environ.get("SEED", "0"))
E = int(os.environ.get("E", "512"))
TOPK = int(os.environ.get("TOPK", "10"))
MS = [int(v) for v in os.environ.get("MS", "1,4,8,16").split(",")]
WANT = os.environ.get("CASES", "gate_up,down").split(",")
DO_UVA = os.environ.get("UVA", "1") == "1"

# q38fn (hidden 2560, moe_intermediate 640, E=512, top_k=10) at TP2, per rank.
#   gate_up: w1 [E, 2*I, H]  N=640  K=2560   down: w2 [E, H, I]  N=2560  K=320
CASES = {"gate_up": (640, 2560), "down": (2560, 320),
         # Not a model shape: N/16 = 5 tiles with ncols = WV*NPW = 8, so the single grid.x
         # block addresses column tiles that do not exist. The dense kernel clamps the tile
         # index here (an unclamped index is a memory fault, not a wrong answer); run
         # CASES=tail to check the MoE kernel does the same.
         "tail": (80, 2560)}


# ---------------------------------------------------------------- library surface
def load(path):
    lib = ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_DEEPBIND)
    lib.r4d_gemm_moe_mxfp4a8_nt_b16.restype = None
    lib.r4d_gemm_moe_mxfp4a8_nt_b16.argtypes = (
        [ctypes.c_long] * 10 + [ctypes.c_int] * 8 + [ctypes.c_long])
    lib.r4d_gemm_moe_mxfp4a8_nt_b16_block.restype = ctypes.c_int
    lib.r4d_gemm_moe_mxfp4a8_nt_b16_group.restype = ctypes.c_int
    return lib


def pick_cfg(N, K):
    """Verbatim from vllm .../linear/mxfp4/r4dhip.py:pick_cfg (no env override)."""
    ntiles = N // 16
    sk = next(s for s in (8, 4, 2, 1) if K % (s * GROUP) == 0)
    npw = 2 if ntiles >= 2 else 1
    wv = max(1, min(32 // sk, 64 // (npw * sk), 8, ntiles))
    return wv, sk, npw


# ---------------------------------------------------------------- layout helpers
def permute_moe(w):
    """[E, N, K/2] checkpoint order -> r4d fragment order, keeping the (E, N, K/2) view.
    Identical per expert to the dense permute (vllm r4dhip.permute_w_gfx1201 /
    libr4d mxfp4_layout.permute_w)."""
    Ee, N, K2 = w.shape
    nt, ks = N // 16, K2 * 2 // 16
    return (w.view(Ee, nt, 16, ks, 2, 4).permute(0, 1, 3, 4, 2, 5)
            .contiguous().view(Ee, N, K2))


def dequant_expert(packed_ckpt_e, s_e):
    """One expert, CHECKPOINT order: [N, K/2] uint8 + [N, K/32] E8M0 -> fp32 [N, K]."""
    N, K2 = packed_ckpt_e.shape
    K = K2 * 2
    codes = torch.empty(N, K, dtype=torch.uint8, device=packed_ckpt_e.device)
    codes[:, 0::2] = packed_ckpt_e & 0x0F
    codes[:, 1::2] = packed_ckpt_e >> 4
    idx = (codes & 0x7).long()
    mag = E2M1.to(codes.device)[idx]
    mag = torch.where((codes & 0x8) > 0, -mag, mag)
    sc = torch.exp2(s_e.float() - 127.0).repeat_interleave(GROUP, dim=1)
    return mag * sc


def quant_fp8_per_token(x):
    """ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True), in torch."""
    amax = x.abs().amax(dim=1).float().clamp(min=1e-12)
    scale = (amax / FP8_MAX).contiguous()
    q = (x.float() / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def moe_align_block_size_torch(topk_ids, block, num_experts):
    """Pure-torch moe_align_block_size. Returns (sorted_ids, expert_ids, num_post_pad).

    sorted_ids  int32 [numel + E*(block-1)]  (clamped to numel*block when numel < E),
                per-expert runs of flat (token*top_k + k) ids, each run padded UP to a
                multiple of `block` with the sentinel `numel`; the tail past
                num_post_pad is sentinel too.
    expert_ids  int32 [ceil(len(sorted_ids)/block)], one expert per block.
    """
    mtk = topk_ids.numel()
    cap = mtk + num_experts * (block - 1)
    if mtk < num_experts:
        cap = min(mtk * block, cap)
    flat = topk_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts)
    blocks = (counts + block - 1) // block
    sorted_ids = torch.full((cap,), mtk, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.zeros((cap + block - 1) // block, dtype=torch.int32,
                             device=topk_ids.device)
    counts_c = counts.tolist()          # one sync, not one per expert
    src = 0
    dst = 0
    for e in range(num_experts):
        c = counts_c[e]
        if c == 0:
            continue
        nb = (c + block - 1) // block
        sorted_ids[dst:dst + c] = order[src:src + c].to(torch.int32)
        expert_ids[dst // block: dst // block + nb] = e
        src += c
        dst += nb * block
    npad = torch.tensor([dst], dtype=torch.int32, device=topk_ids.device)
    return sorted_ids, expert_ids, npad


def check_against_vllm(topk_ids, sorted_ids, expert_ids, npad):
    """Cross-check the pure-torch packing against vllm's own C op, if importable."""
    try:
        from vllm import _custom_ops as ops
    except Exception as exc:                                   # noqa: BLE001
        return f"not checked ({type(exc).__name__})"
    mtk = topk_ids.numel()
    cap = sorted_ids.numel()
    si = torch.empty(cap, dtype=torch.int32, device=topk_ids.device)
    ei = torch.empty(expert_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    np_ = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    try:
        ops.moe_align_block_size(topk_ids, E, BLOCK, si, ei, np_, None)
    except Exception as exc:                                   # noqa: BLE001
        return f"not checked ({type(exc).__name__}: {str(exc)[:40]})"
    n = int(np_.item())
    # vllm's C op is NOT stable within an expert (measured: a 2-token expert can come out
    # [68,41] where a stable sort gives [41,68]). Order inside a block is semantically
    # irrelevant -- the kernel computes each row independently -- so compare block CONTENTS.
    same_blocks = torch.equal(si[:n].view(-1, BLOCK).sort(dim=1).values,
                              sorted_ids[:n].view(-1, BLOCK).sort(dim=1).values)
    ok = (int(npad.item()) == n
          and same_blocks
          and torch.equal(ei[:n // BLOCK], expert_ids[:n // BLOCK])
          and bool((si[n:] == mtk).all()))
    exact = ok and torch.equal(si[:n], sorted_ids[:n])
    if exact:
        return "MATCHES vllm exactly"
    return ("MATCHES vllm up to intra-block order" if ok else "DIFFERS from vllm")


# ---------------------------------------------------------------- case construction
def build_case(N, K, M, top_k, seed):
    """Everything the kernel is handed, in the fork's exact layout, plus the fp32 reference."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    K2, nb = K // 2, K // GROUP

    # E8M0 scales, built so every folded magnitude (2^-d * e2m1, d = wref - e8m0 <= 8) is
    # exactly representable in e4m3 -- the same trick libr4d's own test uses, so the torch
    # reference is exact for the weight and the only slack is accumulate order + bf16 output.
    ref = torch.randint(120, 136, (E, N, 1), generator=g, dtype=torch.int32, device="cuda")
    drop = torch.randint(0, 9, (E, N, nb), generator=g, dtype=torch.int32, device="cuda")
    s = (ref - drop).clamp(0, 254).to(torch.uint8)                     # [E, N, K/32]
    w_ckpt = torch.randint(0, 256, (E, N, K2), generator=g, dtype=torch.uint8, device="cuda")

    wq = permute_moe(w_ckpt)                                           # [E, N, K/2] fragment
    ws_t = s.transpose(1, 2).contiguous()                              # [E, K/32, N]
    wref = s.max(dim=2).values.contiguous()                            # [E, N]

    topk_ids = torch.randint(0, E, (M, top_k), generator=g, dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, npad = moe_align_block_size_torch(topk_ids, BLOCK, E)
    align_note = check_against_vllm(topk_ids, sorted_ids, expert_ids, npad)

    mtk = M * top_k
    x = torch.randn(mtk // top_k if top_k > 1 else mtk, K, generator=g,
                    device="cuda", dtype=torch.float32) * 0.4
    qx, xs = quant_fp8_per_token(x)
    tw = torch.rand(mtk, generator=g, device="cuda", dtype=torch.float32)

    # fp32 reference: row so takes expert topk_ids.flat[so] and activation row so // top_k.
    flat = topk_ids.reshape(-1)
    a_f32 = qx.float() * xs.unsqueeze(1)
    ref_out = torch.zeros(mtk, N, dtype=torch.float32, device="cuda")
    for e in torch.unique(flat).tolist():
        rows = (flat == e).nonzero(as_tuple=True)[0]
        wd = dequant_expert(w_ckpt[e], s[e])                           # [N, K] fp32
        ref_out[rows] = a_f32[(rows // top_k).long()] @ wd.T
        del wd
    return dict(wq=wq, ws_t=ws_t, wref=wref, qx=qx, xs=xs, tw=tw, topk_ids=topk_ids,
                sorted_ids=sorted_ids, expert_ids=expert_ids, npad=npad, mtk=mtk,
                ref=ref_out, align_note=align_note, w_ckpt=w_ckpt, s=s)


def make_call(lib, c, wq_ptr, cse, N, K, top_k, cfg, use_tw):
    wv, sk, npw = cfg
    em = cse["sorted_ids"].numel()
    mtk = cse["mtk"]
    assert em >= 1 and N % 16 == 0 and K % (sk * GROUP) == 0 \
        and wv * sk * 32 <= 1024 and npw in (1, 2, 4, 8) \
        and wv * npw * sk * 1024 <= 64 * 1024
    st = torch.cuda.current_stream().cuda_stream
    f = lib.r4d_gemm_moe_mxfp4a8_nt_b16
    args = (cse["qx"].data_ptr(), cse["xs"].data_ptr(), wq_ptr, cse["ws_t"].data_ptr(),
            cse["wref"].data_ptr(), c.data_ptr(), cse["sorted_ids"].data_ptr(),
            cse["expert_ids"].data_ptr(), cse["npad"].data_ptr(),
            cse["tw"].data_ptr() if use_tw else 0,
            em, mtk, top_k, K, N, wv, sk, npw, st)

    def go():
        f(*args)
    return go


def bench(go, n=25):
    for _ in range(10):
        go()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(n):
        t = time.perf_counter()
        go()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best * 1e6


# ---------------------------------------------------------------- pinned-host (UVA) weight
_hip = None


def uva_device_ptr(t_cuda):
    """Copy a device tensor into PINNED HOST memory and return (host_tensor, device_ptr).

    This is the cold-expert case the hot/cold residency patch produces: the expert weight
    lives in host RAM and the kernel dereferences it over PCIe through a UVA pointer.
    """
    global _hip
    if _hip is None:
        _hip = ctypes.CDLL("libamdhip64.so.7")
        _hip.hipHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                                 ctypes.c_void_p, ctypes.c_uint]
        _hip.hipHostGetDevicePointer.restype = ctypes.c_int
        _hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        _hip.hipHostRegister.restype = ctypes.c_int
    host = torch.empty(t_cuda.shape, dtype=t_cuda.dtype, pin_memory=True)
    host.copy_(t_cuda.cpu())
    dev = ctypes.c_void_p()
    rc = _hip.hipHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(host.data_ptr()), 0)
    if rc != 0:
        nbytes = host.numel() * host.element_size()
        rc2 = _hip.hipHostRegister(ctypes.c_void_p(host.data_ptr()), nbytes, 0x2)
        if rc2 != 0:
            raise RuntimeError(f"hipHostGetDevicePointer rc={rc}, hipHostRegister rc={rc2}")
        rc = _hip.hipHostGetDevicePointer(ctypes.byref(dev),
                                          ctypes.c_void_p(host.data_ptr()), 0)
        if rc != 0:
            raise RuntimeError(f"hipHostGetDevicePointer after register rc={rc}")
    return host, dev.value



# ---------------------------------------------------------------- ABI probes
def probe_abi(lib):
    """Verify the three control-flow contracts a replacement kernel must also honour.

    Each probe pre-fills C with a NaN marker and checks exactly which rows the kernel wrote.
    """
    N, K = CASES["gate_up"]
    cfg = pick_cfg(N, K)
    cse = build_case(N, K, 8, TOPK, SEED)
    mtk = cse["mtk"]
    npad = int(cse["npad"].item())
    si = cse["sorted_ids"]

    def run(sorted_ids, expert_ids, npad_t):
        c = torch.full((mtk, N), float("nan"), dtype=torch.bfloat16, device="cuda")
        saved = (cse["sorted_ids"], cse["expert_ids"], cse["npad"])
        cse["sorted_ids"], cse["expert_ids"], cse["npad"] = sorted_ids, expert_ids, npad_t
        make_call(lib, c, cse["wq"].data_ptr(), cse, N, K, TOPK, cfg, False)()
        torch.cuda.synchronize()
        cse["sorted_ids"], cse["expert_ids"], cse["npad"] = saved
        return ~c.isnan().any(dim=1)          # per-row "was written"

    written = run(si, cse["expert_ids"], cse["npad"])
    live = torch.zeros(mtk, dtype=torch.bool, device="cuda")
    live[si[:npad][si[:npad] < mtk].long()] = True
    print("  probe/baseline      : wrote exactly the live sorted_ids rows: %s (%d rows)"
          % (bool(torch.equal(written, live)), int(live.sum())))

    eid = cse["expert_ids"].clone()
    eid[0] = -1
    w2 = run(si, eid, cse["npad"])
    b0 = si[:BLOCK][si[:BLOCK] < mtk].long()
    print("  probe/expert_id=-1  : block 0 skipped: %s ; other rows unchanged: %s"
          % (bool((~w2[b0]).all()),
             bool(torch.equal(w2[live.clone().index_fill_(0, b0, False)],
                              written[live.clone().index_fill_(0, b0, False)]))))

    short = torch.tensor([BLOCK], dtype=torch.int32, device="cuda")
    w3 = run(si, cse["expert_ids"], short)
    print("  probe/num_post_pad  : honoured on device (only block 0 written): %s"
          % bool(torch.equal(w3.nonzero(as_tuple=True)[0].sort().values, b0.sort().values)))


# ---------------------------------------------------------------- main
def main():
    torch.zeros(1, device="cuda")
    lib = load(LIB_PATH)
    print(f"R4D_LIB={LIB_PATH}  block={lib.r4d_gemm_moe_mxfp4a8_nt_b16_block()} "
          f"group={lib.r4d_gemm_moe_mxfp4a8_nt_b16_group()}  E={E} top_k={TOPK}")
    if os.environ.get("PROBE", "1") == "1":
        print("ABI probes (gate_up, M=8):")
        probe_abi(lib)
        print()
    print("%-9s %6s %6s %4s %5s %-8s %-9s %-10s %-10s %8s %9s"
          % ("case", "N", "K", "M", "mtk", "cfg", "routed_w", "max_abs", "max_rel",
             "us(vram)", "us(uva)"))
    worst = 0.0
    for name in WANT:
        N, K = CASES[name]
        cfg = pick_cfg(N, K)
        # GEMM1 takes M activation rows and top_k>1; GEMM2 takes mtk rows with top_k=1 and
        # applies the routed weight in its epilogue. Exercise both, as the fork does.
        top_k = TOPK if name == "gate_up" else 1
        use_tw = name == "down"
        for M in MS:
            cse = build_case(N, K, M if top_k > 1 else M * TOPK, top_k, SEED)
            mtk = cse["mtk"]
            c = torch.zeros(mtk, N, dtype=torch.bfloat16, device="cuda")
            go = make_call(lib, c, cse["wq"].data_ptr(), cse, N, K, top_k, cfg, use_tw)
            go()
            torch.cuda.synchronize()
            got = c.float()
            exp = cse["ref"] * (cse["tw"].unsqueeze(1) if use_tw else 1.0)
            err = (got - exp).abs()
            denom = exp.abs().amax().clamp(min=1e-30)
            maxabs = err.amax().item()
            maxrel = (maxabs / denom).item()
            worst = max(worst, maxrel)
            t_vram = bench(go)
            t_uva = float("nan")
            if DO_UVA:
                host, dptr = uva_device_ptr(cse["wq"])
                c2 = torch.zeros_like(c)
                go2 = make_call(lib, c2, dptr, cse, N, K, top_k, cfg, use_tw)
                go2()
                torch.cuda.synchronize()
                same = torch.equal(c2.view(torch.int16), c.view(torch.int16))
                if not same:
                    print("   !! UVA weight gave a DIFFERENT result than the VRAM weight")
                t_uva = bench(go2, 10)
                del host, c2
            print("%-9s %6d %6d %4d %5d %-8s %-9s %-10.3e %-10.3e %8.1f %9.1f"
                  % (name, N, K, M if top_k > 1 else M * TOPK, mtk,
                     "/".join(map(str, cfg)), str(use_tw), maxabs, maxrel, t_vram, t_uva))
            if M == MS[0]:
                print("           moe_align cross-check: " + cse["align_note"])
            del cse, c
            torch.cuda.empty_cache()
    print(f"\nworst max_rel (vs |ref|max) over all cases: {worst:.3e}")
    print("A replacement kernel must reach the same order; the residual here is fp32 "
          "accumulate order + bf16 output rounding, the weight itself is exact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
