"""End-to-end numerics for the LRU expert cache, through the REAL r4d grouped GEMM.

Runs the production two-call shape -- resident call over the compact slot buffers +
fallback call over the full UVA host tensor -- for many steps while the LRU kernels move
experts in and out, and compares the summed output against ONE all-UVA call over the same
routing. Same kernel and same weights on both sides, so anything but bit-identical means
the cache handed the GEMM the wrong slot, the wrong scale row, or an incomplete map.

  flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --ipc host --group-add video \
    --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v $REPO_ROOT:/w \
    --entrypoint bash local/q38fn-rocm10:k1build -c 'cd /w/tests/lru && python3 test_numerics_lru.py'
"""
import ctypes
import os
import sys

os.environ.setdefault("E", "64")
os.environ.setdefault("TOPK", "10")
sys.path.insert(0, "/w/tools/routecap")

import numpy as np
import torch

import moe_ref_harness as H

LIB = os.environ.get("R4D_LRU_LIB", "/w/build/kernels/librlu.so")
rlu = ctypes.CDLL(LIB)
rlu.r4d_lru_manage.restype = ctypes.c_int
rlu.r4d_lru_manage.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 9
rlu.r4d_lru_gather.restype = ctypes.c_int
rlu.r4d_lru_gather.argtypes = ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long] * 6 +
                               [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p])
FUSE = os.environ.get("FUSE", "0") == "1"     # use lru_fused instead of manage+align
if FUSE:
    if not hasattr(rlu, "r4d_lru_fused"):
        raise SystemExit(f"FUSE=1 but {LIB} has no r4d_lru_fused (build.sh rebuilds it)")
    rlu.r4d_lru_fused.restype = ctypes.c_int
    rlu.r4d_lru_fused.argtypes = ([ctypes.c_void_p] + [ctypes.c_int] * 5
                                  + [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3
                                  + [ctypes.c_void_p] * 7)
r4d = H.load(os.environ.get("R4D_LIB", "/app/r4dhip/r4d.so"))
BLOCK, E, TOPK = H.BLOCK, H.E, H.TOPK
DEV = "cuda"
FAILS = []


def align_mapped(topk_ids, n_local, mp):
    """moe_align_block_size(..., expert_map=mp, ignore_invalid_experts=True), in torch.
    Rows whose expert maps to -1 are dropped, so the two calls partition the rows."""
    flat = topk_ids.reshape(-1)
    mtk = flat.numel()
    loc = mp[flat.long()]
    cap = mtk + n_local * (BLOCK - 1)
    if mtk < n_local:
        cap = min(mtk * BLOCK, cap)
    sorted_ids = torch.full((cap,), mtk, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.zeros((cap + BLOCK - 1) // BLOCK, dtype=torch.int32,
                             device=topk_ids.device)
    order = torch.argsort(loc.to(torch.int64), stable=True)
    lsorted = loc[order]
    dst = 0
    counts = torch.bincount(lsorted[lsorted >= 0].to(torch.int64),
                            minlength=n_local).tolist()
    src = int((lsorted < 0).sum().item())
    for e in range(n_local):
        c = counts[e]
        if c == 0:
            continue
        nb = (c + BLOCK - 1) // BLOCK
        sorted_ids[dst:dst + c] = order[src:src + c].to(torch.int32)
        expert_ids[dst // BLOCK: dst // BLOCK + nb] = e
        src += c
        dst += nb * BLOCK
    return sorted_ids, expert_ids, torch.tensor([dst], dtype=torch.int32, device=DEV)


def fused_sizes(mtk, n_local):
    """the lengths vllm's wrapper allocates; production passes num_experts=E for BOTH
    calls, so the fused path is checked at exactly those sizes"""
    n = mtk + n_local * (BLOCK - 1)
    if mtk < n_local:
        n = min(mtk * BLOCK, n)
    return n, (n + BLOCK - 1) // BLOCK


def gemm_into(y, qx, xs, wq_ptr, ws_t, wref, align, tw, mtk, top_k, N, K, cfg):
    wv, sk, npw = cfg
    si, ei, npad = align
    r4d.r4d_gemm_moe_mxfp4a8_nt_b16(
        qx.data_ptr(), xs.data_ptr(), wq_ptr, ws_t.data_ptr(), wref.data_ptr(),
        y.data_ptr(), si.data_ptr(), ei.data_ptr(), npad.data_ptr(),
        tw.data_ptr() if tw is not None else 0,
        si.numel(), mtk, top_k, K, N, wv, sk, npw,
        torch.cuda.current_stream().cuda_stream)


def run_case(name, N, K, S, nsteps, M, MAXD=None):
    MAXD = E if MAXD is None else MAXD
    print(f"\n[{name}] N={N} K={K} E={E} slots={S} steps={nsteps} M={M} top_k={TOPK}")
    cfg = H.pick_cfg(N, K)
    cse = H.build_case(N, K, M, TOPK, H.SEED)
    mtk = cse["mtk"]

    # the full expert stack lives in pinned host memory, exactly like --cpu-offload-params
    wq_h, wq_dp = H.uva_device_ptr(cse["wq"])
    ws_h, ws_dp = H.uva_device_ptr(cse["ws_t"])
    wr_h, wr_dp = H.uva_device_ptr(cse["wref"])
    ws_uva = torch.empty(0)          # keep refs alive
    del ws_uva

    # per-expert slab sizes
    b_wq = cse["wq"][0].numel()
    b_ws = cse["ws_t"][0].numel()
    b_wr = cse["wref"][0].numel()
    if b_wr % 16:
        print(f"  wref slab {b_wr} B is not a 16 B multiple -> gather would refuse; skipping")
        return

    hot = sorted(np.random.default_rng(0).choice(E, S, replace=False).tolist())
    hot_t = torch.tensor(hot, dtype=torch.int64, device=DEV)
    table = torch.full((E,), -1, dtype=torch.int32, device=DEV)
    table[hot_t] = torch.arange(S, dtype=torch.int32, device=DEV)
    map_cold = torch.arange(E, dtype=torch.int32, device=DEV)
    map_cold[hot_t] = -1
    slot_expert = hot_t.to(torch.int32).clone()
    slot_stamp = torch.zeros(S, dtype=torch.int64, device=DEV)
    routed = torch.zeros(E, dtype=torch.uint8, device=DEV)
    step = torch.zeros(1, dtype=torch.int64, device=DEV)
    cap = 16
    miss = torch.full((cap, 2), -1, dtype=torch.int32, device=DEV)
    n_miss = torch.zeros(1, dtype=torch.int32, device=DEV)
    # slot buffers, warm-started (this is what _build_hot's index_select produces today)
    s_wq = cse["wq"].index_select(0, hot_t).contiguous()
    s_ws = cse["ws_t"].index_select(0, hot_t).contiguous()
    s_wr = cse["wref"].index_select(0, hot_t).contiguous()
    ident = torch.arange(E, dtype=torch.int32, device=DEV)

    g = torch.Generator(device=DEV).manual_seed(7)
    tot_ins = 0
    bad = 0
    for n in range(nsteps):
        topk_ids = torch.randint(0, E, (M, TOPK), generator=g, dtype=torch.int32, device=DEV)
        st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        ids_p = ctypes.c_void_p(topk_ids.reshape(-1).contiguous().data_ptr())
        state = [ctypes.c_void_p(x.data_ptr()) for x in
                 (table, map_cold, slot_expert, slot_stamp, routed, step, miss, n_miss)]
        fused_aligns = None
        if FUSE:
            nsi, nbi = fused_sizes(mtk, E)
            fo = [torch.empty(nsi, dtype=torch.int32, device=DEV),
                  torch.empty(nbi, dtype=torch.int32, device=DEV),
                  torch.empty(1, dtype=torch.int32, device=DEV),
                  torch.empty(nsi, dtype=torch.int32, device=DEV),
                  torch.empty(nbi, dtype=torch.int32, device=DEV),
                  torch.empty(1, dtype=torch.int32, device=DEV)]
            rc = rlu.r4d_lru_fused(ids_p, topk_ids.numel(), E, S, MAXD, cap, *state,
                                   BLOCK, nsi, nbi,
                                   *[ctypes.c_void_p(t.data_ptr()) for t in fo], st)
            assert rc == 0
            fused_aligns = (tuple(fo[0:3]), tuple(fo[3:6]))
        else:
            rc = rlu.r4d_lru_manage(ids_p, topk_ids.numel(), E, S, MAXD, cap, *state, st)
            assert rc == 0
        # only three buffers in this shape; repeat one to fill the six-slot ABI
        args = []
        for d, s, b in ((s_wq, wq_dp, b_wq), (s_ws, ws_dp, b_ws), (s_wr, wr_dp, b_wr),
                        (s_wq, wq_dp, b_wq), (s_ws, ws_dp, b_ws), (s_wr, wr_dp, b_wr)):
            args += [ctypes.c_void_p(d.data_ptr()), ctypes.c_void_p(s), ctypes.c_long(b)]
        rc = rlu.r4d_lru_gather(*args, ctypes.c_void_p(miss.data_ptr()),
                                ctypes.c_void_p(n_miss.data_ptr()), 8, 16, st)
        assert rc == 0
        torch.cuda.synchronize()
        tot_ins += int(n_miss.item())

        if FUSE:
            a_hot, a_cold = fused_aligns
        else:
            a_hot = align_mapped(topk_ids, S, table)
            a_cold = align_mapped(topk_ids, E, map_cold)
        a_all = align_mapped(topk_ids, E, ident)
        tw = cse["tw"]
        y_split = torch.zeros(mtk, N, dtype=torch.bfloat16, device=DEV)
        gemm_into(y_split, cse["qx"], cse["xs"], s_wq.data_ptr(), s_ws, s_wr,
                  a_hot, tw, mtk, TOPK, N, K, cfg)
        gemm_into(y_split, cse["qx"], cse["xs"], wq_dp, ws_h, wr_h,
                  a_cold, tw, mtk, TOPK, N, K, cfg)
        y_ref = torch.zeros(mtk, N, dtype=torch.bfloat16, device=DEV)
        gemm_into(y_ref, cse["qx"], cse["xs"], wq_dp, ws_h, wr_h,
                  a_all, tw, mtk, TOPK, N, K, cfg)
        torch.cuda.synchronize()
        if not torch.equal(y_split, y_ref):
            d = (y_split.float() - y_ref.float()).abs()
            nz = int((d > 0).sum().item())
            print(f"  step {n}: NOT bit-identical, {nz}/{d.numel()} elems differ, "
                  f"max |d| {d.max().item():.3e}, inserts this step "
                  f"{int(n_miss.item())}")
            FAILS.append(f"{name} step {n}")
            bad += 1
            if bad >= 3:
                return
    print(f"  {nsteps} steps, {tot_ins} inserts ({tot_ins/nsteps:.1f}/step): "
          f"split output bit-identical to the all-UVA reference on every step")


def main():
    torch.cuda.init()
    print("device:", torch.cuda.get_device_name(0))
    print("r4d:", os.environ.get("R4D_LIB", "/app/r4dhip/r4d.so"), "| lru:", LIB)
    run_case("gate_up", 640, 2560, 24, 25, 4)
    run_case("down", 2560, 320, 24, 25, 4)
    run_case("gate_up-tiny-cache", 640, 2560, 8, 25, 4)
    run_case("gate_up-readthrough-gate", 640, 2560, 24, 15, 8, 12)
    print("\n" + ("NUMERICS PASSED (bit-identical)" if not FAILS
                  else f"{len(FAILS)} FAILURES: {FAILS[:5]}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
