"""Does the LRU cache ever hold a slot whose bytes are NOT the expert it claims?

test_numerics_lru only asked "is the two-call split bit-identical to one all-UVA call",
at E=64 with random routing. That can pass while a slot holds subtly wrong bytes, because
both sides of that comparison read the SAME slot. This test compares the slot against the
SOURCE instead, at production geometry, with REAL routing from the captured trace, and
audits the whole residency -- not just the experts inserted on the current step, so a slot
that went stale ten steps ago is still caught.

Checks, every step:
  (a) INSERT   every (expert, slot) the manager emitted: all six buffers byte-equal to source
  (c) MAPS     table/map_cold complementarity over all E experts, table<->slot_expert
               round trip, slot_expert injective, no resident expert marked cold
  (d) NUMERICS split (resident + fallback) vs one all-UVA call, bit-for-bit, with tw
and every AUDIT_EVERY steps:
  (b) RESIDENCY every slot, not just this step's inserts, byte-equal to its claimed expert

All six sources live in pinned host memory reached through hipHostGetDevicePointer, which is
how --cpu-offload-params presents them, so the gather really is reading over PCIe here.

  flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --ipc host --group-add video \
    --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v $REPO_ROOT:/w \
    --entrypoint bash local/q38fn-rocm10:k1build -c 'cd /w/tests/lru && python3 test_slots_prod.py'
Env: E=512 TOPK=10 SLOTS=257 LAYER=7 STEPS=120 AUDIT_EVERY=10 MAXI=64 TRACE=../routes_rank0.npz
"""
import ctypes
import os
import sys

os.environ["E"] = os.environ.get("E", "512")
os.environ["TOPK"] = os.environ.get("TOPK", "10")
sys.path.insert(0, "/w/tools/routecap")
sys.path.insert(0, "/w/tests/lru")

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

import moe_ref_harness as H                                          # noqa: E402
import test_numerics_lru as T                                        # noqa: E402

DEV = "cuda"
E, TOPK, BLOCK = H.E, H.TOPK, H.BLOCK
S = int(os.environ.get("SLOTS", "257"))
LAYER = int(os.environ.get("LAYER", "7"))
STEPS = int(os.environ.get("STEPS", "120"))
AUDIT_EVERY = int(os.environ.get("AUDIT_EVERY", "10"))
MAXI = int(os.environ.get("MAXI", "64"))
TRACE = os.environ.get("TRACE", "/w/artifacts/routes_rank0.npz")
FAULT = int(os.environ.get("FAULT", "-1"))
NAMES = ["w1", "w2", "w1_ws_t", "w1_wref", "w2_ws_t", "w2_wref"]
FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("  FAIL: " + msg, flush=True)


def real_routing(path, layer, nsteps):
    """-> list of [rows, TOPK] int32 topk_ids, real decode steps for one layer."""
    d = np.load(path)
    ids, steps, K = d["ids"], d["steps"], int(d["topk"])
    order = np.argsort(steps)
    ids = ids[order]
    rows = (ids[:, 0] >= 0).sum(axis=1) // K
    modal = np.bincount(rows).argmax()
    out = []
    for t in range(len(ids)):
        if rows[t] != modal:
            continue
        row = ids[t, layer]
        row = row[row >= 0]
        if row.size != modal * K:
            continue
        out.append(torch.tensor(row.reshape(modal, K).astype(np.int32), device=DEV))
        if len(out) >= nsteps:
            break
    return out, modal


def main():
    torch.cuda.init()
    print("device:", torch.cuda.get_device_name(0))
    routes, modal = real_routing(TRACE, LAYER, STEPS)
    print(f"E={E} top_k={TOPK} slots={S} layer={LAYER} max_inserts={MAXI} "
          f"| {len(routes)} real decode steps, {modal} rows/step -> mtk={modal*TOPK}")

    # production geometry: gate_up (N=640,K=2560) and down (N=2560,K=320) in ONE six-buffer
    # gather, exactly as _lru_step issues it
    # gate_up consumes `modal` activation rows with top_k=10; down consumes mtk rows with
    # top_k=1 (it runs on the post-activation buffer), exactly as _apply_split calls them.
    c1 = H.build_case(640, 2560, modal, TOPK, H.SEED)
    c2 = H.build_case(2560, 320, modal * TOPK, 1, H.SEED + 1)
    cfg1, cfg2 = H.pick_cfg(640, 2560), H.pick_cfg(2560, 320)
    src_cuda = [c1["wq"], c2["wq"], c1["ws_t"], c1["wref"], c2["ws_t"], c2["wref"]]
    for n, t in zip(NAMES, src_cuda):
        pe = t[0].numel() * t.element_size()
        print(f"    {n:9s} {tuple(t.shape)} {t.dtype} per-expert {pe} B"
              + ("" if pe % 16 == 0 else "   <-- NOT a 16 B multiple"))

    # sources into pinned host memory, reached by device pointer (== --cpu-offload-params)
    uva = [H.uva_device_ptr(t) for t in src_cuda]
    keep = [h for h, _ in uva]                                       # keep refs alive
    src_dp = [p for _, p in uva]

    hot = sorted(np.random.default_rng(0).choice(E, S, replace=False).tolist())
    hot_t = torch.tensor(hot, dtype=torch.int64, device=DEV)
    dst = [t.index_select(0, hot_t).contiguous() for t in src_cuda]
    table = torch.full((E,), -1, dtype=torch.int32, device=DEV)
    table[hot_t] = torch.arange(S, dtype=torch.int32, device=DEV)
    map_cold = torch.arange(E, dtype=torch.int32, device=DEV)
    map_cold[hot_t] = -1
    slot_expert = hot_t.to(torch.int32).clone()
    # STAMP=rank exercises the production warm start: distinct stamps in [1, S] and step
    # starting at S, so the kernel is checked with non-zero initial stamps, not just zeros.
    if os.environ.get("STAMP") == "rank":
        slot_stamp = 1 + torch.randperm(S, device=DEV).to(torch.int64)
    else:
        slot_stamp = torch.zeros(S, dtype=torch.int64, device=DEV)
    routed = torch.zeros(E, dtype=torch.uint8, device=DEV)
    step = torch.full((1,), S if os.environ.get("STAMP") == "rank" else 0,
                      dtype=torch.int64, device=DEV)
    miss = torch.full((MAXI, 2), -1, dtype=torch.int32, device=DEV)
    n_miss = torch.zeros(1, dtype=torch.int32, device=DEV)
    ident = torch.arange(E, dtype=torch.int32, device=DEV)
    max_distinct = max(1, int(S * 0.5))

    def gather_args():
        a = []
        for d, sp in zip(dst, src_dp):
            a += [ctypes.c_void_p(d.data_ptr()), ctypes.c_void_p(sp),
                  ctypes.c_long(d[0].numel() * d.element_size())]
        return a

    SPECS = [
        dict(name="gate_up N=640 K=2560", N=640, K=2560, cfg=cfg1, case=c1, top_k=TOPK,
             tw=None, hot_wq=dst[0], hot_ws=dst[2], hot_wr=dst[3],
             uva_wq=src_dp[0], uva_ws=keep[2], uva_wr=keep[3]),
        dict(name="down N=2560 K=320", N=2560, K=320, cfg=cfg2, case=c2, top_k=1,
             tw=c2["tw"], hot_wq=dst[1], hot_ws=dst[4], hot_wr=dst[5],
             uva_wq=src_dp[1], uva_ws=keep[4], uva_wr=keep[5]),
    ]

    tot_ins = 0
    for n, topk_ids in enumerate(routes):
        st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = T.rlu.r4d_lru_manage(
            ctypes.c_void_p(topk_ids.reshape(-1).contiguous().data_ptr()),
            topk_ids.numel(), E, S, max_distinct, MAXI,
            ctypes.c_void_p(table.data_ptr()), ctypes.c_void_p(map_cold.data_ptr()),
            ctypes.c_void_p(slot_expert.data_ptr()), ctypes.c_void_p(slot_stamp.data_ptr()),
            ctypes.c_void_p(routed.data_ptr()), ctypes.c_void_p(step.data_ptr()),
            ctypes.c_void_p(miss.data_ptr()), ctypes.c_void_p(n_miss.data_ptr()), st)
        assert rc == 0, f"manage rc={rc}"
        rc = T.rlu.r4d_lru_gather(*gather_args(), ctypes.c_void_p(miss.data_ptr()),
                                  ctypes.c_void_p(n_miss.data_ptr()), 8, 16, st)
        assert rc == 0, f"gather rc={rc}"
        torch.cuda.synchronize()
        nm = int(n_miss.item())
        tot_ins += nm

        # negative control: FAULT=<step> scribbles one byte of one just-inserted slot, so a
        # run that prints PASSED can be shown to be capable of printing FAIL.
        if FAULT >= 0 and n == FAULT and nm > 0:
            sl = int(miss[0, 1].item())
            nb = int(os.environ.get("FAULT_BYTES", "1"))
            dst[0][sl].view(-1)[:nb] ^= 0xFF
            print("  [FAULT] flipped %d byte(s) of w1 slot %d at step %d" % (nb, sl, n),
                  flush=True)

        # (a) every insert this step: all six buffers byte-equal to source
        for j in range(nm):
            e, sl = (int(miss[j, 0].item()), int(miss[j, 1].item()))
            if not (0 <= e < E and 0 <= sl < S):
                fail(f"step {n}: miss[{j}] = ({e},{sl}) out of range")
                continue
            for nm_, d, s in zip(NAMES, dst, src_cuda):
                if not torch.equal(d[sl], s[e]):
                    bad = (d[sl] != s[e]).sum().item()
                    fail(f"step {n}: insert e={e} slot={sl} buffer {nm_}: "
                         f"{bad}/{s[e].numel()} bytes differ from source")

        # (c) maps
        tb, mc, se = table.cpu(), map_cold.cpu(), slot_expert.cpu()
        res = tb >= 0
        if not torch.equal(mc[res], torch.full((int(res.sum()),), -1, dtype=torch.int32)):
            fail(f"step {n}: resident experts not marked -1 in map_cold")
        cold = ~res
        if not torch.equal(mc[cold], ident.cpu()[cold]):
            fail(f"step {n}: cold experts do not map to themselves in map_cold")
        if int(res.sum()) != S:
            fail(f"step {n}: {int(res.sum())} experts resident, expected exactly {S}")
        if se.unique().numel() != S:
            fail(f"step {n}: slot_expert not injective ({se.unique().numel()}/{S})")
        rt = tb[se.long()]
        if not torch.equal(rt, torch.arange(S, dtype=torch.int32)):
            fail(f"step {n}: table[slot_expert[s]] != s for {int((rt != torch.arange(S, dtype=torch.int32)).sum())} slots")

        # (b) full residency audit: every slot, not just this step's
        if n % AUDIT_EVERY == 0 or n == len(routes) - 1:
            se_t = slot_expert.to(torch.int64)
            for nm_, d, s in zip(NAMES, dst, src_cuda):
                if not torch.equal(d, s.index_select(0, se_t)):
                    nbad = int((d != s.index_select(0, se_t)).any(dim=tuple(
                        range(1, d.dim()))).sum().item())
                    fail(f"step {n}: RESIDENCY audit {nm_}: {nbad}/{S} slots hold bytes "
                         f"that are not their claimed expert")

        # (d) split (resident + fallback) vs one all-UVA call, bit-for-bit
        a_hot = T.align_mapped(topk_ids, S, table)
        a_cold = T.align_mapped(topk_ids, E, map_cold)
        a_all = T.align_mapped(topk_ids, E, ident)
        for spec in SPECS:
            mtk = spec["case"]["mtk"]
            ys = torch.zeros(mtk, spec["N"], dtype=torch.bfloat16, device=DEV)
            T.gemm_into(ys, spec["case"]["qx"], spec["case"]["xs"], spec["hot_wq"].data_ptr(),
                        spec["hot_ws"], spec["hot_wr"], a_hot, spec["tw"], mtk,
                        spec["top_k"], spec["N"], spec["K"], spec["cfg"])
            T.gemm_into(ys, spec["case"]["qx"], spec["case"]["xs"], spec["uva_wq"],
                        spec["uva_ws"], spec["uva_wr"], a_cold, spec["tw"], mtk,
                        spec["top_k"], spec["N"], spec["K"], spec["cfg"])
            yr = torch.zeros(mtk, spec["N"], dtype=torch.bfloat16, device=DEV)
            T.gemm_into(yr, spec["case"]["qx"], spec["case"]["xs"], spec["uva_wq"],
                        spec["uva_ws"], spec["uva_wr"], a_all, spec["tw"], mtk,
                        spec["top_k"], spec["N"], spec["K"], spec["cfg"])
            torch.cuda.synchronize()
            if not torch.equal(ys, yr):
                d = (ys.float() - yr.float()).abs()
                fail(f"step {n}: {spec['name']} split != all-UVA, "
                     f"{int((d > 0).sum())}/{d.numel()} elems, max |d| {d.max().item():.3e}, "
                     f"inserts this step {nm}")
        if FAILS and len(FAILS) > 8:
            print("  (stopping after 8 failures)")
            break

    print(f"\n{len(routes)} steps, {tot_ins} inserts ({tot_ins/max(len(routes),1):.1f}/step)")
    print("SLOT/MAP/NUMERICS PASSED" if not FAILS else f"{len(FAILS)} FAILURES")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
