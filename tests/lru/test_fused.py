"""Is the fused kernel a drop-in for (lru_manage + moe_align(table) + moe_align(map_cold))?

The fused kernel runs the manager and then produces both moe_align_block_size outputs from
the table it just rewrote, so it cannot be compared against a fixed reference: the reference
has to be replayed from the SAME starting state. Every check below therefore snapshots the
cache, runs one path, snapshots the result, restores, runs the other.

Checks:
  (a) MANAGER   table/map_cold/slot_expert/slot_stamp/step/miss/n_miss bit-identical to
                r4d_lru_manage from the same starting state
  (b) ALIGN     npad exactly, expert_ids exactly, sorted_ids as a SET PER BLOCK (vllm's
                general path fills sorted_ids with global atomicAdd cursors and is not
                stable within an expert's block), padding entries exactly `mk`
  (c) COVER     hot and cold sorted_ids together contain every token index in [0, mk)
                exactly once -- the split is a partition, which is what makes two GEMM
                calls add up to one
  (d) EDGES     forced read-through (max_distinct = 0) and forced zero-miss steps, which
                are the two paths where an early `return` would leave the previous
                replay's outputs in place
  (e) GRAPH     the same, inside a captured HIP graph replayed with changing routing:
                a normal step followed by a read-through step must not show stale outputs
  (f) CONTROL   the comparator is shown to fail on a perturbed result

  flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --ipc host --group-add video \
    --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v $REPO_ROOT:/w \
    --entrypoint bash local/q38fn-rocm10:k1build -c 'cd /w/tests/lru && python3 test_fused.py'
"""
import ctypes
import os

import numpy as np
import torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import \
    moe_align_block_size

DEV = "cuda"
LIB = os.environ.get("R4D_LRU_LIB", "/w/build/kernels/librlu.so")
E = int(os.environ.get("E", "512"))
S = int(os.environ.get("SLOTS", "257"))
TOPK = int(os.environ.get("TOPK", "10"))
BLOCK = 16
MAXI = int(os.environ.get("MAXI", "64"))
THRESH = float(os.environ.get("THRESH", "0.5"))
TRACE = os.environ.get("TRACE", "/w/artifacts/routes_rank0.npz")
NT = 256

lib = ctypes.CDLL(LIB)
lib.r4d_lru_manage.restype = ctypes.c_int
lib.r4d_lru_manage.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 9
lib.r4d_lru_fused.restype = ctypes.c_int
lib.r4d_lru_fused.argtypes = ([ctypes.c_void_p] + [ctypes.c_int] * 5 +
                              [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3 +
                              [ctypes.c_void_p] * 7)
FAILS = []


def fail(msg):
    FAILS.append(msg)
    print("  FAIL: " + msg, flush=True)


def sizes(mk):
    """exactly what vllm's python wrapper allocates"""
    L = mk + E * (BLOCK - 1)
    if mk < E:
        L = min(mk * BLOCK, L)
    return L, (L + BLOCK - 1) // BLOCK


class Cache:
    """the eight tensors the manager owns, with snapshot/restore"""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)          # slot_stamp below must be reproducible too
        hot = sorted(rng.choice(E, S, replace=False).tolist())
        h = torch.tensor(hot, dtype=torch.int64, device=DEV)
        self.table = torch.full((E,), -1, dtype=torch.int32, device=DEV)
        self.table[h] = torch.arange(S, dtype=torch.int32, device=DEV)
        self.map_cold = torch.arange(E, dtype=torch.int32, device=DEV)
        self.map_cold[h] = -1
        self.slot_expert = h.to(torch.int32).clone()
        self.slot_stamp = (1 + torch.randperm(S, device=DEV)).to(torch.int64)
        self.routed = torch.zeros(E, dtype=torch.uint8, device=DEV)
        self.step = torch.full((1,), S, dtype=torch.int64, device=DEV)
        self.miss = torch.full((MAXI, 2), -1, dtype=torch.int32, device=DEV)
        self.n_miss = torch.zeros(1, dtype=torch.int32, device=DEV)

    NAMES = ("table", "map_cold", "slot_expert", "slot_stamp", "step", "miss", "n_miss")

    def snap(self):
        return {n: getattr(self, n).clone() for n in self.NAMES}

    def restore(self, s):
        for n in self.NAMES:
            getattr(self, n).copy_(s[n])

    def args(self):
        return [ctypes.c_void_p(x.data_ptr()) for x in
                (self.table, self.map_cold, self.slot_expert, self.slot_stamp,
                 self.routed, self.step, self.miss, self.n_miss)]


def call_manage(c, ids, max_distinct, stream=None):
    st = ctypes.c_void_p(stream if stream is not None
                         else torch.cuda.current_stream().cuda_stream)
    rc = lib.r4d_lru_manage(ctypes.c_void_p(ids.data_ptr()), ids.numel(), E, S,
                            max_distinct, MAXI, *c.args(), st)
    assert rc == 0, rc


def call_fused(c, ids, max_distinct, out):
    mk = ids.numel()
    L, NB = sizes(mk)
    st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    rc = lib.r4d_lru_fused(
        ctypes.c_void_p(ids.data_ptr()), mk, E, S, max_distinct, MAXI, *c.args(),
        BLOCK, L, NB,
        *[ctypes.c_void_p(t.data_ptr()) for t in out], st)
    assert rc == 0, rc


def new_out(mk):
    L, NB = sizes(mk)
    return [torch.empty(L, dtype=torch.int32, device=DEV),
            torch.empty(NB, dtype=torch.int32, device=DEV),
            torch.empty(1, dtype=torch.int32, device=DEV),
            torch.empty(L, dtype=torch.int32, device=DEV),
            torch.empty(NB, dtype=torch.int32, device=DEV),
            torch.empty(1, dtype=torch.int32, device=DEV)]


def cmp_align(tag, mk, got, ref):
    """got/ref = (sorted_ids, expert_ids, npad).

    sorted_ids is compared as a multiset PER EXPERT, not per block: vllm's general path
    scatters tokens with a global atomicAdd cursor, so neither the order within an expert
    nor the split of that expert's tokens across its blocks is stable. What must match is
    npad, expert_ids, and which tokens belong to which expert."""
    g_s, g_e, g_n = [x.cpu().numpy() for x in got]
    r_s, r_e, r_n = [x.cpu().numpy() for x in ref]
    bad = 0
    if g_n[0] != r_n[0]:
        fail(f"{tag}: npad {g_n[0]} != {r_n[0]}")
        bad += 1
    if not np.array_equal(g_e, r_e):
        d = int((g_e != r_e).sum())
        fail(f"{tag}: expert_ids differ in {d}/{len(g_e)} blocks")
        bad += 1
    n = int(min(g_n[0], r_n[0]))

    def by_expert(sa, ea):
        d = {}
        for b in range(n // BLOCK):
            d.setdefault(int(ea[b]), []).extend(sa[b * BLOCK:(b + 1) * BLOCK].tolist())
        return {k: sorted(v) for k, v in d.items()}

    ga, ra = by_expert(g_s, g_e), by_expert(r_s, r_e)
    if set(ga) != set(ra):
        fail(f"{tag}: different expert sets ({len(ga)} vs {len(ra)} experts)")
        bad += 1
    else:
        for e in sorted(ga):
            if ga[e] != ra[e]:
                fail(f"{tag}: expert {e} tokens differ "
                     f"({len(ga[e])} vs {len(ra[e])} entries, first diff "
                     f"{next((x for x, y in zip(ga[e], ra[e]) if x != y), None)})")
                bad += 1
                break
    if not (g_s[n:] == mk).all():
        fail(f"{tag}: {int((g_s[n:] != mk).sum())} non-sentinel entries past npad")
        bad += 1
    return bad == 0


def cmp_state(tag, a, b):
    ok = True
    for n in Cache.NAMES:
        if not torch.equal(a[n], b[n]):
            d = int((a[n] != b[n]).sum())
            fail(f"{tag}: manager state '{n}' differs in {d} entries")
            ok = False
    return ok


def cover(tag, mk, out):
    hot = out[0].cpu().numpy()[:int(out[2].item())]
    cold = out[3].cpu().numpy()[:int(out[5].item())]
    tok = np.concatenate([hot[hot != mk], cold[cold != mk]])
    tok.sort()
    if not np.array_equal(tok, np.arange(mk)):
        fail(f"{tag}: hot+cold do not partition [0,{mk}) "
             f"({len(tok)} entries, {len(np.unique(tok))} distinct)")
        return False
    return True


def one_step(c, ids, max_distinct, tag):
    """run both paths from the same state and compare everything"""
    mk = ids.numel()
    start = c.snap()
    out = new_out(mk)
    call_fused(c, ids, max_distinct, out)
    torch.cuda.synchronize()
    fused_state = c.snap()

    c.restore(start)
    call_manage(c, ids, max_distinct)
    torch.cuda.synchronize()
    ref_state = c.snap()
    ids2d = ids.view(-1, TOPK)
    ref_hot = moe_align_block_size(ids2d, BLOCK, E, c.table, ignore_invalid_experts=True)
    ref_cold = moe_align_block_size(ids2d, BLOCK, E, c.map_cold, ignore_invalid_experts=True)
    torch.cuda.synchronize()

    ok = cmp_state(tag, fused_state, ref_state)
    ok &= cmp_align(tag + "/hot", mk, out[0:3], ref_hot)
    ok &= cmp_align(tag + "/cold", mk, out[3:6], ref_cold)
    ok &= cover(tag, mk, out)
    # leave the cache in the fused path's state so the sequence keeps evolving
    c.restore(fused_state)
    return ok, out, (ref_hot, ref_cold)


def real_routing(path, layer, nsteps, want_rows=None):
    d = np.load(path)
    ids, steps, K = d["ids"], d["steps"], int(d["topk"])
    ids = ids[np.argsort(steps)]
    rows = (ids[:, 0] >= 0).sum(axis=1) // K
    modal = np.bincount(rows).argmax() if want_rows is None else want_rows
    out = []
    for t in range(len(ids)):
        if rows[t] != modal:
            continue
        row = ids[t, layer]
        row = row[row >= 0]
        if row.size != modal * K:
            continue
        out.append(torch.tensor(row.astype(np.int32), device=DEV))
        if len(out) >= nsteps:
            break
    return out, modal


def main():
    torch.cuda.init()
    print(f"E={E} S={S} top_k={TOPK} BLOCK={BLOCK} MAXI={MAXI} lib={LIB}")
    md = max(1, int(S * THRESH))

    # ---- (a,b,c) real routing, decode geometry (mk = 50, the deterministic path) --------
    routes, rows = real_routing(TRACE, 7, 60)
    mk = rows * TOPK
    print(f"\n[1] real routing, layer 7, {len(routes)} steps of mk={mk} "
          f"({'deterministic' if mk <= NT else 'cursor'} placement)")
    c = Cache(0)
    ins = 0
    for n, ids in enumerate(routes):
        ok, out, _ = one_step(c, ids, md, f"step {n}")
        ins += int(c.n_miss.item())
        if not ok:
            break
    print(f"  {len(routes)} steps, {ins} inserts total: "
          f"{'OK' if not FAILS else 'FAILED'}")

    # ---- synthetic geometries, including the cursor path and a prefill-sized step -------
    rng = np.random.default_rng(7)
    p = np.arange(1, E + 1, dtype=float) ** -1.1
    p /= p.sum()
    for rowsx in (1, 4, 20, 40, 205):
        mkx = rowsx * TOPK
        c2 = Cache(3)
        okall = True
        for n in range(6):
            v = np.concatenate([rng.choice(E, TOPK, replace=False, p=p)
                                for _ in range(rowsx)])
            ids = torch.tensor(v.astype(np.int32), device=DEV)
            ok, _, _ = one_step(c2, ids, md, f"mk={mkx} step {n}")
            okall &= ok
        print(f"[2] mk={mkx:>5} ({'det' if mkx <= NT else 'cursor'}): "
              f"{'OK' if okall else 'FAILED'}")

    # ---- (d) the two early-exit paths --------------------------------------------------
    print("\n[3] forced edge cases (an early return here = stale outputs)")
    c3 = Cache(11)
    ids = routes[0]
    ok, _, _ = one_step(c3, ids, 0, "read-through")          # max_distinct = 0
    print(f"  read-through (max_distinct=0, no inserts): {'OK' if ok else 'FAILED'}")
    # zero-miss: replay the same routing twice; the second has everything resident
    one_step(c3, ids, md, "warm")
    n_before = int(c3.n_miss.item())
    ok, _, _ = one_step(c3, ids, md, "zero-miss")
    print(f"  zero-miss (same routing replayed, {n_before} -> "
          f"{int(c3.n_miss.item())} inserts): {'OK' if ok else 'FAILED'}")

    # ---- (e) inside a HIP graph --------------------------------------------------------
    print("\n[4] inside a captured HIP graph")
    SEED = 5
    mk = routes[0].numel()
    ids_static = torch.zeros(mk, dtype=torch.int32, device=DEV)
    out = new_out(mk)
    c4 = Cache(SEED)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        ids_static.copy_(routes[0])
        call_fused(c4, ids_static, md, out)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call_fused(c4, ids_static, md, out)
    torch.cuda.synchronize()

    c4.restore(Cache(SEED).snap())          # undo the warm-up call
    torch.cuda.synchronize()
    replayed = []
    for ids in routes[:12]:
        ids_static.copy_(ids)
        g.replay()
        torch.cuda.synchronize()
        replayed.append(([t.clone() for t in out], c4.snap()))

    c6 = Cache(SEED)
    gok = True
    for n, ids in enumerate(routes[:12]):
        o2 = new_out(mk)
        call_fused(c6, ids, md, o2)
        torch.cuda.synchronize()
        got, state = replayed[n]
        for i in range(6):
            if not torch.equal(got[i], o2[i]):
                fail(f"graph replay {n}: output {i} differs from the eager call")
                gok = False
        if not cmp_state(f"graph replay {n}", state, c6.snap()):
            gok = False
        if not gok:
            break
    print(f"  12 replays with changing routing == 12 eager calls, state and outputs: "
          f"{'OK' if gok else 'FAILED'}")

    # the stale-output hazard: a step with no inserts, replayed right after one with
    # inserts, must still rewrite sorted_ids/expert_ids/npad rather than leave the
    # previous replay's values behind.
    ids_static.copy_(routes[0])
    g.replay()
    torch.cuda.synchronize()
    ids_static.copy_(routes[1])
    g.replay()
    torch.cuda.synchronize()
    a1 = [t.clone() for t in out]
    ids_static.copy_(routes[1])             # identical routing -> zero inserts
    g.replay()
    torch.cuda.synchronize()
    n0 = int(c4.n_miss.item())
    c7 = Cache(SEED)
    o3 = new_out(mk)
    for ids in (routes[0], routes[1], routes[1]):
        call_fused(c7, ids, md, o3)
    torch.cuda.synchronize()
    same = all(torch.equal(out[i], o3[i]) for i in range(6))
    if not same:
        fail("zero-insert replay: graph outputs differ from eager (stale?)")
    print(f"  zero-insert step inside the graph ({n0} inserts) matches eager: "
          f"{'OK' if same else 'FAILED'}")

    # ---- (f) negative control ----------------------------------------------------------
    print("\n[5] negative control: the comparator must be able to fail")
    base = len(FAILS)
    mkq = routes[0].numel()
    c7 = Cache(0)
    o = new_out(mkq)
    call_fused(c7, routes[0], md, o)
    torch.cuda.synchronize()
    ref = (o[0].clone(), o[1].clone(), o[2].clone())
    perturbed = [o[0].clone(), o[1].clone(), o[2].clone()]
    perturbed[2] += BLOCK
    cmp_align("control/npad", mkq, perturbed, ref)
    perturbed = [o[0].clone(), o[1].clone(), o[2].clone()]
    perturbed[1][0] = -7
    cmp_align("control/expert_ids", mkq, perturbed, ref)
    perturbed = [o[0].clone(), o[1].clone(), o[2].clone()]
    perturbed[0][0] = perturbed[0][1]
    cmp_align("control/sorted", mkq, perturbed, ref)
    got = len(FAILS) - base
    print(f"  3 perturbations -> {got} reported failures "
          f"({'OK' if got >= 3 else 'CONTROL BROKEN'})")
    ctrl_ok = got >= 3
    del FAILS[base:]

    print("\n" + ("FUSED TEST PASSED" if not FAILS and ctrl_ok
                  else f"FUSED TEST FAILED ({len(FAILS)} problems)"))
    return 0 if not FAILS and ctrl_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
