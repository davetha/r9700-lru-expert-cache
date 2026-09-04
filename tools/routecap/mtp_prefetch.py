"""(b) MTP depth vs cold traffic, and (c) is a next-layer miss prefetch predictable?

Offline replay of the captured routing traces (routecap.py -> routes_rank0.npz).
CPU/numpy only -- touches no GPU.

(b) DEPTH.  A decode step of this server is `modal` rows wide: row 0 is the token the
    target model is actually advancing, rows 1..4 are the MTP drafts.  For r = 1..modal
    we keep only the first r rows of every step and re-run the SAME LRU the production
    kernel implements (warm start from the profile hot set, per-layer caps from the 15 GB
    water-fill).  That gives distinct experts/layer/step and, the number that matters,
    miss bytes/step at each depth -- so the marginal PCIe cost of one extra draft row is
    measured, not guessed.  Caveat stated in the output: the token SEQUENCE is the one
    MTP-4 produced; a real MTP-2 run would generate different text.

(c) PREFETCH.  At step t, before layer L's router runs, what could name layer L's misses?
    Scored against the LRU's own miss stream at 15 GB, second half of each generation:
      last    the experts layer L routed at step t-1
      freq    layer L's globally most frequent experts (profile rank)
      cooc    a learned same-step cross-layer model: score(f) = sum_e C[e,f] over the
              experts layer L-1 routed at THIS step, C trained on the first half
      oracle  the true miss set (upper bound)
    Every predictor is filtered to experts that are NOT resident at prediction time, so
    "predict what the LRU already holds" scores zero by construction.

    python3 mtp_prefetch.py /w/artifacts/routes_rank0.npz
"""
import json
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, "/w/tools/routecap")
os.environ.setdefault("PROFILE", "/w/profiles/hot_profile.json")
import locality_sim as LS                                             # noqa: E402

PER_EXPERT_BYTES = LS.PER_EXPERT_BYTES
GB = float(os.environ.get("GB", "15"))
NLAYER = int(os.environ.get("LAYERS", "48"))
BUDGETS = [int(v) for v in os.environ.get("BUDGETS", "1,2,4,8,16").split(",")]
MAXGEN = int(os.environ.get("MAXGEN", "0"))       # 0 = all


# ------------------------------------------------------------------ loader (per ROW)
def load_rows(path):
    """-> list of (label, ids[nsteps, nlayer, rows, K]) with -1 padding removed by shape."""
    d = np.load(path)
    ids, steps, K = d["ids"], d["steps"], int(d["topk"])
    if int(d["lost_to_wrap"]) != 0:
        print(f"  !! {path}: {int(d['lost_to_wrap'])} step(s) lost to ring wrap")
    order = np.argsort(steps)
    ids, steps = ids[order], steps[order]
    assert (np.diff(steps) == 1).all(), "step counter has gaps"
    rows = (ids[:, 0] >= 0).sum(axis=1) // K
    modal = int(np.bincount(rows).argmax())
    runs, s = [], None
    for t in range(len(rows) + 1):
        ok = t < len(rows) and rows[t] == modal
        if ok and s is None:
            s = t
        elif not ok and s is not None:
            if t - s >= 20:
                runs.append((s, t))
            s = None
    L = min(ids.shape[1], NLAYER)
    out = []
    for i, (a, b) in enumerate(runs):
        lbl = (LS.TRAFFIC_LABELS[i - len(runs) + len(LS.TRAFFIC_LABELS)]
               if i >= len(runs) - len(LS.TRAFFIC_LABELS) else f"gen{i+1}")
        out.append((lbl, ids[a:b, :L, :modal * K].reshape(b - a, L, modal, K)))
    print(f"  {path}: {len(steps)} steps, {ids.shape[1]} routers, modal {modal} rows, K={K}"
          f" -> {len(out)} generation(s): " + ", ".join(f"{l}[{len(t)}]" for l, t in out))
    return modal, K, out


# ------------------------------------------------------------------ the LRU we ship
class LRU:
    """Same policy as lru_manage_k: warm start, evict least-recently-routed, never evict
    something routed this step.  Returns the miss list for the step."""

    def __init__(self, cap, init):
        self.cap = cap
        self.od = OrderedDict()
        for e in init[:cap]:
            self.od[int(e)] = None

    def step(self, ids):
        miss = []
        for e in ids:
            e = int(e)
            if e in self.od:
                self.od.move_to_end(e)
            else:
                miss.append(e)
                self.od[e] = None
        while len(self.od) > self.cap:
            self.od.popitem(last=False)
        return miss


def caps_and_hot():
    prof = LS.load_profile()
    caps, hot, cov = LS.capacities(prof, GB, "global", NLAYER)
    return prof, caps, hot, cov


# ------------------------------------------------------------------ (b) depth sweep
def depth_sweep(gens, modal, caps, hot):
    print("\n" + "=" * 78)
    print(f"(b) MTP depth: keep the first r of {modal} rows, replay the 15 GB LRU")
    print("=" * 78)
    print(f"{'r rows':>7} {'distinct/lyr':>13} {'d(r)-d(r-1)':>12} {'miss/lyr/step':>14}"
          f" {'MB/step':>9} {'dMB/row':>9} {'ms@28GB/s':>10}")
    prev_d = prev_mb = None
    rowsdata = []
    for r in range(1, modal + 1):
        dtot = dn = mtot = 0
        for lbl, arr in gens:
            T, L = arr.shape[0], arr.shape[1]
            for li in range(L):
                c = LRU(caps[li], hot[li])
                sub = arr[:, li, :r, :].reshape(T, -1)
                for t in range(T):
                    u = np.unique(sub[t])
                    u = u[u >= 0]
                    dtot += len(u)
                    dn += 1
                    mtot += len(c.step(u))
        d = dtot / dn
        miss = mtot / dn
        mb = miss * L * PER_EXPERT_BYTES / 1e6      # per step, all layers, one rank
        line = (f"{r:>7} {d:>13.1f} {'' if prev_d is None else f'{d-prev_d:>12.1f}'}"
                f" {miss:>14.3f} {mb:>9.1f}"
                f" {'' if prev_mb is None else f'{mb-prev_mb:>9.1f}'} {mb/28e3*1e3:>10.2f}")
        print(line if prev_d is not None else
              f"{r:>7} {d:>13.1f} {'-':>12} {miss:>14.3f} {mb:>9.1f} {'-':>9}"
              f" {mb/28e3*1e3:>10.2f}")
        rowsdata.append((r, d, miss, mb))
        prev_d, prev_mb = d, mb
    return rowsdata


# ------------------------------------------------------------------ (c) predictability
def predictability(gens, caps, hot, prof, E=512):
    print("\n" + "=" * 78)
    print("(c) prefetch predictability, 15 GB LRU miss stream")
    print("=" * 78)

    # --- P(routed at L,t | routed at L,t-1)
    inter = union = prevtot = 0
    for lbl, arr in gens:
        T, L = arr.shape[0], arr.shape[1]
        for li in range(L):
            flat = arr[:, li].reshape(T, -1)
            prev = None
            for t in range(T):
                u = np.unique(flat[t])
                u = set(u[u >= 0].tolist())
                if prev is not None:
                    inter += len(u & prev)
                    union += len(u)
                    prevtot += len(prev)
                prev = u
    print(f"  P(e routed at L,t | routed at L,t-1) = {inter/prevtot:.3f}"
          f"   (recall of last step's set over this step's = {inter/union:.3f})")

    freq_rank = {li: np.asarray(prof[li][0]) for li in range(NLAYER)}

    tot_miss = 0
    hits = {k: {b: 0 for b in BUDGETS} for k in ("last", "freq", "cooc", "oracle")}
    issued = {k: {b: 0 for b in BUDGETS} for k in ("last", "freq", "cooc", "oracle")}
    # recency of the missed expert's previous use in this generation, same layer
    rec_hist = {"never": 0}
    for lbl, arr in gens:
        T, L = arr.shape[0], arr.shape[1]
        half = T // 2
        # per-step distinct sets, per layer
        sets = [[None] * L for _ in range(T)]
        for li in range(L):
            flat = arr[:, li].reshape(T, -1)
            for t in range(T):
                u = np.unique(flat[t])
                sets[t][li] = u[u >= 0]
        # cross-layer co-occurrence C[L][e_prevlayer, f], trained on the first half
        C = np.zeros((L, E, E), dtype=np.float32)
        for t in range(half):
            for li in range(1, L):
                C[li][np.ix_(sets[t][li - 1], sets[t][li])] += 1.0
        caches = [LRU(caps[li], hot[li]) for li in range(L)]
        lastuse = [dict() for _ in range(L)]
        for t in range(T):
            for li in range(L):
                cur = sets[t][li]
                res_before = set(caches[li].od)
                miss = caches[li].step(cur)
                if t >= half and li >= 1 and miss:
                    ms = set(miss)
                    tot_miss += len(ms)
                    for e in ms:
                        lu = lastuse[li].get(int(e))
                        if lu is None:
                            rec_hist["never"] = rec_hist.get("never", 0) + 1
                        else:
                            k = t - lu
                            key = k if k <= 8 else (16 if k <= 16 else (64 if k <= 64 else 999))
                            rec_hist[key] = rec_hist.get(key, 0) + 1
                    cand = {}
                    cand["last"] = [int(x) for x in sets[t - 1][li]] if t else []
                    cand["freq"] = freq_rank[li].tolist()
                    sc = C[li][sets[t][li - 1]].sum(axis=0)
                    cand["cooc"] = np.argsort(-sc, kind="stable").tolist()
                    cand["oracle"] = list(ms)
                    maxb = max(BUDGETS)
                    for name, lst in cand.items():
                        pick = []
                        for e in lst:
                            if e not in res_before:
                                pick.append(e)
                                if len(pick) >= maxb:
                                    break
                        for b in BUDGETS:
                            p = pick[:b]
                            issued[name][b] += len(p)
                            hits[name][b] += len(ms.intersection(p))
                for e in cur:
                    lastuse[li][int(e)] = t
    print(f"\n  scored on {tot_miss} misses (2nd half of each generation, layers 1..{NLAYER-1})")
    print(f"\n  {'predictor':>9} " + " ".join(f"{'P='+str(b):>16}" for b in BUDGETS))
    print(f"  {'':>9} " + " ".join(f"{'cover / waste':>16}" for _ in BUDGETS))
    for name in ("oracle", "last", "freq", "cooc"):
        cells = []
        for b in BUDGETS:
            cov = hits[name][b] / max(1, tot_miss)
            waste = (issued[name][b] - hits[name][b]) / max(1, hits[name][b]) if hits[name][b] else float("inf")
            cells.append(f"{cov*100:6.1f}% /{waste:7.1f}x")
        print(f"  {name:>9} " + " ".join(f"{c:>16}" for c in cells))
    tot = sum(rec_hist.values())
    print("\n  when a missed expert was last routed by the SAME layer in this generation:")
    for k in sorted(rec_hist, key=lambda x: (x == "never", x if x != "never" else 0)):
        print(f"    {str(k)+' steps ago':>18}: {rec_hist[k]/tot*100:5.1f}%")


def main():
    paths = sys.argv[1:] or ["/w/artifacts/routes_rank0.npz"]
    prof, caps, hot, cov = caps_and_hot()
    print(f"15 GB -> {sum(caps)} slots, mean cap {np.mean(caps):.1f}/layer, profile cov {cov:.3f}")
    modal, K, gens = load_rows(paths[0])
    if MAXGEN:
        gens = gens[:MAXGEN]
    depth_sweep(gens, modal, caps, hot)
    predictability(gens, caps, hot, prof)


if __name__ == "__main__":
    main()
