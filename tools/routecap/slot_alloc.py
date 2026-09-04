"""Two questions the live LRU arm raised:

  (1) Is the per-layer SLOT SPLIT still right? The static hot set used a global water-fill on
      profile frequencies -- that split was chosen to maximise STATIC coverage. Under LRU the
      objective changed: a slot is worth whatever it saves in *reuse* misses, so the right
      split is the one that equalises the marginal miss reduction across layers. Solved exactly
      here (concave-hull greedy on per-layer miss-vs-slots curves), train/test split so the
      answer is not just an in-sample fit.

  (2) Is MAX_INSERTS=64 enough at B=4, and does truncating the miss list in ASCENDING EXPERT ID
      order (which is what the deterministic prefix-scan in lru_manage emits) bias residency
      toward low expert ids? Measured on the merged traces.

Miss curves are computed with STEP-GRANULAR stack distances (Mattson), which is exactly the
kernel's semantics: every expert routed in a step becomes equally-MRU and none can be evicted
by its own step. One O(N log N) pass per layer yields miss(C) for ALL C at once, which is what
makes a 48-layer x 500-cap x train/test sweep tractable at all.

    python3 slot_alloc.py routes_rank0.npz
Env: PROFILE, TOTALS=12207,8000  THRESH=0.5  INSERTS=16,32,64,96,128,256,99999  LAYERS=48
     TRAIN=ab3  (ab3 | traffic -- which segments fit the allocation; the other set scores it)
"""
import heapq
import json
import os
import sys

import numpy as np

os.environ.setdefault("GB", "15")
import locality_sim as ls                                            # noqa: E402

PER_EXPERT_BYTES = ls.PER_EXPERT_BYTES
E_TOTAL = 512
TOTALS = [int(v) for v in os.environ.get("TOTALS", "12207,8000").split(",")]
THRESH = float(os.environ.get("THRESH", "0.5"))
INSERTS = [int(v) for v in os.environ.get("INSERTS", "16,32,64,96,128,256,99999").split(",")]
TRAIN = os.environ.get("TRAIN", "ab3")


# ---------------------------------------------------------------- stack distances
class Fenwick:
    __slots__ = ("n", "t")

    def __init__(self, n):
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & -i

    def pre(self, i):                       # sum of positions [0, i]
        i += 1
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s


def miss_hist(steps_experts, init, ncap=E_TOTAL + 1):
    """steps_experts: list of arrays of distinct expert ids, one per step (one layer).
    init: warm-start expert ids in profile-rank order (rank 0 first).
    -> (hist, n_access) where a real access is a HIT at cap C iff its stack distance < C, so
       misses(C) = n_access - hist[:C].sum().   hist[d] = #accesses with distance exactly d.

    Warm start is modelled by prepending one synthetic step per init expert, in REVERSE rank
    order, so that at cap C the surviving warm entries are exactly ranks 0..C-1 -- i.e. the
    static hot set truncated to C, which is what _build_lru actually loads.
    """
    pre = [np.array([int(e)], dtype=np.int64) for e in reversed(list(init))]
    seq = pre + list(steps_experts)
    nreal_start = len(pre)
    T = len(seq)
    fen = Fenwick(T)
    last = {}
    hist = np.zeros(ncap + 1, dtype=np.int64)          # index ncap == "infinite" / first touch
    n_access = 0
    for t, arr in enumerate(seq):
        real = t >= nreal_start
        moves = []
        for e in arr.tolist():
            p = last.get(e)
            if real:
                n_access += 1
                if p is None:
                    hist[ncap] += 1
                else:
                    d = fen.pre(t - 1) - fen.pre(p)
                    hist[min(d, ncap)] += 1
            moves.append((e, p))
        for e, p in moves:
            if p is not None:
                fen.add(p, -1)
            fen.add(t, 1)
            last[e] = t
    return hist, n_access


def curves(traces, hot, nlayer):
    """-> (hist[nlayer, ncap+1], n_access[nlayer]) over the concatenation of `traces`."""
    H, N = [], []
    for li in range(nlayer):
        steps = [st[li][0] for _, tr in traces for st in tr]
        h, n = miss_hist(steps, hot[li])
        H.append(h)
        N.append(n)
    return np.array(H), np.array(N)


def misses_at(H, caps):
    """total misses for a per-layer cap vector."""
    cum = H.cumsum(axis=1)
    return int(sum(H[li].sum() - cum[li, min(c, H.shape[1] - 1)] for li, c in enumerate(caps)))


# ---------------------------------------------------------------- allocation
def alloc_optimal(H, total, cmax=E_TOTAL):
    """Minimise sum_l misses_l(C_l) subject to sum C_l == total.

    gain_l(C) = misses_l(C) - misses_l(C+1) = H[l, C]. Reuse-distance histograms are only
    ROUGHLY non-increasing, so plain greedy on H is not optimal; take the concave upper hull of
    the cumulative-gain curve per layer and greedy on hull SEGMENT slopes, which is optimal for
    the hull-relaxed problem and integral at every hull vertex.
    """
    L = H.shape[0]
    segs = []
    for li in range(L):
        g = np.concatenate([[0], H[li, :cmax].cumsum()])            # g[C] = misses saved at C
        # upper concave hull of the points (C, g[C]) for C in 0..cmax
        hull = [0]
        for c in range(1, cmax + 1):
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                if (g[b] - g[a]) * (c - b) <= (g[c] - g[b]) * (b - a):
                    hull.pop()
                else:
                    break
            hull.append(c)
        s = []
        for a, b in zip(hull, hull[1:]):
            s.append((b - a, (g[b] - g[a]) / (b - a)))              # (width, slope)
        segs.append(s)
    heap = [(-s[0][1], li, 0) for li, s in enumerate(segs) if s]
    heapq.heapify(heap)
    caps = [0] * L
    left = total
    while left > 0 and heap:
        negslope, li, si = heapq.heappop(heap)
        w, _ = segs[li][si]
        take = min(w, left, cmax - caps[li])
        caps[li] += take
        left -= take
        if si + 1 < len(segs[li]) and caps[li] < cmax:
            heapq.heappush(heap, (-segs[li][si + 1][1], li, si + 1))
    if left > 0:                                                    # everything saturated
        for li in range(L):
            t = min(left, cmax - caps[li])
            caps[li] += t
            left -= t
    return caps


def alloc_equal(total, L, cmax=E_TOTAL):
    c = [total // L] * L
    for i in range(total - sum(c)):
        c[i] += 1
    return [min(x, cmax) for x in c]


def alloc_demand(traces, total, nlayer, cmax=E_TOTAL):
    """proportional to the per-layer working set (distinct experts one generation touches)."""
    ws = np.array([np.mean([len({int(e) for st in tr for e in st[li][0]})
                            for _, tr in traces]) for li in range(nlayer)])
    c = np.floor(ws / ws.sum() * total).astype(int)
    i = 0
    while c.sum() < total:
        c[np.argsort(-(ws / np.maximum(c, 1)))[i % nlayer]] += 1
        i += 1
    return [int(min(x, cmax)) for x in c]


# ---------------------------------------------------------------- kernel-faithful LRU
class KernelLRU:
    """Mirrors lru_manage exactly: batch semantics, never evict an expert routed this step,
    read-through (no inserts, no stamp refresh) when distinct > THRESH*S, and the miss list
    truncated to max_inserts in ASCENDING EXPERT ID order (the prefix-scan's emission order)."""

    def __init__(self, S, init, max_inserts, thresh=THRESH):
        self.S = S
        self.max_inserts = max_inserts
        self.max_distinct = max(1, int(S * thresh))
        self.stamp = {}
        for j, e in enumerate(list(init)[:S]):
            self.stamp[int(e)] = -j          # j == profile rank; rank 0 (hottest) is the
            # most recent, so it is evicted LAST. Getting this backwards costs ~2.7% misses
            # at B=1 and ~10% over the first 20 steps (k1/stamp_ab2.py).
        self.t = 0
        self.readthrough = 0

    def step(self, uniq):
        self.t += 1
        u = [int(e) for e in uniq]
        miss = [e for e in u if e not in self.stamp]
        if len(u) > self.max_distinct:
            self.readthrough += 1
            return miss, 0
        for e in u:
            if e in self.stamp:
                self.stamp[e] = self.t
        routed = set(u)
        ins = sorted(miss)[:self.max_inserts]           # ascending id, as the kernel emits
        for e in ins:
            if len(self.stamp) >= self.S:
                vic = min((s, x) for x, s in self.stamp.items() if x not in routed)[1]
                del self.stamp[vic]
            self.stamp[e] = self.t
            routed.add(e)
        return miss, len(ins)


def merge(traces, B, nlayer):
    """B independent generations batched into one decode step (the kernel pulls the UNION)."""
    pool = [t for _, t in traces]
    out = []
    for g in range(0, len(pool) - B + 1, B):
        grp = pool[g:g + B]
        T = min(len(t) for t in grp)
        mtr = []
        for t in range(T):
            mtr.append([np.unique(np.concatenate([grp[j][t][li][0] for j in range(B)]))
                        for li in range(nlayer)])
        out.append((f"b{g}", mtr))
    return out


def run_kernel_lru(mtraces, caps, hot, nlayer, max_inserts):
    """-> (miss_rate, per-(layer,step) miss counts, inserts, readthrough steps, resident sets)"""
    mc, ins_tot, rt, acc = [], 0, 0, 0
    resid = []
    for _, tr in mtraces:
        pols = [KernelLRU(caps[li], hot[li], max_inserts) for li in range(nlayer)]
        for st in tr:
            for li in range(nlayer):
                m, ni = pols[li].step(st[li])
                mc.append(len(m))
                ins_tot += ni
                acc += len(st[li])
        rt += sum(p.readthrough for p in pols)
        resid += [np.array(sorted(p.stamp)) for p in pols]
    mc = np.array(mc)
    return mc.sum() / max(acc, 1), mc, ins_tot, rt, resid


# ---------------------------------------------------------------- main
def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".npz")]
    if not paths:
        print("usage: slot_alloc.py routes_rank0.npz")
        return 2
    prof = ls.load_profile()
    traces = ls.load_traces(paths)
    nlayer = min(len(traces[0][1][0]), ls.NLAYER)
    traf = [t for t in traces if t[0] in ls.TRAFFIC_LABELS]
    ab3 = [t for t in traces if t[0] not in ls.TRAFFIC_LABELS]
    fit, score = (ab3, traf) if TRAIN == "ab3" else (traf, ab3)
    print(f"\nfit on {[l for l,_ in fit]}\nscore on {[l for l,_ in score]}\n")

    caps_prof15, hot15, cov15 = ls.capacities(prof, 15.0, "global", nlayer)
    Hfit, _ = curves(fit, hot15, nlayer)
    Hsco, Nsco = curves(score, hot15, nlayer)
    Hall, Nall = curves(traces, hot15, nlayer)
    nsteps_score = sum(len(t) for _, t in score)

    # sanity: does the step-granular stack-distance curve agree with the exact LRU replay?
    ex = ls.replay(score, caps_prof15, hot15, prof, want=["lru_ws"])
    ex_mr = ls.aggregate(ex["lru_ws"])[0]
    sd_mr = misses_at(Hsco, caps_prof15) / Nsco.sum()
    print(f"cross-check on the scoring set at the shipped 12207-slot profile split: "
          f"stack-distance {100*sd_mr:.2f}%  vs exact LRU replay {100*ex_mr:.2f}%  "
          f"(diff {100*(sd_mr-ex_mr):+.2f} pp -- batch vs access granularity)\n")

    schemes = {}
    for total in TOTALS:
        gb = total * PER_EXPERT_BYTES / 1e9
        capsp, hotp, covp = ls.capacities(prof, gb, "global", nlayer)
        capsp = capsp + [0] * (nlayer - len(capsp))
        cand = {
            "profile": capsp,
            "equal": alloc_equal(total, nlayer),
            "demand": alloc_demand(fit, total, nlayer),
            "lru-opt(fit)": alloc_optimal(Hfit, total),
            "lru-opt(score)": alloc_optimal(Hsco, total),        # in-sample bound, not shippable
        }
        print(f"=== {total} slots ({gb:.2f} GB/rank of expert weight), scored on "
              f"{[l for l,_ in score]} ({nsteps_score} steps), LRU warm-started from the "
              f"{total}-slot profile set")
        print("    %-16s %10s %9s %11s %12s %9s" %
              ("allocation", "min/med/max", "miss%", "miss/step", "coldMB/step", "vs prof"))
        base = None
        for name, c in cand.items():
            m = misses_at(Hsco, c)
            mr = m / Nsco.sum()
            mps = m / nsteps_score
            if base is None:
                base = mr
            print("    %-16s %10s %8.2f%% %11.1f %12.1f %9s" %
                  (name, f"{min(c)}/{sorted(c)[nlayer//2]}/{max(c)}", 100 * mr, mps,
                   mps * PER_EXPERT_BYTES / 1e6,
                   "baseline" if name == "profile" else f"{100*(mr-base)/base:+.1f}%"))
        schemes[total] = cand
        print()

    # what does the optimal split actually look like vs the profile's?
    total = TOTALS[0]
    c_opt, c_prof = schemes[total]["lru-opt(fit)"], schemes[total]["profile"]
    d = np.array(c_opt) - np.array(c_prof)
    order = np.argsort(d)
    print(f"=== per-layer slots, {total} total: lru-opt(fit) vs profile water-fill")
    print("    biggest LOSERS  (layer:prof->opt): " +
          " ".join(f"{li}:{c_prof[li]}->{c_opt[li]}" for li in order[:8]))
    print("    biggest WINNERS (layer:prof->opt): " +
          " ".join(f"{li}:{c_prof[li]}->{c_opt[li]}" for li in order[::-1][:8]))
    print(f"    mean |delta| {np.abs(d).mean():.1f} slots, "
          f"corr(prof, opt) {np.corrcoef(c_prof, c_opt)[0,1]:.3f}\n")

    json.dump({str(t): {k: list(map(int, v)) for k, v in s.items()}
               for t, s in schemes.items()}, open("slot_alloc.json", "w"), indent=1)

    # ------------------------------------------------------------ inserts / THRESH
    print("=== distinct experts routed per layer per step, and the read-through gate "
          f"(THRESH={THRESH}, fires above THRESH*S)")
    caps = schemes[TOTALS[0]]["profile"]
    Smed = sorted(caps)[nlayer // 2]
    print("    %-4s %8s %8s %8s %8s %8s %10s" %
          ("B", "mean", "p50", "p90", "p99", "max", "gate@S=%d" % Smed))
    merged = {}
    for B in (1, 2, 4, 8):
        mt = merge(traces, B, nlayer)
        if not mt:
            continue
        merged[B] = mt
        u = np.array([len(st[li]) for _, tr in mt for st in tr for li in range(nlayer)])
        gate = int(Smed * THRESH)
        print("    %-4d %8.1f %8d %8d %8d %8d %9.2f%%" %
              (B, u.mean(), np.percentile(u, 50), np.percentile(u, 90),
               np.percentile(u, 99), u.max(), 100 * (u > gate).mean()))
    print()

    print("=== per-(layer,step) MISS count at the shipped split, and what MAX_INSERTS truncates")
    print("    %-4s %-8s %8s %8s %8s %8s %9s %10s %10s" %
          ("B", "inserts", "miss%", "p50", "p90", "p99", "max", ">64 steps", "readthru"))
    for B in sorted(merged):
        for mi in INSERTS:
            mr, mc, ins, rt, resid = run_kernel_lru(merged[B], caps, hot15, nlayer, mi)
            tag = "inf" if mi > E_TOTAL else str(mi)
            print("    %-4d %-8s %8.2f%% %8d %8d %8d %9d %9.2f%% %9d" %
                  (B, tag, 100 * mr, np.percentile(mc, 50), np.percentile(mc, 90),
                   np.percentile(mc, 99), mc.max(), 100 * (mc > 64).mean(), rt))
        print()

    # id bias from ascending-order truncation
    print("=== residency bias from truncating the miss list in ascending expert-id order")
    print("    (mean expert id of the resident set at end of trace; unbiased = 255.5)")
    print("    %-4s %-8s %12s %12s" % ("B", "inserts", "mean id", "median id"))
    for B in sorted(merged):
        for mi in INSERTS:
            if mi > 256 and mi <= E_TOTAL:
                continue
            _, _, _, _, resid = run_kernel_lru(merged[B], caps, hot15, nlayer, mi)
            allr = np.concatenate(resid)
            print("    %-4d %-8s %12.1f %12.1f" %
                  (B, "inf" if mi > E_TOTAL else str(mi), allr.mean(), np.median(allr)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
