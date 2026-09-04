"""Would a dynamic expert cache beat the static profile hot set at equal VRAM?

Offline replay of captured routing traces (routecap.py -> routes_rank<N>.npz) through a set of
per-layer cache policies at a fixed slot budget. CPU/numpy only.

A "slot" holds one (layer, expert) pair's MXFP4 weight for BOTH GEMMs:
819200 (gate_up wq) + 409600 (down wq) = 1,228,800 B, matching the production
`per_expert_bytes` (15 GB / 1.2288 MB = 12207 slots/rank).

Policies, all at the same total slot budget and the same per-layer split:
  static   the profile hot set, chosen exactly as r4d_mxfp4_moe._hot_sets does
           (global water-fill on the normalised per-layer counts). No inserts.
  lru      per-layer LRU, COLD start (empty cache)
  lru_ws   per-layer LRU, WARM start: preloaded with the static hot set, then free to adapt.
           This is the honest "dynamic cache we would actually ship" arm.
  lfu      per-layer LFU with exponential decay, warm start
  hybrid   per-layer: pin the top HYBRID_PIN fraction of slots from the profile (never
           evicted), run LRU over the rest, warm start
  belady   per-layer optimal (evict the farthest next use), warm start -- an upper bound on
           what ANY dynamic policy could do on this trace, not an implementable policy

A miss costs one expert pull over PCIe (1.2288 MB). For the dynamic policies the insert IS
that same pull, so misses == inserts and the PCIe byte cost is directly comparable to static;
the dynamic policies additionally pay a VRAM write and eviction bookkeeping per insert.

    python3 locality_sim.py routes_rank0.npz
    SYNTH=iid python3 locality_sim.py          # null control: no temporal locality at all
Env: PROFILE, GB=15,16,17,18  DIST=global  HYBRID_PIN=0.5  DECAY=0.99  SYNTH_STEPS=1000
     LAYERS=48 (cap on layers scored; the 49th router is the MTP head, reported separately)
"""
import heapq
import json
import os
import sys
from collections import OrderedDict

import numpy as np

PROFILE = os.environ.get("PROFILE", "/w/profiles/hot_profile.json")
PER_EXPERT_BYTES = 1_228_800
GBS = [float(v) for v in os.environ.get("GB", "15,16,17,18").split(",")]
DISTS = os.environ.get("DIST", "global").split(",")
HYBRID_PIN = float(os.environ.get("HYBRID_PIN", "0.5"))
DECAY = float(os.environ.get("DECAY", "0.99"))
SYNTH = os.environ.get("SYNTH", "")
SYNTH_STEPS = int(os.environ.get("SYNTH_STEPS", "1000"))
NLAYER = int(os.environ.get("LAYERS", "48"))
ONLY_TRAFFIC = os.environ.get("TRAFFIC_TABLE", "") == "1"
TRAFFIC_LABELS = ["code1", "prose1", "json1", "code2", "prose2", "json2"]


# ------------------------------------------------------------------ inputs
def load_profile():
    prof = json.load(open(PROFILE))["layers"]
    return {int(k): (np.asarray(v["ranked"], dtype=np.int64),
                     np.asarray(v["counts"], dtype=np.float64)) for k, v in prof.items()}


def load_traces(paths):
    """-> list of (label, trace); trace = list of steps, step = list of per-layer arrays of
    DISTINCT routed expert ids.

    Segmentation: a decode step of this server is exactly `modal` rows wide (1 target token +
    num_speculative_tokens MTP drafts, one sequence). Prefill steps are not captured at all
    (too wide), so a request boundary shows up as a short run of off-modal row counts, not as a
    gap in the step counter. Maximal runs of modal-width steps == one generation each.
    """
    out = []
    for p in paths:
        d = np.load(p)
        ids, steps = d["ids"], d["steps"]
        K = int(d["topk"])
        if int(d["lost_to_wrap"]) != 0:
            print(f"  !! {p}: {int(d['lost_to_wrap'])} step(s) lost to ring wrap")
        order = np.argsort(steps)
        ids, steps = ids[order], steps[order]
        assert (np.diff(steps) == 1).all(), "step counter has gaps"
        rows = (ids[:, 0] >= 0).sum(axis=1) // K
        modal = np.bincount(rows).argmax()
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
        for i, (a, b) in enumerate(runs):
            lbl = (TRAFFIC_LABELS[i - len(runs) + len(TRAFFIC_LABELS)]
                   if i >= len(runs) - len(TRAFFIC_LABELS) else f"ab3_{i+1}")
            tr = [[np.unique(row[row >= 0], return_counts=True) for row in ids[t, :L]]
                  for t in range(a, b)]
            out.append((lbl, tr))
        print(f"  {p}: {len(steps)} steps, {ids.shape[1]} routers, modal width {modal} rows"
              f" -> {len(runs)} generation(s): "
              + ", ".join(f"{l}[{len(t)}]" for l, t in out[-len(runs):]))
    return out


def synth_traces(prof, nseg=6):
    rng = np.random.default_rng(0)
    out = []
    for j in range(nseg):
        tr = []
        for _ in range(SYNTH_STEPS // nseg):
            step = []
            for li in range(NLAYER):
                r, c = prof[li]
                p = c / c.sum()
                step.append(np.unique(rng.choice(r, size=50, replace=True, p=p),
                                      return_counts=True))
            tr.append(step)
        out.append((f"iid_{j+1}", tr))
    return out


# ------------------------------------------------------------------ capacity split
def capacities(prof, gb, how, nlayer):
    """Replicates r4d_mxfp4_moe._hot_sets exactly. Returns (per-layer caps, hot ids, cov)."""
    budget = int(gb * 1e9) // PER_EXPERT_BYTES
    layers, ranked, counts = [], [], []
    for li in range(nlayer):
        r, c = prof[li]
        layers.append(np.full(len(r), li))
        ranked.append(r)
        counts.append(c / c.sum())
    layers, ranked, counts = map(np.concatenate, (layers, ranked, counts))
    if how == "equal":
        caps = [budget // nlayer] * nlayer
        hot = {li: prof[li][0][:caps[li]].tolist() for li in range(nlayer)}
        cov = float(sum(prof[li][1][:caps[li]].sum() / prof[li][1].sum()
                        for li in range(nlayer))) / nlayer
        return caps, hot, cov
    keep = np.argsort(-counts, kind="stable")[:budget]
    hot = {li: [] for li in range(nlayer)}
    for li, e in zip(layers[keep].tolist(), ranked[keep].tolist()):
        hot[li].append(e)
    cov = float(counts[keep].sum()) / nlayer
    return [len(hot[li]) for li in range(nlayer)], hot, cov


# ------------------------------------------------------------------ policies
class Static:
    def __init__(self, cap, init=None, **_):
        self.res = init

    def step(self, ids):
        miss = [i for i, e in enumerate(ids) if e not in self.res]
        return miss


class LRU:
    def __init__(self, cap, init=None, prof_rank=None, pin=0, **_):
        self.cap = cap
        self.pin = set(prof_rank[:pin].tolist()) if pin else set()
        self.free = cap - len(self.pin)
        self.od = OrderedDict()
        for e in (init or ()):            # warm start, oldest-first in profile rank order
            if e not in self.pin and len(self.od) < self.free:
                self.od[int(e)] = None

    def step(self, ids):
        miss = []
        for i, e in enumerate(ids):
            e = int(e)
            if e in self.pin:
                continue
            if e in self.od:
                self.od.move_to_end(e)
            else:
                miss.append(i)
                self.od[e] = None
                while len(self.od) > self.free:
                    self.od.popitem(last=False)
        return miss


class LFU:
    """LFU with exponential decay, done with a growing boost instead of decaying every entry."""
    def __init__(self, cap, init=None, **_):
        self.cap = cap
        self.score = {int(e): 0.0 for e in (init or ())}
        self.heap = [(0.0, e) for e in self.score]
        heapq.heapify(self.heap)
        self.boost = 1.0

    def step(self, ids):
        self.boost /= DECAY
        miss = []
        fresh = []
        for i, e in enumerate(ids):
            e = int(e)
            if e not in self.score:
                miss.append(i)
            self.score[e] = self.score.get(e, 0.0) + self.boost
            fresh.append(e)
        for e in fresh:
            heapq.heappush(self.heap, (self.score[e], e))
        keep = set(fresh)
        back = []
        while len(self.score) > self.cap and self.heap:
            s, e = heapq.heappop(self.heap)
            if e not in self.score or self.score[e] != s:
                continue            # stale entry, drop it
            if e in keep:
                back.append((s, e))  # touched this step: not evictable, but keep it findable
                continue
            del self.score[e]
        for it in back:
            heapq.heappush(self.heap, it)
        return miss


class Belady:
    def __init__(self, cap, init=None, nexts=None, **_):
        self.cap = cap
        self.nexts = nexts
        self.res = set(int(e) for e in (init or ()))
        self.t = -1

    def step(self, ids):
        self.t += 1
        miss = []
        nx = self.nexts[self.t]
        for i, e in enumerate(ids):
            if int(e) not in self.res:
                miss.append(i)
                self.res.add(int(e))
        cur = set(int(e) for e in ids)
        if len(self.res) > self.cap:
            ranked = sorted(self.res, key=lambda e: -nx.get(e, 1 << 30))
            for e in ranked:
                if len(self.res) <= self.cap:
                    break
                if e in cur and len(cur) <= self.cap:
                    continue
                self.res.discard(e)
        return miss


_NEXTS = {}


def belady_nexts(trace, li):
    key = (id(trace), li)
    if key in _NEXTS:
        return _NEXTS[key]
    T = len(trace)
    nxt = [None] * T
    seen = {}
    for t in range(T - 1, -1, -1):
        nxt[t] = dict(seen)
        for e in trace[t][li][0]:
            seen[int(e)] = t
    _NEXTS[key] = nxt
    return nxt


_CONCAT = {}


def _concat(traces):
    key = tuple(id(t) for _, t in traces)
    if key not in _CONCAT:
        _CONCAT[key] = sum((t for _, t in traces), [])
    return _CONCAT[key]


WANT = os.environ.get("POLICIES", "static,lru,lru_ws,lfu,hybrid,belady").split(",")


def make(name, li, caps, hot, prof, tr):
    cap, init = caps[li], hot[li]
    if name == "static":
        return Static(cap, init=set(init))
    if name == "lru":
        return LRU(cap)
    if name == "lru_ws":
        return LRU(cap, init=init)
    if name == "lfu":
        return LFU(cap, init=init)
    if name == "hybrid":
        return LRU(cap, init=init, prof_rank=prof[li][0], pin=int(cap * HYBRID_PIN))
    if name == "belady":
        return Belady(cap, init=init, nexts=belady_nexts(tr, li))
    raise KeyError(name)


def replay(traces, caps, hot, prof, want=WANT, carry=False):
    """-> {policy: [ (label, [(miss, routed), ...]) ]}.  carry=True keeps ONE cache across all
    segments (context switches hit a warm cache); carry=False resets per segment."""
    nlayer = len(caps)
    allsteps = _concat(traces) if carry else None
    res = {}
    for name in want:
        segs = []
        pols = None
        off = 0
        for lbl, tr in traces:
            if pols is None or not carry or name == "belady":
                pols = [make(name, li, caps, hot, prof,
                             allsteps if carry else tr) for li in range(nlayer)]
                if name == "belady" and carry:
                    for p in pols:
                        p.t = off - 1
            off += len(tr)
            series = []
            for step in tr:
                m = r = wm = wr = 0
                for li in range(nlayer):
                    u, mult = step[li]
                    idx = pols[li].step(u)
                    m += len(idx)
                    r += len(u)
                    wm += int(mult[idx].sum()) if idx else 0
                    wr += int(mult.sum())
                series.append((m, r, wm, wr))
            segs.append((lbl, series))
        res[name] = segs
    return res


def aggregate(segs, skip=0, only=None):
    """-> (distinct miss rate, misses/step, steps, routing-weighted miss rate)"""
    m = r = n = wm = wr = 0
    for lbl, series in segs:
        if only is not None and lbl not in only:
            continue
        for t, row in enumerate(series):
            if t < skip:
                continue
            m += row[0]; r += row[1]; wm += row[2]; wr += row[3]; n += 1
    return (m / max(r, 1), m / max(n, 1), n, wm / max(wr, 1))


def warmup_len(segs, tail_frac=0.5):
    """First step index after which the windowed miss rate stays within 10% of steady state,
    measured on the longest segment."""
    lbl, series = max(segs, key=lambda s: len(s[1]))
    m = np.array([s[0] for s in series], float)
    r = np.array([s[1] for s in series], float)
    n = len(m)
    steady = m[int(n * tail_frac):].sum() / max(r[int(n * tail_frac):].sum(), 1)
    if steady <= 0:
        return 0
    win = max(10, n // 10)
    for t in range(n - win):
        w = m[t:t + win].sum() / max(r[t:t + win].sum(), 1)
        if abs(w - steady) / steady <= 0.10:
            return t
    return n


def table(traces, caps, hot, prof, cov, gb, how, only=None, note=""):
    r = replay(traces, caps, hot, prof)
    warm = warmup_len(r["lru" if "lru" in r else WANT[-1]])
    base = aggregate(r["static"], 0, only)[0]
    print(f"=== {gb:g} GB/rank = {int(gb*1e9)//PER_EXPERT_BYTES} slots, {how} split "
          f"(per-layer min/med/max {min(caps)}/{sorted(caps)[len(caps)//2]}/{max(caps)}), "
          f"profile-estimated coverage {cov:.3f} -> predicted miss {100*(1-cov):.1f}%{note}")
    print("    %-8s %9s %9s %9s %10s %11s %12s %11s" %
          ("policy", "miss%", "wmiss%", "miss%ss", "miss/step", "insert/step",
           "coldMB/step", "vs static"))
    rows = {}
    for name in WANT:
        # miss/step and coldMB/step are ALL-INCLUSIVE (every scored step, no warm-up skip):
        # they are the per-step PCIe bill production actually pays. miss%ss is a diagnostic
        # only -- skipping each segment's first `warm` steps also drops the short segments,
        # which reweights the mix and is NOT a like-for-like byte count.
        mr_all, mps, _, wmr = aggregate(r[name], 0, only)
        mr = aggregate(r[name], warm, only)[0]
        ips = 0.0 if name == "static" else mps
        rows[name] = (mr_all, mr, mps, wmr)
        print("    %-8s %8.2f%% %8.2f%% %8.2f%% %9.1f %11.1f %12.1f %11s" %
              (name, 100 * mr_all, 100 * wmr, 100 * mr, mps, ips,
               mps * PER_EXPERT_BYTES / 1e6,
               "baseline" if name == "static" else f"{100*(mr_all-base)/base:+.1f}%"))
    print(f"    (cold-start warm-up skip for the miss%ss column: {warm} steps)")
    print()
    return rows


def main():
    prof = load_profile()
    paths = [a for a in sys.argv[1:] if a.endswith(".npz")]
    if SYNTH:
        print(f"SYNTHETIC trace ({SYNTH}) -- null control, no temporal locality")
        traces = synth_traces(prof)
    else:
        if not paths:
            print("usage: locality_sim.py routes_rank0.npz [...]   (or SYNTH=iid)")
            return 2
        traces = load_traces(paths)
    nlayer = len(traces[0][1][0])
    nsteps = sum(len(t) for _, t in traces)
    routed = np.mean([len(s[li][0]) for _, t in traces for s in t for li in range(nlayer)])
    print(f"\n{len(traces)} generation(s), {nsteps} decode steps, {nlayer} MoE layers scored, "
          f"mean DISTINCT experts routed per layer per step {routed:.2f} "
          f"(of {nlayer * 512} (layer,expert) pairs)\n")

    # ---- why: how wide is one generation's working set compared to the cache?
    caps0, _, _ = capacities(prof, GBS[0], DISTS[0], nlayer)
    print("=== working set: DISTINCT experts a single generation touches, per layer "
          f"(cache holds {sorted(caps0)[nlayer//2]} of 512 at {GBS[0]:g} GB)")
    print("    %-8s %6s %10s %10s %10s" % ("prompt", "steps", "min", "median", "max"))
    for lbl, tr in traces:
        ws = [len(set(int(e) for st in tr for e in st[li][0])) for li in range(nlayer)]
        print("    %-8s %6d %10d %10d %10d"
              % (lbl, len(tr), min(ws), int(np.median(ws)), max(ws)))
    print()

    traf = [t for t in traces if t[0] in TRAFFIC_LABELS] or traces
    for gb in GBS:
        for how in DISTS:
            caps, hot, cov = capacities(prof, gb, how, nlayer)
            table(traces, caps, hot, prof, cov, gb, how)
            if ONLY_TRAFFIC:
                table([t for t in traces if t[0] in TRAFFIC_LABELS], caps, hot, prof, cov,
                      gb, how, note="  [the 6 labelled traffic.py prompts only]")

    # ---- calibration: profile-predicted vs actual, all traffic and per prompt
    print("=== calibration of the static hot set: profile-predicted miss vs MEASURED "
          "routing-selection-weighted miss (like for like), per prompt")
    print("    %-8s" % "GB" + "".join("%10s" % l for l in
                                      ["predict"] + [l for l, _ in traf] + ["all"]))
    for gb in GBS:
        caps, hot, cov = capacities(prof, gb, DISTS[0], nlayer)
        r = replay(traces, caps, hot, prof, want=["static"])
        cells = [aggregate(r["static"], 0, {l})[3] for l, _ in traf]
        cells.append(aggregate(r["static"], 0)[3])
        print("    %-8g" % gb + "%9.1f%%" % (100 * (1 - cov))
              + "".join("%9.1f%%" % (100 * c) for c in cells))
    print()

    # ---- context-switch sensitivity
    if len(traf) > 1:
        caps, hot, cov = capacities(prof, GBS[0], DISTS[0], nlayer)
        rc = replay(traf, caps, hot, prof, carry=True)
        rs = replay(traf, caps, hot, prof, carry=False)
        print(f"=== context switches ({len(traf)} prompts back to back, {GBS[0]:g} GB, "
              f"{DISTS[0]} split): one cache carried across prompts vs a cache re-seeded from "
              f"the profile at each prompt")
        print("    %-8s %12s %12s %12s" % ("policy", "carried%", "reseeded%", "delta"))
        for name in WANT:
            a = aggregate(rc[name])[0]
            b = aggregate(rs[name])[0]
            print("    %-8s %11.2f%% %11.2f%% %10.2f pp" %
                  (name, 100 * a, 100 * b, 100 * (a - b)))
        print()
        # per-prompt first-N-steps penalty after a switch
        print("    miss rate in the first 20 steps of each prompt (carried cache):")
        print("    %-8s" % "policy" + "".join("%9s" % l for l, _ in traf))
        for name in WANT:
            cells = []
            for lbl, series in rc[name]:
                m = sum(s[0] for s in series[:20])
                rr = sum(s[1] for s in series[:20])
                cells.append(100 * m / max(rr, 1))
            print("    %-8s" % name + "".join("%8.1f%%" % c for c in cells))
    # ---- concurrency: the capture was single-stream. Batch B independent generations into
    # one step (the kernel pulls the UNION of their cold experts) and re-measure.
    print("=== concurrency sensitivity: B independent generations batched into one decode step "
          f"({GBS[0]:g} GB, {DISTS[0]} split). The capture itself is B=1.")
    caps, hot, cov = capacities(prof, GBS[0], DISTS[0], nlayer)
    print("    %-4s %8s" % ("B", "uniq/L") + "".join("%10s" % n for n in WANT))
    for B in (1, 2, 4):
        pool = [t for _, t in traces]
        n = min(len(t) for t in pool[:max(B, 1) * 3]) if B > 1 else None
        merged = []
        for g in range(0, len(pool) - B + 1, B):
            grp = pool[g:g + B]
            T = min(len(t) for t in grp)
            mtr = []
            for t in range(T):
                step = []
                for li in range(nlayer):
                    cat = np.concatenate([grp[j][t][li][0] for j in range(B)])
                    rep = np.concatenate([grp[j][t][li][1] for j in range(B)])
                    u, inv = np.unique(cat, return_inverse=True)
                    step.append((u, np.bincount(inv, weights=rep).astype(np.int64)))
                mtr.append(step)
            merged.append((f"b{g}", mtr))
        rr = replay(merged, caps, hot, prof)
        uq = np.mean([len(st[li][0]) for _, t in merged for st in t for li in range(nlayer)])
        print("    %-4d %8.1f" % (B, uq)
              + "".join("%9.2f%%" % (100 * aggregate(rr[n2])[0]) for n2 in WANT))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
