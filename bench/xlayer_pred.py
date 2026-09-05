#!/usr/bin/env python3
"""Cross-layer expert predictability on captured routes (k1/routes_rank0.npz).

Question: can layer L's routing (known before layer L+1's gather) predict which experts layer L+1
will route, well enough to prefetch L+1's MISSES during layer L's compute?
Predictors trained on the first 70% of 5-row steps, scored on the last 30%:
  freq      per-layer popularity of L+1 (no information from L)          -- the floor
  cooc      set-level: score(e') = sum_{e in S_L} P(e' at L+1 | e at L)
  tok       token-level: same, but pairs are formed within the same token's 10 experts
  hybrid    tok + 0.5*cooc
Scored per (step, layer-pair) at prefetch budget K: recall of S_{L+1}, coverage of LRU misses
(S=257 slots/layer, warm-started with training popularity, run in time order over all steps),
and over-fetch = experts prefetched that were neither resident nor needed / misses."""
import numpy as np, collections
import sys; z = np.load(sys.argv[1] if len(sys.argv) > 1 else "/w/k1/routes_rank0.npz")   # a tools/routecap dump; ids = z["ids"].reshape(z["ids"].shape[0], -1, 32, 10)
live = (ids[:, 0, :, 0] >= 0).sum(1); steps = np.where(live == 5)[0]
E, NL, ROWS = 512, 48, 5
rows = ids[steps][:, :NL, :ROWS, :]                     # [T, 48, 5, 10]
T = len(steps); ntr = int(0.7 * T)
print(f"{T} five-row steps, train {ntr}, eval {T-ntr}; layers 0..{NL-1}")

# --- train
freq = np.zeros((NL, E)); C = np.zeros((NL - 1, E, E)); Ct = np.zeros((NL - 1, E, E))
for t in range(ntr):
    for L in range(NL):
        s = np.unique(rows[t, L]); freq[L, s] += 1
        if L < NL - 1:
            s2 = np.unique(rows[t, L + 1])
            C[L][np.ix_(s, s2)] += 1
            for r in range(ROWS):
                a, b = rows[t, L, r], rows[t, L + 1, r]
                Ct[L][np.ix_(a, b)] += 1
Cn = C / np.maximum(C.sum(2, keepdims=True), 1); Ctn = Ct / np.maximum(Ct.sum(2, keepdims=True), 1)

def predict(kind, t, L):
    s = np.unique(rows[t, L])
    if kind == "freq": return freq[L + 1]
    if kind == "cooc": return Cn[L][s].sum(0)
    if kind == "tok":
        sc = np.zeros(E)
        for r in range(ROWS): sc += Ctn[L][rows[t, L, r]].sum(0)
        return sc
    if kind == "hybrid": return predict("tok", t, L) + 0.5 * predict("cooc", t, L)

# --- LRU simulation per layer (S slots), replay all steps in order; misses recorded for eval steps
S = 257
def simulate(kind, K):
    res = [set(np.argsort(-freq[L])[:S].tolist()) for L in range(NL)]
    stamp = [dict((e, 0) for e in r) for r in res]
    tot_miss = cov = fetched_extra = 0; recall_num = recall_den = 0; n = 0
    for t in range(T):
        # predictions for L+1 made from layer L's routing, before layer L+1 runs
        pred = {}
        if t >= ntr:
            for L in range(NL - 1):
                sc = predict(kind, t, L).copy()
                pred[L + 1] = set(np.argpartition(-sc, K)[:K].tolist()) if K < E else set(range(E))
        for L in range(NL):
            need = set(np.unique(rows[t, L]).tolist())
            miss = need - res[L]
            if t >= ntr and L >= 1:
                P = pred[L]
                tot_miss += len(miss); cov += len(miss & P)
                fetched_extra += len(P - res[L] - need)
                recall_num += len(need & P); recall_den += len(need); n += 1
            # LRU update: insert misses (and, for the prefetch variants, the prefetched extras would
            # also occupy slots; we account bytes above but keep the residency policy identical
            # so coverage numbers are comparable across predictors)
            for e in miss:
                if len(res[L]) >= S:
                    victim = min((e2 for e2 in res[L] if e2 not in need), key=lambda e2: stamp[L][e2])
                    res[L].discard(victim); del stamp[L][victim]
                res[L].add(e)
            for e in need: stamp[L][e] = t * NL + L + 1
    return recall_num / recall_den, cov / max(tot_miss, 1), fetched_extra / max(tot_miss, 1), tot_miss / (T - ntr)

print(f"{'predictor':9s} {'K':>3s} {'recall(S_L+1)':>14s} {'miss coverage':>14s} {'over-fetch x':>13s}  misses/step")
for kind in ("freq", "cooc", "tok", "hybrid"):
    for K in (8, 16, 32):
        r, c, w, m = simulate(kind, K)
        print(f"{kind:9s} {K:3d} {r:14.3f} {c:14.3f} {w:13.1f}  {m:6.1f}")
