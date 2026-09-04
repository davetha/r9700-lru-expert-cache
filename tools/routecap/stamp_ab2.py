"""Three warm-start stamp policies for the LRU cache, on the real traces.

  zero      what the shipped kernel does: every warm slot stamp 0, so the (stamp, slot)
            tiebreak evicts the lowest slot index == the lowest expert id first.
  inverted  what slot_alloc.KernelLRU actually did (stamp = -(S-j), j=0 the hottest):
            evicts the HOTTEST profile expert first. This was a simulator bug.
  rank      the intended policy: stamp = -j, so the hottest warm expert is evicted last.
"""
import os, sys
os.environ.setdefault("GB", "15")
sys.path.insert(0, "/w/tools/routecap")
import numpy as np
import locality_sim as ls
import slot_alloc as SA


def make(kind):
    class P(SA.KernelLRU):
        def __init__(self, S, init, max_inserts, thresh=SA.THRESH):
            super().__init__(S, init, max_inserts, thresh)
            init = list(init)[:S]
            for j, e in enumerate(init):          # j == profile rank, 0 = hottest
                if kind == "zero":
                    self.stamp[int(e)] = 0
                elif kind == "inverted":
                    self.stamp[int(e)] = -(len(init) - j)
                else:
                    self.stamp[int(e)] = -j
    return P


def run(cls, mtraces, caps, hot, nlayer, mi):
    mc, acc, ins = 0, 0, 0
    for _, tr in mtraces:
        pols = [cls(caps[li], hot[li], mi) for li in range(nlayer)]
        for st in tr:
            for li in range(nlayer):
                m, ni = pols[li].step(st[li])
                mc += len(m); ins += ni; acc += len(st[li])
    return mc / max(acc, 1), mc, ins


def main():
    prof = ls.load_profile()
    traces = ls.load_traces([sys.argv[1]])
    nlayer = min(len(traces[0][1][0]), ls.NLAYER)
    caps, hot, _ = ls.capacities(prof, 15.0, "global", nlayer)
    caps = caps + [0] * (nlayer - len(caps))
    # is hot[li] really in descending profile-count order?
    ok = all(list(hot[li]) == [int(e) for e in prof[li][0][:len(hot[li])]] for li in range(nlayer))
    print(f"hot sets are in descending profile-rank order: {ok}")
    print(f"slots min/med/max {min(caps)}/{sorted(caps)[nlayer//2]}/{max(caps)}\n")
    print("  %-4s %-10s %8s %10s %10s %9s" % ("B", "policy", "miss%", "misses", "inserts", "vs zero"))
    for B in (1, 2, 4):
        mt = SA.merge(traces, B, nlayer)
        base = None
        for kind in ("zero", "inverted", "rank"):
            r, c, i = run(make(kind), mt, caps, hot, nlayer, 64)
            if base is None:
                base = r
            print("  %-4d %-10s %7.3f%% %10d %10d %9s"
                  % (B, kind, 100*r, c, i,
                     "baseline" if kind == "zero" else "%+.2f%%" % (100*(r-base)/base)))
        print()
    print("  first-K steps, B=1 (the warm start can only matter early):")
    mt1 = SA.merge(traces, 1, nlayer)
    for K in (20, 50, 100, 200):
        sub = [(l, t[:K]) for l, t in mt1]
        out = []
        for kind in ("zero", "inverted", "rank"):
            r, c, _ = run(make(kind), sub, caps, hot, nlayer, 64)
            out.append((kind, r, c))
        b = out[0][1]
        print("    K=%-4d " % K + "  ".join(
            "%s %6.3f%% (%+.2f%%)" % (k, 100*r, 100*(r-b)/b) for k, r, _ in out))


main()
