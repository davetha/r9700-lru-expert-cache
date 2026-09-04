"""Step 0 for k1/lru/FUSE.md: what does ONE more kernel node actually cost inside a
~3250-node HIP graph?

FUSE.md estimates the fusion win as 5 launches x 52 MoE layer invocations x 3.72 us
(the median inter-kernel gap measured in production) = ~0.96 ms/step. 3.72 us is a
*median gap*, not a *marginal cost*: it says nothing about what happens to replay time
when a node is removed. This measures the marginal cost directly.

Method: capture graphs that contain
    52 x [ lru_manage, lru_gather, k x (moe_align(table), moe_align(map_cold), tw cast) ]
  + NFILL filler nodes (realistic-duration elementwise kernels, interleaved)
for k = 0,1,2,3, and time replay. Everything except the k copies is identical, so
slope over k = the cost of 52 x 5 = 260 launches at a graph size that brackets 3250.
k=0 is the fused ideal (bookkeeping folded into the manager), k=1 is today.

Run:
  flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --ipc host --group-add video \
    --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v $REPO_ROOT:/w \
    --entrypoint bash local/q38fn-rocm10:k1build -c 'cd /w/tests/lru && python3 step0_nodecost.py'
"""
import os
import sys

os.environ.setdefault("NLAYER", "52")
sys.path.insert(0, "/w/tools/routecap")
sys.path.insert(0, "/w/tests/lru")

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import \
    moe_align_block_size                                              # noqa: E402

import test_graph_lru as G                                            # noqa: E402

DEV = "cuda"
NLAYER = G.NLAYER
E, S = G.E, G.S
M = int(os.environ.get("M", "5"))          # target forward at MTP-4: 1 target + 4 drafts
TOPK = int(os.environ.get("TOPK", "10"))
BLOCK = 16
NFILL = int(os.environ.get("NFILL", "2886"))   # 2886 + 52*7 = 3250 = the measured step
KMAX = int(os.environ.get("KMAX", "3"))
REPS = int(os.environ.get("REPS", "40"))


def timeit(g, n=REPS):
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    st = torch.cuda.current_stream()
    a.record(st)
    for _ in range(n):
        g.replay()
    b.record(st)
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000.0 / n      # us


def calibrate():
    """Pick a filler element count whose kernel lasts ~13.5 us -- the real step's
    kernel-busy/kernel-count ratio (47 ms / 3490). A filler that is much shorter would
    exaggerate the share of dispatch, much longer would hide it."""
    target = float(os.environ.get("FILL_US", "13.5"))
    best = None
    for elems in [1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21]:
        a = torch.randn(elems, device=DEV)
        b = torch.randn(elems, device=DEV)
        c = torch.empty(elems, device=DEV)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                torch.add(a, b, out=c)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(256):
                torch.add(a, b, out=c)
        per = timeit(g, 20) / 256.0
        del g, a, b, c
        torch.cuda.empty_cache()
        print(f"  filler {elems:>9} elems -> {per:6.2f} us/node")
        if best is None or abs(per - target) < abs(best[1] - target):
            best = (elems, per)
    print(f"  chosen: {best[0]} elems ({best[1]:.2f} us/node, target {target})")
    return best


class TLayer(G.Layer):
    """G.Layer's tensors, but the six slot buffers are shared across layers. This is a
    TIMING harness: 52 private copies would be 16.5 GB and the box has a server resident.
    Every layer still gets its own table/stamps/miss list, so the manager does exactly the
    work it does in production; only the gather's destination is shared."""

    def __init__(self, rng, src_d, shared):
        hot = sorted(rng.choice(E, S, replace=False).tolist())
        h = torch.tensor(hot, dtype=torch.int64, device=DEV)
        self.table = torch.full((E,), -1, dtype=torch.int32, device=DEV)
        self.table[h] = torch.arange(S, dtype=torch.int32, device=DEV)
        self.map_cold = torch.arange(E, dtype=torch.int32, device=DEV)
        self.map_cold[h] = -1
        self.slot_expert = h.to(torch.int32).clone()
        self.slot_stamp = torch.zeros(S, dtype=torch.int64, device=DEV)
        self.routed = torch.zeros(E, dtype=torch.uint8, device=DEV)
        self.step = torch.zeros(1, dtype=torch.int64, device=DEV)
        self.miss = torch.full((G.MAXI, 2), -1, dtype=torch.int32, device=DEV)
        self.n_miss = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.dst = shared
        self.src_d = src_d


class Bench:
    def __init__(self, fill_elems):
        rng = np.random.default_rng(0)
        self.src_h, self.src_d = [], []
        for b in G.SIZES:
            h, dp = G.uva(E * b)
            self.src_h.append(h)
            self.src_d.append(dp)
        shared = [torch.zeros(S * b, dtype=torch.uint8, device=DEV) for b in G.SIZES]
        self.shared = shared
        self.layers = [TLayer(rng, self.src_d, shared) for _ in range(NLAYER)]
        self.ids = [torch.zeros(M * TOPK, dtype=torch.int32, device=DEV)
                    for _ in range(NLAYER)]
        self.ids2d = [t.view(M, TOPK) for t in self.ids]
        self.tw = [torch.rand(M, TOPK, dtype=torch.bfloat16, device=DEV)
                   for _ in range(NLAYER)]
        self.fa = torch.randn(fill_elems, device=DEV)
        self.fb = torch.randn(fill_elems, device=DEV)
        self.fc = torch.empty(fill_elems, device=DEV)
        self.t1 = torch.zeros(1, device=DEV)
        self.fill_routing(1)

    def fill_routing(self, seed):
        r = np.random.default_rng(seed)
        p = np.arange(1, E + 1, dtype=float) ** -1.1
        p /= p.sum()
        for li in range(NLAYER):
            v = np.concatenate([r.choice(E, TOPK, replace=False, p=p) for _ in range(M)])
            self.ids[li].copy_(torch.tensor(v, dtype=torch.int32))

    def body(self, k, trivial=False):
        """One decode step's worth of MoE bookkeeping with k copies of the fusable part."""
        keep = []
        per_layer_fill = NFILL // NLAYER
        extra = NFILL - per_layer_fill * NLAYER
        for li in range(NLAYER):
            L = self.layers[li]
            L.launch(self.ids[li])                       # manage + gather (2 nodes)
            for _ in range(k):
                if trivial:
                    # control arm: 5 nodes of (almost) pure dispatch, to split the
                    # measured marginal cost into launch overhead vs the removed
                    # kernels' own execution. A fused kernel only recovers the launch
                    # overhead plus whatever of the work it does more cheaply.
                    for _ in range(5):
                        self.t1.add_(1.0)
                    continue
                keep.append(moe_align_block_size(self.ids2d[li], BLOCK, E, L.table,
                                                 ignore_invalid_experts=True))
                keep.append(moe_align_block_size(self.ids2d[li], BLOCK, E, L.map_cold,
                                                 ignore_invalid_experts=True))
                keep.append(self.tw[li].to(torch.float32).reshape(-1).contiguous())
            n = per_layer_fill + (1 if li < extra else 0)
            for _ in range(n):
                torch.add(self.fa, self.fb, out=self.fc)
        return keep

    def capture(self, k, trivial=False):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.body(k, trivial)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.keep = self.body(k, trivial)
        torch.cuda.synchronize()
        return g


def main():
    print("calibrating the filler kernel:")
    fill_elems, fill_us = calibrate()
    bench = Bench(fill_elems)
    print(f"\n{NLAYER} layers, M={M} top_k={TOPK} (mtk={M*TOPK}), "
          f"{NFILL} filler nodes at {fill_us:.2f} us")
    graphs = [bench.capture(k) for k in range(KMAX + 1)]
    nodes = [NLAYER * (2 + 5 * k) + NFILL for k in range(KMAX + 1)]
    # round-robin the k values so any drift (clocks, a neighbour waking up) hits all of
    # them equally instead of landing on one arm.
    ROUNDS = int(os.environ.get("ROUNDS", "5"))
    samples = [[] for _ in graphs]
    for r in range(ROUNDS):
        for k, g in enumerate(graphs):
            samples[k].append(timeit(g))
    times = [float(np.median(v)) for v in samples]
    print(f"{'k':>2} {'nodes':>7} {'us/replay':>11} {'spread':>8} "
          f"{'delta us':>10} {'us/node':>9}")
    for k in range(KMAX + 1):
        v = np.array(samples[k])
        d = times[k] - times[0]
        pern = d / (nodes[k] - nodes[0]) if k else float("nan")
        print(f"{k:>2} {nodes[k]:>7} {times[k]:>11.1f} "
              f"{100*(v.max()-v.min())/times[k]:>7.1f}% {d:>10.1f} {pern:>9.3f}")
    A = np.polyfit(nodes, times, 1)
    print(f"\nlinear fit: {A[0]*1000:.3f} ns/node  (intercept {A[1]:.1f} us)")
    saved = A[0] * 5 * NLAYER
    print(f"marginal cost of one launch node in a ~{nodes[1]}-node graph: {A[0]:.3f} us")
    print(f"removing 5 launches x {NLAYER} layers = {5*NLAYER} nodes "
          f"-> {saved:.0f} us/step = {saved/1000:.2f} ms/step")
    print(f"  (FUSE.md's estimate at the 3.72 us median gap was "
          f"{3.72*5*NLAYER/1000:.2f} ms/step)")
    print(f"  measured k=1 -> k=0 delta: {(times[1]-times[0])/1000:.3f} ms/step")

    for g in graphs:
        del g
    graphs = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print("\ncontrol: the same node COUNT, but 5 one-element kernels instead of the "
          "real bookkeeping\n(marginal cost here is launch overhead only -- the real "
          "arm also includes\nthe removed kernels' execution, which a fused kernel "
          "still has to do somewhere)")
    cg = [bench.capture(k, trivial=True) for k in range(KMAX + 1)]
    cs = [[] for _ in cg]
    for r in range(ROUNDS):
        for k, g in enumerate(cg):
            cs[k].append(timeit(g))
    ct = [float(np.median(v)) for v in cs]
    print(f"{'k':>2} {'nodes':>7} {'us/replay':>11} {'delta us':>10} {'us/node':>9}")
    for k in range(KMAX + 1):
        d = ct[k] - ct[0]
        pern = d / (nodes[k] - nodes[0]) if k else float("nan")
        print(f"{k:>2} {nodes[k]:>7} {ct[k]:>11.1f} {d:>10.1f} {pern:>9.3f}")
    B = np.polyfit(nodes, ct, 1)
    print(f"\nlaunch overhead alone: {B[0]:.3f} us/node "
          f"-> {B[0]*5*NLAYER/1000:.2f} ms/step for {5*NLAYER} nodes")
    print(f"execution of the real bookkeeping kernels: "
          f"{(A[0]-B[0])*5*NLAYER/1000:.2f} ms/step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
