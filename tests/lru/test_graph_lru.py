"""HIP-graph capture test for the LRU expert cache.

The whole point of the device-side design: after capture there is no Python and no host
sync, yet the cache must still ADAPT -- table/slot_expert/stamps must keep changing across
replays as the routing (a static input tensor the caller overwrites) changes.

Also the honest timing: per-layer GPU cost inside a graph, which is what production pays.
"""
import ctypes
import os

import numpy as np
import torch

LIB = os.environ.get("R4D_LRU_LIB", "/w/build/kernels/librlu.so")
lib = ctypes.CDLL(LIB)
hip = ctypes.CDLL("libamdhip64.so")
lib.r4d_lru_manage.restype = ctypes.c_int
lib.r4d_lru_manage.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 9
lib.r4d_lru_gather.restype = ctypes.c_int
lib.r4d_lru_gather.argtypes = ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long] * 6 +
                               [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_int, ctypes.c_void_p])
DEV = "cuda:0"
E, S, NLAYER = 512, 257, int(os.environ.get("NLAYER", "48"))
SIZES = [819200, 409600, 51200, 640, 25600, 2560]
PER = sum(SIZES)
MAXI = 64
CHUNKS = int(os.environ.get("CHUNKS", "16"))
LANES = int(os.environ.get("LANES", "64"))


def uva(nbytes):
    t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dp = ctypes.c_void_p()
    rc = hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(t.data_ptr()),
                                     ctypes.c_uint(0))
    assert rc == 0
    return t, dp.value


class Layer:
    def __init__(self, rng, src_d):
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
        self.miss = torch.full((MAXI, 2), -1, dtype=torch.int32, device=DEV)
        self.n_miss = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.dst = [torch.zeros(S * b, dtype=torch.uint8, device=DEV) for b in SIZES]
        self.src_d = src_d

    def launch(self, ids_t, max_distinct=128):
        st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = lib.r4d_lru_manage(
            ctypes.c_void_p(ids_t.data_ptr()), ids_t.numel(), E, S, max_distinct, MAXI,
            ctypes.c_void_p(self.table.data_ptr()), ctypes.c_void_p(self.map_cold.data_ptr()),
            ctypes.c_void_p(self.slot_expert.data_ptr()),
            ctypes.c_void_p(self.slot_stamp.data_ptr()),
            ctypes.c_void_p(self.routed.data_ptr()), ctypes.c_void_p(self.step.data_ptr()),
            ctypes.c_void_p(self.miss.data_ptr()), ctypes.c_void_p(self.n_miss.data_ptr()), st)
        assert rc == 0
        args = []
        for i in range(6):
            args += [ctypes.c_void_p(self.dst[i].data_ptr()),
                     ctypes.c_void_p(self.src_d[i]), ctypes.c_long(SIZES[i])]
        rc = lib.r4d_lru_gather(*args, ctypes.c_void_p(self.miss.data_ptr()),
                                ctypes.c_void_p(self.n_miss.data_ptr()), CHUNKS, LANES, st)
        assert rc == 0


def main():
    rng = np.random.default_rng(0)
    src_h, src_d = [], []
    for b in SIZES:
        h, dp = uva(E * b)
        h.copy_(torch.randint(0, 256, (E * b,), dtype=torch.uint8,
                              generator=torch.Generator().manual_seed(b)))
        src_h.append(h); src_d.append(dp)
    layers = [Layer(rng, src_d) for _ in range(NLAYER)]
    # static input the graph reads; the caller overwrites its CONTENTS between replays
    ids = [torch.zeros(50, dtype=torch.int32, device=DEV) for _ in range(NLAYER)]

    def fill(seed):
        r = np.random.default_rng(seed)
        p = np.arange(1, E + 1, dtype=float) ** -1.1
        p /= p.sum()
        for li in range(NLAYER):
            v = np.concatenate([r.choice(E, 10, replace=False, p=p) for _ in range(5)])
            ids[li].copy_(torch.tensor(v, dtype=torch.int32))

    def step():
        for li in range(NLAYER):
            layers[li].launch(ids[li])

    # eager warm-up on a side stream, then capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for k in range(3):
            fill(1000 + k); step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    fill(2000)
    with torch.cuda.graph(g):
        step()
    torch.cuda.synchronize()
    print(f"captured a {NLAYER}-layer manage+gather graph "
          f"({2*NLAYER} kernel launches, {2} per layer)")

    base_step = int(layers[0].step.item())
    tbl0 = layers[0].table.clone()
    se0 = layers[0].slot_expert.clone()
    tot = 0
    for k in range(20):
        fill(3000 + k)
        g.replay()
        torch.cuda.synchronize()
        tot += sum(int(l.n_miss.item()) for l in layers)
    adv = int(layers[0].step.item()) - base_step
    print(f"20 replays: step counter advanced {adv} (want 20), "
          f"{tot} inserts total ({tot/20:.1f}/step over {NLAYER} layers)")
    ok = adv == 20 and tot > 0
    ok &= not torch.equal(tbl0, layers[0].table)
    ok &= not torch.equal(se0, layers[0].slot_expert)
    print("  cache mutated across replays:",
          "YES" if not torch.equal(se0, layers[0].slot_expert) else "NO (BROKEN)")

    # residency invariants after all that replaying
    for li, l in enumerate(layers):
        t = l.table.cpu().numpy(); c = l.map_cold.cpu().numpy(); se = l.slot_expert.cpu().numpy()
        assert (((t >= 0) & (c == -1)) | ((t < 0) & (c == np.arange(E)))).all(), li
        assert len(np.unique(se)) == len(se), li
        assert (t[se] == np.arange(S)).all(), li
    print("  table/map_cold complementary and slot_expert a bijection on all layers: OK")

    # bytes actually landed
    bad = 0
    l = layers[0]
    se = l.slot_expert.cpu().numpy()
    # only slots written since capture are verifiable; check the ones in the last miss list
    nm = int(l.n_miss.item())
    for j in range(nm):
        e, sl = [int(x) for x in l.miss[j].cpu()]
        for i, b in enumerate(SIZES):
            if not torch.equal(l.dst[i][sl*b:(sl+1)*b].cpu(), src_h[i][e*b:(e+1)*b]):
                bad += 1
    print(f"  bit-identity of the last replay's {nm} gathered experts x 6 buffers: "
          f"{'OK' if bad == 0 else str(bad) + ' MISMATCHES'}")
    ok &= bad == 0

    # ---- timing inside the graph -------------------------------------------------------
    def timeit(n=30):
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
        return a.elapsed_time(b) * 1000 / n

    fill(7777)
    g.replay(); torch.cuda.synchronize()          # let it settle into steady state
    steady = sum(int(l.n_miss.item()) for l in layers)
    us = timeit()
    print(f"\n[graph timing] {NLAYER} layers x (manage + gather), "
          f"{steady} inserts on the settled step")
    print(f"  {us:8.1f} us/step total  = {us/NLAYER:.2f} us/layer")
    # floor: same graph with routing that hits every time (no inserts at all)
    for _ in range(6):
        g.replay()
    torch.cuda.synchronize()
    zero = sum(int(l.n_miss.item()) for l in layers)
    us0 = timeit()
    print(f"  floor with {zero} inserts (identical routing replayed): {us0:8.1f} us/step "
          f"= {us0/NLAYER:.2f} us/layer  <- the price paid on a perfect-hit step")
    print("\n" + ("GRAPH TEST PASSED" if ok else "GRAPH TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
