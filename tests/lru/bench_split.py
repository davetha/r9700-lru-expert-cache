"""Where do the added microseconds go? Time manage-only, gather-only (0 inserts) and both,
inside a HIP graph, and sweep the gather grid."""
import ctypes, os
import numpy as np, torch
exec(open("/w/tests/lru/test_graph_lru.py").read().split("def main()")[0])

def graph_time(fn, n=50):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    st = torch.cuda.current_stream(); a.record(st)
    for _ in range(n): g.replay()
    b.record(st); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / n

rng = np.random.default_rng(0)
src_h, src_d = [], []
for b in SIZES:
    h, dp = uva(E * b); src_h.append(h); src_d.append(dp)
NL = 4
layers = [Layer(rng, src_d) for _ in range(NL)]
ids = [torch.zeros(50, dtype=torch.int32, device=DEV) for _ in range(NL)]
r = np.random.default_rng(5)
for li in range(NL):
    # route ONLY to experts that are already resident -> steady state, zero inserts
    res = layers[li].slot_expert.cpu().numpy()[:50]
    ids[li].copy_(torch.tensor(res.astype(np.int32)))

def mk(nl, do_manage, do_gather, chunks, lanes):
    def f():
        st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        for li in range(nl):
            l = layers[li]
            if do_manage:
                lib.r4d_lru_manage(
                    ctypes.c_void_p(ids[li].data_ptr()), 50, E, S, 128, MAXI,
                    ctypes.c_void_p(l.table.data_ptr()), ctypes.c_void_p(l.map_cold.data_ptr()),
                    ctypes.c_void_p(l.slot_expert.data_ptr()),
                    ctypes.c_void_p(l.slot_stamp.data_ptr()),
                    ctypes.c_void_p(l.routed.data_ptr()), ctypes.c_void_p(l.step.data_ptr()),
                    ctypes.c_void_p(l.miss.data_ptr()), ctypes.c_void_p(l.n_miss.data_ptr()), st)
            if do_gather:
                args = []
                for i in range(6):
                    args += [ctypes.c_void_p(l.dst[i].data_ptr()),
                             ctypes.c_void_p(l.src_d[i]), ctypes.c_long(SIZES[i])]
                lib.r4d_lru_gather(*args, ctypes.c_void_p(l.miss.data_ptr()),
                                   ctypes.c_void_p(l.n_miss.data_ptr()), chunks, lanes, st)
    return f

print("per-layer us, zero-insert steady state, inside a HIP graph (%d layers, scaled)" % NL)
for label, dm, dg in (("manage only", 1, 0), ("gather only (empty)", 0, 1), ("both", 1, 1)):
    t = graph_time(mk(NL, dm, dg, CHUNKS, LANES))
    print("  %-22s %6.2f us/layer  -> %6.0f us/step over 48 layers" % (label, t/NL, t/NL*48))
print("\ngather grid sweep (empty gather only):")
for chunks in (4, 8, 16, 32):
    for lanes in (8, 16, 32, 64):
        t = graph_time(mk(NL, 0, 1, chunks, lanes))
        print("  chunks=%2d lanes=%2d (%5d blocks): %5.2f us/layer" %
              (chunks, lanes, chunks*lanes, t/NL))
