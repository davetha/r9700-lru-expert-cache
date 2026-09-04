"""Closes the loop between LOCALITY.md and the shipped kernel: replay the FULL captured
production trace through the REAL lru_manage kernel, with the REAL 15 GB profile hot set as
the warm start, and count the inserts it actually performs. If the kernel is the policy the
study simulated, this must land on the simulator's numbers (static 457.9 misses/step ->
LRU 52.6 inserts/step over 48 layers).

Manager only (no gather), so it needs no slot VRAM.
"""
import ctypes, json, os, time
import numpy as np, torch

LIB = os.environ.get("R4D_LRU_LIB", "/w/build/kernels/librlu.so")
lib = ctypes.CDLL(LIB)
lib.r4d_lru_manage.restype = ctypes.c_int
lib.r4d_lru_manage.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 9
DEV = "cuda:0"
PER = 1_228_800
GB = float(os.environ.get("GB", "15"))
THRESH = float(os.environ.get("THRESH", "0.5"))
MAXI = int(os.environ.get("MAXI", "64"))

prof = json.load(open(os.environ.get("PROFILE", "/hp/hot_profile.json")))["layers"]
L = len(prof)
E = 512
budget = int(GB * 1e9) // PER
lay, rk, co = [], [], []
for li in range(L):
    c = np.asarray(prof[str(li)]["counts"], float)
    lay.append(np.full(E, li)); rk.append(np.asarray(prof[str(li)]["ranked"])); co.append(c / c.sum())
lay, rk, co = map(np.concatenate, (lay, rk, co))
keep = np.argsort(-co, kind="stable")[:budget]
hot = {li: [] for li in range(L)}
for li, e in zip(lay[keep].tolist(), rk[keep].tolist()):
    hot[li].append(e)
hot = {li: sorted(v) for li, v in hot.items()}
caps = [len(hot[li]) for li in range(L)]
print(f"{GB:g} GB -> {budget} slots, per-layer min/med/max {min(caps)}/"
      f"{sorted(caps)[L//2]}/{max(caps)}, thresh {THRESH} max_inserts {MAXI}")

d = np.load(os.environ.get("TRACE", "/w/artifacts/routes_rank0.npz"))
ids_all = d["ids"]
K = int(d["topk"])
rows = (ids_all[:, 0] >= 0).sum(axis=1) // K
modal = np.bincount(rows).argmax()
sel = np.flatnonzero(rows == modal)          # decode steps only, same as the simulator
print(f"trace: {len(sel)} decode steps of {ids_all.shape[0]} captured")

state = []
for li in range(L):
    S = caps[li]
    h = torch.tensor(hot[li], dtype=torch.int64, device=DEV)
    table = torch.full((E,), -1, dtype=torch.int32, device=DEV)
    table[h] = torch.arange(S, dtype=torch.int32, device=DEV)
    mc = torch.arange(E, dtype=torch.int32, device=DEV)
    mc[h] = -1
    state.append(dict(
        S=S, hot=set(hot[li]), table=table, map_cold=mc,
        se=h.to(torch.int32).clone(), st=torch.zeros(S, dtype=torch.int64, device=DEV),
        routed=torch.zeros(E, dtype=torch.uint8, device=DEV),
        step=torch.zeros(1, dtype=torch.int64, device=DEV),
        miss=torch.full((MAXI, 2), -1, dtype=torch.int32, device=DEV),
        nm=torch.zeros(1, dtype=torch.int32, device=DEV),
        maxd=max(1, int(S * THRESH))))

ins = np.zeros(L, np.int64)
static_miss = np.zeros(L, np.int64)
routed_tot = np.zeros(L, np.int64)
t0 = time.time()
stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
for n, t in enumerate(sel):
    nm_batch = []
    for li in range(L):
        r = ids_all[t, li]
        r = r[r >= 0].astype(np.int32)
        u = np.unique(r)
        s = state[li]
        static_miss[li] += sum(1 for e in u if int(e) not in s["hot"])
        routed_tot[li] += len(u)
        ids_t = torch.tensor(r, dtype=torch.int32, device=DEV)
        rc = lib.r4d_lru_manage(
            ctypes.c_void_p(ids_t.data_ptr()), ids_t.numel(), E, s["S"], s["maxd"], MAXI,
            ctypes.c_void_p(s["table"].data_ptr()), ctypes.c_void_p(s["map_cold"].data_ptr()),
            ctypes.c_void_p(s["se"].data_ptr()), ctypes.c_void_p(s["st"].data_ptr()),
            ctypes.c_void_p(s["routed"].data_ptr()), ctypes.c_void_p(s["step"].data_ptr()),
            ctypes.c_void_p(s["miss"].data_ptr()), ctypes.c_void_p(s["nm"].data_ptr()), stream)
        assert rc == 0
        nm_batch.append(s["nm"])
    torch.cuda.synchronize()
    for li in range(L):
        ins[li] += int(nm_batch[li].item())
    if n % 400 == 0:
        print(f"  step {n}/{len(sel)}  ({time.time()-t0:.0f}s)", flush=True)

N = len(sel)
print(f"\nreplayed {N} steps x {L} layers in {time.time()-t0:.0f}s")
print(f"  static hot set : {static_miss.sum()/N:8.1f} misses/step  "
      f"{100*static_miss.sum()/routed_tot.sum():6.2f}%  "
      f"{static_miss.sum()/N*PER/1e6:7.1f} MB/step")
print(f"  LRU kernel     : {ins.sum()/N:8.1f} inserts/step  "
      f"{100*ins.sum()/routed_tot.sum():6.2f}%  "
      f"{ins.sum()/N*PER/1e6:7.1f} MB/step   "
      f"({100*(ins.sum()-static_miss.sum())/static_miss.sum():+.1f}% vs static)")
# residency invariants after 2329 real steps on all 48 layers
for li in range(L):
    s = state[li]
    tb = s["table"].cpu().numpy(); mc = s["map_cold"].cpu().numpy(); se = s["se"].cpu().numpy()
    assert (((tb >= 0) & (mc == -1)) | ((tb < 0) & (mc == np.arange(E)))).all(), li
    assert len(np.unique(se)) == len(se), li
    assert (tb[se] == np.arange(s["S"])).all(), li
    assert int(s["step"].item()) == N, (li, int(s["step"].item()))
print("  invariants hold on all 48 layers after the full trace")
