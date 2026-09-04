"""What does one insert cost in lru_manage_k?

The victim loop is serial: per insert it scans all S slots and then does a 256-thread
blkMin, which is 8 reduction rounds -> ~10 __syncthreads().  This measures the marginal
cost of an insert directly.

Method: restore the pristine state, launch the manager, N times, timed with events.  The
routing is fixed and contains exactly k experts that are not resident, so every call does
exactly k inserts.  t(k) - t(0) divided by N*k is the per-insert cost; the restore copies
and the launch overhead are identical in both arms and cancel.

  flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --ipc host --group-add video \
    --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v $REPO_ROOT:/w \
    --entrypoint bash local/q38fn-rocm10:k1build -c 'cd /w/tests/lru && python3 bench_victim.py'
"""
import ctypes
import os
import sys

import numpy as np
import torch

LIB = os.environ.get("R4D_LRU_LIB", "/w/build/kernels/librlu.so")
SYM = os.environ.get("SYM", "r4d_lru_manage")
lib = ctypes.CDLL(LIB)
fn = getattr(lib, SYM)
fn.restype = ctypes.c_int
fn.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 9

DEV = "cuda"
E = int(os.environ.get("E", "512"))
S = int(os.environ.get("S", "257"))
MK = int(os.environ.get("MK", "50"))
MAXI = 64
N = int(os.environ.get("N", "20"))
ROUNDS = int(os.environ.get("ROUNDS", "5"))
KS = [int(v) for v in os.environ.get("KS", "0,1,2,4,8,16,32").split(",")]


SEED = int(os.environ.get("SEED", "0"))


def build(k, seed=SEED):
    """State lives in one contiguous buffer so restoring it is a single copy node."""
    rng = np.random.default_rng(seed)
    hot = np.sort(rng.choice(E, S, replace=False))
    cold = np.setdiff1d(np.arange(E), hot)
    assert k <= len(cold) and MK - k <= S
    routed = np.concatenate([rng.choice(cold, k, replace=False),
                             rng.choice(hot, MK - k, replace=False)])
    rng.shuffle(routed)

    spec = [("slot_stamp", S, torch.int64), ("step", 1, torch.int64),
            ("table", E, torch.int32), ("map_cold", E, torch.int32),
            ("slot_expert", S, torch.int32), ("miss", MAXI * 2, torch.int32),
            ("n_miss", 1, torch.int32), ("routed", E, torch.uint8)]
    sz = {torch.int64: 8, torch.int32: 4, torch.uint8: 1}
    nb = sum(n * sz[d] for _, n, d in spec)
    nb = (nb + 7) // 8 * 8
    buf = torch.zeros(nb, dtype=torch.uint8, device=DEV)
    st, off = {}, 0
    for name, n, d in spec:
        st[name] = buf[off:off + n * sz[d]].view(d)
        off += n * sz[d]

    ht = torch.as_tensor(hot, device=DEV).long()
    st["table"].fill_(-1)
    st["table"][ht] = torch.arange(S, dtype=torch.int32, device=DEV)
    st["map_cold"].copy_(torch.arange(E, dtype=torch.int32, device=DEV))
    st["map_cold"][ht] = -1
    st["slot_expert"].copy_(ht.to(torch.int32))
    # distinct stamps so the victim order is a total order, like production
    st["slot_stamp"].copy_(torch.as_tensor(rng.permutation(S) + 1, device=DEV).to(torch.int64))
    st["step"].fill_(S + 1)
    st["miss"].fill_(-1)
    st["n_miss"].zero_()
    st["routed"].zero_()
    ids = torch.as_tensor(routed, device=DEV).to(torch.int32)
    return st, ids, buf


ORDER = ["table", "map_cold", "slot_expert", "slot_stamp", "routed", "step", "miss", "n_miss"]


def call(st, ids, md=S):
    return fn(ctypes.c_void_p(ids.data_ptr()), MK, E, S, md, MAXI,
              *[ctypes.c_void_p(st[n].data_ptr()) for n in ORDER],
              ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))


def prep(k):
    st, ids, buf = build(k)
    pristine = buf.clone()
    rc = call(st, ids)
    torch.cuda.synchronize()
    assert rc == 0, rc
    got = int(st["n_miss"].item())
    assert got == k, f"k={k} but n_miss={got}"
    return st, ids, buf, pristine


REP = int(os.environ.get("REP", "100"))


def graph(arm):
    """R iterations of [restore state, manage] captured once.  Eager launches cost ~47 us
    of CPU each, which hides the whole kernel; inside a graph there is no host in the loop."""
    st, ids, buf, pristine = arm
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            buf.copy_(pristine)
            call(st, ids)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(REP):
            buf.copy_(pristine)
            call(st, ids)
    return g


def main():
    print(f"lib={LIB} sym={SYM} E={E} S={S} mk={MK} N={N} rep={REP} rounds={ROUNDS}")
    arms = {k: prep(k) for k in KS}
    gs = {k: graph(arms[k]) for k in KS}
    for k in KS:
        gs[k].replay()
    torch.cuda.synchronize()
    times = {k: [] for k in KS}
    for _ in range(ROUNDS):                       # round-robin so clock drift hits all arms
        for k in KS:
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(N):
                gs[k].replay()
            b.record()
            torch.cuda.synchronize()
            times[k].append(a.elapsed_time(b) * 1e3 / (N * REP))
    med = {k: float(min(times[k])) for k in KS}   # min: interference only adds
    base = med[KS[0]]
    print(f"{'inserts':>8} {'us/call':>9} {'spread':>8} {'-t(0)':>9} {'us/insert':>10}")
    for k in KS:
        sp = max(times[k]) - min(times[k])
        print(f"{k:>8} {med[k]:>9.3f} {sp:>8.3f} {med[k]-base:>9.3f} "
              f"{'-' if k == KS[0] else f'{(med[k]-base)/k:>10.3f}'}")
    ks = np.array([k for k in KS], dtype=float)
    ts = np.array([med[k] for k in KS])
    slope, icept = np.polyfit(ks, ts, 1)
    print(f"\n  least squares: {slope:.3f} us/insert, intercept {icept:.3f} us")


if __name__ == "__main__":
    main()
