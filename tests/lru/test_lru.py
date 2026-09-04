"""Validation for the device-side LRU expert cache (librlu.so).

(a) policy: drive the REAL kernels with random routing and with the captured production
    trace, and check table / map_cold / slot_expert / slot_stamp / miss list / n_miss
    against an independent numpy LRU reference after EVERY step.
(b) data:   the gathered slabs must be bit-identical to the UVA source rows.
(c) timing: manager + gather at production shapes.

Runs on ONE GPU. Never touches GPUs 0/3 -- launch with HIP_VISIBLE_DEVICES=1,2 and hold
$REPO_ROOT/gpu.lock.
"""
import ctypes
import os
import sys

import numpy as np
import torch

LIB = os.environ.get("R4D_LRU_LIB", "/w/build/kernels/librlu.so")
lib = ctypes.CDLL(LIB)
hip = ctypes.CDLL("libamdhip64.so")

lib.r4d_lru_manage.restype = ctypes.c_int
lib.r4d_lru_manage.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + \
    [ctypes.c_void_p] * 8 + [ctypes.c_void_p]
lib.r4d_lru_gather.restype = ctypes.c_int
lib.r4d_lru_gather.argtypes = ([ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long] * 6 +
                               [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_int, ctypes.c_void_p])

DEV = "cuda:0"
FAIL = []


def chk(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("  FAIL:", msg)
    return cond


def uva(nbytes):
    """Pinned host buffer + its device-visible pointer (what --cpu-offload-params experts
    hands the kernels)."""
    t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dp = ctypes.c_void_p()
    rc = hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(t.data_ptr()),
                                     ctypes.c_uint(0))
    assert rc == 0, f"hipHostGetDevicePointer rc={rc}"
    return t, dp.value


class State:
    def __init__(self, E, S, hot):
        self.E, self.S = E, S
        self.table = torch.full((E,), -1, dtype=torch.int32, device=DEV)
        self.map_cold = torch.arange(E, dtype=torch.int32, device=DEV)
        self.slot_expert = torch.full((S,), -1, dtype=torch.int32, device=DEV)
        self.slot_stamp = torch.zeros(S, dtype=torch.int64, device=DEV)
        self.routed = torch.zeros(E, dtype=torch.uint8, device=DEV)
        self.step = torch.zeros(1, dtype=torch.int64, device=DEV)
        self.n_miss = torch.zeros(1, dtype=torch.int32, device=DEV)
        h = torch.tensor(hot, dtype=torch.int64, device=DEV)
        self.table[h] = torch.arange(len(hot), dtype=torch.int32, device=DEV)
        self.map_cold[h] = -1
        self.slot_expert[:len(hot)] = h.to(torch.int32)
        # reference mirrors
        self.r_table = self.table.cpu().numpy().copy()
        self.r_cold = self.map_cold.cpu().numpy().copy()
        self.r_se = self.slot_expert.cpu().numpy().copy()
        self.r_st = np.zeros(S, dtype=np.int64)
        self.r_step = 0

    def alloc_miss(self, cap):
        self.miss = torch.full((cap, 2), -7, dtype=torch.int32, device=DEV)
        self.cap = cap


def ref_step(s, ids, max_distinct, max_inserts):
    """Independent numpy model of lru_manage_k. Returns the expected miss list."""
    s.r_step += 1
    routed = np.unique(ids[ids >= 0])
    if len(routed) > max_distinct:
        return []
    misses = []
    for e in routed:                       # ascending expert id == the kernel's scan order
        sl = s.r_table[e]
        if sl >= 0:
            s.r_st[sl] = s.r_step
        else:
            misses.append(int(e))
    misses = misses[:max_inserts]
    rset = set(routed.tolist())
    out = []
    for e in misses:
        cand = [i for i in range(s.S) if s.r_se[i] < 0 or int(s.r_se[i]) not in rset]
        if not cand:
            break
        sl = min(cand, key=lambda i: (s.r_st[i], i))
        old = int(s.r_se[sl])
        if old >= 0:
            s.r_table[old] = -1
            s.r_cold[old] = old
        s.r_se[sl] = e
        s.r_st[sl] = s.r_step
        s.r_table[e] = sl
        s.r_cold[e] = -1
        out.append((e, sl))
    return out


def run_manage(s, ids_t, max_distinct, max_inserts):
    rc = lib.r4d_lru_manage(
        ctypes.c_void_p(ids_t.data_ptr()), ids_t.numel(), s.E, s.S,
        max_distinct, max_inserts,
        ctypes.c_void_p(s.table.data_ptr()), ctypes.c_void_p(s.map_cold.data_ptr()),
        ctypes.c_void_p(s.slot_expert.data_ptr()), ctypes.c_void_p(s.slot_stamp.data_ptr()),
        ctypes.c_void_p(s.routed.data_ptr()), ctypes.c_void_p(s.step.data_ptr()),
        ctypes.c_void_p(s.miss.data_ptr()), ctypes.c_void_p(s.n_miss.data_ptr()),
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    assert rc == 0, f"r4d_lru_manage rc={rc}"


def compare(s, exp, tag, step_no):
    torch.cuda.synchronize()
    nm = int(s.n_miss.item())
    ok = chk(nm == len(exp), f"{tag} step {step_no}: n_miss {nm} != ref {len(exp)}")
    if ok and nm:
        got = s.miss[:nm].cpu().numpy()
        want = np.array(exp, dtype=np.int32)
        chk((got == want).all(),
            f"{tag} step {step_no}: miss list {got.tolist()} != {want.tolist()}")
    chk((s.table.cpu().numpy() == s.r_table).all(), f"{tag} step {step_no}: table")
    chk((s.map_cold.cpu().numpy() == s.r_cold).all(), f"{tag} step {step_no}: map_cold")
    chk((s.slot_expert.cpu().numpy() == s.r_se).all(), f"{tag} step {step_no}: slot_expert")
    chk((s.slot_stamp.cpu().numpy() == s.r_st).all(), f"{tag} step {step_no}: slot_stamp")
    chk(int(s.step.item()) == s.r_step, f"{tag} step {step_no}: step counter")
    # invariant that makes the two GEMM calls disjoint and complete
    t = s.table.cpu().numpy(); c = s.map_cold.cpu().numpy()
    chk((((t >= 0) & (c == -1)) | ((t < 0) & (c == np.arange(s.E)))).all(),
        f"{tag} step {step_no}: table/map_cold are not complementary")
    se = s.slot_expert.cpu().numpy()
    live = se[se >= 0]
    chk(len(np.unique(live)) == len(live), f"{tag} step {step_no}: duplicate expert in slots")
    chk((t[live] == np.nonzero(se >= 0)[0]).all(), f"{tag} step {step_no}: table/slot mismatch")


def policy_suite(name, E, S, gen, nsteps, max_distinct, max_inserts, seed=0):
    print(f"\n[{name}] E={E} S={S} steps={nsteps} max_distinct={max_distinct} "
          f"max_inserts={max_inserts}")
    rng = np.random.default_rng(seed)
    hot = sorted(rng.choice(E, S, replace=False).tolist())
    s = State(E, S, hot)
    s.alloc_miss(max_inserts)
    before = len(FAIL)
    tot_miss = 0
    for n in range(nsteps):
        ids = gen(rng, n)
        ids_t = torch.tensor(ids.ravel(), dtype=torch.int32, device=DEV)
        exp = ref_step(s, ids.ravel(), max_distinct, max_inserts)
        run_manage(s, ids_t, max_distinct, max_inserts)
        compare(s, exp, name, n)
        tot_miss += len(exp)
        if len(FAIL) > before:
            print(f"  aborting {name} at step {n}")
            return s
    print(f"  OK: {nsteps} steps, {tot_miss} inserts ({tot_miss/nsteps:.2f}/step), "
          f"reference match on every field every step")
    return s


def main():
    torch.cuda.init()
    print("device:", torch.cuda.get_device_name(0), "| lib:", LIB)

    # ---- (a) policy vs numpy reference ------------------------------------------------
    # 1. tiny + high churn: every step misses, forces eviction on every step
    policy_suite("tiny-churn", 32, 8,
                 lambda rng, n: rng.choice(32, size=(2, 5), replace=True), 200, 8, 8)
    # 2. skewed routing at production-ish shape (5 rows x top-10, E=512)
    def skew(rng, n):
        p = np.arange(1, 513, dtype=float) ** -1.1
        p /= p.sum()
        return np.stack([rng.choice(512, size=10, replace=False, p=p) for _ in range(5)])
    policy_suite("skewed-512", 512, 257, skew, 300, 128, 64)
    # 3. cap pressure: more distinct misses than max_inserts allows
    policy_suite("insert-capped", 128, 32,
                 lambda rng, n: rng.choice(128, size=(8, 10), replace=True), 150, 128, 4)
    # 4. read-through gate: distinct above max_distinct must change nothing
    policy_suite("read-through", 128, 32,
                 lambda rng, n: rng.choice(128, size=(16, 10), replace=True), 60, 8, 16)
    # 5. padding sentinels (-1) must be ignored
    def padded(rng, n):
        a = rng.choice(64, size=(4, 10), replace=True)
        a[2:] = -1
        return a
    policy_suite("padded-rows", 64, 16, padded, 120, 64, 16)
    # 6. the real captured trace, real shapes
    tr = os.environ.get("TRACE", "/w/artifacts/routes_rank0.npz")
    if os.path.exists(tr):
        d = np.load(tr)
        ids_all = d["ids"]
        L = int(os.environ.get("TRACE_LAYER", "7"))
        rows = [ids_all[t, L][ids_all[t, L] >= 0].astype(np.int64) for t in range(400)]
        policy_suite("real-trace-L%d" % L, 512, 257,
                     lambda rng, n: rows[n].reshape(1, -1), len(rows), 128, 64)
    else:
        print("\n[real-trace] SKIPPED, no", tr)

    # ---- (b) gathered bytes must be bit-identical --------------------------------------
    print("\n[gather] bit-identity of the six per-expert buffers")
    E, S = 512, 257
    sizes = [819200, 409600, 51200, 640, 25600, 2560]   # w1 w2 ws1 ref1 ws2 ref2
    src_h, src_d, dst = [], [], []
    g = torch.Generator().manual_seed(1)
    for b in sizes:
        h, dp = uva(E * b)
        h.copy_(torch.randint(0, 256, (E * b,), dtype=torch.uint8, generator=g))
        src_h.append(h); src_d.append(dp)
        dst.append(torch.zeros(S * b, dtype=torch.uint8, device=DEV))
    st = State(E, S, sorted(np.random.default_rng(3).choice(E, S, replace=False).tolist()))
    st.alloc_miss(64)
    rng = np.random.default_rng(11)
    for n in range(40):
        ids = np.stack([rng.choice(512, size=10, replace=False) for _ in range(5)])
        exp = ref_step(st, ids.ravel(), 128, 64)
        ids_t = torch.tensor(ids.ravel(), dtype=torch.int32, device=DEV)
        run_manage(st, ids_t, 128, 64)
        compare(st, exp, "gather-drive", n)
        args = []
        for i in range(6):
            args += [ctypes.c_void_p(dst[i].data_ptr()), ctypes.c_void_p(src_d[i]),
                     ctypes.c_long(sizes[i])]
        rc = lib.r4d_lru_gather(*args, ctypes.c_void_p(st.miss.data_ptr()),
                                ctypes.c_void_p(st.n_miss.data_ptr()), 16, 64,
                                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        assert rc == 0, f"r4d_lru_gather rc={rc}"
        torch.cuda.synchronize()
        for (e, sl) in exp:
            for i, b in enumerate(sizes):
                got = dst[i][sl * b:(sl + 1) * b].cpu()
                want = src_h[i][e * b:(e + 1) * b]
                if not torch.equal(got, want):
                    chk(False, f"gather step {n} expert {e} slot {sl} buf {i} MISMATCH")
                    break
    # every slot the cache claims to hold must contain that expert's bytes
    torch.cuda.synchronize()
    se = st.slot_expert.cpu().numpy()
    nver = 0
    for sl in range(S):
        e = int(se[sl])
        if e < 0:
            continue
        i, b = 0, sizes[0]
        if torch.equal(dst[i][sl * b:(sl + 1) * b].cpu(), src_h[i][e * b:(e + 1) * b]):
            nver += 1
    print(f"  40 steps gathered; {nver}/{S} slots hold w1 bytes matching slot_expert "
          f"(warm-start slots were never written, so a mismatch there is expected)")

    # ---- (c) timing --------------------------------------------------------------------
    print("\n[timing] production shapes, one layer")
    per_expert = sum(sizes)
    for n_ins in (1, 2, 8, 52, 64):
        st.n_miss.fill_(n_ins)
        st.miss[:n_ins, 0] = torch.arange(n_ins, dtype=torch.int32, device=DEV)
        st.miss[:n_ins, 1] = torch.arange(n_ins, dtype=torch.int32, device=DEV)
        args = []
        for i in range(6):
            args += [ctypes.c_void_p(dst[i].data_ptr()), ctypes.c_void_p(src_d[i]),
                     ctypes.c_long(sizes[i])]
        tail = [ctypes.c_void_p(st.miss.data_ptr()), ctypes.c_void_p(st.n_miss.data_ptr()),
                16, 64, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)]
        for _ in range(3):
            lib.r4d_lru_gather(*args, *tail)
        torch.cuda.synchronize()
        a, b_ = torch.cuda.Event(True), torch.cuda.Event(True)
        stream = torch.cuda.current_stream()
        a.record(stream)
        R = 20
        for _ in range(R):
            lib.r4d_lru_gather(*args, *tail)
        b_.record(stream)
        torch.cuda.synchronize()
        us = a.elapsed_time(b_) * 1000 / R
        gb = n_ins * per_expert / (us * 1e-6) / 1e9
        print(f"  gather {n_ins:3d} experts: {us:8.1f} us  {gb:6.1f} GB/s  "
              f"({n_ins*per_expert/1e6:.2f} MB)")
    # manager alone, decode-shaped
    ids = torch.tensor(np.stack([np.random.default_rng(k).choice(512, 10, replace=False)
                                 for k in range(5)]).ravel(), dtype=torch.int32, device=DEV)
    for _ in range(5):
        run_manage(st, ids, 128, 64)
    torch.cuda.synchronize()
    a, b_ = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(torch.cuda.current_stream())
    for _ in range(200):
        run_manage(st, ids, 128, 64)
    b_.record(torch.cuda.current_stream())
    torch.cuda.synchronize()
    print(f"  manage (50 routings, steady state): {a.elapsed_time(b_)*1000/200:.1f} us/call")

    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILURES"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
