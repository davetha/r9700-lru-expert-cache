"""Is the new victim selection the SAME cache as the old serial argmin loop?

Runs the old library (serial argmin, always) and the new one (serial argmin
for <= NSER inserts, batched ranking above that) from bit-identical
pristine state on the same routing, and compares every byte of state they touch:
table, map_cold, slot_expert, slot_stamp, miss, n_miss.  Both r4d_lru_manage and
r4d_lru_fused.  Ends with a negative control so the comparator is known to bite.

  flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --ipc host --group-add video \
    --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v $REPO_ROOT:/w \
    --entrypoint bash local/q38fn-rocm10:k1build -c 'cd /w/tests/lru && python3 test_victim_equiv.py'
"""
import ctypes
import os
import sys

import numpy as np
import torch

OLD = os.environ.get("OLD_LIB", "/w/build/kernels/librlu_old.so")
NEW = os.environ.get("NEW_LIB", "/w/build/kernels/librlu.so")
DEV = "cuda"
BLOCK = 16
FAIL = 0


def bind(path):
    lib = ctypes.CDLL(path)
    lib.r4d_lru_manage.restype = ctypes.c_int
    lib.r4d_lru_manage.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 9
    lib.r4d_lru_fused.restype = ctypes.c_int
    lib.r4d_lru_fused.argtypes = ([ctypes.c_void_p] + [ctypes.c_int] * 5 +
                                  [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3 +
                                  [ctypes.c_void_p] * 7)
    return lib


def align_sizes(mk, E, bs=BLOCK):
    L = mk + E * (bs - 1)
    if mk < E:
        L = min(mk * bs, L)
    return L, (L + bs - 1) // bs


ORDER = ["table", "map_cold", "slot_expert", "slot_stamp", "routed", "step", "miss", "n_miss"]


class Case:
    def __init__(self, E, S, mk, k_cold, maxi, seed, resident=None, empty=0,
                 tie=False, md=None):
        rng = np.random.default_rng(seed)
        self.E, self.S, self.mk, self.maxi = E, S, mk, maxi
        self.md = E if md is None else md
        nres = S - empty if resident is None else resident
        hot = np.sort(rng.choice(E, nres, replace=False))
        cold = np.setdiff1d(np.arange(E), hot)
        k = min(k_cold, len(cold))
        nh = min(mk - k, len(hot))
        routed = np.concatenate([rng.choice(cold, k, replace=False),
                                 rng.choice(hot, nh, replace=False)])
        if len(routed) < mk:                       # pad with repeats, like real top-k
            routed = np.concatenate([routed, rng.choice(routed, mk - len(routed))])
        rng.shuffle(routed)
        self.ids = torch.as_tensor(routed[:mk].astype(np.int32), device=DEV)
        st = {}
        st["table"] = torch.full((E,), -1, dtype=torch.int32, device=DEV)
        ht = torch.as_tensor(hot, device=DEV).long()
        st["table"][ht] = torch.arange(nres, dtype=torch.int32, device=DEV)
        st["map_cold"] = torch.arange(E, dtype=torch.int32, device=DEV)
        st["map_cold"][ht] = -1
        se = torch.full((S,), -1, dtype=torch.int32, device=DEV)
        se[:nres] = ht.to(torch.int32)
        st["slot_expert"] = se
        stamps = (np.zeros(S, dtype=np.int64) if tie
                  else rng.permutation(S).astype(np.int64) + 1)
        st["slot_stamp"] = torch.as_tensor(stamps, device=DEV)
        st["routed"] = torch.zeros(E, dtype=torch.uint8, device=DEV)
        st["step"] = torch.full((1,), int(stamps.max()) + 1, dtype=torch.int64, device=DEV)
        st["miss"] = torch.full((maxi * 2,), -1, dtype=torch.int32, device=DEV)
        st["n_miss"] = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.pristine = st

    def fresh(self):
        return {n: self.pristine[n].clone() for n in ORDER}

    def run_manage(self, lib, md=None):
        st = self.fresh()
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = lib.r4d_lru_manage(ctypes.c_void_p(self.ids.data_ptr()), self.mk, self.E,
                                self.S, self.md if md is None else md, self.maxi,
                                *[ctypes.c_void_p(st[n].data_ptr()) for n in ORDER], stream)
        torch.cuda.synchronize()
        assert rc == 0, f"manage rc={rc}"
        return st

    def run_fused(self, lib, md=None):
        st = self.fresh()
        L, NB = align_sizes(self.mk, self.E)
        out = [torch.empty(L, dtype=torch.int32, device=DEV),
               torch.empty(NB, dtype=torch.int32, device=DEV),
               torch.empty(1, dtype=torch.int32, device=DEV),
               torch.empty(L, dtype=torch.int32, device=DEV),
               torch.empty(NB, dtype=torch.int32, device=DEV),
               torch.empty(1, dtype=torch.int32, device=DEV)]
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = lib.r4d_lru_fused(ctypes.c_void_p(self.ids.data_ptr()), self.mk, self.E,
                               self.S, self.md if md is None else md, self.maxi,
                               *[ctypes.c_void_p(st[n].data_ptr()) for n in ORDER],
                               BLOCK, L, NB,
                               *[ctypes.c_void_p(o.data_ptr()) for o in out], stream)
        torch.cuda.synchronize()
        assert rc == 0, f"fused rc={rc}"
        return st, out


def cmp_state(tag, a, b, nm):
    global FAIL
    bad = []
    for n in ORDER:
        if n == "routed":
            continue
        x, y = a[n], b[n]
        if n == "miss":                       # only the first n_miss pairs are defined
            x, y = x[:2 * nm], y[:2 * nm]
        if not torch.equal(x, y):
            d = int((x != y).sum())
            bad.append(f"{n}({d} differ)")
    if bad:
        FAIL += 1
        print(f"    FAIL {tag}: " + ", ".join(bad))
    return not bad


def main():
    old, new = bind(OLD), bind(NEW)
    print(f"old={OLD}\nnew={NEW}\n")
    cases = [
        ("prod B=1", dict(E=512, S=257, mk=50, k_cold=2, maxi=64, seed=1)),
        ("prod B=4", dict(E=512, S=257, mk=200, k_cold=13, maxi=64, seed=2)),
        ("at the cliff", dict(E=512, S=257, mk=50, k_cold=14, maxi=64, seed=3)),
        ("past the cliff", dict(E=512, S=257, mk=120, k_cold=40, maxi=64, seed=4)),
        ("zero misses", dict(E=512, S=257, mk=50, k_cold=0, maxi=64, seed=5)),
        ("serial n=4", dict(E=512, S=257, mk=50, k_cold=4, maxi=64, seed=18)),
        ("rank n=5", dict(E=512, S=257, mk=50, k_cold=5, maxi=64, seed=19)),
        ("serial, ev=0", dict(E=512, S=24, mk=60, k_cold=3, maxi=64, seed=20)),
        ("serial, ev=2", dict(E=512, S=26, mk=60, k_cold=4, maxi=64, seed=21)),
        ("capped by maxi", dict(E=512, S=257, mk=300, k_cold=200, maxi=8, seed=6)),
        ("tiny maxi", dict(E=512, S=257, mk=100, k_cold=60, maxi=3, seed=15)),
        ("empty slots", dict(E=512, S=257, mk=50, k_cold=10, maxi=64, seed=7, empty=100)),
        ("stamp ties", dict(E=512, S=257, mk=50, k_cold=12, maxi=64, seed=8, tie=True)),
        ("tiny cache", dict(E=512, S=16, mk=50, k_cold=30, maxi=64, seed=9)),
        ("evict-all", dict(E=512, S=24, mk=60, k_cold=50, maxi=64, seed=16)),
        ("evict exhausted", dict(E=512, S=40, mk=200, k_cold=60, maxi=64, seed=10)),
        ("S=1024", dict(E=1024, S=1024, mk=400, k_cold=100, maxi=64, seed=11,
                        resident=600)),
        ("maxi=1024", dict(E=1024, S=1024, mk=900, k_cold=500, maxi=1024,
                           seed=17, resident=400)),
        ("maxi=1", dict(E=512, S=257, mk=50, k_cold=20, maxi=1, seed=12)),
    ]
    print("[1] manage: serial loop vs pickVictims, identical pristine state")
    for name, kw in cases:
        c = Case(**kw)
        a, b = c.run_manage(old), c.run_manage(new)
        na, nb = int(a["n_miss"].item()), int(b["n_miss"].item())
        ok = (na == nb) and cmp_state(name, a, b, na)
        print(f"    {'ok  ' if ok else 'BAD '} {name:<16} n_miss old={na} new={nb}")
        if na != nb:
            globals()["FAIL"] = FAIL + 1

    print("\n[2] fused: same comparison, plus the align outputs must be untouched")
    for name, kw in cases:
        c = Case(**kw)
        a, oa = c.run_fused(old)
        b, ob = c.run_fused(new)
        na, nb = int(a["n_miss"].item()), int(b["n_miss"].item())
        ok = (na == nb) and cmp_state(name, a, b, na)
        for i, (x, y) in enumerate(zip(oa, ob)):
            if not torch.equal(x, y):
                ok = False
                print(f"    FAIL {name}: align output {i} differs")
        print(f"    {'ok  ' if ok else 'BAD '} {name:<16} n_miss old={na} new={nb}")

    print("\n[3] read-through (max_distinct = 0) must still insert nothing")
    c = Case(E=512, S=257, mk=50, k_cold=10, maxi=64, seed=13)
    a, b = c.run_manage(old, md=0), c.run_manage(new, md=0)
    ok = cmp_state("read-through", a, b, 0) and int(b["n_miss"].item()) == 0
    print(f"    {'ok' if ok else 'BAD'} n_miss={int(b['n_miss'].item())}")

    print("\n[4] negative control: the comparator must catch a perturbed state")
    c = Case(E=512, S=257, mk=50, k_cold=5, maxi=64, seed=14)
    a, b = c.run_manage(old), c.run_manage(new)
    n = int(a["n_miss"].item())
    caught = 0
    for name, fn in (("slot_stamp", lambda s: s["slot_stamp"].add_(1)),
                     ("table", lambda s: s["table"].add_(1)),
                     ("miss", lambda s: s["miss"][:2 * n].add_(1))):
        bb = {k: v.clone() for k, v in b.items()}
        fn(bb)
        before = FAIL
        cmp_state(f"control/{name}", a, bb, n)
        caught += (FAIL > before)
    globals()["FAIL"] = FAIL - caught          # controls are expected failures
    print(f"    caught {caught}/3 perturbations")
    if caught != 3:
        globals()["FAIL"] = FAIL + 1

    print("\n" + ("VICTIM EQUIVALENCE PASSED" if FAIL == 0 else f"FAILED ({FAIL})"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
