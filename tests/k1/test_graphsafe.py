"""Prove the two claims routecap.py rests on, before anyone runs it against a live server.

  1. a Python body inside BaseRouter.capture_fn runs ONLY at graph capture, never on replay
     -- so histext.py-style Python histogramming silently records one step and freezes;
  2. the tensor-op ring buffer routecap uses DOES replay: the device step counter advances,
     the slot index follows it, and each step's topk_ids land in the right slot.
"""
import os
os.environ.setdefault("ROUTECAP_RING", "8")
os.environ.setdefault("ROUTECAP_MAXROWS", "4")
os.environ.setdefault("ROUTECAP_DUMP_S", "36000")
os.environ.setdefault("ROUTECAP_DIR", "/w/artifacts/_test")

import numpy as np
import torch
import routecap

RING, MAXROWS, TOPK, ROWS, L = 8, 4, 10, 2, 2
routecap._routers.extend([None] * L)
fns = [routecap._make_fn(i) for i in range(L)]

dev = "cuda:0"
torch.zeros(1, device=dev)
static = torch.zeros(ROWS, TOPK, dtype=torch.int32, device=dev)
py = {"n": 0}


def body():
    py["n"] += 1                       # the Python side-effect under test
    for f in fns:
        f(static)


# eager warm-up: this is what allocates the ring (never allocate inside a capture)
static.fill_(7)
body()
st = routecap._st
assert st is not None, "ring was not allocated on the eager path"
print(f"eager: ctr={int(st.ctr.item())} python_calls={py['n']}")

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        body()
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    body()
torch.cuda.synchronize()
ctr_after_capture = int(st.ctr.item())
py_after_capture = py["n"]

NREP = 5
expect = {}
for r in range(NREP):
    static.copy_(torch.arange(ROWS * TOPK, device=dev, dtype=torch.int32).reshape(ROWS, TOPK)
                 + 100 * r)
    g.replay()
    torch.cuda.synchronize()
    expect[ctr_after_capture + 1 + r] = (np.arange(ROWS * TOPK) + 100 * r).astype(np.int16)

ctr = int(st.ctr.item())
steps = st.step.cpu().numpy()
ring = st.ring.cpu().numpy()
print(f"after {NREP} replays: device ctr {ctr_after_capture} -> {ctr} "
      f"(advanced {ctr - ctr_after_capture}), python calls {py_after_capture} -> {py['n']}")

ok_py = py["n"] == py_after_capture
ok_ctr = ctr - ctr_after_capture == NREP
print(f"  [1] python body did NOT re-run on replay : {ok_py}")
print(f"  [2] device step counter advanced per replay: {ok_ctr}")

bad = []
for step, payload in expect.items():
    slot = step % RING
    if int(steps[slot]) != step:
        bad.append(f"step {step}: step-ring slot {slot} holds {int(steps[slot])}")
        continue
    for l in range(L):
        got = ring[l, slot, : ROWS * TOPK]
        if not np.array_equal(got, payload):
            bad.append(f"step {step} layer {l}: {got[:4]} != {payload[:4]}")
        if not (ring[l, slot, ROWS * TOPK:] == -1).all():
            bad.append(f"step {step} layer {l}: padding not -1")
print(f"  [3] every replayed step's topk_ids landed in slot=step%RING with -1 padding: "
      f"{not bad}")
for b in bad:
    print("      " + b)

f = routecap.flush()
print(f"  [4] flush wrote {f}")
raise SystemExit(0 if (ok_py and ok_ctr and not bad) else 1)
