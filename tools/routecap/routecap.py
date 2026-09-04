"""RouteCap -- capture every MoE layer's topk_ids, per decode step, under HIP graphs.

Why this is not histext.py
--------------------------
`BaseRouter.capture_fn` is a *Python* callable invoked from `_select_experts`. Production runs
with cudagraph_mode FULL, so that Python body executes only while the graph is being CAPTURED;
on every subsequent replay the GPU just re-runs the recorded kernels and no Python happens.
A histogram written with `index_add_` from Python therefore records one step and then freezes.

So the capture fn issues only *tensor ops*, which ARE recorded into the graph and DO replay:

    step counter   ctr += 1                      (layer 0 only, so once per step)
    slot           slot = ctr % RING             (device tensor -- a Python int would be baked in)
    pad            buf.fill_(-1)                 (sentinel; rows vary between graphs)
    payload        buf[:rows*k].copy_(topk_ids)
    store          ring[layer].index_copy_(0, slot, buf)

A background thread then D2Hs the ring on its own stream every ROUTECAP_DUMP_S seconds and
accumulates completed slots on the host, so traces longer than RING steps are not lost.

Installation is automatic: `--worker-extension-cls routecap.RouteCap` makes vLLM
`resolve_obj_by_qualname` this module inside every worker, BEFORE the model is built, and the
module-level patch of `BaseRouter.__init__` then catches every router as it is constructed.
No collective_rpc is needed, so it works behind the plain HTTP server.

Env:
    ROUTECAP=1                 master switch (default on when the module is imported)
    ROUTECAP_DIR=/w/tools/routecap         where routes_rank<N>.npz is written
    ROUTECAP_RING=1024         steps held on device (1024 x 48 x 32 x 10 int16 = 31.5 MiB)
    ROUTECAP_MAXROWS=32        steps with more rows than this are not captured (prefill)
    ROUTECAP_DUMP_S=20         seconds between host flushes
"""
import os
import threading
import time

import numpy as np
import torch

ENABLE = os.environ.get("ROUTECAP", "1") == "1"
OUT_DIR = os.environ.get("ROUTECAP_DIR", "/w/tools/routecap")
RING = int(os.environ.get("ROUTECAP_RING", "1024"))
MAXROWS = int(os.environ.get("ROUTECAP_MAXROWS", "32"))
DUMP_S = float(os.environ.get("ROUTECAP_DUMP_S", "20"))

_routers = []          # every BaseRouter, in construction order == layer order
_st = None             # device state, allocated lazily on the first EAGER call
_lock = threading.Lock()


class _State:
    def __init__(self, dev, nlayer, topk):
        self.dev, self.nlayer, self.topk = dev, nlayer, topk
        self.width = MAXROWS * topk
        self.ctr = torch.zeros(1, dtype=torch.int64, device=dev)
        self.ring = torch.full((nlayer, RING, self.width), -1,
                               dtype=torch.int16, device=dev)
        self.step = torch.full((RING,), -1, dtype=torch.int64, device=dev)
        self.buf = torch.full((nlayer, self.width), -1, dtype=torch.int16, device=dev)
        self.dump_stream = torch.cuda.Stream(device=dev)
        # host side
        self.saved_steps = []
        self.saved_ids = []
        self.high = -1          # highest step index already saved
        self.dumps = 0
        self.wrapped = 0


def _ensure(dev, topk):
    """Allocate the ring on the first eager call. NEVER allocate inside a graph capture."""
    global _st
    if _st is not None:
        return _st
    if torch.cuda.is_current_stream_capturing():
        return None
    with _lock:
        if _st is None:
            _st = _State(dev, len(_routers), topk)
            t = threading.Thread(target=_dumper, daemon=True, name="routecap")
            t.start()
    return _st


def _make_fn(idx):
    def fn(topk_ids):
        if not ENABLE:
            return
        st = _st or _ensure(topk_ids.device, topk_ids.shape[-1])
        if st is None or idx >= st.nlayer:
            return
        rows, k = topk_ids.shape[0], topk_ids.shape[-1]
        if rows > MAXROWS or k != st.topk:
            return                      # prefill / an unexpected shape: not our business
        if idx == 0:
            st.ctr.add_(1)
            st.step.index_copy_(0, st.ctr % RING, st.ctr)
        slot = st.ctr % RING
        b = st.buf[idx]
        b.fill_(-1)
        b[: rows * k].copy_(topk_ids.reshape(-1).to(torch.int16))
        st.ring[idx].index_copy_(0, slot, b.view(1, -1))
    return fn


def _rank():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:                                              # noqa: BLE001
        pass
    for v in ("VLLM_WORKER_RANK", "RANK", "LOCAL_RANK"):
        if os.environ.get(v):
            return int(os.environ[v])
    return torch.cuda.current_device()


def flush():
    """D2H the ring on a side stream and append every newly-completed slot to the host trace."""
    st = _st
    if st is None:
        return None
    torch.cuda.set_device(st.dev)
    with torch.cuda.stream(st.dump_stream):
        ctr = int(st.ctr.item())
        steps = st.step.to("cpu", copy=True).numpy()
        ring = st.ring.to("cpu", copy=True).numpy()
    st.dump_stream.synchronize()
    # a slot is complete once the step AFTER it has started; the newest slot may be mid-write
    order = np.argsort(steps)
    for s in order:
        v = int(steps[s])
        if v <= st.high or v < 0 or v >= ctr:
            continue
        st.saved_steps.append(v)
        st.saved_ids.append(ring[:, s, :].copy())
        st.high = v
    if st.saved_steps and st.saved_steps[-1] - len(st.saved_steps) + 1 > 0:
        st.wrapped = st.saved_steps[-1] + 1 - len(st.saved_steps)   # steps lost to wrap
    st.dumps += 1
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"routes_rank{_rank()}.npz")
    tmp = p + ".tmp"
    with open(tmp, "wb") as fh:          # a bare path would get ".npz" appended to ".tmp"
        np.savez(fh,
                 steps=np.asarray(st.saved_steps, dtype=np.int64),
                 ids=(np.stack(st.saved_ids) if st.saved_ids
                      else np.zeros((0, st.nlayer, st.width), dtype=np.int16)),
                 nlayer=np.int64(st.nlayer), topk=np.int64(st.topk),
                 maxrows=np.int64(MAXROWS), ring=np.int64(RING),
                 ctr=np.int64(ctr), lost_to_wrap=np.int64(st.wrapped))
    os.replace(tmp, p)
    return dict(path=p, ctr=ctr, saved=len(st.saved_steps), lost_to_wrap=st.wrapped)


def _dumper():
    while True:
        time.sleep(DUMP_S)
        try:
            flush()
        except Exception as exc:                                   # noqa: BLE001
            print(f"[routecap] flush failed: {type(exc).__name__}: {exc}", flush=True)


def _install():
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    if getattr(BaseRouter, "_routecap_patched", False):
        return
    orig = BaseRouter.__init__

    def patched(self, *a, **kw):
        orig(self, *a, **kw)
        idx = len(_routers)
        _routers.append(self)
        self.capture_fn = _make_fn(idx)

    BaseRouter.__init__ = patched
    BaseRouter._routecap_patched = True
    print(f"[routecap] patched BaseRouter.__init__ (ring={RING} maxrows={MAXROWS} "
          f"dir={OUT_DIR} dump={DUMP_S}s)", flush=True)


class RouteCap:
    """Worker extension. Its only job is to get this module imported in the worker;
    the capture is installed by the module-level patch above. The two methods are
    available over collective_rpc if a driver-side caller ever wants them."""

    def routecap_flush(self):
        return flush()

    def routecap_status(self):
        st = _st
        if st is None:
            return dict(installed=len(_routers), armed=False)
        return dict(installed=len(_routers), armed=True, nlayer=st.nlayer, topk=st.topk,
                    ctr=int(st.ctr.item()), saved=len(st.saved_steps), dumps=st.dumps)


if ENABLE:
    _install()
