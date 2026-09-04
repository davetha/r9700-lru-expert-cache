"""Does the fused kernel actually pay? Two graphs, same layers, same filler:

  A (today)  52 x [ lru_manage, lru_gather, moe_align(table), moe_align(map_cold) ]  6 kernels
  B (fused)  52 x [ lru_fused,  lru_gather ]                                          2 kernels

plus NFILL identical filler nodes so both sit at a realistic graph size. The difference is
what production would save: 4 launches per layer, minus whatever extra execution the fused
kernel costs by doing the align work in one workgroup.

  flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --ipc host --group-add video \
    --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v $REPO_ROOT:/w \
    --entrypoint bash local/q38fn-rocm10:k1build -c 'cd /w/tests/lru && python3 bench_fused.py'
"""
import ctypes
import os
import sys

os.environ.setdefault("NLAYER", "52")
sys.path.insert(0, "/w/tools/routecap")
sys.path.insert(0, "/w/tests/lru")

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import \
    moe_align_block_size                                              # noqa: E402

import step0_nodecost as Z                                            # noqa: E402
import test_graph_lru as G                                            # noqa: E402
import test_fused as F                                                # noqa: E402

DEV = "cuda"
NLAYER = G.NLAYER
E, S = G.E, G.S
M = int(os.environ.get("M", "5"))
TOPK = int(os.environ.get("TOPK", "10"))
BLOCK = 16
NFILL = int(os.environ.get("NFILL", "2886"))
ROUNDS = int(os.environ.get("ROUNDS", "5"))
MD = max(1, int(S * 0.5))


class B(Z.Bench):
    def __init__(self, fill_elems):
        super().__init__(fill_elems)
        mk = M * TOPK
        self.mk = mk
        self.L, self.NB = F.sizes(mk)
        self.out = [[torch.empty(self.L, dtype=torch.int32, device=DEV),
                     torch.empty(self.NB, dtype=torch.int32, device=DEV),
                     torch.empty(1, dtype=torch.int32, device=DEV),
                     torch.empty(self.L, dtype=torch.int32, device=DEV),
                     torch.empty(self.NB, dtype=torch.int32, device=DEV),
                     torch.empty(1, dtype=torch.int32, device=DEV)]
                    for _ in range(NLAYER)]

    def body(self, k, trivial=False):
        """k=0 -> arm A (manage + gather + 2 aligns), k=1 -> arm B (fused + gather)"""
        keep = []
        per = NFILL // NLAYER
        extra = NFILL - per * NLAYER
        for li in range(NLAYER):
            L = self.layers[li]
            st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            args = [ctypes.c_void_p(x.data_ptr()) for x in
                    (L.table, L.map_cold, L.slot_expert, L.slot_stamp,
                     L.routed, L.step, L.miss, L.n_miss)]
            if k == 0:
                rc = F.lib.r4d_lru_manage(ctypes.c_void_p(self.ids[li].data_ptr()),
                                          self.mk, E, S, MD, G.MAXI, *args, st)
                assert rc == 0
            else:
                rc = F.lib.r4d_lru_fused(
                    ctypes.c_void_p(self.ids[li].data_ptr()), self.mk, E, S, MD, G.MAXI,
                    *args, BLOCK, self.L, self.NB,
                    *[ctypes.c_void_p(t.data_ptr()) for t in self.out[li]], st)
                assert rc == 0
            ga = []
            for i in range(6):
                ga += [ctypes.c_void_p(L.dst[i].data_ptr()),
                       ctypes.c_void_p(L.src_d[i]), ctypes.c_long(G.SIZES[i])]
            rc = F.lib.r4d_lru_gather(*ga, ctypes.c_void_p(L.miss.data_ptr()),
                                      ctypes.c_void_p(L.n_miss.data_ptr()),
                                      G.CHUNKS, G.LANES, st)
            assert rc == 0
            if k == 0:
                keep.append(moe_align_block_size(self.ids2d[li], BLOCK, E, L.table,
                                                 ignore_invalid_experts=True))
                keep.append(moe_align_block_size(self.ids2d[li], BLOCK, E, L.map_cold,
                                                 ignore_invalid_experts=True))
            n = per + (1 if li < extra else 0)
            for _ in range(n):
                torch.add(self.fa, self.fb, out=self.fc)
        return keep


def main():
    print("calibrating the filler kernel:")
    fill_elems, fill_us = Z.calibrate()
    b = B(fill_elems)
    print(f"\n{NLAYER} layers, mk={b.mk}, {NFILL} filler nodes at {fill_us:.2f} us")
    graphs = [b.capture(0), b.capture(1)]
    nodes = [NLAYER * 6 + NFILL, NLAYER * 2 + NFILL]
    samples = [[], []]
    for r in range(ROUNDS):
        for k, g in enumerate(graphs):
            samples[k].append(Z.timeit(g))
    t = [float(np.median(v)) for v in samples]
    names = ["A today  (manage+gather+2 aligns)", "B fused  (fused+gather)         "]
    for k in range(2):
        v = np.array(samples[k])
        print(f"  {names[k]}  {nodes[k]:>5} nodes  {t[k]:9.1f} us/step "
              f"(spread {100*(v.max()-v.min())/t[k]:.1f}%)")
    d = t[0] - t[1]
    print(f"\nsaved: {d:.0f} us/step = {d/1000:.3f} ms/step over {NLAYER} layers "
          f"({d/NLAYER:.2f} us/layer, {nodes[0]-nodes[1]} nodes removed)")
    print(f"  pure-dispatch value of those nodes at 2.40 us/node: "
          f"{2.40*(nodes[0]-nodes[1])/1000:.3f} ms/step")
    print(f"  -> the fused kernel's extra execution costs "
          f"{(2.95*(nodes[0]-nodes[1]) - d)/NLAYER:.2f} us/layer vs the 5 kernels it absorbs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
