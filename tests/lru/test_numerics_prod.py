"""test_numerics_lru at PRODUCTION shapes.

test_numerics_lru ran E=64, S=24, mtk=40. That already answers the interesting question --
"does it matter which of the two calls computes a given (token,k) row?" -- because it compares
the two-call split against ONE all-UVA call over the same routing, and the LRU kernels reshuffle
the partition on every step. It came out bit-identical, i.e. the r4d grouped GEMM is
partition-invariant at that size.

What it did NOT cover, and what the live server actually runs:
  E=512, S~257 slots/layer, MAX_INSERTS=64, and the mtk values a real step produces --
  mtk=50   one sequence, target forward with 4 MTP drafts (5 rows x top_k 10)
  mtk=10   one sequence, a single MTP draft-model forward
  mtk=200  four concurrent sequences (B=4 x 5 rows)
  mtk=20480 a 2048-token prefill chunk (NBT 2048), where the read-through gate is engaged
None of those is a multiple of BLOCK=16 except the prefill one, and the expert-block count in
each align is an order of magnitude larger than anything covered so far.

  flock -w 3600 $REPO_ROOT/gpu.lock docker run --rm --ipc host --group-add video \
    --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=1 -v $REPO_ROOT:/w \
    --entrypoint bash local/q38fn-rocm10:k1build -c 'cd /w/tests/lru && python3 test_numerics_prod.py'
"""
import os
import sys

os.environ["E"] = os.environ.get("E", "512")
os.environ["TOPK"] = os.environ.get("TOPK", "10")
sys.path.insert(0, "/w/tools/routecap")
sys.path.insert(0, "/w/tests/lru")

import test_numerics_lru as T                                        # noqa: E402
import torch                                                         # noqa: E402


def main():
    torch.cuda.init()
    print("device:", torch.cuda.get_device_name(0))
    print(f"E={T.E} top_k={T.TOPK} BLOCK={T.BLOCK}")
    S = int(os.environ.get("SLOTS", "257"))
    NS = int(os.environ.get("STEPS", "30"))
    # (label, N, K) -- the two production GEMMs, from the repack log
    for gname, N, K in (("gate_up", 640, 2560), ("down", 2560, 320)):
        for mlabel, M in (("decode-mtp4", 5), ("draft", 1), ("B4-decode", 20)):
            T.run_case(f"{gname}-{mlabel}", N, K, S, NS, M)
    # prefill chunk: wide enough to trip the read-through gate (distinct > S/2), so the
    # partition is frozen for the step -- exercises the "no inserts" path at a huge mtk.
    if os.environ.get("PREFILL", "1") == "1":
        for gname, N, K in (("gate_up", 640, 2560), ("down", 2560, 320)):
            T.run_case(f"{gname}-prefill2048", N, K, S, 3, 2048, MAXD=S // 2)
    print("\n" + ("PROD NUMERICS PASSED (bit-identical)" if not T.FAILS
                  else f"{len(T.FAILS)} FAILURES: {T.FAILS[:5]}"))
    return 1 if T.FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
