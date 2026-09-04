"""CPU proof of the weight-layout algebra the kernel assumes:
byte (n,k) with n = 16*nt + r, k = 32*k32 + 16*d3 + 8*d4 + c lands at
  (((nt*(K/32) + k32)*2 + d4)*256 + r*16 + d3*8 + c)
in shuffle_weight_gfx1201(W), so lane l (r = l&15, d4 = l>>4) reading 16 contiguous bytes
at r*16 gets the d3=0 fragment in bytes 0..7 and the d3=1 fragment in bytes 8..15 of the
same 32-K step."""
import sys, torch
sys.path.insert(0, "/app/vllm")
from vllm.model_executor.kernels.linear.scaled_mm.fp8hip import shuffle_weight_gfx1201

N, K = 64, 128
W = torch.arange(N * K, dtype=torch.int32).remainder(251).to(torch.uint8).view(N, K)
S = shuffle_weight_gfx1201(W).contiguous().view(-1)
bad = 0
for nt in range(N // 16):
    for r in range(16):
        for k32 in range(K // 32):
            for d3 in range(2):
                for d4 in range(2):
                    for c in range(8):
                        n = 16 * nt + r
                        k = 32 * k32 + 16 * d3 + 8 * d4 + c
                        off = (((nt * (K // 32) + k32) * 2 + d4) * 256 + r * 16 + d3 * 8 + c)
                        if int(S[off]) != int(W[n, k]):
                            bad += 1
                            if bad < 5:
                                print("MISMATCH", nt, r, k32, d3, d4, c, int(S[off]), int(W[n, k]))
print("LAYOUT", "OK" if bad == 0 else f"FAILED ({bad} mismatches)")
sys.exit(1 if bad else 0)
