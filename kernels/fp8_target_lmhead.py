# SPDX-License-Identifier: Apache-2.0
"""fp8_target_lmhead: 128x128 block-scaled fp8 lm_head for the TARGET logits on gfx1201.

The target head is one 124160 x 2560 bf16 GEMM per decode step, 636 MB per rank read at
the bandwidth roofline (~1 ms of a ~29 ms step). Halving its bytes with e4m3 weights and
128x128 block scales is the same operand set the attention/GDN projections already use
through the closed ``fp8hip_gemm_w8a8_launch`` (K3: at 91-96% of a pure-read control), so
this module only quantizes the weight ONCE at first use and re-routes small-M calls.

Unlike the draft W4 head this DOES change the target distribution slightly (block-scaled
e4m3 on the logit projection). It is therefore env-gated (VLLM_TARGET_FP8_LMHEAD=1) and the
adoption call rests on the teacher-forced logprob probe against the restart-noise floor.

Quantization: per 128(N) x 128(K) block, scale = absmax / 448, w_q = round(w / scale) in
e4m3 (OCP e4m3fn; gfx12 uses the OCP format, not fnuz). Activation: per-token, per-128-K
group dynamic e4m3 via vLLM's per_token_group_quant_fp8, exactly what the fp8 linear path
does. The weight is stored pre-shuffled in gfx1201 WMMA fragment order (shuffle_weight_gfx1201)
with the canonical (N, K) view, as fp8hip requires.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_BLK = 128
_MAX_M = int(os.environ.get("VLLM_TARGET_FP8_LMHEAD_MAX_M", "64"))


def available() -> bool:
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.fp8hip import (  # noqa: F401
            Fp8HipBlockScaledMMKernel, shuffle_weight_gfx1201)
    except Exception:  # noqa: BLE001
        return False
    ok, _ = Fp8HipBlockScaledMMKernel.is_supported()
    return bool(ok)


@torch.no_grad()
def pack_fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (shuffled e4m3 [N, K], fp32 block scales [N/128, K/128])."""
    from vllm.model_executor.kernels.linear.scaled_mm.fp8hip import shuffle_weight_gfx1201
    N, K = w.shape
    assert N % _BLK == 0 and K % _BLK == 0, (N, K)
    # Row-chunked so the fp32 temporaries stay ~40 MB: a whole-weight .float() is 1.27 GB,
    # and because this runs inside vLLM's memory-profiling forward it would be booked as
    # activation peak and taken straight out of the KV budget (it killed KV sizing once).
    out = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=w.device)
    ws = torch.empty(N // _BLK, K // _BLK, dtype=torch.float32, device=w.device)
    rows = 4096                                                # multiple of 128 and of 16
    for r0 in range(0, N, rows):
        r1 = min(N, r0 + rows)
        wf = w[r0:r1].float().view((r1 - r0) // _BLK, _BLK, K // _BLK, _BLK)
        amax = wf.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
        scale = amax / 448.0                                   # e4m3 max
        wq = (wf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(r1 - r0, K)
        # The fragment shuffle permutes within 16-row tiles only, so shuffling each row
        # chunk in place is exact and avoids a second whole-weight copy.
        out[r0:r1] = shuffle_weight_gfx1201(wq)
        ws[r0 // _BLK: r1 // _BLK] = scale.view((r1 - r0) // _BLK, K // _BLK)
        del wf, amax, scale, wq
    return out, ws


def gemm_fp8_block(x: torch.Tensor, wq_shuf: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """bf16 [M, K] @ dequant(W)^T -> bf16 [M, N] through fp8hip; caller guarantees M <= _MAX_M."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    qx, xs = per_token_group_quant_fp8(x.contiguous(), _BLK, dtype=torch.float8_e4m3fn)
    return torch.ops.vllm.fp8hip_block_scaled_mm(qx, wq_shuf, xs, ws)


class Fp8TargetHead:
    """Drop-in replacement for the lm_head's quant_method.apply at decode M."""

    def __init__(self, orig_method, weight: torch.Tensor):
        self.orig = orig_method
        self.wq, self.ws = pack_fp8_block(weight)
        self.n_fp8 = 0

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] > _MAX_M or bias is not None or x.dtype != torch.bfloat16:
            return self.orig.apply(layer, x, bias=bias)
        y = gemm_fp8_block(flat, self.wq, self.ws)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    def __getattr__(self, name):        # everything else (embedding, create_weights...) passes through
        return getattr(self.orig, name)
