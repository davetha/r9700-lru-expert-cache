# SPDX-License-Identifier: Apache-2.0
"""One Triton kernel for the MoE intermediate's `silu_and_mul` + per-token fp8 quant.

The r4d MoE writes ic1 [mtk, 2I], runs `torch.ops._C.silu_and_mul` into a bf16
ic2, then throws ic2 away after `scaled_fp8_quant` reads it back (k4 hitlist
#11). Two launches and a full bf16 round-trip for one elementwise pass. At 52
MoE blocks per decode step (48 layers + 4 MTP drafts) that is 52 launches.

Bit-identical to the pair it replaces, which is why the rounding is spelled out:
`vllm::act_and_mul_kernel` rounds silu to the storage dtype before multiplying
and rounds the product on store, and
`vllm::dynamic_per_token_scaled_fp8_quant_kernel` takes the row absmax of that
*stored* bf16, floors the scale at 1/(FP8_MAX*512), and DIVIDES by the scale
(it does not multiply by a reciprocal). Skipping any of those steps drifts.
"""

import torch
import triton
import triton.language as tl

# vllm's min_scaling_factor<fp8_type>::val()
_MIN_SCALE_DIV = 512.0
_MAX_BLOCK = 8192


@triton.jit
def _silu_mul_quant_kernel(
    q_ptr,
    s_ptr,
    y_ptr,
    x_ptr,
    d,
    stride_x,
    FP8_MAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
    WRITE_Y: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < d
    xb = x_ptr + row * stride_x
    g = tl.load(xb + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(xb + d + cols, mask=mask, other=0.0).to(tl.float32)
    # Round silu to the storage dtype before the multiply, and the product on
    # store: that is what act_and_mul_kernel does through c10::BFloat16.
    s = (g / (1.0 + tl.exp(-g))).to(x_ptr.dtype.element_ty).to(tl.float32)
    a = (s * u).to(x_ptr.dtype.element_ty)
    if WRITE_Y:
        tl.store(y_ptr + row * d + cols, a, mask=mask)
    af = a.to(tl.float32)
    amax = tl.max(tl.where(mask, tl.abs(af), 0.0), axis=0)
    scale = tl.maximum(amax / FP8_MAX, MIN_SCALE)
    tl.store(s_ptr + row, scale)
    qv = af / scale
    qv = tl.maximum(-FP8_MAX, tl.minimum(qv, FP8_MAX))
    tl.store(q_ptr + row * d + cols, qv.to(q_ptr.dtype.element_ty), mask=mask)


def supported(x: torch.Tensor) -> bool:
    """Whether `x` is a [rows, 2*d] gated-activation input this kernel handles.

    One program owns a whole row, so the half-row must fit one Triton block.
    """
    return (
        x.is_cuda
        and x.ndim == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.shape[1] % 2 == 0
        and x.stride(1) == 1
        and triton.next_power_of_2(x.shape[1] // 2) <= _MAX_BLOCK
    )


def silu_mul_quant(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    write_act: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """`silu_and_mul(x)` and its per-token fp8 quant in one launch.

    Returns (act, q, scale); `act` is the bf16 activation and is None unless
    `write_act`, because the r4d MoE never reads it after the quant.
    Caller checks supported() first.
    """
    rows = x.shape[0]
    d = x.shape[1] // 2
    fp8_max = torch.finfo(fp8_dtype).max
    q = torch.empty((rows, d), dtype=fp8_dtype, device=x.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    act = torch.empty((rows, d), dtype=x.dtype, device=x.device) if write_act else None
    if rows:
        _silu_mul_quant_kernel[(rows,)](
            q, scale, act if act is not None else q, x, d, x.stride(0),
            FP8_MAX=fp8_max,
            MIN_SCALE=1.0 / (fp8_max * _MIN_SCALE_DIV),
            WRITE_Y=act is not None,
            BLOCK=triton.next_power_of_2(d),
            num_warps=4,
        )
    return act, q, scale
