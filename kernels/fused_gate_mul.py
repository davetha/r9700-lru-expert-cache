# SPDX-License-Identifier: Apache-2.0
"""One Triton kernel for the shared expert's `sigmoid(gate) * out`.

Inductor never fuses these two aten kernels: the shared expert runs on an aux
stream, which breaks the graph around them (k4 hitlist #7). At 48 MoE layers
that is 96 launches per decode step for ~0.25 ms of real work.

The rounding is deliberately the same as `F.sigmoid(g) * out`: sigmoid is
computed in fp32 and rounded to the gate's dtype before the multiply, exactly
as aten's opmath does, so the result is bit-identical to the stock path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_gate_mul_kernel(
    y_ptr,
    out_ptr,
    gate_ptr,
    n_cols,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # Round sigmoid to the storage dtype first: aten computes sigmoid in fp32
    # and stores bf16, then multiplies. Skipping that round would drift.
    g = tl.load(gate_ptr + row).to(tl.float32)
    s = tl.sigmoid(g).to(gate_ptr.dtype.element_ty).to(tl.float32)
    base = out_ptr + row * stride_row
    ybase = y_ptr + row * n_cols
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        o = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(ybase + cols, (o * s).to(y_ptr.dtype.element_ty), mask=mask)


def supported(gate: torch.Tensor, out: torch.Tensor) -> bool:
    """Whether this pair is the [T, 1] x [T, H] shared-expert-gate shape."""
    return (
        gate.is_cuda
        and out.is_cuda
        and gate.dtype == out.dtype
        and gate.ndim == 2
        and out.ndim == 2
        and gate.shape[1] == 1
        and gate.shape[0] == out.shape[0]
        and gate.stride(0) == 1
        and out.stride(1) == 1
    )


def sigmoid_gate_mul(gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """`sigmoid(gate) * out` in one launch. Caller checks supported() first.

    Writes a fresh tensor rather than mutating `out`: `out` is a down_proj
    result that may be an all-reduce output buffer, and clobbering one of those
    in place is not ours to assume.
    """
    n_rows, n_cols = out.shape
    y = torch.empty_like(out)
    if n_rows:
        _sigmoid_gate_mul_kernel[(n_rows,)](
            y, out, gate, n_cols, out.stride(0),
            BLOCK=min(1024, triton.next_power_of_2(n_cols)),
            num_warps=4,
        )
    return y
