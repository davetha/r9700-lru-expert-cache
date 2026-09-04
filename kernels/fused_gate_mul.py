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


# ---------------------------------------------------------------------------
# Mode 2: fold the shared_expert_gate GEMV in as well (k4 hitlist, 2026-09-04).
#
# `shared_expert_gate` is a ReplicatedLinear(hidden_size, 1): m=1 output column,
# a handful of rows, bf16, no bias. No skinny-GEMM path will take it (`m > 8`
# and `m % 4` both fail), so `F.linear` lands on a hipBLASLt Tensile kernel that
# splits K 32 ways over that single column -- 23 us for 5 dot products, 52 of
# them per decode step. Doing the dot inside this kernel removes the GEMM launch
# entirely.
#
# NOT bit-identical, and it cannot be: hipBLASLt's split-K fp32 reduction order
# is not observable from outside. Everything after the dot rounds exactly as the
# stock path does (round to bf16 as F.linear would, sigmoid in fp32, round,
# multiply), so the only divergence is the last bit of the gate scalar.
# ---------------------------------------------------------------------------


@triton.jit
def _gemv_sigmoid_gate_mul_kernel(
    y_ptr,
    out_ptr,
    x_ptr,
    w_ptr,
    K,
    N,
    stride_x,
    stride_out,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    xbase = x_ptr + row * stride_x
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, K, BLOCK_K):
        cols = off + tl.arange(0, BLOCK_K)
        mask = cols < K
        a = tl.load(xbase + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(a * b, axis=0)
    # F.linear returns bf16, so round the dot before the sigmoid sees it.
    g = acc.to(x_ptr.dtype.element_ty).to(tl.float32)
    s = tl.sigmoid(g).to(x_ptr.dtype.element_ty).to(tl.float32)
    obase = out_ptr + row * stride_out
    ybase = y_ptr + row * N
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        o = tl.load(obase + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(ybase + cols, (o * s).to(y_ptr.dtype.element_ty), mask=mask)


def supported_gemv(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
                   max_rows: int) -> bool:
    """Whether this is the [T, K] x [1, K] -> [T, 1] shared-expert-gate GEMV.

    `max_rows` keeps prefill on hipBLASLt: one workgroup per row is the right
    shape for a decode step's handful of rows and the wrong one for thousands.
    """
    return (
        x.is_cuda
        and weight.is_cuda
        and out.is_cuda
        and x.dtype == weight.dtype == out.dtype
        and x.ndim == 2
        and weight.ndim == 2
        and out.ndim == 2
        and weight.shape[0] == 1
        and weight.shape[1] == x.shape[1]
        and weight.stride(1) == 1
        and x.shape[0] == out.shape[0]
        and x.stride(1) == 1
        and out.stride(1) == 1
        and 0 < x.shape[0] <= max_rows
    )


def gemv_sigmoid_gate_mul(x: torch.Tensor, weight: torch.Tensor,
                          out: torch.Tensor) -> torch.Tensor:
    """`sigmoid(x @ weight.T) * out` in one launch. Caller checks supported_gemv().

    Writes a fresh tensor for the same reason `sigmoid_gate_mul` does: `out` is a
    down_proj result and may be an all-reduce buffer.
    """
    n_rows, k = x.shape
    n_cols = out.shape[1]
    y = torch.empty_like(out)
    _gemv_sigmoid_gate_mul_kernel[(n_rows,)](
        y, out, x, weight, k, n_cols, x.stride(0), out.stride(0),
        BLOCK_K=min(1024, triton.next_power_of_2(k)),
        BLOCK_N=min(1024, triton.next_power_of_2(n_cols)),
        num_warps=8,
    )
    return y
