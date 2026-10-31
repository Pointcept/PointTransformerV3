# Vendored from Pointcept2 src/pointcept2/ops/_norm_triton.py @ 771a949 (changed: do_not_specialize on row counts, so a warm-up compiles every variant the model needs).
"""Triton kernels and ``triton_op``s of ``layer_norm`` (imports triton; ``layer_norm`` imports
this module on first use).

``point_transformer_v3::layer_norm`` returns ``(y, mean, rstd)`` and has a fake impl and an autograd formula
built from ``point_transformer_v3::layer_norm_backward``; under torch.compile, torch.export and AOTInductor
the kernels are captured as user-defined Triton kernels. Eager calls take ``LayerNormFunction``
(the same launchers from a plain ``torch.autograd.Function``, no custom-op dispatch; I5).

- ``_fwd_kernel``: a program normalizes ``BLOCK_M`` rows held in registers. fp32 mean, then the
  fp32 variance of the centered row (two passes over registers, no cancellation), ``rstd =
  rsqrt(var + eps)``; ``y = (x - mean) * rstd * w + b`` with fp32 weight and bias, rounded once
  to the output dtype. mean and rstd are kept in fp32 for the backward.
- ``_bwd_kernel``: a fixed number of programs per GPU (``_BWD_PROGRAMS_PER_SM`` x SMs) walk the
  row blocks; each writes ``dx`` and keeps fp32 partial sums of ``dweight = sum(dy * xhat)`` and
  ``dbias = sum(dy)``, which one ``sum(0)`` reduces. The grid and the row assignment depend only
  on the GPU and N, so the result is bitwise reproducible.

Every address is computed in int64 from the row and column; no index comes from data. Launch
configs are a fixed table per channel tile (a bounded offline sweep: research/accel/r1/nm/sweep.py,
bf16 at 1M rows on an RTX PRO 6000), never autotuned. Prototype: research/accel/r1/nm/fused_ln.py.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.library import triton_op

from point_transformer_v3.ops._triton import device_guard, raw, traceable

__all__ = ["LayerNormFunction", "layer_norm_backward", "layer_norm_forward"]

# (BLOCK_M rows, num_warps) per channel tile BLOCK_C = next_pow2(C); other tiles: _config().
_FWD = {32: (256, 4), 64: (32, 4), 128: (32, 4), 256: (16, 8), 512: (16, 8)}
_BWD = {32: (256, 4), 64: (128, 4), 128: (64, 4), 256: (32, 4), 512: (16, 4)}
_BWD_PROGRAMS_PER_SM = 2


@triton.jit(do_not_specialize=["N"])
def _fwd_kernel(X, W, B, Y, MEAN, RSTD, N, EPS: tl.constexpr, C: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):  # fmt: skip
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    row_ok = rows < N
    col_ok = cols < C
    mask = row_ok[:, None] & col_ok[None, :]
    offs = rows[:, None] * C + cols[None, :]
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / C
    d = tl.where(mask, x - mean[:, None], 0.0)
    rstd = tl.math.rsqrt(tl.sum(d * d, axis=1) / C + EPS)  # EPS rounded to fp32, as in torch
    w = tl.load(W + cols, mask=col_ok, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_ok, other=0.0).to(tl.float32)
    y = d * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(MEAN + rows, mean, mask=row_ok)
    tl.store(RSTD + rows, rstd, mask=row_ok)


@triton.jit(do_not_specialize=["N"])
def _bwd_kernel(DY, X, W, MEAN, RSTD, DX, PARTIAL, N, C: tl.constexpr, BLOCK_M: tl.constexpr,
                BLOCK_C: tl.constexpr):  # fmt: skip
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_ok = cols < C
    w = tl.load(W + cols, mask=col_ok, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK_C], dtype=tl.float32)
    db = tl.zeros([BLOCK_C], dtype=tl.float32)
    for block in range(pid, tl.cdiv(N, BLOCK_M), tl.num_programs(0)):
        rows = (block * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)  # N < 2**31 rows
        row_ok = rows < N
        mask = row_ok[:, None] & col_ok[None, :]
        offs = rows[:, None] * C + cols[None, :]
        dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(MEAN + rows, mask=row_ok, other=0.0)
        rstd = tl.load(RSTD + rows, mask=row_ok, other=0.0)
        xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
        wdy = w[None, :] * dy
        c1 = tl.sum(xhat * wdy, axis=1) / C
        c2 = tl.sum(wdy, axis=1) / C
        dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
        tl.store(DX + offs, dx.to(DX.dtype.element_ty), mask=mask)
        dw += tl.sum(dy * xhat, axis=0)
        db += tl.sum(dy, axis=0)
    out = PARTIAL + pid.to(tl.int64) * (2 * C) + cols
    tl.store(out, dw, mask=col_ok)
    tl.store(out + C, db, mask=col_ok)


# ---- ops (traced graphs) and the eager Function


@triton_op("point_transformer_v3::layer_norm", mutates_args=())
def layer_norm_forward(
    x: Tensor, weight: Tensor, bias: Tensor, eps: float, out_dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    """``(y, mean, rstd)``: y the LayerNorm of every row in ``out_dtype``, mean and rstd fp32.

    Shapes:
        x: ``(N, C)`` fp32/fp16/bf16, contiguous. weight, bias: ``(C,)`` float (used in fp32).
        returns: y ``(N, C)``; mean, rstd ``(N,)``.
    """
    return _forward(x, weight, bias, eps, out_dtype, traceable)


@layer_norm_forward.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _forward_fake(
    x: Tensor, weight: Tensor, bias: Tensor, eps: float, out_dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    stats = x.new_empty(x.shape[0], dtype=torch.float32)
    return x.new_empty(x.shape, dtype=out_dtype), stats, torch.empty_like(stats)


@triton_op("point_transformer_v3::layer_norm_backward", mutates_args=())
def layer_norm_backward(
    grad_y: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor
) -> tuple[Tensor, Tensor]:
    """``(grad_x, grad_affine)``: grad_x in x's dtype; grad_affine ``(2, C)`` fp32, the weight
    gradient then the bias gradient.

    Shapes: grad_y, x: ``(N, C)``; weight: ``(C,)``; mean, rstd: ``(N,)`` fp32.
    """
    return _backward_launch(grad_y, x, weight, mean, rstd, traceable)


@layer_norm_backward.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _backward_fake(
    grad_y: Tensor, x: Tensor, weight: Tensor, mean: Tensor, rstd: Tensor
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x), x.new_empty(2, x.shape[1], dtype=torch.float32)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, ...]) -> None:
    x, weight, bias = inputs[:3]
    _, mean, rstd = output
    ctx.save_for_backward(x, weight, mean, rstd)
    ctx.dtypes = (weight.dtype, bias.dtype)


def _backward(
    ctx: Any, grad_y: Tensor, *_stats_grads: Any, launch: Callable[..., Any] = layer_norm_backward
) -> tuple[Tensor | None, ...]:
    """Gradients of ``(x, weight, bias, eps, out_dtype)`` (mean and rstd get none): through the op
    (traced graphs keep it) or through the launcher (``LayerNormFunction``)."""
    x, weight, mean, rstd = ctx.saved_tensors
    grad_x, grad_affine = launch(grad_y.contiguous(), x, weight, mean, rstd)
    return grad_x, grad_affine[0].to(ctx.dtypes[0]), grad_affine[1].to(ctx.dtypes[1]), None, None


layer_norm_forward.register_autograd(  # pyright: ignore[reportFunctionMemberAccess]
    _backward, setup_context=_setup_context
)


class LayerNormFunction(torch.autograd.Function):
    """Eager fast path of ``point_transformer_v3::layer_norm``: the op's kernels launched directly
    (``ops._tracing.tracing`` picks it). Arguments as ``layer_norm_forward``; returns y only."""

    @staticmethod
    def forward(
        ctx: Any, x: Tensor, weight: Tensor, bias: Tensor, eps: float, out_dtype: torch.dtype
    ) -> Tensor:
        y, mean, rstd = _forward(x, weight, bias, eps, out_dtype)
        _setup_context(ctx, (x, weight, bias), (y, mean, rstd))
        return y

    # once_differentiable's annotation makes ctx positional-only, which pyright flags.
    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_y: Tensor) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _backward(ctx, grad_y, launch=_backward_launch)


# ---- launchers: the op bodies (wrap=traceable) and the eager Function (wrap=raw)


def _forward(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float,
    out_dtype: torch.dtype,
    wrap: Callable[[Any], Any] = raw,
) -> tuple[Tensor, Tensor, Tensor]:
    n, c = x.shape
    y = x.new_empty(n, c, dtype=out_dtype)
    mean = x.new_empty(n, dtype=torch.float32)
    rstd = x.new_empty(n, dtype=torch.float32)
    if n == 0:
        return y, mean, rstd
    block_c, (block_m, warps) = _config(c, _FWD)
    with device_guard(x):
        wrap(_fwd_kernel)[(triton.cdiv(n, block_m),)](
            x, weight, bias, y, mean, rstd, n,
            EPS=eps, C=c, BLOCK_M=block_m, BLOCK_C=block_c, num_warps=warps,
        )  # fmt: skip
    return y, mean, rstd


def _backward_launch(
    grad_y: Tensor,
    x: Tensor,
    weight: Tensor,
    mean: Tensor,
    rstd: Tensor,
    wrap: Callable[[Any], Any] = raw,
) -> tuple[Tensor, Tensor]:
    n, c = x.shape
    grad_x = torch.empty_like(x)
    programs = _BWD_PROGRAMS_PER_SM * _sm_count(x.device)
    partial = x.new_empty(programs, 2, c, dtype=torch.float32)
    block_c, (block_m, warps) = _config(c, _BWD)
    with device_guard(x):
        wrap(_bwd_kernel)[(programs,)](
            grad_y, x, weight, mean, rstd, grad_x, partial, n,
            C=c, BLOCK_M=block_m, BLOCK_C=block_c, num_warps=warps,
        )  # fmt: skip
    return grad_x, partial.sum(0)


def _config(c: int, table: dict[int, tuple[int, int]]) -> tuple[int, tuple[int, int]]:
    """``(BLOCK_C, (BLOCK_M, num_warps))``: the table entry of the channel tile, else about 4096
    elements per program (4 warps; 8 from 1024 channels)."""
    block_c = max(16, 1 << (c - 1).bit_length())  # next power of two
    if block_c in table:
        return block_c, table[block_c]
    return block_c, (max(1, 4096 // block_c), 4 if block_c < 1024 else 8)


@functools.cache
def _sm_count(device: torch.device) -> int:
    """SMs of a CUDA device (cached per device); 4 for the Triton interpreter on CPU."""
    if device.type != "cuda":
        return 4
    return torch.cuda.get_device_properties(device).multi_processor_count
