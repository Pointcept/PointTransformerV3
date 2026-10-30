# Vendored from Pointcept2 src/pointcept2/ops/gather.py @ 771a949 (simplified for pointtransformerv3).
"""Row gathers (``x[index]`` along dim 0) that avoid a slow torch kernel on CUDA.

torch 2.12 sends ``index_select``/``x[index]`` to ``vectorized_gather_kernel`` when
``fast_gather_kernel_eligible`` (``ATen/native/cuda/IndexKernelUtils.h``) holds: a contiguous index
(int32 or int64), contiguous rows, 16-byte-aligned source and output pointers, and a row size,
source row stride and output row stride that are multiples of 16 bytes. On sm_120 that kernel
takes ~0.48 ms per 1M gathered rows for every row size from 16 to 256 bytes, so it is the slow path
exactly for the common widths (bf16 C = 8..128, fp32 C = 4..64). Breaking any one condition sends
the gather to the elementwise kernel. Two exact ways to do that from the outside:

- strided index (``strided_index``): a copy of the index with element stride 2. It costs an index
  copy, so it suits gathers of about as many rows as the source has (permutations, unpooling).
- padded source (``gather_source``): a copy of x whose row stride is not a multiple of 16 bytes.
  It costs a copy of x but the gather from it is faster, so it suits sources that are read many
  times per row (kernel maps); it also appends zero rows, the target of "no neighbor" entries.

``gather_rows`` checks the conditions on the host (``slow_gather``) and picks one of the two per
call: a padded source when the index has at least 4 rows per source row, else a strided index.
Above 256 bytes per row the vectorized kernel is as fast or faster, so it is kept. Scatters
(``index_copy_``, ``index_add_``, ``index_put_``) have no such trap and need no wrapper.

Backward: ``gather_rows`` uses autograd's ``x[index]`` backward, a sorted, deterministic
``index_put_(accumulate=True)`` (not ``index_select``'s atomic ``index_add_``). Its sort makes it
the expensive half: a 1M-row gather of bf16 qkv rows (C=48, 288 B) takes 2.17 ms fwd+bwd. A
permutation needs no accumulation, so ``permute_rows`` scatters the gradient back with
``index_copy_`` instead: 0.95 ms for the same rows (``out[inverse]``, 96 B: 0.86 -> 0.39 ms).

Measured on an RTX PRO 6000 (sm_120), torch 2.12.1, ms per gather, medians of 25 interleaved runs
(plain / padded source incl. its copy / strided index incl. its copy):

    rows                         bf16 C=16      C=32           C=64           C=128
    permutation 1M of 1M         0.48/0.08/0.08 0.49/0.24/0.19 0.48/0.50/0.27 0.49/0.96/0.45
    unpool 1M of 419k            0.49/0.08/0.09 0.49/0.12/0.14 0.49/0.26/0.22 0.49/0.50/0.34
    kernel map 2.7M of 100k      1.25/0.13/0.17 1.25/0.27/0.26 1.26/0.38/0.48 1.26/0.61/0.76

int32 and int64 indices, views with a 16-byte-multiple row stride (``qkv[:, :C]``) and
``(N, H, D)`` rows all take the slow kernel; a source pointer that is not 16-byte aligned does not
(it is then the fastest plain gather, and ``slow_gather`` says False). On the 1M-point ScanNet
pack, the k=3 kernel-map gathers of a whole conv (27M rows, bf16, int32 index) take 0.96 / 1.71 /
3.19 / 6.22 ms for C = 16 / 32 / 64 / 128 from a padded source, and 1.28 / 2.19 / 3.49 / 6.49 ms
with a strided index; ``ops/sparse_conv/torch_impl.py`` therefore gathers from a
``gather_source`` with a contiguous index.

Under ``torch.compile``/``torch.export`` the plain op is traced: Inductor writes its own gather.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

__all__ = ["gather_rows", "gather_source", "permute_rows", "slow_gather", "strided_index"]

# Rows above this size gather as fast through the vectorized kernel (see the module doc).
_MAX_ROW_BYTES = 256
# Gathers of at least this many rows per source row copy the source instead of the index.
_SOURCE_COPY_RATIO = 4
_ALIGN = 16  # bytes; the vectorized kernel's load width


def gather_rows(x: Tensor, index: Tensor) -> Tensor:
    """``x[index]`` along dim 0, through a strided index or a padded source where torch's
    vectorized gather would be slow (``slow_gather``); plain ``x[index]`` everywhere else.

    Shapes:
        x: ``(N, ...)``, any strides.
        index: ``(M,)`` int64 or int32 on x's device, values in ``[0, N)`` (no negative indexing).
        returns: ``(M, ...)`` contiguous, x's dtype; bit-identical to ``x[index]``.

    Differentiable in x with autograd's ``x[index]`` backward (sorted ``index_put_``,
    deterministic). Sync-free; ``PTV3_DEBUG=1`` checks the index range (a host sync).
    """
    _check(x, index)
    if not slow_gather(x, index):
        return x[index]
    if torch.is_grad_enabled() and x.requires_grad:
        return _GatherRows.apply(x, index)
    return _fast_gather(x, index)


def permute_rows(x: Tensor, perm: Tensor) -> Tensor:
    """``x[perm]`` for a permutation ``perm`` of x's rows (serialization orders, their inverses,
    pooling sorts): ``gather_rows`` forward, and a backward that scatters the gradient back with
    ``index_copy_`` (exact, deterministic, no sort; about twice as fast as ``gather_rows``'s).

    Shapes:
        x: ``(N, ...)``, any strides.
        perm: ``(N,)`` int64 or int32 on x's device, a permutation of ``[0, N)``.
        returns: ``(N, ...)`` contiguous, x's dtype; bit-identical to ``x[perm]``.

    Sync-free; ``PTV3_DEBUG=1`` checks that perm is a permutation (a host sync).
    """
    _check(x, perm)
    if perm.shape[0] != x.shape[0]:
        raise ValueError(
            f"perm must have one entry per row of x ({x.shape[0]}), got {perm.shape[0]}; use "
            "gather_rows for other gathers"
        )
    if os.environ.get("PTV3_DEBUG") == "1":
        rows = torch.arange(perm.shape[0], device=perm.device)
        if not torch.equal(perm.sort().values, rows.to(perm.dtype)):
            raise ValueError("perm must be a permutation of the rows; use gather_rows instead")
    if torch.is_grad_enabled() and x.requires_grad:
        return _PermuteRows.apply(x, perm)
    return gather_rows(x, perm)


def gather_source(x: Tensor, zero_rows: int = 0) -> Tensor:
    """``(N + zero_rows, ...)`` copy of x to gather from many times, with ``zero_rows`` zero rows
    appended (the target of "no neighbor" indices).

    On CUDA, when x's rows would take the slow vectorized gather, the copy is a view of a buffer
    one element wider per row, so its row stride is not a multiple of 16 bytes and
    ``index_select(0, index)`` from it takes the elementwise kernel. The values are exact; only
    the strides differ from a contiguous copy.
    """
    if isinstance(zero_rows, bool) or not isinstance(zero_rows, int) or zero_rows < 0:
        raise ValueError(f"zero_rows must be a non-negative int, got {zero_rows!r}")
    if not isinstance(x, Tensor) or x.dim() == 0:
        raise TypeError("x must be a tensor with a leading row dimension")
    n, width = x.shape[0], math.prod(x.shape[1:])
    flat = x.reshape(n, width)
    pad = int(x.is_cuda and _slow_row_bytes(width * x.element_size()))
    out = flat.new_empty(n + zero_rows, width + pad)[:, :width]
    out[:n] = flat
    if zero_rows:
        out[n:].zero_()  # no CPU scalar operand: one less node in a captured graph
    return out.view(n + zero_rows, *x.shape[1:])


def strided_index(index: Tensor) -> Tensor:
    """A copy of a 1-D index with element stride 2 (same values), so a gather with it skips the
    vectorized kernel. An index that is already non-contiguous is returned as is, so a caller that
    gathers with the same index many times can build it once."""
    if index.dim() != 1:
        raise ValueError(f"index must be 1-D, got shape {tuple(index.shape)}")
    if index.stride(0) != 1:
        return index
    out = index.new_empty(index.shape[0], 2)[:, 0]
    out.copy_(index)
    return out


def slow_gather(x: Tensor, index: Tensor) -> bool:
    """Whether ``x.index_select(0, index)`` would take torch's vectorized gather at a row size where
    it is slow (CUDA, 16..256-byte rows; see the module doc). A host check of devices, sizes,
    strides and the data pointer: no sync. False while ``torch.compile``/``torch.export`` trace."""
    if not x.is_cuda or x.dim() == 0 or x.shape[0] == 0 or index.shape[0] == 0:
        return False
    if torch.compiler.is_compiling():
        return False
    size = x.element_size()
    if not _slow_row_bytes(math.prod(x.shape[1:]) * size):
        return False
    return (
        index.stride(0) == 1
        and _rows_contiguous(x)
        and x.stride(0) * size % _ALIGN == 0
        and x.data_ptr() % _ALIGN == 0
    )


# ---- internals


class _GatherRows(torch.autograd.Function):
    """``_fast_gather`` forward; the backward is autograd's ``x[index]`` backward."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, index: Tensor) -> Tensor:
        ctx.save_for_backward(index)
        ctx.num_rows = x.shape[0]
        return _fast_gather(x, index)

    # once_differentiable's annotation makes ctx positional-only, which pyright flags.
    @staticmethod
    @once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, *grad_outputs: Tensor
    ) -> tuple[Tensor, None]:
        (grad,) = grad_outputs
        (index,) = ctx.saved_tensors
        out = grad.new_zeros(ctx.num_rows, *grad.shape[1:])
        return out.index_put_((index,), grad, accumulate=True), None


class _PermuteRows(torch.autograd.Function):
    """``x[perm]`` for a permutation; the backward is an ``index_copy_`` of the gradient."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, perm: Tensor) -> Tensor:
        ctx.save_for_backward(perm)
        return gather_rows(x, perm)

    @staticmethod
    @once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, *grad_outputs: Tensor
    ) -> tuple[Tensor, None]:
        (grad,) = grad_outputs
        (perm,) = ctx.saved_tensors
        return grad.new_empty(grad.shape).index_copy_(0, perm.long(), grad), None  # int64 only


def _fast_gather(x: Tensor, index: Tensor) -> Tensor:
    """Gather off the vectorized kernel: copy the source when it is read many times per row,
    else the index."""
    if index.shape[0] >= _SOURCE_COPY_RATIO * x.shape[0]:
        return gather_source(x).index_select(0, index)
    return x.index_select(0, strided_index(index))


def _slow_row_bytes(row_bytes: int) -> bool:
    return 0 < row_bytes <= _MAX_ROW_BYTES and row_bytes % _ALIGN == 0


def _rows_contiguous(x: Tensor) -> bool:
    """Each row ``x[i]`` is one dense block (size-1 dims ignored), without building a view."""
    expected = 1
    for size, stride in zip(reversed(x.shape[1:]), reversed(x.stride()[1:]), strict=True):
        if size != 1 and stride != expected:
            return False
        expected *= size
    return True


def _check(x: Tensor, index: Tensor) -> None:
    if not isinstance(x, Tensor) or x.dim() == 0:
        raise TypeError("x must be a tensor with a leading row dimension")
    if (
        not isinstance(index, Tensor)
        or index.dim() != 1
        or index.dtype not in (torch.int64, torch.int32)
    ):
        got = f"{index.dtype} {tuple(index.shape)}" if isinstance(index, Tensor) else repr(index)
        raise TypeError(f"index must be a 1-D int64 or int32 tensor, got {got}")
    if index.device != x.device:
        raise ValueError(f"index is on {index.device} but x is on {x.device}")
    if os.environ.get("PTV3_DEBUG") == "1" and index.numel():
        low, high = int(index.min()), int(index.max())
        if low < 0 or high >= x.shape[0]:
            raise ValueError(f"index values must lie in [0, {x.shape[0]}), got [{low}, {high}]")
