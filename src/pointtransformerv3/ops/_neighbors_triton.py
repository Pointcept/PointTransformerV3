# Vendored from Pointcept2 src/pointcept2/ops/_neighbors_triton.py @ 771a949 (changed: do_not_specialize on row counts, so a warm-up compiles every variant the model needs).
"""Triton kernels and ``triton_op`` of ``kmap_from_coarse`` (imports triton; ``neighbors``
imports this module on first use on CUDA).

The same top-down lookup as the torch path of ``neighbors.kmap_from_coarse`` (I3), fused: the
torch path writes seven ``(N, K)`` intermediates (about 7 GB of traffic for K=125 at 1M rows),
these kernels write the ``(N, K)`` map once and keep the two small tables in L2.

- ``_children_kernel``: ``children[inverse[p] * 8 + slot[p]] = p``, where ``slot`` is the child
  slot ``(x & 1) << 2 | (y & 1) << 1 | (z & 1)`` of the row's cell (ocnn's ``key % 8``).
- ``_kmap_kernel``: for row p and offset ``d = kernel_offsets(k)[j]``, with ``t = slot + d + 2``
  per axis: the neighbor's coarse cell is column ``t // 2`` (as a 3^3 offset + 1) of the coarse
  map, its child slot ``t % 2``; ``kmap[p, j] = children[coarse_kmap[inverse[p], column] * 8 +
  child slot]``, -1 where the coarse neighbor or the child is missing. The lookup tables of the
  torch path become this arithmetic.

Index safety: every load and store is masked by the range of what it indexes (rows in [0, N),
``inverse`` and coarse rows in [0, M)), so corrupt values give a wrong map, never an access
outside the tensors; ``PTV3_DEBUG=1`` checks the values before the launch (``neighbors``). The
column and child-slot arithmetic stays inside a 27-wide coarse row and an 8-slot child row only for
k <= 5 and a ``(M, 27)`` coarse map: ``kmap_from_coarse`` and the op body check the shapes,
dtypes, devices and kernel size first (no device read). Every address is int64. One launch
writes N * K int32 (1M rows, K=125: 500 MB, about 0.6 ms).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op

from pointtransformerv3.ops._triton import device_guard, raw, traceable
from pointtransformerv3.ops.neighbors import _check_from_coarse_args

__all__ = ["kmap_from_coarse_op", "launch"]

_ROWS = 256  # rows per program of the children kernel


@triton.jit(do_not_specialize=["N", "M"], do_not_specialize_on_alignment=["INVERSE"])
def _children_kernel(GRID, INVERSE, CHILDREN, N, M, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = rows < N
    x = tl.load(GRID + rows * 3, mask=ok, other=0)
    y = tl.load(GRID + rows * 3 + 1, mask=ok, other=0)
    z = tl.load(GRID + rows * 3 + 2, mask=ok, other=0)
    slot = ((x & 1) << 2) | ((y & 1) << 1) | (z & 1)
    parent = tl.load(INVERSE + rows, mask=ok, other=-1).to(tl.int64)
    ok = ok & (parent >= 0) & (parent < M)
    tl.store(CHILDREN + parent * 8 + slot, rows.to(tl.int32), mask=ok)


@triton.jit(do_not_specialize=["N", "M"], do_not_specialize_on_alignment=["INVERSE"])
def _kmap_kernel(GRID, INVERSE, COARSE, CHILDREN, KMAP, N, M, KS: tl.constexpr,
                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):  # fmt: skip
    K: tl.constexpr = KS * KS * KS
    R: tl.constexpr = KS // 2
    rows = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    j = tl.arange(0, BLOCK_K)
    j_ok = j < K
    # offset j of kernel_offsets(KS): dx slowest, dz fastest; + 2 keeps slot + d + 2 >= 0
    dx = j // (KS * KS) - R + 2
    dy = (j // KS) % KS - R + 2
    dz = j % KS - R + 2
    tx = (tl.load(GRID + rows * 3, mask=row_ok, other=0) & 1)[:, None] + dx[None, :]
    ty = (tl.load(GRID + rows * 3 + 1, mask=row_ok, other=0) & 1)[:, None] + dy[None, :]
    tz = (tl.load(GRID + rows * 3 + 2, mask=row_ok, other=0) & 1)[:, None] + dz[None, :]
    column = (tx // 2) * 9 + (ty // 2) * 3 + tz // 2
    child = (tx % 2) * 4 + (ty % 2) * 2 + tz % 2
    parent = tl.load(INVERSE + rows, mask=row_ok, other=-1).to(tl.int64)
    parent_ok = row_ok & (parent >= 0) & (parent < M)
    mask = parent_ok[:, None] & j_ok[None, :]
    cell = tl.load(COARSE + parent[:, None] * 27 + column, mask=mask, other=-1).to(tl.int64)
    cell_ok = mask & (cell >= 0) & (cell < M)
    q = tl.load(CHILDREN + cell * 8 + child, mask=cell_ok, other=-1)
    tl.store(KMAP + rows[:, None] * K + j[None, :], q, mask=row_ok[:, None] & j_ok[None, :])


@triton_op("pointtransformerv3::kmap_from_coarse", mutates_args=())
def kmap_from_coarse_op(
    coarse_kmap: Tensor, inverse: Tensor, grid: Tensor, kernel_size: int
) -> Tensor:
    """``neighbors.kmap_from_coarse`` on CUDA. Shapes, dtypes, devices and the kernel size are
    checked first (``ValueError`` / ``TypeError``, no device read), also on a direct call.

    Shapes:
        coarse_kmap: ``(M, 27)`` int32. inverse: ``(N,)`` int. grid: ``(N, 3)`` int.
        kernel_size: 1, 3 or 5. returns: ``(N, kernel_size**3)`` int32.
    """
    _check_from_coarse_args(coarse_kmap, inverse, grid, kernel_size)
    return launch(coarse_kmap, inverse, grid, kernel_size, traceable)


@kmap_from_coarse_op.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _fake(coarse_kmap: Tensor, inverse: Tensor, grid: Tensor, kernel_size: int) -> Tensor:
    _check_from_coarse_args(coarse_kmap, inverse, grid, kernel_size)
    return grid.new_empty(grid.shape[0], kernel_size**3, dtype=torch.int32)


def launch(
    coarse_kmap: Tensor,
    inverse: Tensor,
    grid: Tensor,
    kernel_size: int,
    wrap: Callable[[Any], Any] = raw,
) -> Tensor:
    """The two launches: the child table, then the map. Eager calls of ``kmap_from_coarse`` use
    this directly, after its checks; the op body (``wrap=traceable``) checks first."""
    n, m = grid.shape[0], coarse_kmap.shape[0]
    num_offsets = kernel_size**3
    kmap = grid.new_empty(n, num_offsets, dtype=torch.int32)
    children = grid.new_full((m * 8,), -1, dtype=torch.int32)
    grid, inverse = grid.contiguous(), inverse.contiguous()
    block_k = max(16, 1 << (num_offsets - 1).bit_length())
    block_n = max(16, 4096 // block_k)  # 128 rows for K=27, 32 for K=125
    with device_guard(grid):
        wrap(_children_kernel)[(triton.cdiv(n, _ROWS),)](grid, inverse, children, n, m, BLOCK=_ROWS)
        wrap(_kmap_kernel)[(triton.cdiv(n, block_n),)](
            grid, inverse, coarse_kmap.contiguous(), children, kmap, n, m,
            KS=kernel_size, BLOCK_N=block_n, BLOCK_K=block_k,
        )  # fmt: skip
    return kmap
