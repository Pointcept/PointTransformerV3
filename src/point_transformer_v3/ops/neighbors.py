# Vendored from Pointcept2 src/pointcept2/ops/neighbors.py @ 771a949 (simplified for point_transformer_v3).
"""Neighbor tables on integer grid cells: cubic kernel offsets and the submanifold kernel map.

A kernel map ``kmap`` of shape ``(N, K)`` lists, for every row ``p`` and kernel offset ``k``, the
row ``q`` that sits at ``grid[p] + offsets[k]`` in the same sample, or -1. ``build_kmap`` builds it
with one sort of the packed keys and ``searchsorted`` lookups (the reference, and the only way for
a level without a coarser one); ``kmap_from_coarse`` builds the same map from the 3^3 map of the
stride-2 coarse level with two table lookups (ocnn's top-down construction; on CUDA two fused
Triton kernels, I3). Both are sync-free on CUDA. Offsets follow the spconv weight order, which
makes the map symmetric: ``kmap[p, k] == q`` exactly when ``kmap[q, K - 1 - k] == p``; the conv
backward relies on this (``check_kmap``).
"""

from __future__ import annotations

import functools
import os

import torch
from torch import Tensor

from point_transformer_v3.ops._tracing import assume_constant_result, tracing
from point_transformer_v3.ops._triton import TRITON_MIN_CAPABILITY, available
from point_transformer_v3.ops.keys import AXIS_BITS, check_key_range, pack_keys

__all__ = ["build_kmap", "check_kmap", "kernel_offsets", "kmap_from_coarse"]

# Lookups per chunk of sorted rows. Bounds the temporaries to ~130 MiB; on a 704k-point
# ScanNet pack (RTX PRO 6000) 2**22 was the fastest of 2**20..2**23: k=3 ~1.3 ms, k=5 ~3.9 ms.
_QUERIES_PER_CHUNK = 1 << 22


def kernel_offsets(kernel_size: int, *, device: torch.device | str | None = None) -> Tensor:
    """``(K, 3)`` int64 offsets of a cubic kernel, ``K = kernel_size ** 3``.

    Rows follow ``itertools.product(range(-r, r + 1), repeat=3)`` (dx slowest, dz fastest), the
    order of the flattened spconv weight ``(C_out, k, k, k, C_in)``; row ``K - 1 - k`` is the
    negation of row ``k``. Built on ``device`` without a host copy.
    """
    r = _radius(kernel_size)
    axis = torch.arange(-r, r + 1, device=device)
    return torch.cartesian_prod(axis, axis, axis).view(-1, 3)


def build_kmap(grid: Tensor, batch: Tensor, kernel_size: int) -> Tensor:
    """Submanifold kernel map of unique grid cells.

    Args:
        grid: ``(N, 3)`` int grid coordinates, ``0 <= grid < 2**16 - 2 * (kernel_size // 2)``.
        batch: ``(N,)`` int sample index of each row. ``(batch, grid)`` pairs must be unique.
        kernel_size: odd edge length of the cubic kernel.

    Returns:
        ``(N, K)`` int32: ``kmap[p, k]`` is the row ``q`` with ``grid[q] == grid[p] + offsets[k]``
        and ``batch[q] == batch[p]``, or -1 when there is none (``offsets = kernel_offsets(...)``).

    Value checks (key range, uniqueness) read the device back and run only under ``PTV3_DEBUG=1``.
    """
    r = _radius(kernel_size)
    if batch.device != grid.device:
        raise ValueError(f"grid is on {grid.device} but batch is on {batch.device}")
    keys = pack_keys(grid, batch, margin=r)
    n, num_offsets = keys.shape[0], kernel_size**3
    debug = os.environ.get("PTV3_DEBUG") == "1"
    if debug and n:
        if batch.min() < 0:
            raise ValueError("batch indices must be non-negative")
        check_key_range(grid, int(batch.max()) + 1, r)
    kmap = torch.empty(n, num_offsets, dtype=torch.int32, device=grid.device)
    if n == 0:
        return kmap

    sorted_keys, order = torch.sort(keys)
    if debug and bool((sorted_keys[1:] == sorted_keys[:-1]).any()):
        raise ValueError("build_kmap needs unique (batch, grid) rows; grid-sample the points first")
    offsets = kernel_offsets(kernel_size, device=grid.device)
    # Adding a packed offset moves each axis inside its own field: the margin keeps every field in
    # [0, 2**16), so no borrow or carry crosses into the neighboring axis or the batch bits.
    delta = (offsets[:, 0] << 2 * AXIS_BITS) + (offsets[:, 1] << AXIS_BITS) + offsets[:, 2]
    rows = order.int()
    step = max(1, _QUERIES_PER_CHUNK // num_offsets)
    # Walking the rows in key order keeps the binary searches of a chunk on nearby cache lines.
    for start in range(0, n, step):
        query = sorted_keys[start : start + step, None] + delta
        pos = torch.searchsorted(sorted_keys, query).clamp_(max=n - 1)
        kmap[order[start : start + step]] = torch.where(sorted_keys[pos] == query, rows[pos], -1)
    return kmap


def kmap_from_coarse(
    coarse_kmap: Tensor, inverse: Tensor, grid: Tensor, kernel_size: int = 3
) -> Tensor:
    """Submanifold kernel map of a level from the 3^3 map of its stride-2 coarse level.

    ocnn's top-down construction (``Octree.construct_neigh_by_parent`` and ``Octree._get_lut``,
    ocnn-pytorch commit 7521c22, MIT, Copyright (c) 2022 Peng-Shuai Wang) on a stride-2 pooling
    hierarchy instead of an octree. A cell ``g = 2 * P + s`` (child slot ``s`` in {0, 1}^3) has
    its neighbor ``g + d`` in the coarse cell ``P + floor((s + d) / 2)``, at child slot
    ``(s + d) mod 2``. For ``|d| <= 2`` the coarse offset lies in {-1, 0, 1}^3, so the coarse 3^3
    map serves kernel sizes 1, 3 and 5. With two ``(8, K)`` lookup tables indexed by the slot:

        c = coarse_kmap[inverse[p], lut_coarse[slot[p], k]]   # coarse row holding the neighbor
        q = children[c, lut_child[slot[p], k]]                 # its child at that slot, or -1

    ``children (M + 1, 8)`` holds the row of every (coarse row, slot) and -1 for empty slots; its
    last row stands for missing coarse neighbors. Two gathers and a scatter: no sort, no search,
    no mask, no host sync, and no key-range limit beyond the coarse map's.

    On CUDA (with a working Triton) the same lookups run as two Triton kernels that write the map
    once (``_neighbors_triton``; 1M rows on an RTX PRO 6000: K=125 about 0.6 ms against 5.2 ms
    for ``build_kmap``), recorded as ``point_transformer_v3::kmap_from_coarse`` in traced graphs; elsewhere
    the torch ops below. Both give the same map, bit for bit.

    Args:
        coarse_kmap: ``(M, 27)`` int32 kernel map of the coarse level (``build_kmap(..., 3)`` or
            ``kmap_from_coarse`` of the next coarser level).
        inverse: ``(N,)`` int coarse row of every row (``PoolIndex.inverse`` of
            ``PointGrid.downsample(2)``): the cell ``grid[i] >> 1`` of row i's sample.
        grid: ``(N, 3)`` int grid coordinates of the rows, unique per sample.
        kernel_size: 1, 3 or 5.

    Returns:
        ``(N, kernel_size**3)`` int32, equal to ``build_kmap(grid, batch, kernel_size)``.

    Peak memory: the torch path two ``(N, K)`` index buffers plus the ``(M + 1, 8)`` child table;
    the Triton path the map plus the child table.

    Checks under ``PTV3_DEBUG=1`` (they read the device back): ``inverse`` and the coarse map in
    range, one row per (coarse row, slot), one parent cell ``grid >> 1`` per coarse row, and every
    neighbor found at ``grid + offset``. They do not see samples: a coarse level that merges two
    samples' cells passes.
    """
    _check_from_coarse_args(coarse_kmap, inverse, grid, kernel_size)
    n, num_offsets = grid.shape[0], kernel_size**3
    if n == 0:
        return torch.empty(0, num_offsets, dtype=torch.int32, device=grid.device)
    debug = os.environ.get("PTV3_DEBUG") == "1"
    if debug:
        _check_coarse(coarse_kmap, inverse.long(), _slot_address(grid, inverse)[1], grid)
    if _triton_ready(grid.device):  # the fused kernels (I3)
        from point_transformer_v3.ops import _neighbors_triton

        if tracing(grid):  # traced graphs keep the op (D15)
            kmap = torch.ops.point_transformer_v3.kmap_from_coarse(
                coarse_kmap, inverse, grid, kernel_size
            )
        else:  # eager: no custom-op dispatch (I5)
            kmap = _neighbors_triton.launch(coarse_kmap, inverse, grid, kernel_size)
    else:
        kmap = _from_coarse_torch(coarse_kmap, inverse, grid, kernel_size)
    if debug:
        _check_geometry(kmap, grid, kernel_size)
    return kmap


def _check_from_coarse_args(
    coarse_kmap: Tensor, inverse: Tensor, grid: Tensor, kernel_size: int
) -> None:
    """Kernel size, ranks, shapes, dtypes and devices of ``kmap_from_coarse``'s inputs (no device
    read, safe under tracing). Also the checks of the ``point_transformer_v3::kmap_from_coarse`` op
    body and its fake impl: its kernels index the coarse map by row * 27 + column and the child
    table by coarse row * 8 + slot, which stay inside the tensors only for these shapes and
    k <= 5."""
    if isinstance(kernel_size, bool) or kernel_size not in (1, 3, 5):
        raise ValueError(
            f"kernel_size must be 1, 3 or 5 (the coarse 3^3 map reaches offsets |d| <= 2), got "
            f"{kernel_size!r}; use build_kmap for larger kernels"
        )
    if coarse_kmap.dtype != torch.int32 or coarse_kmap.dim() != 2 or coarse_kmap.shape[1] != 27:
        raise TypeError(
            f"coarse_kmap must be an int32 (M, 27) kernel map (kernel size 3), got "
            f"{coarse_kmap.dtype} {tuple(coarse_kmap.shape)}"
        )
    if grid.dtype.is_floating_point or grid.dim() != 2 or grid.shape[1] != 3:
        raise TypeError(f"grid must be an (N, 3) int tensor, got {grid.dtype} {tuple(grid.shape)}")
    n = grid.shape[0]
    if inverse.dtype.is_floating_point or inverse.shape != (n,):
        raise TypeError(
            f"inverse must be an ({n},) int tensor (PoolIndex.inverse), got {inverse.dtype} "
            f"{tuple(inverse.shape)}"
        )
    if not grid.device == inverse.device == coarse_kmap.device:
        raise ValueError(
            f"grid, inverse and coarse_kmap must share a device, got {grid.device}, "
            f"{inverse.device}, {coarse_kmap.device}"
        )


def _slot_address(grid: Tensor, inverse: Tensor) -> tuple[Tensor, Tensor]:
    """``(slot, address)``: each row's child slot in its stride-2 parent cell (ocnn's key % 8,
    x, y, z bit order) and ``inverse * 8 + slot``, int64."""
    g = grid.long() & 1
    slot = (g[:, 0] << 2) | (g[:, 1] << 1) | g[:, 2]
    return slot, (inverse.long() << 3) | slot


def _from_coarse_torch(
    coarse_kmap: Tensor, inverse: Tensor, grid: Tensor, kernel_size: int
) -> Tensor:
    """``kmap_from_coarse`` in torch ops (CPU, MPS, and CUDA without Triton)."""
    n, m = grid.shape[0], coarse_kmap.shape[0]
    num_offsets = kernel_size**3
    slot, address = _slot_address(grid, inverse)
    inverse = inverse.long()
    # ocnn build_octree: children[(parent << 3) | slot] = row. Addresses are unique (one row per
    # cell), so the scatter is deterministic; row m (all -1) is the missing coarse neighbor.
    children = torch.full(((m + 1) * 8,), -1, dtype=torch.int32, device=grid.device)
    children.scatter_(0, address, torch.arange(n, dtype=torch.int32, device=grid.device))
    index_dtype = torch.int32 if (m + 1) * 27 < 1 << 31 else torch.int64
    # the process-wide cache must hold real tensors only: under non-strict torch.export or make_fx
    # the inputs (and so the tables) are FakeTensors
    luts = _octree_luts if type(grid) is Tensor else _build_octree_luts
    lut_coarse, lut_child = luts(kernel_size, grid.device)
    # flat index into the coarse map: inverse * 27 + lut_coarse[slot]                (N, K)
    index = lut_coarse.index_select(0, slot).to(index_dtype)
    index += (inverse * 27).to(index_dtype)[:, None]
    parent = coarse_kmap.reshape(-1).index_select(0, index.view(-1)).view(n, num_offsets)
    del index  # at most two (N, K) buffers live at a time
    parent.masked_fill_(parent < 0, m)  # a missing coarse neighbor: row m of children (all -1)
    # flat index into children: parent * 8 + lut_child[slot]                        (N, K)
    index = lut_child.index_select(0, slot).to(index_dtype)
    index.add_(parent, alpha=8)
    del parent
    return children.index_select(0, index.view(-1)).view(n, num_offsets)


@assume_constant_result
def _triton_ready(device: torch.device) -> bool:
    """Whether ``kmap_from_coarse`` runs its Triton kernels on ``device`` (a CUDA device where
    Triton works; their ``triton_op`` is registered here). Runs eagerly while torch.compile /
    torch.export trace, so the toolchain probe never enters a graph."""
    if device.type != "cuda" or not available(device, min_capability=TRITON_MIN_CAPABILITY):
        return False
    from point_transformer_v3.ops import _neighbors_triton  # noqa: F401  (imports triton, registers the op)

    return True


def check_kmap(kmap: Tensor) -> None:
    """Raise ``ValueError`` unless ``kmap`` is a valid submanifold kernel map.

    Checks that it is an ``(N, K)`` int32 tensor with K odd, every entry in ``[-1, N)`` and the
    map symmetric (``kmap[p, k] == q`` iff ``kmap[q, K - 1 - k] == p``), which the conv backward
    relies on. Reads the device back (host syncs): call it under ``PTV3_DEBUG=1`` or in tests.
    """
    if kmap.dtype != torch.int32 or kmap.dim() != 2 or kmap.shape[1] % 2 == 0:
        raise TypeError(
            f"kmap must be an (N, K) int32 tensor with K odd, got {kmap.dtype} {tuple(kmap.shape)}"
        )
    n = kmap.shape[0]
    if n == 0:
        return
    low, high = torch.stack([kmap.min(), kmap.max()]).tolist()
    if low < -1 or high >= n:
        raise ValueError(
            f"kmap entries must lie in [-1, {n}) (row indices or -1), got [{low}, {high}]; build "
            "it with build_kmap or kmap_from_coarse for these rows"
        )
    # back[p, k] = kmap[kmap[p, k], K - 1 - k]: the mirrored entry, which must be p again
    back = torch.gather(kmap.flip(1), 0, kmap.clamp(min=0).long())
    rows = torch.arange(n, dtype=torch.int32, device=kmap.device)[:, None]
    bad = (kmap >= 0) & (back != rows)
    if bool(bad.any()):
        p, k = (int(i) for i in bad.nonzero()[0])
        q = int(kmap[p, k])
        raise ValueError(
            f"kmap is not symmetric: kmap[{p}, {k}] = {q} but kmap[{q}, {kmap.shape[1] - 1 - k}] "
            f"= {int(kmap[q, kmap.shape[1] - 1 - k])}; the conv backward needs kmap[p, k] == q "
            "iff kmap[q, K-1-k] == p (build_kmap / kmap_from_coarse of unique grid cells)"
        )


def _build_octree_luts(kernel_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """``(lut_coarse, lut_child)``, each ``(8, K)`` int32 on ``device``.

    ``lut_coarse[s, k]`` is the coarse 3^3 column holding offset k of a cell in child slot s,
    ``lut_child[s, k]`` the child slot inside that coarse cell. For k=3 these are ocnn's
    ``Octree._get_lut`` tables. Built with device arithmetic (no host copy).
    """
    two = torch.arange(2, device=device)
    # slot order x << 2 | y << 1 | z; +2 keeps center + offset >= 0, so // is a floor division
    center = torch.cartesian_prod(two, two, two) + 2  # (8, 3)
    target = center[:, None, :] + kernel_offsets(kernel_size, device=device)[None]  # (8, K, 3)
    coarse = target // 2  # coarse offset + 1, in {0, 1, 2}
    child = target % 2
    lut_coarse = coarse[..., 0] * 9 + coarse[..., 1] * 3 + coarse[..., 2]
    lut_child = child[..., 0] * 4 + child[..., 1] * 2 + child[..., 2]
    return lut_coarse.int().contiguous(), lut_child.int().contiguous()


_octree_luts = functools.cache(_build_octree_luts)  # per (kernel_size, device); real tensors only


def _check_coarse(coarse_kmap: Tensor, inverse: Tensor, address: Tensor, grid: Tensor) -> None:
    """PTV3_DEBUG checks of ``kmap_from_coarse`` inputs (host syncs)."""
    m = coarse_kmap.shape[0]
    low, high = torch.stack([inverse.min(), inverse.max()]).tolist()
    if low < 0 or high >= m:
        raise ValueError(f"inverse must lie in [0, {m}) (coarse rows), got [{low}, {high}]")
    if m and not -1 <= int(coarse_kmap.min()) <= int(coarse_kmap.max()) < m:
        raise ValueError(f"coarse_kmap entries must lie in [-1, {m})")
    if torch.unique(address).shape[0] != address.shape[0]:
        raise ValueError(
            "two rows share a coarse row and child slot: the rows must be unique grid cells and "
            "inverse must map each to its stride-2 parent cell (grid >> 1)"
        )
    # one parent cell per coarse row: catches a coarse level of another stride or grid origin
    cell = grid.long() >> 1
    rows = inverse[:, None].expand(-1, 3)
    low_cell = cell.new_zeros(m, 3).scatter_reduce_(0, rows, cell, "amin", include_self=False)
    high_cell = cell.new_zeros(m, 3).scatter_reduce_(0, rows, cell, "amax", include_self=False)
    if not torch.equal(low_cell, high_cell):
        raise ValueError(
            "rows of one coarse row lie in different parent cells grid >> 1: inverse must map "
            "each row to its stride-2 parent cell (PointGrid.downsample(2).pool.inverse)"
        )


def _check_geometry(kmap: Tensor, grid: Tensor, kernel_size: int) -> None:
    """PTV3_DEBUG check of a ``kmap_from_coarse`` result (host sync): every neighbor it found sits
    at ``grid + offset``. Catches a coarse map and an inverse taken from different levels."""
    g = grid.long()
    offsets = kernel_offsets(kernel_size, device=grid.device)
    bad = torch.zeros((), dtype=torch.bool, device=grid.device)
    for k in range(kmap.shape[1]):  # one column at a time: (N, 3) temporaries
        q = kmap[:, k]
        moved = (g.index_select(0, q.clamp(min=0).long()) != g + offsets[k]).any(1)
        bad |= (moved & (q >= 0)).any()
    if bool(bad):
        raise ValueError(
            "kmap_from_coarse found neighbors at the wrong offsets: coarse_kmap and inverse must "
            "come from the same coarse level (its kmap(3) and pool.inverse), the stride-2 pooling "
            "of these grid rows"
        )


def _radius(kernel_size: int) -> int:
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError(f"kernel_size must be an int, got {kernel_size!r}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}")
    return kernel_size // 2
