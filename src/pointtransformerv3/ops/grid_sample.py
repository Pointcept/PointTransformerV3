# Vendored from Pointcept2 @ 771a949 (now src/pointcept2/ops/grid_sample.py, renamed in 36bac04; simplified for pointtransformerv3).
"""Grid sampling: one representative row (point) per occupied grid cell, per sequence of a pack.

Unrelated to ``torch.nn.functional.grid_sample`` (bilinear interpolation of an image or volume at
given locations): the name is Pointcept's ``GridSample``. What it keeps are discrete points, one
per cell, not cubes.

``grid_sample`` reproduces Pointcept's ``GridSample`` (fnv hash, ``return_grid_coord=True``): the
lattice is ``floor(coord / grid_size)`` in float64, re-based by the per-sequence minimum, and one
row is kept per occupied cell. Instead of hashing, it sorts one integer key per row (stable) and
run-length encodes the sorted keys, so cells come out ordered by (sequence, x, y, z), the order of
``pack_keys``, and the rows of a cell stay in input order.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Literal, NamedTuple, get_args

import torch
from torch import Tensor

from pointtransformerv3.ops.keys import AXIS_BITS, BATCH_SHIFT
from pointtransformerv3.ops.seqlens import cu_seqlens_of

__all__ = ["GridSampled", "Pick", "grid_sample", "lattice", "unique_grid"]

Pick = Literal["random", "first", "center"]

_MAX_SEQS = 1 << (63 - BATCH_SHIFT)
_GRID_LIMIT = 1 << AXIS_BITS
_PACKED = (1 << 2 * AXIS_BITS, 1 << AXIS_BITS, 1 << BATCH_SHIFT)  # pack_keys weights of x, y, b
_INT32_LIMIT = 1 << 31
_MIN_SLOTS = 256  # power of two, see _segment_bounds


class GridSampled(NamedTuple):
    """Result of ``grid_sample`` / ``unique_grid``: M kept rows, one per occupied cell, from N
    input rows in B sequences."""

    # (M,) int64 representative input row per cell (shadows tuple.index, as the spec names it)
    index: Tensor  # pyright: ignore[reportIncompatibleMethodOverride]
    grid_coord: Tensor  # (M, 3) int32, per-sequence minimum subtracted
    origin: Tensor  # (B, 3) int32 per-sequence minimum of the lattice
    seq_lens: tuple[int, ...]  # kept rows (occupied cells) per sequence (host)
    cu_seqlens: Tensor  # (B + 1,) int32 over kept rows
    inverse: Tensor | None  # (N,) int64 input row -> kept row of its cell
    rank: Tensor | None  # (N,) int64 rank of the row among its cell's rows (input order)


def lattice(coord: Tensor, grid_size: float) -> Tensor:
    """Grid cell of every point: ``(N, 3)`` float coord -> ``(N, 3)`` int64
    ``floor(coord.double() / grid_size)``.

    This is Pointcept's ``GridSample`` lattice: a true float64 division (not a multiply by the
    reciprocal) on every device; float32 picks a different cell for about 1 row in 10^4. MPS has
    no float64, so it computes on the CPU there.
    """
    _check_coord(coord)
    _check_grid_size(grid_size)
    if coord.device.type == "mps":
        return lattice(coord.cpu(), grid_size).to(coord.device)
    return _floor_div(coord.to(torch.float64, copy=True), grid_size)


def grid_sample(
    coord: Tensor,
    grid_size: float,
    *,
    seq_lens: Sequence[int] | None = None,
    pick: Pick = "random",
    generator: torch.Generator | None = None,
    return_inverse: bool = False,
    return_rank: bool = False,
) -> GridSampled:
    """Keep one row per occupied cell of ``lattice(coord, grid_size)`` in every sequence.

    Pointcept's ``GridSample``; unrelated to ``torch.nn.functional.grid_sample`` (bilinear
    interpolation).

    Args:
        coord: ``(N, 3)`` float, B sequences of ``seq_lens`` rows concatenated (one if None).
        pick: the representative row of a cell. "first": its lowest input row. "random": a
            uniform draw ``start + floor(u * count)`` with float64 ``u`` from ``generator`` (on
            coord's device). "center": the row nearest to the cell center ``(g + 0.5) *
            grid_size`` in float64, ties to the lowest row.
        return_inverse / return_rank: also return ``inverse`` / ``rank`` (else None).

    The re-based grid must fit in 16 bits per axis (2^16 cells, 1.3 km at 2 cm). On CUDA, reading
    the cell counts back to the host is the one sync.
    """
    _check_coord(coord)
    _check_grid_size(grid_size)
    _check_pick(pick)
    if coord.device.type == "mps":
        if generator is not None and generator.device.type != "cpu":
            raise ValueError("grid_sample runs on the CPU for MPS inputs; pass a CPU generator")
        out = grid_sample(
            coord.cpu(),
            grid_size,
            seq_lens=seq_lens,
            pick=pick,
            generator=generator,
            return_inverse=return_inverse,
            return_rank=return_rank,
        )
        return _to(out, coord.device)
    seq_lens = _check_seq_lens(seq_lens, coord.shape[0])
    grid = _floor_div(coord.to(torch.float64, copy=True), grid_size)
    dist = None
    if pick == "center":
        diff = coord.double() - (grid.double() + 0.5) * grid_size
        diff.square_()
        dist = diff[:, 0] + diff[:, 1] + diff[:, 2]  # fixed summation order on every device
    batch = _segment_ids(seq_lens, coord.device)
    low, high = _segment_bounds(grid, batch, len(seq_lens))
    return _unique(
        grid,
        batch,
        seq_lens,
        origin=low,
        low=torch.zeros_like(low),
        high=high - low,
        pick=pick,
        generator=generator,
        dist=dist,
        return_inverse=return_inverse,
        return_rank=return_rank,
    )


def unique_grid(
    grid: Tensor,
    *,
    seq_lens: Sequence[int] | None = None,
    pick: Pick = "first",
    generator: torch.Generator | None = None,
    return_inverse: bool = False,
) -> GridSampled:
    """Deduplicate an integer grid that is already on one lattice (e.g. a Mix3D union).

    ``grid`` is ``(N, 3)`` int in ``[0, 2**16)``; it is not re-based (``origin`` is zero).
    ``pick`` is "first" or "random" as in ``grid_sample`` ("center" needs coordinates).
    """
    if grid.dtype.is_floating_point or grid.dtype == torch.bool or grid.dim() != 2:
        raise TypeError(
            f"grid must be an (N, 3) integer tensor, got {grid.dtype} {tuple(grid.shape)}"
        )
    if grid.shape[1] != 3:
        raise TypeError(f"grid must have 3 columns, got shape {tuple(grid.shape)}")
    _check_pick(pick)
    if pick == "center":
        raise ValueError('unique_grid has no coordinates to find a center; use "first" or "random"')
    seq_lens = _check_seq_lens(seq_lens, grid.shape[0])
    grid = grid.long()
    batch = _segment_ids(seq_lens, grid.device)
    low, high = _segment_bounds(grid, batch, len(seq_lens))
    return _unique(
        grid,
        batch,
        seq_lens,
        origin=torch.zeros_like(low),
        low=low,
        high=high,
        pick=pick,
        generator=generator,
        dist=None,
        return_inverse=return_inverse,
        return_rank=False,
    )


def _unique(
    grid: Tensor,
    batch: Tensor | None,
    seq_lens: tuple[int, ...],
    *,
    origin: Tensor,
    low: Tensor,
    high: Tensor,
    pick: Pick,
    generator: torch.Generator | None,
    dist: Tensor | None,
    return_inverse: bool,
    return_rank: bool,
) -> GridSampled:
    """Sort + run-length encode ``(N, 3)`` int64 ``grid - origin[sequence]``.

    ``low`` / ``high`` are the ``(B, 3)`` per-sequence bounds of ``grid - origin``. ``dist`` is the
    ``(N,)`` float64 squared distance of every row to its cell center, given for pick "center".
    """
    n, device = grid.shape[0], grid.device
    bounds = torch.stack([low.amin(), high.amax()])
    weights, key_dtype = _PACKED, torch.int64
    if device.type == "cpu":
        _check_bounds(*bounds.tolist())
        # The CPU sorts int32 keys about twice as fast: ravel by the actual extent when it fits.
        ex, ey, ez = (high.amax(0) + 1).tolist()
        if len(seq_lens) * ex * ey * ez < _INT32_LIMIT:
            weights, key_dtype = (ey * ez, ez, ex * ey * ez), torch.int32
    wx, wy, wb = weights
    offset = torch.arange(len(seq_lens), device=device) * wb
    offset -= origin[:, 0] * wx + origin[:, 1] * wy + origin[:, 2]
    key = grid[:, 0] * wx
    key.add_(grid[:, 1], alpha=wy).add_(grid[:, 2])
    key += offset[0] if batch is None else offset[batch]
    key_sorted, order = torch.sort(key.to(key_dtype), stable=True)

    head = torch.ones(n, dtype=torch.bool, device=device)
    torch.ne(key_sorted[1:], key_sorted[:-1], out=head[1:])
    # cell_end[p + 1]: number of cells that start at sorted position <= p
    cell_end = torch.zeros(n + 1, dtype=torch.int64, device=device)
    torch.cumsum(head, 0, out=cell_end[1:])
    cu = cell_end.index_select(0, cu_seqlens_of(seq_lens, device))
    if device.type == "cpu":
        cu_host = cu.tolist()
        start = head.nonzero().squeeze(1)
    else:  # the one sync: cell counts per sequence, plus the range check
        *cu_host, lo, hi = torch.cat([cu, bounds]).tolist()
        _check_bounds(lo, hi)
        start = torch.nonzero_static(head, size=cu_host[-1]).squeeze(1)
    m = cu_host[-1]
    cell = cell_end[1:] - 1  # (N,) cell of every sorted row

    if dist is not None:  # "center": nearest row, ties to the lowest sorted (= input) row
        d = dist[order]
        best = torch.full((m,), math.inf, dtype=torch.float64, device=device)
        best.scatter_reduce_(0, cell, d, "amin")
        rows = torch.arange(n, device=device)
        pos = torch.full((m,), n, dtype=torch.int64, device=device)
        pos.scatter_reduce_(0, cell, torch.where(d == best[cell], rows, n), "amin")
    elif pick == "random":
        count = torch.diff(start, append=start.new_full((1,), n))
        u = torch.rand(m, dtype=torch.float64, device=device, generator=generator)
        pos = start + (u * count).long()  # u * count < count in round-to-nearest
    else:
        pos = start

    k = key_sorted[start]
    if len(seq_lens) > 1:
        k = k % wb
    grid_coord = torch.stack([k // wx, k % wx // wy, k % wy], dim=1).int()
    inverse = rank = None
    if return_inverse:
        inverse = torch.empty_like(order).scatter_(0, order, cell)
    if return_rank:
        rows = torch.arange(n, device=device)
        rank = torch.empty_like(order).scatter_(0, order, rows - start[cell])
    return GridSampled(
        index=order[pos],
        grid_coord=grid_coord,
        origin=origin.int(),
        seq_lens=tuple(b - a for a, b in itertools.pairwise(cu_host)),
        cu_seqlens=cu.int(),
        inverse=inverse,
        rank=rank,
    )


def _floor_div(x: Tensor, grid_size: float) -> Tensor:
    """In place on float64 ``x``: ``floor(x / grid_size)`` as int64. The divisor is a device
    tensor because CUDA turns division by a CPU scalar into a multiply by its reciprocal."""
    x.div_(torch.full((), grid_size, dtype=torch.float64, device=x.device))
    return x.floor_().long()


def _segment_ids(seq_lens: tuple[int, ...], device: torch.device) -> Tensor | None:
    """``(N,)`` int64 sequence of every row, or None for a single sequence."""
    if len(seq_lens) == 1:
        return None
    return torch.repeat_interleave(
        torch.arange(len(seq_lens), device=device),
        cu_seqlens_of(seq_lens, device).diff(),
        output_size=sum(seq_lens),
    )


def _segment_bounds(grid: Tensor, batch: Tensor | None, num_seqs: int) -> tuple[Tensor, Tensor]:
    """``(B, 3)`` int64 per-sequence min and max of ``(N, 3)`` int64 grid (0 when empty).

    Row i of sequence b reduces into slot ``(b, i % _MIN_SLOTS)`` first: a plain scatter onto B
    rows serializes its CUDA atomics (2.4 ms for 1.2M rows in 9 sequences, 0.1 ms this way).
    """
    if batch is None:
        if grid.shape[0] == 0:
            return grid.new_zeros(1, 3), grid.new_zeros(1, 3)
        low, high = torch.aminmax(grid, dim=0, keepdim=True)
        return low, high
    slot = torch.arange(grid.shape[0], device=grid.device)
    slot = slot.bitwise_and_(_MIN_SLOTS - 1).add_(batch, alpha=_MIN_SLOTS)[:, None].expand(-1, 3)
    out = []
    limits = torch.iinfo(torch.int64)
    for reduce, empty in (("amin", limits.max), ("amax", limits.min)):
        acc = grid.new_full((num_seqs * _MIN_SLOTS, 3), empty)
        acc.scatter_reduce_(0, slot, grid, reduce)
        acc = acc.view(num_seqs, _MIN_SLOTS, 3)
        acc = acc.amin(1) if reduce == "amin" else acc.amax(1)
        out.append(acc.masked_fill_(acc == empty, 0))
    return out[0], out[1]


def _check_bounds(low: int, high: int) -> None:
    if low < 0 or high >= _GRID_LIMIT:
        raise ValueError(
            f"grid must lie in [0, {_GRID_LIMIT}) per axis after re-basing, got [{low}, {high}]; "
            "use a coarser grid_size or split the sequence"
        )


def _to(v: GridSampled, device: torch.device) -> GridSampled:
    return GridSampled(
        v.index.to(device),
        v.grid_coord.to(device),
        v.origin.to(device),
        v.seq_lens,
        v.cu_seqlens.to(device),
        None if v.inverse is None else v.inverse.to(device),
        None if v.rank is None else v.rank.to(device),
    )


def _check_coord(coord: Tensor) -> None:
    if not coord.dtype.is_floating_point or coord.dim() != 2 or coord.shape[1] != 3:
        raise TypeError(
            f"coord must be an (N, 3) float tensor, got {coord.dtype} {tuple(coord.shape)}"
        )


def _check_grid_size(grid_size: float) -> None:
    if isinstance(grid_size, bool) or not isinstance(grid_size, int | float):
        raise TypeError(f"grid_size must be a float, got {type(grid_size).__name__}")
    if not (math.isfinite(grid_size) and grid_size > 0):
        raise ValueError(f"grid_size must be positive and finite, got {grid_size}")


def _check_pick(pick: str) -> None:
    if pick not in get_args(Pick):
        raise ValueError(f"pick must be one of {get_args(Pick)}, got {pick!r}")


def _check_seq_lens(seq_lens: Sequence[int] | None, n: int) -> tuple[int, ...]:
    if seq_lens is None:
        return (n,)
    out = tuple(seq_lens)
    if not out or any(type(s) is not int or s < 0 for s in out) or sum(out) != n:
        raise ValueError(f"seq_lens must be non-negative ints summing to {n} rows, got {out}")
    if len(out) > _MAX_SEQS:
        raise ValueError(f"at most {_MAX_SEQS} sequences fit in a key, got {len(out)}")
    return out
