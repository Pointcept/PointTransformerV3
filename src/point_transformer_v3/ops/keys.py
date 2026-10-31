# Vendored from Pointcept2 src/pointcept2/ops/keys.py @ 771a949 (simplified for point_transformer_v3).
"""Packed int64 keys of (batch, x, y, z) integer grid coordinates.

Layout: ``key = b << 48 | (x + margin) << 32 | (y + margin) << 16 | (z + margin)``. Sorting keys
orders rows by sample, then x, y, z. ``margin`` shifts every axis so that neighbor offsets in
``[-margin, margin]`` stay inside their 16-bit field (kernel maps use ``kernel_size // 2``).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["AXIS_BITS", "BATCH_SHIFT", "check_key_range", "pack_keys", "unpack_keys"]

BATCH_SHIFT = 48
AXIS_BITS = 16
_AXIS_MASK = (1 << AXIS_BITS) - 1
_MAX_BATCH = 1 << (63 - BATCH_SHIFT)
_INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def pack_keys(grid: Tensor, batch: Tensor, margin: int = 0) -> Tensor:
    """Pack grid ``(N, 3)`` int and batch ``(N,)`` int into ``(N,)`` int64 keys.

    Values are not range-checked here (that needs a host sync); see ``check_key_range``.
    """
    _check_grid(grid)
    if batch.dtype not in _INT_DTYPES or batch.shape != grid.shape[:1]:
        raise TypeError(
            f"batch must be an integer tensor of shape ({grid.shape[0]},), "
            f"got {batch.dtype} {tuple(batch.shape)}"
        )
    g = grid.long()
    if margin:
        g = g + margin
    key = batch.long() << BATCH_SHIFT
    key |= g[:, 0] << 2 * AXIS_BITS
    key |= g[:, 1] << AXIS_BITS
    key |= g[:, 2]
    return key


def unpack_keys(keys: Tensor, margin: int = 0) -> tuple[Tensor, Tensor]:
    """Inverse of ``pack_keys``: ``(N,)`` int64 keys -> grid ``(N, 3)`` int32, batch ``(N,)``
    int64."""
    if keys.dtype != torch.int64 or keys.dim() != 1:
        raise TypeError(f"keys must be a 1-D int64 tensor, got {keys.dtype} {tuple(keys.shape)}")
    grid = torch.stack(
        [(keys >> 2 * AXIS_BITS) & _AXIS_MASK, (keys >> AXIS_BITS) & _AXIS_MASK, keys & _AXIS_MASK],
        dim=1,
    )
    if margin:
        grid -= margin
    return grid.int(), keys >> BATCH_SHIFT


def check_key_range(grid: Tensor, batch_size: int, margin: int) -> None:
    """Raise ``ValueError`` unless ``pack_keys(grid, batch, margin)`` is lossless.

    Needs ``0 <= grid``, ``grid.max() < 2**16 - 2 * margin`` and ``batch_size < 2**15``. Reads
    the grid's range back to the host (a sync on CUDA): call it in data workers or under
    ``PTV3_DEBUG=1``.
    """
    _check_grid(grid)
    if not 0 <= batch_size < _MAX_BATCH:
        raise ValueError(f"batch_size must be in [0, {_MAX_BATCH}), got {batch_size}")
    if grid.numel() == 0:
        return
    low, high = torch.stack([grid.min(), grid.max()]).tolist()
    limit = (1 << AXIS_BITS) - 2 * margin
    if low < 0 or high >= limit:
        raise ValueError(
            f"grid values must lie in [0, {limit}) for margin {margin}, got [{low}, {high}]; "
            "subtract the per-sample minimum or use a coarser grid_size"
        )


def _check_grid(grid: Tensor) -> None:
    if grid.dtype not in _INT_DTYPES or grid.dim() != 2 or grid.shape[1] != 3:
        raise TypeError(
            f"grid must be an (N, 3) integer tensor, got {grid.dtype} {tuple(grid.shape)}"
        )
