# Vendored from Pointcept2 src/pointcept2/ops/seqlens.py @ 771a949 (simplified for pointtransformerv3).
"""Sequence offsets of a pack: ``cu_seqlens`` built on the host from ``seq_lens``.

Lives in ``ops`` because ops build packed outputs (``grid_sample``) and must not import
``structures``; ``structures.batch`` re-exports it for ``Ragged`` users.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = ["cu_seqlens_of"]


def cu_seqlens_of(seq_lens: Sequence[int], device: torch.device | str) -> Tensor:
    """``(B + 1,)`` int32 cumulative offsets ``[0, l0, l0 + l1, ...]`` of host ``seq_lens`` on
    ``device``. Built on the host and copied from pinned memory without a sync."""
    device = torch.device(device)
    offsets = [0, *itertools.accumulate(seq_lens)]
    if offsets[-1] >= 1 << 31:
        raise ValueError(f"{offsets[-1]} rows overflow int32 cu_seqlens; pack fewer rows")
    cu = torch.tensor(offsets, dtype=torch.int32)
    if device.type == "cuda":
        return cu.pin_memory().to(device, non_blocking=True)
    return cu.to(device)
