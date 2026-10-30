# Vendored from Pointcept2 src/pointcept2/ops/pooling.py @ 771a949 (simplified for pointtransformerv3).
"""Pooling between grid levels: one coarse row per parent cell, segment reductions, unpooling.

The index (which fine rows form each coarse row) is geometry. ``build_pool_index`` builds it once
per level, from rows already sorted by their parent key, with the level's one host sync, and
``PointGrid.downsample`` caches it as a ``PoolIndex``. ``pool`` and ``unpool`` take that index and
the features only: no host sync, no data-dependent shapes, so the feature network stays
capturable by ``torch.compile(fullgraph=True)`` and ``torch.export``.

The recipe is the one shared by PTv2 ``GridPool`` and PTv3 ``SerializedPooling``/``GridPooling``
(``torch.unique`` over a packed 1-D key, a sort of the cluster ids, ``ptr = cumsum(counts)``,
``head = sort[ptr[:-1]]``, ``segment_csr`` reductions, unpooling by indexing with the inverse),
minus its two sorts: ``PointGrid.downsample`` passes a serialization order it already has.
Children of a parent cell are contiguous in every serialization order (prefix property:
``serialize(g, b, d, o) >> 3 == serialize(g >> 1, b, d - 1, o)``), so the clusters are the runs of
``code[order] >> 3`` and ``order`` is the cluster sort. The run starts, cluster ids, per-sample
counts and the inverse come from one cumsum; ``ptr`` from a ``searchsorted`` once the host knows
the coarse size.

Measured on real ScanNet packs (``tests/bench/make_packs.py``), 4 levels of stride 2 from depth 15
on the 1M pack (1,000,000 -> 419,002 -> 122,214 -> 32,591 -> 8,233 rows), RTX PRO 6000 (sm_120),
torch 2.12.1, bf16. The GPU is shared with other jobs: wall times are medians of 30 interleaved
runs taken while the host was quiet, GPU times come from the profiler. One downsample = index,
coarse grid, mean coord, coarse codes and orders of 2 serializations, incl. the sync (ms):

    variant                                     L0->1  L1->2  L2->3  L3->4   GPU L0 (kernels)
    unique(dim=0) of (batch, x, y, z) (m2, m3)   3.89   3.18   2.05   1.99   1.48 (85)
    unique of a packed 1-D key + sort (PC m1)    1.47   1.33   1.60   1.43   0.47 (75)
    one stable sort of the packed key            1.33   1.21   1.15   1.16   0.38 (66)
    runs of a serialization order (this)         0.97   0.77   0.78   0.87   0.18 (39)

All four give the same parent cells, coarse grid, mean coord and codes; the ``unique`` variants
also need a second sync to read the coarse sample sizes. Wall time is mostly kernel
launches and the sync (WSL2), so it hardly changes with the level or the pack size (400k: within
0.2 ms). Coarse rows come out in the reused order, so their ``Serialized.order`` for it is the
identity and the next downsample along the same order finds them already grouped
(``PoolIndex.grouped``): no gather in ``pool``/``unpool``.

Gathers go through ``ops.gather`` (``permute_rows`` for ``pool``'s sort, ``gather_rows`` for
``unpool``), which avoids torch 2.12's slow vectorized gather on sm_120 (~0.48 ms per 1M rows of
16..256 bytes). 1M rows, bf16, plain / ``gather_rows`` ms: permutation C=32 0.49 / 0.19, C=64
0.48 / 0.27, C=128 0.49 / 0.45; unpool-like 1M rows from 419k C=32 0.49 / 0.14, C=64 0.49 / 0.22,
C=128 0.49 / 0.34. With them, ``tests/bench/bench_ops.py`` (1M pack, level 0 -> 1, bf16 fwd+bwd)
measures pool max C=32 / 48: 0.62-0.65 / 0.95-0.97 ms and unpool 0.51-0.72 / 0.57-0.76 ms (two
runs on a shared GPU). The table below was measured before, with a padded-source gather for rows
<= 128 B (fwd+bwd, 1M pack, wall ms):

                                          C=32   C=64   C=128
    level 0 -> 1 (1M -> 419k, not grouped)
      pool max, autograd feat[sort]       1.17   1.64   2.64
      pool max, this                      0.70   1.43   2.26
      pool max, had the rows been grouped 0.37   0.74   1.43
      unpool, autograd feat[inverse]      0.72   0.83   0.97
      unpool, this                        0.56   0.66   0.97  (plain indexing above 128 B rows)
    level 1 -> 2 (419k -> 122k, grouped)
      unpool, autograd feat[inverse]      0.39   0.40   0.44  (GPU 0.29 / 0.31 / 0.34)
      unpool, this                        0.34   0.31   0.38  (GPU 0.05 / 0.13 / 0.32)

``pool``'s gather is a permutation, so its backward is an ``index_copy_`` (C=128: 0.41 ms against
0.77 for autograd's ``index_add_``). ``unpool``'s backward is a segment sum when the fine rows are
grouped and torch's sort-based ``index_put_(accumulate=True)`` otherwise; both are deterministic,
unlike ``index_add_`` with repeated indices.

MPS has no ``segment_reduce`` (torch 2.12 falls back to the CPU, or raises without
``PYTORCH_ENABLE_MPS_FALLBACK=1``), so on MPS ``pool`` reduces with ``scatter_reduce`` over
``index.inverse`` (``"mean"`` as a sum divided by the child counts) and ``unpool``'s grouped
backward with ``index_add_``.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Any, Literal, get_args

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

from pointtransformerv3.ops.gather import gather_rows, permute_rows, slow_gather

__all__ = ["REDUCES", "PoolIndex", "Reduce", "build_pool_index", "pool", "unpool"]

Reduce = Literal["max", "mean", "min", "sum"]
REDUCES: tuple[Reduce, ...] = get_args(Reduce)


@dataclass(frozen=True)
class PoolIndex:
    """Which fine rows form each coarse row (one coarse row per parent cell).

    Coarse row j has the fine rows ``sort[ptr[j]:ptr[j + 1]]``; fine row i belongs to coarse row
    ``inverse[i]``.
    """

    inverse: Tensor  # (N,) int64 coarse row of every fine row; unpool gathers with it
    sort: Tensor  # (N,) int64 permutation of the fine rows that groups them by coarse row
    ptr: Tensor  # (M + 1,) int64 offsets of each coarse row's children in ``sort``
    head: Tensor  # (M,) int64 first child of each coarse row, ``sort[ptr[:-1]]``
    grouped: bool = False  # host: fine rows are already grouped, ``sort`` is the identity

    def __post_init__(self) -> None:
        for name in ("inverse", "sort", "ptr", "head"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.dtype != torch.int64 or value.dim() != 1:
                raise TypeError(
                    f"PoolIndex.{name} must be a 1-D int64 tensor, got {_describe(value)}"
                )
        n, m = self.inverse.shape[0], self.head.shape[0]
        if self.sort.shape[0] != n or self.ptr.shape[0] != m + 1:
            raise ValueError(
                f"PoolIndex shapes disagree: inverse ({n},), sort {tuple(self.sort.shape)}, "
                f"ptr {tuple(self.ptr.shape)}, head ({m},); expected sort ({n},), ptr ({m + 1},)"
            )
        devices = {t.device for t in (self.inverse, self.sort, self.ptr, self.head)}
        if len(devices) > 1:
            raise ValueError(
                f"PoolIndex tensors are on several devices: {sorted(map(str, devices))}"
            )
        debug = os.environ.get("PTV3_DEBUG") == "1"
        if debug and self.grouped and not torch.equal(self.sort, torch.arange(n).to(self.sort)):
            raise ValueError("grouped=True needs the fine rows already grouped (sort == arange)")

    @property
    def num_fine(self) -> int:
        return self.inverse.shape[0]

    @property
    def num_coarse(self) -> int:
        return self.head.shape[0]


def build_pool_index(
    keys: Tensor, sort: Tensor, cu_seqlens: Tensor, *, grouped: bool = False
) -> tuple[PoolIndex, Tensor, tuple[int, ...]]:
    """Group rows into clusters of equal keys (geometry side; one host sync on CUDA).

    Args:
        keys: ``(N,)`` int64 cluster key at each sorted position (``keys[i]`` is the key of row
            ``sort[i]``). Equal keys must be contiguous and must not cross samples: sample b's rows
            fill positions ``cu_seqlens[b]:cu_seqlens[b + 1]`` of the sorted order, as they do when
            the sample index sits in the key's top bits (``serialize`` codes, packed keys).
        sort: ``(N,)`` int64 permutation of the rows into that order.
        cu_seqlens: ``(B + 1,)`` int32 sample offsets of the rows.
        grouped: the rows are already in that order (``sort`` is the identity); saves a scatter
            here and every gather in ``pool``/``unpool``.

    Returns:
        The index, coarse ``cu_seqlens`` ``(B + 1,)`` int32 on the keys' device, and the coarse
        ``seq_lens`` (host). Clusters are numbered in sorted order. Reading the coarse sizes back
        is the only host sync; key checks that need more run under ``PTV3_DEBUG=1``.
    """
    n = keys.shape[0]
    for name, value in (("keys", keys), ("sort", sort)):
        if not isinstance(value, Tensor) or value.dtype != torch.int64 or value.shape != (n,):
            raise TypeError(f"{name} must be an ({n},) int64 tensor, got {_describe(value)}")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.dim() != 1:
        raise TypeError(f"cu_seqlens must be a 1-D int32 tensor, got {_describe(cu_seqlens)}")
    if not keys.device == sort.device == cu_seqlens.device:
        raise ValueError(
            f"keys, sort and cu_seqlens must share a device, got {keys.device}, {sort.device}, "
            f"{cu_seqlens.device}"
        )
    if os.environ.get("PTV3_DEBUG") == "1":
        _check_keys(keys, sort, cu_seqlens)

    new = torch.ones(n, dtype=torch.bool, device=keys.device)  # a cluster starts here
    torch.ne(keys[1:], keys[:-1], out=new[1:])
    starts = F.pad(new.cumsum(0), (1, 0))  # (N + 1,) clusters starting before each position
    cu = starts[cu_seqlens]
    cluster = starts[1:].sub_(1)  # (N,) cluster of each sorted position, non-decreasing
    inverse = cluster if grouped else torch.empty_like(cluster).scatter_(0, sort, cluster)
    bounds = cu.tolist()  # the host sync: coarse sample sizes
    seq_lens = tuple(b - a for a, b in itertools.pairwise(bounds))
    num = bounds[-1]
    ptr = torch.searchsorted(cluster, torch.arange(num + 1, device=keys.device))
    head = ptr[:-1] if grouped else sort[ptr[:-1]]
    index = PoolIndex(inverse=inverse, sort=sort, ptr=ptr, head=head, grouped=grouped)
    return index, cu.int(), seq_lens


def pool(feat: Tensor, index: PoolIndex, reduce: Reduce = "max") -> Tensor:
    """Reduce the fine rows of each coarse row (``torch.segment_reduce`` over ``index.ptr``).

    Shapes:
        feat: ``(N, ...)`` one row per fine row.
        returns: ``(M, ...)`` one row per coarse row, in feat's dtype.

    Sync-free. ``"max"``/``"min"`` split the gradient evenly between tied children.
    """
    if reduce not in REDUCES:
        raise ValueError(f"reduce must be one of {REDUCES}, got {reduce!r}")
    _check_rows(feat, index.num_fine, "feat", "fine")
    if feat.device.type == "mps":
        return _pool_scatter(feat, index, reduce)
    src = feat if index.grouped else permute_rows(feat, index.sort)
    return torch.segment_reduce(src, reduce, offsets=index.ptr, axis=0, unsafe=True)


_SCATTER_REDUCE = {"max": "amax", "min": "amin", "sum": "sum", "mean": "sum"}


def _pool_scatter(feat: Tensor, index: PoolIndex, reduce: Reduce) -> Tensor:
    """``pool`` without ``segment_reduce`` (MPS): a ``scatter_reduce`` of each fine row into its
    coarse row. Every coarse row has a child, so ``include_self=False`` leaves no initial value
    in the result. Ties in ``"max"``/``"min"`` share the gradient evenly (``segment_reduce`` does
    so only for positive gradients). The initial value is -inf / +inf for them: torch 2.12's
    backward also counts an initial value equal to the result as a tie."""
    target = index.inverse.view(-1, *(1,) * (feat.dim() - 1)).expand_as(feat)
    fill = {"max": float("-inf"), "min": float("inf")}.get(reduce, 0.0)
    out = feat.new_full((index.num_coarse, *feat.shape[1:]), fill)
    out = out.scatter_reduce(0, target, feat, _SCATTER_REDUCE[reduce], include_self=False)
    if reduce == "mean":
        counts = index.ptr.diff().to(out.dtype)
        out = out / counts.view(-1, *(1,) * (feat.dim() - 1))
    return out


def unpool(feat: Tensor, index: PoolIndex) -> Tensor:
    """Copy every coarse row to its fine rows: ``feat[index.inverse]``.

    Shapes:
        feat: ``(M, ...)`` one row per coarse row.
        returns: ``(N, ...)`` one row per fine row.

    Sync-free; the backward sums each coarse row's gradients deterministically.
    """
    _check_rows(feat, index.num_coarse, "feat", "coarse")
    if not (index.grouped or slow_gather(feat, index.inverse)):
        return feat[index.inverse]  # torch's backward: a sorted, deterministic index_put_
    return _Unpool.apply(feat, index.inverse, index.ptr, index.grouped)


# ---- autograd


class _Unpool(torch.autograd.Function):
    """``x[inverse]``; the backward sums over each coarse row's children without atomics."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, inverse: Tensor, ptr: Tensor, grouped: bool) -> Tensor:
        ctx.save_for_backward(inverse, ptr)
        ctx.grouped = grouped
        ctx.num_coarse = x.shape[0]
        return gather_rows(x, inverse)

    @staticmethod
    @once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, None, None, None]:
        (grad,) = grad_outputs
        inverse, ptr = ctx.saved_tensors
        if ctx.grouped and grad.device.type == "mps":  # no segment_reduce on MPS
            out = grad.new_zeros(ctx.num_coarse, *grad.shape[1:])
            out.index_add_(0, inverse, grad)
        elif ctx.grouped:
            out = torch.segment_reduce(grad, "sum", offsets=ptr, axis=0, unsafe=True)
        else:  # torch sorts the indices first, so the sums are deterministic
            out = grad.new_zeros(ctx.num_coarse, *grad.shape[1:])
            out.index_put_((inverse,), grad, accumulate=True)
        return out, None, None, None


# ---- validation


def _describe(x: object) -> str:
    if isinstance(x, Tensor):
        return f"{x.dtype} tensor of shape {tuple(x.shape)} on {x.device}"
    return type(x).__name__


def _check_rows(feat: Tensor, rows: int, name: str, level: str) -> None:
    if not isinstance(feat, Tensor) or feat.dim() == 0:
        raise TypeError(
            f"{name} must be a tensor with a leading row dimension, got {_describe(feat)}"
        )
    if feat.shape[0] != rows:
        raise ValueError(f"{name} has {feat.shape[0]} rows but the index has {rows} {level} rows")


def _check_keys(keys: Tensor, sort: Tensor, cu_seqlens: Tensor) -> None:
    """PTV3_DEBUG checks (host syncs): sort is a permutation, clusters stay inside samples."""
    n = keys.shape[0]
    if not torch.equal(sort.sort().values, torch.arange(n, device=sort.device)):
        raise ValueError("sort must be a permutation of the rows")
    cu = cu_seqlens.long()
    if cu.shape[0] < 1 or int(cu[0]) != 0 or int(cu[-1]) != n:
        raise ValueError(f"cu_seqlens must run from 0 to {n}, got {cu.tolist()}")
    starts = cu[:-1][cu[:-1] < cu[1:]]  # first sorted position of every non-empty sample
    inner = starts[starts > 0]
    if bool((keys[inner] == keys[inner - 1]).any()):
        raise ValueError("a cluster key spans two samples; put the sample index in the key")
    _, counts = torch.unique_consecutive(keys, return_counts=True)
    if counts.shape[0] != torch.unique(keys).shape[0]:
        raise ValueError("equal keys must be contiguous in the sorted order")
