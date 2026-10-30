# Vendored from Pointcept2 src/pointcept2/structures/point.py @ 771a949 (simplified for pointtransformerv3).
"""Point clouds as the model sees them: ``Point`` (the user-facing attribute dict) and
``PointGrid`` / ``Level`` (the geometry of a packed batch across grid levels).

``Point`` is a ``dict`` with attribute access, the input and output type of
``PointTransformerV3.forward`` (Pointcept's ``Point``, without addict: a missing key raises
``AttributeError`` / ``KeyError`` instead of returning an empty dict).

``PointGrid`` holds the cached index tables of each level.

``PointGrid`` holds the points (one per grid cell) of B samples at one resolution and builds,
once per level and cached, every index table the feature network needs there: the sample index of
each row, kernel maps, serialization orders, attention patches and the pooling index to the next
level. All data-dependent sizes are found here. Feature-side ops (``subm_conv3d``,
``varlen_attention``, ``pool``/``unpool``) take only these tensors and host ints, so a model
forward given a built level hierarchy has no host sync and can be captured by
``torch.compile(fullgraph=True)`` or ``torch.export``. The only host syncs are the coarse sample
sizes in ``downsample``, one per level.

Cache keys (``PointGrid.cache``) are tuples that start with the method name: ``("batch_index",)``,
``("kmap", k)``, ``("serialized", order)``, ``("patch_layout", patch_size, tail)``,
``("patches", patch_size, order, tail)`` and ``("downsample", stride, policy, order)``. The spconv
backend keeps its rulebooks in the same dict under ``("spconv", ...)``.

Row order: level 0 keeps the rows it was given. A downsampled level has one row per parent cell,
in the serialization order it was pooled along (``downsample(order=...)``, default ``"z"``), and
records that order as ``row_order``. Its ``serialized(row_order).order`` is then the identity, and
the next ``downsample`` along the same order finds its rows already grouped by parent cell, so
``pool``/``unpool`` skip their gathers.
"""

from __future__ import annotations

import os
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import torch
from torch import Tensor

from pointtransformerv3._util import suggest
from pointtransformerv3.ops.neighbors import build_kmap, kmap_from_coarse
from pointtransformerv3.ops.patches import PatchLayout, Tail, build_patches
from pointtransformerv3.ops.pooling import PoolIndex, build_pool_index
from pointtransformerv3.ops.pooling import pool as pool_rows
from pointtransformerv3.ops.serialization import MAX_DEPTH, ORDERS, Order, serialize

__all__ = [
    "POLICIES",
    "Level",
    "PatchPartition",
    "Point",
    "PointGrid",
    "Policy",
    "Serialized",
    "auto_depth",
    "offset_to_seq_lens",
]

Policy = Literal["shift", "reencode"]
POLICIES: tuple[Policy, ...] = get_args(Policy)

_TAILS = get_args(Tail)


@dataclass(frozen=True)
class Serialized:
    """The rows of a level along one space-filling curve."""

    code: Tensor  # (N,) int64 serialize(grid_coord, batch, depth, order): sample bits on top
    order: Tensor  # (N,) int64 rows sorted by code
    inverse: Tensor  # (N,) int64 position of each row in ``order`` (``order[inverse] == arange``)


@dataclass(frozen=True)
class PatchPartition:
    """Attention patches of a level along one serialization, as rows (``PatchLayout`` positions
    mapped through ``Serialized.order``). Patch i has the queries ``q_rows[cu_q[i]:cu_q[i + 1]]``
    and the keys/values ``kv_rows[cu_k[i]:cu_k[i + 1]]``; ``q_rows`` is ``Serialized.order``, so
    every row is a query exactly once and ``out[Serialized.inverse]`` puts outputs back in rows."""

    q_rows: Tensor  # (N,) int64
    kv_rows: Tensor  # (Tk,) int64, Tk >= N (tail patches of the "overlap" mode repeat keys)
    cu_q: Tensor  # (P + 1,) int32
    cu_k: Tensor  # (P + 1,) int32
    max_q: int  # host
    max_k: int  # host


@dataclass(eq=False)
class PointGrid:
    """Unique grid cells of B samples at one resolution, with their cached index tables.

    Implements ``SparseGrid`` (``ops/sparse_conv``). Rows are sample-contiguous.

    Attributes:
        coord: ``(N, 3)`` float32 point (or child-mean) coordinates in meters.
        grid_coord: ``(N, 3)`` int32 grid coordinates at this level, unique per sample, in
            ``[0, 2**depth)``.
        cu_seqlens: ``(B + 1,)`` int32 sample offsets on the rows' device.
        seq_lens: host mirror of the sample sizes.
        grid_size: grid cell edge in meters at this level.
        depth: serialization bits per axis at this level (level 0: the model's Config value, or
            the batch's own with ``serialization_depth="auto"``, see ``auto_depth``).
        level: 0 for the input resolution, +1 per ``downsample``.
        parent: the finer level this one was pooled from.
        pool: the ``PoolIndex`` from ``parent``'s rows to this level's rows.
        row_order: the serialization the rows are sorted by, if any (set by ``downsample``; a
            caller may set it for level 0 when its rows already follow that curve).
        cache: index tables built so far (see the module doc for the keys).
    """

    coord: Tensor
    grid_coord: Tensor
    cu_seqlens: Tensor
    seq_lens: tuple[int, ...]
    grid_size: float
    depth: int
    level: int = 0
    parent: PointGrid | None = field(default=None, repr=False)
    pool: PoolIndex | None = field(default=None, repr=False)
    row_order: Order | None = None
    cache: dict[Hashable, Any] = field(default_factory=dict, repr=False)
    _code_shift: int | None = field(default=None, init=False, repr=False)  # "shift" codes

    def __post_init__(self) -> None:
        self.seq_lens = tuple(self.seq_lens)
        if any(type(n) is not int or n < 0 for n in self.seq_lens):
            raise ValueError(f"seq_lens must be non-negative host ints, got {self.seq_lens}")
        b, n = len(self.seq_lens), sum(self.seq_lens)
        _check_tensor("coord", self.coord, torch.float32, (n, 3), "sum(seq_lens)")
        _check_tensor("grid_coord", self.grid_coord, torch.int32, (n, 3), "sum(seq_lens)")
        _check_tensor("cu_seqlens", self.cu_seqlens, torch.int32, (b + 1,), "len(seq_lens) + 1")
        devices = {t.device for t in (self.coord, self.grid_coord, self.cu_seqlens)}
        if len(devices) > 1:
            raise ValueError(
                "coord, grid_coord and cu_seqlens must share a device, got "
                f"{sorted(map(str, devices))}"
            )
        if isinstance(self.grid_size, bool) or not isinstance(self.grid_size, int | float):
            raise TypeError(f"grid_size must be a float (meters), got {self.grid_size!r}")
        self.grid_size = float(self.grid_size)
        if not self.grid_size > 0:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise TypeError(f"depth must be an int (bits per axis), got {self.depth!r}")
        if not 1 <= self.depth <= MAX_DEPTH:
            raise ValueError(f"depth must be in 1..{MAX_DEPTH}, got {self.depth}")
        if self.row_order is not None:
            _check_order(self.row_order)
        if os.environ.get("PTV3_DEBUG") == "1":
            self._debug_checks()

    @property
    def num_rows(self) -> int:
        return self.grid_coord.shape[0]

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)

    @property
    def device(self) -> torch.device:
        return self.grid_coord.device

    def batch_index(self) -> Tensor:
        """``(N,)`` int64 sample index of every row (cached, sync-free)."""
        key = ("batch_index",)
        if key not in self.cache:
            self.cache[key] = torch.repeat_interleave(
                torch.arange(self.batch_size, device=self.device),
                self.cu_seqlens.diff(),
                output_size=self.num_rows,
            )
        return self.cache[key]

    def kmap(self, kernel_size: int) -> Tensor:
        """``(N, kernel_size**3)`` int32 submanifold kernel map (cached, sync-free).

        Top-down (I3) when this level has a stride-2 coarse level already (``downsample(2)`` was
        called): ``kmap_from_coarse`` of the coarse level's ``kmap(3)`` for kernel sizes 1, 3 and
        5; otherwise ``build_kmap``. Both give the same map; the top-down one is several times
        cheaper (1M rows, k=5: about 0.6 against 5.2 ms on CUDA), so build the level chain
        before the first conv (PT-v3m1 does).
        """
        key = ("kmap", kernel_size)
        if key not in self.cache:
            coarse = self._coarse_stride2() if kernel_size in (1, 3, 5) else None
            if coarse is not None and coarse.pool is not None:
                self.cache[key] = kmap_from_coarse(
                    coarse.kmap(3), coarse.pool.inverse, self.grid_coord, kernel_size
                )
            else:
                self.cache[key] = build_kmap(self.grid_coord, self.batch_index(), kernel_size)
        return self.cache[key]

    def serialized(self, order: Order) -> Serialized:
        """Codes and row order along ``order`` at this level's depth (cached, sync-free).

        On a level made by ``downsample(policy="shift")`` the codes are the finer level's codes
        shifted right by 3 bits per halving, when the finer level has them; otherwise they are
        encoded here. Both give the same codes (prefix property).
        """
        key = ("serialized", order)
        if key in self.cache:
            return self.cache[key]
        _check_order(order)
        code = self._shifted_code(order)
        if code is None:
            batch_size = self.batch_size or None  # host bit-budget check; nothing to check if 0
            code = serialize(
                self.grid_coord, self.batch_index(), self.depth, order, batch_size=batch_size
            )
        if order == self.row_order:
            rows = torch.arange(self.num_rows, device=self.device)
            result = Serialized(code, rows, rows)
        else:
            perm = torch.argsort(code, stable=True)
            inverse = torch.empty_like(perm).scatter_(
                0, perm, torch.arange(self.num_rows, device=self.device)
            )
            result = Serialized(code, perm, inverse)
        self.cache[key] = result
        return result

    def patch_layout(self, patch_size: int, tail: Tail = "overlap") -> PatchLayout:
        """Patch positions of this level's sample sizes (``build_patches``; cached). They do not
        depend on the serialization, so all orders share one layout. It holds both forms the
        attention backends read: the packed layout (``cu_q``, ``cu_k``, ``kv_pos``) and the
        range table of the ``triton_d16`` kernels (``table``), built in one host pass and one
        transfer."""
        key = ("patch_layout", patch_size, tail)
        if key not in self.cache:
            self.cache[key] = build_patches(self.seq_lens, patch_size, tail, self.device)
        return self.cache[key]

    def patches(self, patch_size: int, order: Order, tail: Tail = "overlap") -> PatchPartition:
        """Attention patches along ``order`` as rows: ``q_rows = order``,
        ``kv_rows = order[kv_pos]`` (cached, sync-free)."""
        key = ("patches", patch_size, order, tail)
        if key not in self.cache:
            if tail not in _TAILS:
                raise ValueError(f"tail must be one of {_TAILS}, got {tail!r}")
            layout = self.patch_layout(patch_size, tail)
            rows = self.serialized(order).order
            kv_rows = rows if layout.kv_pos is layout.q_pos else rows[layout.kv_pos]
            self.cache[key] = PatchPartition(
                q_rows=rows,
                kv_rows=kv_rows,
                cu_q=layout.cu_q,
                cu_k=layout.cu_k,
                max_q=layout.max_q,
                max_k=layout.max_k,
            )
        return self.cache[key]

    def downsample(
        self,
        stride: int = 2,
        policy: Policy = "shift",
        *,
        order: Order = "z",
        depth: int | None = None,
    ) -> PointGrid:
        """The next coarser level: one row per parent cell ``grid_coord >> log2(stride)`` of each
        sample (cached). Reading the coarse sample sizes back is the one host sync of a level.

        Coarse rows follow ``order``: the clusters are the runs of
        ``serialized(order).code[order] >> 3 * log2(stride)``, so no extra sort is needed (see
        ``ops/pooling.py`` for the measured variants). The coarse ``coord`` is the mean of the
        children's, ``grid_coord`` is the parent cell, ``depth`` drops by ``log2(stride)`` and
        ``pool`` maps this level's rows to the coarse rows.

        Args:
            stride: a power of two >= 2, below ``2**depth``.
            policy: how the coarse level's codes are made. ``"shift"`` (PTv3 m1): the child codes
                shifted right by ``3 * log2(stride)`` bits. ``"reencode"`` (m2): ``serialize`` the
                coarse grid at the coarse depth. For power-of-two strides the two are equal (prefix
                property); ``"shift"`` saves the encode.
            order: the serialization the coarse rows follow (use the model's first order).
            depth: ``"reencode"`` only: the coarse level's serialization depth instead of
                ``self.depth - log2(stride)`` (``serialization_depth="auto"``: the coarse grid's
                own, see ``auto_depth``). It must hold the coarse grid.
        """
        key = ("downsample", stride, policy, order, depth)
        if key in self.cache:
            return self.cache[key]
        bits = _stride_bits(stride, self.depth)
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        if depth is not None and policy != "reencode":
            raise ValueError('downsample(depth=...) needs policy="reencode" ("shift" shifts codes)')
        s = self.serialized(order)
        grouped = order == self.row_order
        keys = (s.code if grouped else s.code[s.order]) >> 3 * bits
        index, cu_seqlens, seq_lens = build_pool_index(
            keys, s.order, self.cu_seqlens, grouped=grouped
        )
        coarse = PointGrid(
            coord=pool_rows(self.coord, index, "mean"),
            grid_coord=self.grid_coord.index_select(0, index.head) >> bits,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            grid_size=self.grid_size * stride,
            depth=self.depth - bits if depth is None else depth,
            level=self.level + 1,
            parent=self,
            pool=index,
            row_order=order,
        )
        if policy == "shift":
            coarse._code_shift = 3 * bits
        self.cache[key] = coarse
        return coarse

    def _coarse_stride2(self) -> PointGrid | None:
        """A cached ``downsample(2, ...)`` of this level (any policy and order: the same cells),
        or None."""
        for key, value in self.cache.items():
            if isinstance(key, tuple) and key[:2] == ("downsample", 2):
                return value
        return None

    def _shifted_code(self, order: Order) -> Tensor | None:
        """ "shift" policy: the finer level's cached codes at the head rows, shifted; else None."""
        parent, index, shift = self.parent, self.pool, self._code_shift
        if parent is None or index is None or shift is None:
            return None
        finer = parent.cache.get(("serialized", order))
        if finer is None:
            return None
        return finer.code[index.head] >> shift

    def _debug_checks(self) -> None:
        """PTV3_DEBUG=1 value checks (host syncs)."""
        cu = self.cu_seqlens.tolist()
        expected = [0]
        for n in self.seq_lens:
            expected.append(expected[-1] + n)
        if cu != expected:
            raise ValueError(f"cu_seqlens {cu} does not match seq_lens {self.seq_lens}")
        if self.num_rows:
            low, high = int(self.grid_coord.min()), int(self.grid_coord.max())
            if low < 0 or high >= 1 << self.depth:
                raise ValueError(
                    f"grid_coord must lie in [0, 2**{self.depth}) for depth={self.depth}, got "
                    f"[{low}, {high}]; subtract the per-sample minimum or raise depth"
                )
        if self.row_order is not None and self.num_rows > 1:
            code = serialize(self.grid_coord, self.batch_index(), self.depth, self.row_order)
            if bool((code[1:] < code[:-1]).any()):
                raise ValueError(f"rows are not sorted along row_order={self.row_order!r}")


@dataclass(eq=False)
class Level:
    """Features on one ``PointGrid`` level; ``parent`` is the finer ``Level`` it was pooled from.

    Shapes:
        feat: ``(N, ...)``, row i at ``grid.grid_coord[i]``.
    """

    feat: Tensor
    grid: PointGrid
    parent: Level | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feat, Tensor) or self.feat.dim() == 0:
            raise TypeError("feat must be a tensor with a leading row dimension")
        if self.feat.shape[0] != self.grid.num_rows:
            raise ValueError(
                f"feat has {self.feat.shape[0]} rows but the grid has {self.grid.num_rows}"
            )
        if self.parent is not None and self.parent.grid is not self.grid.parent:
            raise ValueError("parent must be the Level on the finer level grid.parent")


def _check_tensor(
    name: str, value: object, dtype: torch.dtype, shape: tuple[int, ...], rows: str
) -> None:
    """dtype and trailing dims -> TypeError; a wrong row count -> ValueError."""
    if not isinstance(value, Tensor) or value.dtype != dtype or value.shape[1:] != shape[1:]:
        got = f"{value.dtype} {tuple(value.shape)}" if isinstance(value, Tensor) else repr(value)
        raise TypeError(f"{name} must be a {dtype} tensor of shape {shape}, got {got}")
    if value.shape[0] != shape[0]:
        raise ValueError(f"{name} has {value.shape[0]} rows, expected {rows} = {shape[0]}")


def _check_order(order: str) -> None:
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}{suggest(str(order), ORDERS)}; one of {ORDERS}")


def _stride_bits(stride: int, depth: int) -> int:
    """log2(stride) for a power-of-two stride that leaves at least one bit of depth."""
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise TypeError(f"stride must be an int, got {stride!r}")
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    bits = stride.bit_length() - 1
    if bits >= depth:
        raise ValueError(f"stride {stride} needs depth > {bits}, this level has depth {depth}")
    return bits


# ---- the user-facing point dict


class Point(dict):
    """A point cloud (or a packed batch of them) as a ``dict`` with attribute access.

    Keys the model reads: ``coord`` ``(N, 3)`` float meters, ``feat`` ``(N, C)`` float,
    ``grid_coord`` ``(N, 3)`` int (or ``grid_size`` to compute it from ``coord``), and ``offset``
    ``(B,)`` (cumulative sample ends, Pointcept's convention) or ``batch`` ``(N,)`` (sample index
    of each row). Any other key is carried along. The model's output is a ``Point`` too; an
    encoder-only model's output also holds ``pooling_parent`` (the finer level's ``Point``) and
    ``pooling_inverse`` (``(N_parent,)`` int64 row of this level for every parent row), as in
    Pointcept / Sonata, so ``pointtransformerv3.upcast`` and existing code can walk back to level 0.

    Unlike Pointcept's addict-based ``Point``, a missing attribute raises ``AttributeError``.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"Point has no key {name!r}{suggest(name, list(self.keys()))}; keys: "
                f"{sorted(self.keys())}"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self) -> list[str]:
        return [*super().__dir__(), *(k for k in self.keys() if isinstance(k, str))]

    def copy(self) -> Point:  # dict.copy would return a plain dict
        return Point(self)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={_brief(v)}" for k, v in self.items())
        return f"Point({items})"


def _brief(v: Any) -> str:
    if isinstance(v, Tensor):
        return f"<{str(v.dtype).removeprefix('torch.')} {tuple(v.shape)} {v.device}>"
    if isinstance(v, Point):
        return "Point(...)"
    return repr(v)


def offset_to_seq_lens(offset: Tensor) -> tuple[int, ...]:
    """Host sample sizes of Pointcept's ``offset`` (cumulative sample ends, ``(B,)``)."""
    ends = [int(v) for v in offset.tolist()]
    lens, start = [], 0
    for end in ends:
        if end < start:
            raise ValueError(f"offset must be non-decreasing cumulative sample ends, got {ends}")
        lens.append(end - start)
        start = end
    return tuple(lens)


def auto_depth(max_coord: int, *, rule: str, minimum: int = 1) -> int:
    """The serialization depth of a grid whose largest coordinate is ``max_coord``.

    ``rule``: ``"auto"`` (PTv3 ``model.py`` / Pointcept PT-v3m1: ``max_coord.bit_length()``, the
    smallest cube holding the grid) or ``"auto+1"`` (Sonata / Pointcept PT-v3m2:
    ``(max_coord + 1).bit_length()``, one more bit when ``max_coord + 1`` is a power of two).
    A depth below ``minimum`` (tiny clouds, where the pooling levels need more bits) is raised in
    steps of 3: Hilbert and z-order codes of a grid are the same at depth d and d + 3, so the
    curve does not change.
    """
    if rule not in ("auto", "auto+1"):
        raise ValueError(f"unknown depth rule {rule!r}; use 'auto' or 'auto+1'")
    depth = (max_coord + (rule == "auto+1")).bit_length()
    while depth < max(minimum, 1):
        depth += 3
    return depth
