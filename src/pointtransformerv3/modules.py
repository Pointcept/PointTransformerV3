# Vendored from Pointcept2 src/pointcept2/models/point_transformer_v3/modules.py @ 771a949 (simplified for pointtransformerv3; Rope3d added for Utonia).
"""PTv3 building blocks of ``PointTransformerV3`` (``model.py``). Behavior changes keep the
golden outputs of every preset (``tests/golden``) or update them on purpose.

Every module takes features and the cached geometry of its level (``PointGrid``), never builds an
index itself and issues no host sync, so a forward over a built level hierarchy can be captured.
The Pointcept (PC) counterparts are named in each docstring.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, get_args

import torch
from torch import Tensor, nn
from torch.amp.autocast_mode import is_autocast_available

from pointtransformerv3.layers.drop_path import DropPath
from pointtransformerv3.layers.layer_scale import LayerScale
from pointtransformerv3.layers.mlp import Mlp
from pointtransformerv3.layers.norm import NormKind, make_norm
from pointtransformerv3.layers.sparse_conv import SubMConv3d
from pointtransformerv3.ops.attention import serialized_attention
from pointtransformerv3.ops.patches import Tail
from pointtransformerv3.ops.pooling import PoolIndex, Reduce, pool, unpool
from pointtransformerv3.ops.serialization import MAX_DEPTH, ORDERS, Order
from pointtransformerv3.structure import POLICIES, Level, PointGrid

__all__ = [
    "AUTO_DEPTHS",
    "STEMS",
    "WEIGHT_INITS",
    "AutoDepth",
    "Block",
    "Cpe",
    "DecoderStage",
    "EncoderStage",
    "LinearEmbedding",
    "OrderRows",
    "Pooling",
    "Rope3d",
    "SerializedAttention",
    "Stem",
    "StemEmbedding",
    "Unpooling",
    "WeightInit",
    "check_config",
    "check_input",
    "check_orders",
    "draw_level_orders",
    "drop_path_rates",
    "select_orders",
]

_TAILS: tuple[Tail, ...] = get_args(Tail)
AutoDepth = Literal["auto", "auto+1"]
AUTO_DEPTHS: tuple[AutoDepth, ...] = get_args(AutoDepth)
Stem = Literal["conv", "linear"]
STEMS: tuple[Stem, ...] = get_args(Stem)
WeightInit = Literal["torch", "trunc_normal"]
WEIGHT_INITS: tuple[WeightInit, ...] = get_args(WeightInit)


class OrderRows(NamedTuple):
    """A serialization given by its row permutation instead of its name: ``order`` (rows sorted
    along the curve) and ``inverse`` (``order[inverse] == arange``). ``select_orders`` builds them
    from a device index, so a captured forward does not specialize on which order a block uses."""

    order: Tensor
    inverse: Tensor


# ---- embeddings


class StemEmbedding(nn.Module):
    """``GELU(norm(conv(x)))`` with a bias-free ``SubMConv3d`` stem (PC m1 ``Embedding``: k=5,
    BatchNorm). Under autocast the conv runs in the autocast dtype, as in PC: the released m1
    input (color, normal) holds no coordinates, and an fp32 stem costs 11.1 instead of 6.9 ms
    forward + backward on the 1M pack (``research/m2/p2/bench_stem.log``)."""

    def __init__(
        self, in_channels: int, channels: int, norm: NormKind, kernel_size: int = 5
    ) -> None:
        super().__init__()
        self.conv = SubMConv3d(in_channels, channels, kernel_size, bias=False)
        self.norm = make_norm(norm, channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor, grid: PointGrid) -> Tensor:
        """
        Shapes:
            x: ``(N, in_channels)`` float, rows of ``grid``.
            returns: ``(N, channels)``, in the autocast dtype under autocast.
        """
        return self.act(self.norm(self.conv(x, grid)))


class LinearEmbedding(nn.Module):
    """``GELU(norm(proj(x)))`` with a Linear stem (PC m2 ``Embedding``, without the mask token).

    The input holds coordinates (Sonata: coord, color, normal), so ``proj`` runs in its weight's
    dtype (fp32) with autocast off (C10: bf16 resolves 3.1 cm at 4-8 m, coarser than a grid
    cell). The norm then casts its output to the autocast dtype.
    """

    def __init__(self, in_channels: int, channels: int, norm: NormKind) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, channels)
        self.norm = make_norm(norm, channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x: ``(N, in_channels)`` float.
            returns: ``(N, channels)``, in the autocast dtype under autocast.
        """
        device = x.device.type
        if is_autocast_available(device) and torch.is_autocast_enabled(device):
            with torch.autocast(device, enabled=False):
                x = self.proj(x.to(self.proj.weight.dtype))
        else:
            x = self.proj(x)
        return self.act(self.norm(x))


# ---- block


class Cpe(nn.Module):
    """Conditional positional encoding ``norm(proj(conv(x)))``: k=3 ``SubMConv3d`` with bias,
    Linear, LayerNorm (PC ``Block.cpe``)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = SubMConv3d(channels, channels, 3, bias=True)
        self.proj = nn.Linear(channels, channels)
        self.norm = make_norm("ln", channels)

    def forward(self, x: Tensor, grid: PointGrid) -> Tensor:
        """
        Shapes:
            x: ``(N, C)`` float, rows of ``grid``.
            returns: ``(N, C)``.
        """
        return self.norm(self.proj(self.conv(x, grid)))


class Rope3d(nn.Module):
    """3D rotary position embedding of queries and keys (Utonia's ``Point3DRoPE``): each head
    splits into three chunks, for x, y and z; a chunk turns its two halves (GPT-NeoX
    ``rotate_half``) by the angles ``coord[axis] * inv_freq``, with ``inv_freq = 1 / base **
    (arange(0, chunk, 2) / chunk)`` repeated for both halves. The angles, cos and sin are fp32
    (``coord`` is fp32), the rotation runs in the promoted dtype, the result has qkv's dtype.

    ``inv_freq`` is a buffer, as in the original (its checkpoints hold ``attn.rope.inv_freq``)."""

    inv_freq: Tensor

    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        if head_dim % 6:
            raise ValueError(
                f"3D RoPE splits a head into three chunks of even size: head_dim must be a "
                f"multiple of 6, got {head_dim}"
            )
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("inv_freq", torch.empty(head_dim // 6))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        chunk = self.head_dim // 3
        steps = torch.arange(0, chunk, 2, device=self.inv_freq.device).float()
        with torch.no_grad():
            self.inv_freq.copy_(1.0 / (self.base ** (steps / chunk)))

    def forward(self, qkv: Tensor, coord: Tensor) -> Tensor:
        """
        Shapes:
            qkv: ``(N, 3 * H * head_dim)``, ``[q | k | v]`` each head-major. coord: ``(N, 3)``
                float32, the rows' coordinates.
            returns: qkv's shape and dtype, q and k rotated.
        """
        chunk = self.head_dim // 3
        angle = coord[:, :, None] * self.inv_freq  # (N, 3, chunk / 2)
        angle = torch.cat([angle, angle], dim=-1)[:, None, None]  # (N, 1, 1, 3, chunk)
        qk, v = qkv.unflatten(1, (3, -1, 3, chunk)).split([2, 1], dim=1)  # (N, 2, H, 3, chunk)
        half = qk.split(chunk // 2, dim=-1)
        rotated = torch.cat([-half[1], half[0]], dim=-1)
        qk = qk * angle.cos() + rotated * angle.sin()
        return torch.cat([qk.to(qkv.dtype), v], dim=1).flatten(1)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, base={self.base:g}"


class SerializedAttention(nn.Module):
    """Multi-head attention inside patches of ``patch_size`` consecutive rows along one
    serialization order (PC ``SerializedAttention`` without RPE, dropout or upcasting).

    ``ops.attention.serialized_attention`` runs the qkv Linear and the attention core; ``proj``
    follows. On CUDA in bf16/fp16 (head_dim <= 64) the ``triton_d16`` kernels read the patches'
    rows through the order, with no permutation or key gather. Elsewhere (CPU, fp32, other head
    dims) the rows are permuted into the order before the qkv Linear, keys and values of "overlap"
    tail patches are gathered by ``PatchLayout.kv_pos``, ``varlen_attn`` or padded SDPA runs on the
    packed patches and the output rows are permuted back before ``proj``; on a level whose rows
    already follow the order (``PointGrid.row_order``) both permutations are skipped. The two
    paths differ by rounding only (D29, D34: the bf16 forward by 1-2 ulp). qkv channels are
    ``[q | k | v]``, each head-major (PC's layout); ``scale = head_dim ** -0.5``.

    ``rope_base`` (Utonia, PC m3): q and k are rotated by ``Rope3d`` on the level's rows before
    attention, with the level's (child-mean) coordinates.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        *,
        qkv_bias: bool = True,
        tail: Tail = "overlap",
        rope_base: float | None = None,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        if tail not in _TAILS:
            raise ValueError(f"tail must be one of {_TAILS}, got {tail!r}")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.patch_size = patch_size
        self.tail: Tail = tail
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(channels, 3 * channels, bias=qkv_bias)
        self.rope = None if rope_base is None else Rope3d(self.head_dim, rope_base)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: Tensor, grid: PointGrid, order: Order | OrderRows) -> Tensor:
        """``order``: a serialization name, or its ``OrderRows`` (then both permutations always
        run, also when they are the identity).

        Shapes:
            x: ``(N, C)`` float, rows of ``grid``.
            returns: ``(N, C)``, rows of ``grid`` (in the autocast dtype under autocast).
        """
        if isinstance(order, OrderRows):
            perm, inverse, in_order = order.order, order.inverse, False
        else:
            ser = grid.serialized(order)
            perm, inverse, in_order = ser.order, ser.inverse, order == grid.row_order
        qkv: Any = self.qkv
        if self.rope is not None:  # rotate on the level's rows; attention then takes qkv as is
            x, qkv = self.rope(self.qkv(x), grid.coord), _identity
        out = serialized_attention(
            x,
            qkv,
            perm,
            inverse,
            grid.patch_layout(self.patch_size, self.tail),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            in_order=in_order,
        )
        return self.proj(out)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, num_heads={self.num_heads}, "
            f"patch_size={self.patch_size}, tail={self.tail!r}"
        )


def _identity(x: Tensor) -> Tensor:
    return x


class Block(nn.Module):
    """Pre-norm PTv3 block (PC ``Block`` with ``pre_norm=True``)::

        x = x + cpe(x)
        x = x + drop_path(ls1(attn(norm1(x))))
        x = x + drop_path(ls2(mlp(norm2(x))))

    ``ls1``/``ls2`` are ``LayerScale`` when ``layer_scale`` is set, else identities. The two
    ``drop_path`` calls draw independent masks. Under autocast the residual stream takes the
    branches' dtype: bf16 (the norms emit bf16, C10), or fp32 with LayerScale (its fp32 gamma
    promotes). PC's stream was fp32 (its LayerNorms returned fp32).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        tail: Tail = "overlap",
        rope_base: float | None = None,
    ) -> None:
        super().__init__()
        self.cpe = Cpe(channels)
        self.norm1 = make_norm("ln", channels)
        self.attn = SerializedAttention(
            channels, num_heads, patch_size, qkv_bias=qkv_bias, tail=tail, rope_base=rope_base
        )
        self.ls1 = nn.Identity() if layer_scale is None else LayerScale(channels, layer_scale)
        self.norm2 = make_norm("ln", channels)
        self.mlp = Mlp(channels, int(channels * mlp_ratio))
        self.ls2 = nn.Identity() if layer_scale is None else LayerScale(channels, layer_scale)
        self.drop_path = DropPath(drop_path)

    def forward(
        self,
        x: Tensor,
        grid: PointGrid,
        order: Order | OrderRows,
        cpe_input: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns the block output and its MLP branch ``drop_path(ls2(mlp(norm2(x))))``.

        ``cpe_input`` replaces x as the input of the CPE conv (the residual still adds to x):
        legacy PC models fed it stale features (``EncoderStage``, ``DecoderStage``).

        Shapes:
            x, cpe_input: ``(N, C)`` float, rows of ``grid``.
            returns: two ``(N, C)`` tensors.
        """
        x = x + self.cpe(x if cpe_input is None else cpe_input, grid)
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), grid, order)))
        branch = self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x + branch, branch


# ---- pooling


class Pooling(nn.Module):
    """``GELU(norm(pool(proj(x))))`` from a level's rows to the next coarser level (PC m1
    ``SerializedPooling`` / m2 ``GridPooling``: Linear first, then the segment reduction).
    The coarse geometry (mean coordinates, codes) is the coarse ``PointGrid``'s."""

    def __init__(
        self, in_channels: int, out_channels: int, norm: NormKind, reduce: Reduce = "max"
    ) -> None:
        super().__init__()
        self.reduce: Reduce = reduce
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = make_norm(norm, out_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor, index: PoolIndex) -> Tensor:
        """
        Shapes:
            x: ``(N, in_channels)``, the fine rows.
            returns: ``(M, out_channels)``, the coarse rows.
        """
        return self.act(self.norm(pool(self.proj(x), index, self.reduce)))

    def extra_repr(self) -> str:
        return f"reduce={self.reduce!r}"


class Unpooling(nn.Module):
    """``skip + unpool(up)`` with ``up = GELU(norm(proj(x)))`` on the coarse rows and
    ``skip = GELU(skip_norm(skip_proj(skip)))`` on the fine rows (PC ``SerializedUnpooling`` /
    ``GridUnpooling``: ``proj`` and ``proj_skip`` sequences)."""

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, norm: NormKind
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = make_norm(norm, out_channels)
        self.skip_proj = nn.Linear(skip_channels, out_channels)
        self.skip_norm = make_norm(norm, out_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor, skip: Tensor, index: PoolIndex) -> tuple[Tensor, Tensor]:
        """Returns the fine features and the skip branch alone.

        Shapes:
            x: ``(M, in_channels)``, the coarse rows. skip: ``(N, skip_channels)``, the fine rows.
            returns: two ``(N, out_channels)`` tensors.
        """
        up = self.act(self.norm(self.proj(x)))
        skip = self.act(self.skip_norm(self.skip_proj(skip)))
        return skip + unpool(up, index), skip


# ---- stages


class EncoderStage(nn.Module):
    """``down`` (absent at level 0), then the blocks of one level; block i attends along
    ``orders[i % len(orders)]``.

    ``stale_cpe_input`` reproduces PC release 314afb3 (2023-12-31; fixed by 74c4fc1, 2024-03-26:
    ``Block.forward`` dropped the result of ``replace_feature``): every block after the first
    feeds its CPE conv the previous block's MLP branch instead of its input.
    """

    def __init__(
        self, blocks: Sequence[Block], down: Pooling | None = None, *, stale_cpe_input: bool = False
    ) -> None:
        super().__init__()
        self.down = down
        self.blocks = nn.ModuleList(blocks)
        self.stale_cpe_input = stale_cpe_input

    def forward(self, x: Tensor, grid: PointGrid, orders: Sequence[Order | OrderRows]) -> Tensor:
        """
        Shapes:
            x: ``(N_parent, C_in)`` on ``grid.parent``'s rows when the stage has ``down``, else
                ``(N, C)`` on ``grid``'s rows.
            returns: ``(N, C)`` on ``grid``'s rows.
        """
        if self.down is not None:
            x = self.down(x, _pool_index(grid))
        return _run_blocks(self.blocks, x, grid, orders, None, self.stale_cpe_input)

    def extra_repr(self) -> str:
        return f"stale_cpe_input={self.stale_cpe_input}" if self.stale_cpe_input else ""


class DecoderStage(nn.Module):
    """``up`` from a coarse level to its parent, then the blocks of the parent level; block i
    attends along ``orders[i % len(orders)]``.

    ``skip_cpe_input``: the first block's CPE conv sees the skip branch of ``up`` instead of its
    input. PC m1 ``SerializedUnpooling`` (every revision, 314afb3 2023-12-31 .. HEAD) never
    refreshed the sparse tensor after adding the unpooled features; PC m2 ``GridUnpooling``
    (10b7490, 2025-03-21) does. ``stale_cpe_input``: the later blocks see the previous block's MLP
    branch (PC release 314afb3, as in ``EncoderStage``).
    """

    def __init__(
        self,
        up: Unpooling,
        blocks: Sequence[Block],
        *,
        skip_cpe_input: bool = False,
        stale_cpe_input: bool = False,
    ) -> None:
        super().__init__()
        self.up = up
        self.blocks = nn.ModuleList(blocks)
        self.skip_cpe_input = skip_cpe_input
        self.stale_cpe_input = stale_cpe_input

    def forward(
        self, x: Tensor, skip: Tensor, grid: PointGrid, orders: Sequence[Order | OrderRows]
    ) -> Tensor:
        """
        Shapes:
            x: ``(M, C_in)`` on ``grid``'s rows (the coarse level).
            skip: ``(N, C_skip)`` on ``grid.parent``'s rows.
            returns: ``(N, C)`` on ``grid.parent``'s rows.
        """
        parent = grid.parent
        if parent is None:
            raise ValueError("DecoderStage needs a grid made by PointGrid.downsample (no parent)")
        x, skip_branch = self.up(x, skip, _pool_index(grid))
        cpe_input = skip_branch if self.skip_cpe_input else None
        return _run_blocks(self.blocks, x, parent, orders, cpe_input, self.stale_cpe_input)

    def extra_repr(self) -> str:
        flags = ("skip_cpe_input", "stale_cpe_input")
        return ", ".join(f"{f}=True" for f in flags if getattr(self, f))


def _run_blocks(
    blocks: nn.ModuleList,
    x: Tensor,
    grid: PointGrid,
    orders: Sequence[Order | OrderRows],
    cpe_input: Tensor | None,
    stale: bool,
) -> Tensor:
    for i, block in enumerate(blocks):
        x, branch = block(x, grid, orders[i % len(orders)], cpe_input)
        cpe_input = branch if stale else None
    return x


def _pool_index(grid: PointGrid) -> PoolIndex:
    if grid.pool is None:
        raise ValueError("this level has no pooling index; build it with PointGrid.downsample")
    return grid.pool


# ---- schedules


def drop_path_rates(
    depths: Sequence[int], drop_path: float, *, reverse: bool = False
) -> list[list[float]]:
    """Per-stage DropPath rates: ``torch.linspace(0, drop_path, sum(depths))`` in fp32, cut into
    stages; ``reverse`` reverses each stage's slice (PC m1 encoder ``:610-617``, decoder
    ``:659-668``: the decoder's own linspace, reversed within each stage)."""
    rates = torch.linspace(0.0, drop_path, sum(depths), dtype=torch.float32, device="cpu")
    out, start = [], 0
    for depth in depths:
        stage = rates[start : start + depth].tolist()
        out.append(stage[::-1] if reverse else stage)
        start += depth
    return out


def draw_level_orders(num_orders: int, num_levels: int, *, shuffle: bool) -> Tensor:
    """The order indices of each level: ``(num_levels, num_orders)`` int64 on the CPU; block i of
    level s attends along ``orders[out[s, i % num_orders]]``.

    Without ``shuffle``: ``arange(num_orders)`` at every level, no RNG draw. With it: one
    ``torch.randperm(num_orders)`` per level, drawn in level order from torch's global CPU
    generator. PC draws one permutation in ``Point.serialization`` and one in every pooling; PC m2
    applies each to the unshuffled orders (as here), PC m1 to the parent level's shuffled ones, a
    composition with the same distribution (independent uniform permutations). A decoder stage
    reuses its level's row (PC decodes on the encoder's ``Point``). CPU only: no device sync.
    """
    if shuffle:
        return torch.stack([torch.randperm(num_orders, device="cpu") for _ in range(num_levels)])
    return torch.arange(num_orders, device="cpu").repeat(num_levels, 1)


def check_orders(orders: Tensor, num_levels: int, num_orders: int) -> None:
    """Host checks of a ``forward(..., orders=)`` argument: a ``(num_levels, num_orders)`` int64
    CPU tensor; outside ``torch.compile`` also that every row is a permutation."""
    if not isinstance(orders, Tensor) or orders.dtype != torch.int64 or orders.device.type != "cpu":
        got = f"{orders.dtype} on {orders.device}" if isinstance(orders, Tensor) else repr(orders)
        raise TypeError(f"orders must be an int64 CPU tensor (from model.draw_orders()), got {got}")
    if tuple(orders.shape) != (num_levels, num_orders):
        raise ValueError(
            f"orders must be (levels={num_levels}, len(cfg.orders)={num_orders}), got "
            f"{tuple(orders.shape)}; draw it with model.draw_orders()"
        )
    if torch.compiler.is_compiling():
        return
    expected = list(range(num_orders))
    bad = [row for row in orders.tolist() if sorted(row) != expected]
    if bad:
        raise ValueError(f"every row of orders must be a permutation of {expected}, got {bad}")


def select_orders(grid: PointGrid, orders: Sequence[Order], index: Tensor) -> list[OrderRows]:
    """``OrderRows`` of ``grid`` along ``orders[index[j]]`` for each j, chosen on the device: the
    serializations of every order are stacked and ``index`` (a device tensor) selects among them,
    so a captured graph does not specialize on the index values.

    Shapes:
        index: ``(J,)`` int64 on ``grid.device``, values in ``[0, len(orders))``.
        returns: J ``OrderRows`` of ``(N,)`` int64 tensors.
    """
    sers = [grid.serialized(o) for o in orders]
    perms = torch.stack([s.order for s in sers]).index_select(0, index)
    inverses = torch.stack([s.inverse for s in sers]).index_select(0, index)
    return [OrderRows(p, i) for p, i in zip(perms.unbind(0), inverses.unbind(0), strict=True)]


# ---- validation


def check_input(point: Level, in_channels: int, depth: int | None) -> PointGrid:
    """A model input's level-0 grid, after checking ``point.feat`` against ``in_channels`` and the
    grid against the model's ``serialization_depth`` (host checks, no sync; ``depth=None``: any
    depth, for ``serialization_depth="auto"``)."""
    grid = point.grid
    if point.feat.dim() != 2 or point.feat.shape[1] != in_channels:
        raise ValueError(
            f"point.feat must be (N, in_channels={in_channels}), got {tuple(point.feat.shape)}"
        )
    if not point.feat.is_floating_point():
        raise TypeError(f"point.feat must be a float tensor, got {point.feat.dtype}")
    if grid.level != 0 or (depth is not None and grid.depth != depth):
        raise ValueError(
            f"point.grid must be a level-0 PointGrid of depth serialization_depth={depth}, got "
            f"level {grid.level}, depth {grid.depth}"
        )
    return grid


def check_config(cfg: Any) -> None:
    """Validate a ``PointTransformerV3m1Config``: per-stage tuple lengths (``len(strides) + 1``
    encoder stages, one decoder stage fewer, or ``dec_depths=()`` for an encoder-only model, whose
    other ``dec_*`` fields are then unused), channels divisible by heads, power-of-two strides, a
    level-0 depth that leaves the coarsest level at least one bit, the option literals and the
    bool switches, and head dims that are multiples of 6 with ``rope_base``. A head_dim that is
    not a multiple of 8 is allowed (the attention backends pad it)."""
    _int("in_channels", cfg.in_channels, 1)
    if not isinstance(cfg.orders, tuple) or not cfg.orders:
        raise TypeError(f"orders must be a non-empty tuple of {ORDERS}, got {cfg.orders!r}")
    bad = [o for o in cfg.orders if o not in ORDERS]
    if bad:
        raise ValueError(f"unknown orders {bad}; use orders from {ORDERS}")
    for name in (
        "shuffle_orders",
        "qkv_bias",
        "legacy_cpe_stale_input",
        "legacy_dec_cpe_skip_input",
    ):
        if not isinstance(getattr(cfg, name), bool):
            raise TypeError(f"{name} must be a bool, got {getattr(cfg, name)!r}")
    for name, allowed in (
        ("stem", STEMS),
        ("norm", ("bn", "ln")),
        ("pool_policy", POLICIES),
        ("init", WEIGHT_INITS),
    ):
        if getattr(cfg, name) not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {getattr(cfg, name)!r}")
    scale = cfg.layer_scale
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, int | float)):
        raise TypeError(f"layer_scale must be a float or None, got {scale!r}")
    if scale is not None and not scale > 0:
        raise ValueError(f"layer_scale must be positive (or None for no LayerScale), got {scale}")
    strides = _ints("strides", cfg.strides, 2, None)
    if any(s & (s - 1) for s in strides):
        raise ValueError(f"strides must be powers of two (2, 4, ...), got {strides}")
    bits = sum(s.bit_length() - 1 for s in strides)
    if cfg.serialization_depth in AUTO_DEPTHS:
        pass  # chosen per batch (structure.auto_depth)
    elif isinstance(cfg.serialization_depth, str):
        raise ValueError(
            f"serialization_depth must be an int or one of {AUTO_DEPTHS}, got "
            f"{cfg.serialization_depth!r}"
        )
    elif not bits < _int("serialization_depth", cfg.serialization_depth, 1) <= MAX_DEPTH:
        raise ValueError(
            f"serialization_depth must be in {bits + 1}..{MAX_DEPTH} for strides {strides} "
            f"(each stride 2**k takes k bits), got {cfg.serialization_depth}"
        )
    stages = len(strides) + 1
    _stages("enc", cfg.enc_depths, cfg.enc_channels, cfg.enc_num_heads, cfg.enc_patch_sizes, stages)
    if cfg.dec_depths != ():
        _stages(
            "dec", cfg.dec_depths, cfg.dec_channels, cfg.dec_num_heads, cfg.dec_patch_sizes,
            stages - 1,
        )  # fmt: skip
    if isinstance(cfg.mlp_ratio, bool) or not isinstance(cfg.mlp_ratio, int | float):
        raise TypeError(f"mlp_ratio must be a float, got {cfg.mlp_ratio!r}")
    channels = (*cfg.enc_channels, *(cfg.dec_channels if cfg.dec_depths else ()))
    if int(min(channels) * cfg.mlp_ratio) < 1:
        raise ValueError(f"mlp_ratio {cfg.mlp_ratio} leaves an MLP without hidden channels")
    if isinstance(cfg.drop_path, bool) or not isinstance(cfg.drop_path, int | float):
        raise TypeError(f"drop_path must be a float, got {cfg.drop_path!r}")
    if not 0.0 <= cfg.drop_path <= 1.0:
        raise ValueError(f"drop_path must be in [0, 1], got {cfg.drop_path}")
    if cfg.patch_tail not in _TAILS:
        raise ValueError(f"patch_tail must be one of {_TAILS}, got {cfg.patch_tail!r}")
    rope = cfg.rope_base
    if rope is not None:
        if isinstance(rope, bool) or not isinstance(rope, int | float) or not rope > 0:
            raise ValueError(
                f"rope_base must be a positive number (or None: no RoPE), got {rope!r}"
            )
        heads = (*cfg.enc_num_heads, *(cfg.dec_num_heads if cfg.dec_depths else ()))
        dims = sorted({c // h for c, h in zip(channels, heads, strict=True)})
        if any(d % 6 for d in dims):
            raise ValueError(
                f"rope_base (3D RoPE: three even chunks per head) needs head dims "
                f"(channels / num_heads) that are multiples of 6, got {dims}"
            )


def _stages(
    prefix: str,
    depths: tuple[int, ...],
    channels: tuple[int, ...],
    num_heads: tuple[int, ...],
    patch_sizes: tuple[int, ...],
    stages: int,
) -> None:
    _ints(f"{prefix}_depths", depths, 0, stages)
    _ints(f"{prefix}_channels", channels, 1, stages)
    _ints(f"{prefix}_num_heads", num_heads, 1, stages)
    _ints(f"{prefix}_patch_sizes", patch_sizes, 1, stages)
    bad = [(c, h) for c, h in zip(channels, num_heads, strict=True) if c % h]
    if bad:
        raise ValueError(
            f"{prefix}_channels must be divisible by {prefix}_num_heads per stage, got "
            f"(channels, heads) {bad}"
        )


def _ints(name: str, value: object, minimum: int, length: int | None) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of ints, got {value!r}")
    for v in value:
        _int(name, v, minimum)
    if length is not None and len(value) != length:
        raise ValueError(
            f"{name} needs one entry per stage ({length}: len(strides) + 1 encoder stages, one "
            f"decoder stage fewer), got {len(value)}: {value}"
        )
    return value


def _int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name}: expected an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name}: expected an int >= {minimum}, got {value}")
    return value
