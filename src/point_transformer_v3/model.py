# Vendored from Pointcept2 src/pointcept2/models/point_transformer_v3/point_transformer_v3m1_base.py
# @ 771a949 (simplified for point_transformer_v3: no registry; compat constructor, dict input and
# output, per-batch serialization depth, precision handling, heads and rope_base added).
"""Point Transformer V3: one class, configured by ``Config`` options and ``presets``.

A U-Net of PTv3 blocks over a ``PointGrid`` hierarchy: a stem, encoder stages joined by max
pooling (Linear + norm + GELU), decoder stages joined by unpooling with skip connections, or an
encoder only (Sonata). Attention runs in patches along serialization orders that change from
block to block and, in training, are shuffled once per level and step (``draw_orders``); in eval
the orders are fixed, so inference is deterministic.

Ways to build it:

- ``PointTransformerV3(in_channels=6, ...)``: the original ``model.py`` constructor (Pointcept
  ``PT-v3m1``), kwargs mapped onto ``Config`` (``convert.pc_backbone_config``).
- ``PointTransformerV3(config=Config(...))`` or ``PointTransformerV3.from_preset("sonata")``.
- ``point_transformer_v3.load("sonata")``: a preset with its released weights.

``Config`` defaults are Pointcept2's recipe for new training: the released ScanNet semantic
segmentation recipe with every compat switch off. The compat switches reproduce Pointcept
behaviors that trained weights depend on:

- ``legacy_dec_cpe_skip_input``: the first block of every decoder stage feeds its CPE conv the
  skip branch of the unpooling, not the summed features (Pointcept m1 and this repository's
  ``model.py`` at every revision).
- ``legacy_cpe_stale_input``: Pointcept release 314afb3 dropped the result of
  ``replace_feature()`` at the end of ``Block.forward``, so every later block's CPE conv saw the
  previous block's MLP branch. Fixed by 74c4fc1 (2024-03-26). Needed by the released ScanNet
  weights (``presets.PC_M1_RELEASE``); this repository's ``model.py`` has the fix, which is why
  those weights did not reproduce with it.

Init: every module's own ``reset_parameters`` (torch's defaults for Linear and the norms,
spconv's for ``SubMConv3d``), then for ``init="trunc_normal"`` ``trunc_normal(std=0.02)`` on every
Linear and ``SubMConv3d`` weight with zero biases. The init is drawn on the CPU under a seed taken
from the global CPU generator in ``__init__``, so an eager build (CPU or CUDA) and a meta build +
``to_empty(device=...)`` + ``_init_weights`` give the same weights.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.amp.autocast_mode import is_autocast_available

from point_transformer_v3._util import suggest
from point_transformer_v3.convert import convert_state_dict, is_native, pc_backbone_config
from point_transformer_v3.layers.init import (
    build_rng,
    drawn_on_cpu,
    init_seed,
    seeded_rng,
    trunc_normal_init_,
)
from point_transformer_v3.layers.norm import NormKind
from point_transformer_v3.modules import (
    AUTO_DEPTHS,
    AutoDepth,
    Block,
    DecoderStage,
    EncoderStage,
    LinearEmbedding,
    OrderRows,
    Pooling,
    Stem,
    StemEmbedding,
    Unpooling,
    WeightInit,
    check_config,
    check_input,
    check_orders,
    draw_level_orders,
    drop_path_rates,
    select_orders,
)
from point_transformer_v3.ops.patches import Tail
from point_transformer_v3.ops.serialization import Order
from point_transformer_v3.structure import (
    Level,
    Point,
    PointGrid,
    Policy,
    auto_depth,
)

__all__ = ["Config", "PointTransformerV3", "Precision"]

log = logging.getLogger(__name__)

Precision = Literal["auto", "fp32", "fp16", "bf16"]
PRECISIONS: tuple[str, ...] = ("auto", "fp32", "fp16", "bf16")
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# positional order of the original model.py constructor
_ORIGINAL_ARGS = (
    "in_channels", "order", "stride", "enc_depths", "enc_channels", "enc_num_head",
    "enc_patch_size", "dec_depths", "dec_channels", "dec_num_head", "dec_patch_size",
    "mlp_ratio", "qkv_bias", "qk_scale", "attn_drop", "proj_drop", "drop_path", "pre_norm",
    "shuffle_orders", "enable_rpe", "enable_flash", "upcast_attention", "upcast_softmax",
    "cls_mode", "pdnorm_bn", "pdnorm_ln", "pdnorm_decouple", "pdnorm_adaptive", "pdnorm_affine",
    "pdnorm_conditions",
)  # fmt: skip


@dataclass(frozen=True, kw_only=True)
class Config:
    """Every option of ``PointTransformerV3``. Defaults are Pointcept2's recipe for new training
    (PTv3 ScanNet ``semseg-pt-v3m1-0-base`` with every compat switch off); presets:
    ``presets.py``.

    Attributes:
        in_channels: input feature channels (ScanNet PTv3: color + normal; Sonata: coord + color
            + normal).
        orders: serialization orders; block i of a stage attends along ``orders[i % len]``.
        shuffle_orders: in training, permute ``orders`` once per level and step.
        serialization_depth: bits per axis at level 0, or chosen per batch: ``"auto"`` (the
            smallest cube holding the batch, ``max(grid_coord).bit_length()``, as ``model.py`` /
            Pointcept m1) or ``"auto+1"`` (``(max + 1).bit_length()`` at every level, as Sonata /
            Pointcept m2). A fixed int makes the result independent of the other samples of a
            batch (packing invariance) and saves a host sync; ``"auto"`` matches what the
            released weights were trained with. Each stride ``2**k`` takes k bits from the next
            level.
        strides: pooling strides between levels (powers of two).
        enc_depths, enc_channels, enc_num_heads, enc_patch_sizes: per encoder stage.
        dec_depths, dec_channels, dec_num_heads, dec_patch_sizes: per decoder stage, finest
            first; ``dec_depths=()`` builds an encoder only (the other ``dec_*`` are unused).
        mlp_ratio: MLP hidden channels per channel.
        qkv_bias: bias of the qkv Linear.
        drop_path: largest DropPath rate. Encoder: linspace over all encoder blocks; decoder: its
            own linspace over all decoder blocks, cut per stage and reversed within each stage.
        patch_tail: the last partial patch of a sample: "overlap" (Pointcept: keys from the
            previous patch) or "shrink".
        stem: "conv" (PTv3: bias-free k=5 ``SubMConv3d`` in the autocast dtype) or "linear"
            (Sonata: a Linear in fp32, for inputs that hold coordinates).
        norm: the norm of the stem, poolings and unpoolings, "bn" (PTv3) or "ln" (Sonata); the
            blocks always use LayerNorm.
        pool_policy: codes of a pooled level, "shift" (PTv3: the finer codes shifted) or
            "reencode" (Sonata: serialized again at the coarse depth).
        layer_scale: initial LayerScale gamma of both block branches, or None for none.
        init: "torch" (every module's default, PTv3) or "trunc_normal" (Sonata).
        rope_base: 3D rotary position embedding of q and k in every block with this base (Utonia:
            10; ``modules.Rope3d``), or None for none. Head dims must be multiples of 6.
        legacy_cpe_stale_input: compat, Pointcept release 314afb3 stale CPE inputs (module doc).
        legacy_dec_cpe_skip_input: compat, the first decoder block's CPE input (module doc).
    """

    in_channels: int = 6
    orders: tuple[Order, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")
    shuffle_orders: bool = True
    serialization_depth: int | AutoDepth = 15
    strides: tuple[int, ...] = (2, 2, 2, 2)
    enc_depths: tuple[int, ...] = (2, 2, 2, 6, 2)
    enc_channels: tuple[int, ...] = (32, 64, 128, 256, 512)
    enc_num_heads: tuple[int, ...] = (2, 4, 8, 16, 32)
    enc_patch_sizes: tuple[int, ...] = (1024, 1024, 1024, 1024, 1024)
    dec_depths: tuple[int, ...] = (2, 2, 2, 2)
    dec_channels: tuple[int, ...] = (64, 64, 128, 256)
    dec_num_heads: tuple[int, ...] = (4, 4, 8, 16)
    dec_patch_sizes: tuple[int, ...] = (1024, 1024, 1024, 1024)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_path: float = 0.3
    patch_tail: Tail = "overlap"
    stem: Stem = "conv"
    norm: NormKind = "bn"
    pool_policy: Policy = "shift"
    layer_scale: float | None = None
    init: WeightInit = "torch"
    rope_base: float | None = None
    legacy_cpe_stale_input: bool = False
    legacy_dec_cpe_skip_input: bool = False

    def __post_init__(self) -> None:
        check_config(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Config:
        """A Config from a plain dict (lists become tuples); unknown keys raise with a hint."""
        names = [f.name for f in fields(cls)]
        unknown = [k for k in values if k not in names]
        if unknown:
            raise TypeError(
                "; ".join(f"unknown Config option {k!r}{suggest(k, names)}" for k in unknown)
            )
        kwargs: dict[str, Any] = {
            k: tuple(v) if isinstance(v, list) else v for k, v in values.items()
        }
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PointTransformerV3(nn.Module):
    """Point Transformer V3 (module doc).

    State-dict tree: ``embedding.{conv,norm}`` (stem "conv") or ``embedding.{proj,norm}`` (stem
    "linear"), ``enc.{s}.down.{proj,norm}`` (s > 0), ``enc.{s}.blocks.{i}.*`` (with
    ``ls1/ls2.gamma`` when ``layer_scale`` is set), ``dec.{s}.up.{proj,norm,skip_proj,skip_norm}``,
    ``dec.{s}.blocks.{i}.*`` (s = the level the decoder stage outputs, 0 = finest; blocks hold
    ``attn.rope.inv_freq`` when ``rope_base`` is set), and
    ``seg_head.{weight,bias}`` when a segmentation head is attached. ``load_state_dict`` also
    takes the original ``model.py`` / Pointcept / Sonata key names and converts them.

    Attributes:
        cfg: the ``Config``.
        precision: "auto" (fp16 autocast on CUDA in eval, bf16 autocast on CUDA in training, fp32
            elsewhere), "fp32", "fp16" or "bf16". An autocast the caller already entered wins.
        preset: the preset name (``from_preset`` / ``load``) or None.
        weights_info: where the loaded weights came from, their sha256 and license (``load``).
        seg_head: an ``nn.Linear`` segmentation head over the (upcast) features, or None.
    """

    Config = Config

    def __init__(
        self,
        *args: Any,
        config: Config | Mapping[str, Any] | None = None,
        precision: Precision = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            if args or kwargs:
                raise TypeError(
                    "pass either config=... or the original model.py keyword arguments, not both"
                )
            cfg = config if isinstance(config, Config) else Config.from_dict(config)
        else:
            if len(args) > len(_ORIGINAL_ARGS):
                raise TypeError(f"PointTransformerV3 takes at most {len(_ORIGINAL_ARGS)} args")
            dup = [k for k in _ORIGINAL_ARGS[: len(args)] if k in kwargs]
            if dup:
                raise TypeError(f"got multiple values for {dup}")
            original = dict(zip(_ORIGINAL_ARGS, args, strict=False)) | kwargs
            values, runtime = pc_backbone_config(original, "PT-v3m1")
            cfg = Config.from_dict(values)
            if precision == "auto":
                precision = runtime.get("precision", "auto")
        _check_precision(precision)
        self.cfg = cfg
        self.precision: str = precision
        self.preset: str | None = None
        self.weights_info: dict[str, Any] = {}
        self._init_seed = init_seed()
        # The modules' construction-time init is replaced by _init_weights; keep it off the global
        # CPU and default-device streams so every build path leaves the generators at one state.
        with build_rng():
            self._build()
        self.seg_head: nn.Linear | None = None
        self._init_weights()

    # ---- construction

    @classmethod
    def from_preset(cls, name: str, **overrides: Any) -> PointTransformerV3:
        """A randomly initialized model of preset ``name`` (``presets.PRESETS``) with Config
        ``overrides``. ``point_transformer_v3.load(name)`` also loads the released weights."""
        from point_transformer_v3.presets import get_preset

        preset = get_preset(name)
        model = cls(config=Config.from_dict({**preset.config, **overrides}))
        model.preset = name
        return model

    def _build(self) -> None:
        cfg = self.cfg
        stale = cfg.legacy_cpe_stale_input
        channels = cfg.enc_channels[0]
        self.embedding = (
            StemEmbedding(cfg.in_channels, channels, cfg.norm)
            if cfg.stem == "conv"
            else LinearEmbedding(cfg.in_channels, channels, cfg.norm)
        )
        self.enc = nn.ModuleList()
        for s, rates in enumerate(drop_path_rates(cfg.enc_depths, cfg.drop_path)):
            channels = cfg.enc_channels[s]
            down = Pooling(cfg.enc_channels[s - 1], channels, cfg.norm) if s else None
            blocks = [
                self._block(channels, cfg.enc_num_heads[s], cfg.enc_patch_sizes[s], rate)
                for rate in rates
            ]
            self.enc.append(EncoderStage(blocks, down, stale_cpe_input=stale))
        self.dec = nn.ModuleList()
        if cfg.dec_depths:
            channels_in = (*cfg.dec_channels[1:], cfg.enc_channels[-1])
            rates = drop_path_rates(cfg.dec_depths, cfg.drop_path, reverse=True)
            for s, channels in enumerate(cfg.dec_channels):
                up = Unpooling(channels_in[s], cfg.enc_channels[s], channels, cfg.norm)
                blocks = [
                    self._block(channels, cfg.dec_num_heads[s], cfg.dec_patch_sizes[s], rate)
                    for rate in rates[s]
                ]
                stage = DecoderStage(
                    up,
                    blocks,
                    skip_cpe_input=cfg.legacy_dec_cpe_skip_input,
                    stale_cpe_input=stale,
                )
                self.dec.append(stage)

    def _block(self, channels: int, num_heads: int, patch_size: int, drop_path: float) -> Block:
        cfg = self.cfg
        return Block(
            channels,
            num_heads,
            patch_size,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_path=drop_path,
            layer_scale=cfg.layer_scale,
            tail=cfg.patch_tail,
            rope_base=cfg.rope_base,
        )

    def _init_weights(self) -> None:
        """Every submodule's ``reset_parameters`` in ``modules()`` order (Linear, norms and their
        running statistics, ``SubMConv3d``, LayerScale), then ``trunc_normal_init_(std=0.02)``
        for ``init="trunc_normal"``; drawn on the CPU (``drawn_on_cpu``) under ``seeded_rng``
        with the seed of ``__init__``, so the weights do not depend on the device or GPU model.
        After a meta-device build: ``to_empty(device=...)``, then this."""
        devices = {t.device for t in (*self.parameters(), *self.buffers())}
        with seeded_rng(self._init_seed, devices), drawn_on_cpu(self.modules()):
            for module in self.modules():
                reset = getattr(module, "reset_parameters", None)
                if module is not self and callable(reset):
                    reset()
            if self.cfg.init == "trunc_normal":
                trunc_normal_init_(self, std=0.02)

    def attach_head(self, num_classes: int, in_channels: int | None = None) -> nn.Linear:
        """Add a linear segmentation head ``seg_head`` over the upcast features (``in_channels``
        defaults to ``feature_channels``), on the model's device and dtype."""
        in_channels = self.feature_channels if in_channels is None else in_channels
        ref = next(self.parameters())
        self.seg_head = nn.Linear(in_channels, num_classes, device=ref.device, dtype=ref.dtype)
        return self.seg_head

    # ---- properties

    @property
    def out_channels(self) -> int:
        """Channels of the forward output: ``dec_channels[0]``, or ``enc_channels[-1]`` without
        decoder."""
        return self.cfg.dec_channels[0] if self.cfg.dec_depths else self.cfg.enc_channels[-1]

    @property
    def feature_channels(self) -> int:
        """Channels of ``upcast(forward(...))``: the decoder output, or for an encoder the
        concatenation of every level (Sonata: 1232)."""
        return self.out_channels if self.cfg.dec_depths else sum(self.cfg.enc_channels)

    @property
    def num_levels(self) -> int:
        """Grid levels (encoder stages): ``len(strides) + 1``."""
        return len(self.cfg.strides) + 1

    # ---- forward

    def draw_orders(self) -> Tensor:
        """This step's order indices, ``(num_levels, len(orders))`` int64 on the CPU: in training
        with ``shuffle_orders`` one ``torch.randperm`` per level from the global CPU generator
        (``draw_level_orders``), else the unshuffled orders (no RNG draw)."""
        shuffle = self.training and self.cfg.shuffle_orders
        return draw_level_orders(len(self.cfg.orders), self.num_levels, shuffle=shuffle)

    def forward(self, data: Mapping[str, Any], orders: Tensor | None = None) -> Point:
        """
        Args:
            data: a ``Point`` or dict with ``feat`` ``(N, in_channels)`` float, ``grid_coord``
                ``(N, 3)`` int (unique grid cells per sample, non-negative; or ``coord`` and
                ``grid_size`` to compute it as the original did), ``coord`` ``(N, 3)`` float
                meters, and ``offset`` ``(B,)`` (cumulative sample ends) or ``batch`` ``(N,)``
                (rows sample-contiguous); one sample if neither. ``point_transformer_v3.prepare``
                makes it from raw points. Other keys are carried to the output.
            orders: ``(num_levels, len(cfg.orders))`` int64 CPU order indices
                (``draw_orders()``); None draws them here.

        Returns:
            A ``Point``. With a decoder: ``feat (N, out_channels)`` on the input rows. Encoder
            only: the coarsest level (``feat``, ``coord``, ``grid_coord``, ``offset``,
            ``batch``), whose ``pooling_parent`` / ``pooling_inverse`` chain leads back to level
            0 (``point_transformer_v3.upcast``).
        """
        point = data if isinstance(data, Point) else Point(data)
        device = next(self.parameters()).device
        with self._autocast(device):
            grid, feat = self._level0(point)
            levels = self._forward_levels(Level(feat, grid), orders)
        out = _to_point(point, levels, has_decoder=bool(self.dec))
        _release(levels)
        return out

    def _autocast(self, device: torch.device) -> contextlib.AbstractContextManager[Any]:
        kind = device.type
        if is_autocast_available(kind) and torch.is_autocast_enabled(kind):
            return contextlib.nullcontext()  # the caller's autocast wins
        precision = self.precision
        if precision == "auto":
            precision = ("bf16" if self.training else "fp16") if kind == "cuda" else "fp32"
        if precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(kind, dtype=_DTYPES[precision])

    def _forward_levels(self, point: Level, orders: Tensor | None) -> list[Level]:
        """Pointcept2's ``PT-v3m1.forward`` on a level-0 ``Level``: the encoder levels, then (with
        a decoder) the level-0 output appended."""
        cfg = self.cfg
        depth = cfg.serialization_depth
        auto = not isinstance(depth, int)  # "auto" / "auto+1"
        grid = check_input(point, cfg.in_channels, None if not isinstance(depth, int) else depth)
        if orders is None:
            orders = self.draw_orders()
        else:
            check_orders(orders, self.num_levels, len(cfg.orders))
        level_orders = _LevelOrders(cfg.orders, orders, grid.device)
        grids = [grid]  # the level chain first: kernel maps are then built top-down
        max_coord = getattr(grid, "_max_coord", None)
        for i, stride in enumerate(cfg.strides):
            depth = None
            if auto and cfg.pool_policy == "reencode":
                bits = stride.bit_length() - 1
                max_coord = None if max_coord is None else max_coord >> bits
                nxt = cfg.strides[i + 1].bit_length() if i + 1 < len(cfg.strides) else 1
                depth = auto_depth(max_coord or 0, rule=str(cfg.serialization_depth), minimum=nxt)
            grids.append(
                grids[-1].downsample(stride, cfg.pool_policy, order=cfg.orders[0], depth=depth)
            )
        x = self.embedding(point.feat, grid) if cfg.stem == "conv" else self.embedding(point.feat)
        levels: list[Level] = []
        for s, stage in enumerate(self.enc):
            grid = grids[s]
            x = stage(x, grid, level_orders(s, grid))
            levels.append(Level(x, grid, levels[-1] if levels else None))
        if not self.dec:
            return levels
        for s in reversed(range(len(self.dec))):
            x = self.dec[s](x, levels[s].feat, levels[s + 1].grid, level_orders(s, levels[s].grid))
        levels.append(Level(x, point.grid))
        return levels

    def _level0(self, point: Point) -> tuple[PointGrid, Tensor]:
        """The level-0 ``PointGrid`` and the input features of a point dict: one host sync (the
        sample sizes and, for ``serialization_depth="auto"``, the largest grid coordinate)."""
        cfg = self.cfg
        device = next(self.parameters()).device
        if "feat" not in point:
            raise KeyError(
                "the input needs 'feat' (N, in_channels); build it with "
                "point_transformer_v3.prepare"
            )
        feat = _tensor(point["feat"], device)
        if feat.dim() != 2 or feat.shape[1] != cfg.in_channels or not feat.is_floating_point():
            raise ValueError(
                f"feat must be a float (N, in_channels={cfg.in_channels}) tensor, got "
                f"{feat.dtype} {tuple(feat.shape)}"
            )
        n = feat.shape[0]
        grid_size = point.get("grid_size", None)
        if isinstance(grid_size, Tensor):
            grid_size = float(grid_size.reshape(-1)[0])
        if "grid_coord" in point:
            grid_coord = _tensor(point["grid_coord"], device)
        elif "coord" in point and grid_size is not None:
            coord = _tensor(point["coord"], device)  # the original's batch-global rule
            grid_coord = torch.div(coord - coord.min(0)[0], grid_size, rounding_mode="trunc").int()
        else:
            raise KeyError("the input needs 'grid_coord', or 'coord' and 'grid_size'")
        if grid_coord.dtype.is_floating_point or grid_coord.shape != (n, 3):
            raise ValueError(
                f"grid_coord must be an integer ({n}, 3) tensor, got {grid_coord.dtype} "
                f"{tuple(grid_coord.shape)}"
            )
        grid_coord = grid_coord.int()
        coord = (
            _tensor(point["coord"], device).float()
            if "coord" in point
            else grid_coord.float() * (grid_size or 1.0)
        )
        if coord.shape != (n, 3):
            raise ValueError(f"coord must be ({n}, 3), got {tuple(coord.shape)}")
        # one host read: sample ends and (auto depth) the largest grid coordinate
        if "offset" in point:
            ends = _tensor(point["offset"], device).reshape(-1).long()
        elif "batch" in point:
            batch = _tensor(point["batch"], device).reshape(-1).long()
            if batch.shape[0] != n:
                raise ValueError(f"batch must be ({n},), got {tuple(batch.shape)}")
            ends = torch.bincount(batch).cumsum(0) if n else batch.new_zeros(1)
        else:
            ends = torch.full((1,), n, dtype=torch.long, device=device)
        auto = cfg.serialization_depth in AUTO_DEPTHS
        top = grid_coord.max().long().view(1) if auto and n else ends.new_zeros(1)
        host = torch.cat([ends.to(top.device), top]).tolist()
        *ends_host, max_coord = host
        seq_lens, start = [], 0
        for end in ends_host:
            if end < start:
                raise ValueError(f"offset must be non-decreasing sample ends, got {ends_host}")
            seq_lens.append(end - start)
            start = end
        if start != n:
            raise ValueError(f"offset/batch cover {start} rows but feat has {n}")
        if auto:
            bits = sum(s.bit_length() - 1 for s in cfg.strides)
            rule = str(cfg.serialization_depth)
            first = cfg.strides[0].bit_length() if cfg.strides else 1
            minimum = bits + 1 if cfg.pool_policy == "shift" else first
            depth = auto_depth(max_coord, rule=rule, minimum=minimum)
        else:
            depth = int(cfg.serialization_depth)
        from point_transformer_v3.ops.seqlens import cu_seqlens_of

        grid = PointGrid(
            coord=coord.contiguous(),
            grid_coord=grid_coord.contiguous(),
            cu_seqlens=cu_seqlens_of(seq_lens, device),
            seq_lens=tuple(seq_lens),
            grid_size=float(grid_size) if grid_size is not None else 1.0,
            depth=depth,
        )
        grid._max_coord = max_coord if auto else None  # type: ignore[attr-defined]
        return grid, feat

    # ---- convenience

    def embed(
        self,
        coord: Any,
        color: Any = None,
        normal: Any = None,
        *,
        upcast: bool = True,
        **prepare_kwargs: Any,
    ) -> Tensor | list[Tensor]:
        """Per-point features of raw point clouds: ``prepare``, forward, ``upcast`` (encoders:
        the concatenated features of every level) and map back to the input points.

        ``coord`` ``(N, 3)`` (numpy or torch; meters), ``color`` ``(N, 3)`` RGB in [0, 255],
        ``normal`` ``(N, 3)``; or lists of them for a batch (one forward, one result per cloud).
        ``prepare_kwargs`` go to ``point_transformer_v3.prepare`` (``grid_size``, ``pick``, ...).
        Returns ``(N, feature_channels)`` float tensors on the model's device (eval mode, no
        grad).
        """
        from point_transformer_v3.heads import upcast as upcast_point
        from point_transformer_v3.preprocess import collate, prepare

        device = next(self.parameters()).device
        many = isinstance(coord, list | tuple)
        coords = list(coord) if many else [coord]
        colors = list(color) if many and color is not None else [color] * len(coords)
        normals = list(normal) if many and normal is not None else [normal] * len(coords)
        spec = dict(prepare_kwargs)
        if self.preset is not None and "preset" not in spec and "feat" not in spec:
            spec["preset"] = self.preset
        items = [
            prepare(c, color=k, normal=m, device=device, **spec)
            for c, k, m in zip(coords, colors, normals, strict=True)
        ]
        point = collate([p for p, _ in items])
        with torch.inference_mode():
            out = self(point)
            if upcast:
                out = upcast_point(out)
            feat, offset = out.feat, out.offset.tolist()
        results, start = [], 0
        for (_, inverse), end in zip(items, offset, strict=True):
            results.append(feat[start:end][inverse].float())
            start = end
        return results if many else results[0]

    def predict(self, coord: Any, color: Any = None, normal: Any = None, **kwargs: Any) -> Any:
        """Per-point class logits of raw point clouds through ``seg_head`` (``embed`` + head)."""
        if self.seg_head is None:
            raise RuntimeError(
                "this model has no segmentation head; load it with a head, e.g. "
                "point_transformer_v3.load('sonata', head='scannet20'), or call attach_head(...)"
            )
        feats = self.embed(coord, color, normal, **kwargs)
        with torch.inference_mode():
            head = self.seg_head
            if isinstance(feats, list):
                return [head(f.to(head.weight.dtype)) for f in feats]
            return head(feats.to(head.weight.dtype))

    # ---- state dict

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """``nn.Module.load_state_dict`` that also takes the key names of the original
        ``model.py``, Pointcept (``PT-v3m1`` / ``PT-v3m2``, segmentor checkpoints with
        ``backbone.`` / ``seg_head.``) and Sonata, converting them first (strictly: every key is
        mapped or explicitly dropped, every parameter filled)."""
        if not is_native(state_dict):
            target = super().state_dict(keep_vars=False)
            state_dict, dropped = convert_state_dict(state_dict, target)
            for key, reason in dropped.items():
                log.info("load_state_dict: dropped %s (%s)", key, reason)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def extra_repr(self) -> str:
        return f"preset={self.preset!r}, precision={self.precision!r}"


class _LevelOrders:
    """The orders of each level's blocks for one forward: names eagerly (``orders.tolist()`` of a
    CPU tensor, no sync), ``OrderRows`` selected on the device while ``torch.compile`` traces."""

    def __init__(self, names: tuple[Order, ...], orders: Tensor, device: torch.device) -> None:
        compiling = torch.compiler.is_compiling()
        self.names = names
        self.index = orders.to(device, non_blocking=True) if compiling else None
        self.rows: list[list[int]] = [] if compiling else orders.tolist()
        self.selected: dict[int, list[OrderRows]] = {}

    def __call__(self, level: int, grid: PointGrid) -> list[Order] | list[OrderRows]:
        if self.index is None:
            return [self.names[j] for j in self.rows[level]]
        if level not in self.selected:
            self.selected[level] = select_orders(grid, self.names, self.index[level])
        return self.selected[level]


def _check_precision(precision: str) -> None:
    if precision not in PRECISIONS:
        raise ValueError(
            f"precision must be one of {PRECISIONS}, got {precision!r}{suggest(precision, PRECISIONS)}"
        )


def _tensor(value: Any, device: torch.device) -> Tensor:
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value)
    return value.to(device, non_blocking=True)


def _to_point(point: Point, levels: Sequence[Level], *, has_decoder: bool) -> Point:
    """The output ``Point``: the input's keys with the level-0 features (decoder), or the coarsest
    level with its ``pooling_parent`` chain down to level 0 (encoder only)."""
    base = Point(point)
    for key in ("feat", "grid_coord", "offset", "batch"):
        base.pop(key, None)
    if has_decoder:
        return _level_point(levels[-1], base)
    out = _level_point(levels[0], base)
    for level in levels[1:]:
        grid = level.grid
        nxt = _level_point(level, Point())
        nxt["pooling_parent"] = out
        nxt["pooling_inverse"] = grid.pool.inverse  # type: ignore[union-attr]
        out = nxt
    return out


def _release(levels: Sequence[Level]) -> None:
    """Break the reference cycles of the level chain (a grid caches its coarse grid, which points
    back to it), so the forward's index tables are freed when the call returns instead of at the
    next garbage collection. The output holds the tensors it needs, autograd its saved ones."""
    seen: set[int] = set()
    for level in levels:
        grid: PointGrid | None = level.grid
        while grid is not None and id(grid) not in seen:
            seen.add(id(grid))
            parent = grid.parent
            grid.cache.clear()
            grid.parent = None
            grid = parent


def _level_point(level: Level, base: Point) -> Point:
    grid = level.grid
    out = Point(base)
    out["feat"] = level.feat
    out["coord"] = grid.coord
    out["grid_coord"] = grid.grid_coord
    out["offset"] = grid.cu_seqlens[1:].long()
    out["batch"] = grid.batch_index()
    if grid.level:
        out["grid_size"] = grid.grid_size
    return out
