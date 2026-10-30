# Vendored from Pointcept2 src/pointcept2/models/point_transformer_v3/convert.py @ 771a949
# (simplified for pointtransformerv3: no CLI, no train.log dating, no config-file reader, no
# checkpoint writing; the presets fix the legacy switches; PC PT-v3m3 / Utonia added).
"""Pointcept (PC) PTv3 checkpoints and constructor kwargs -> ``PointTransformerV3``.

Sources: the original ``model.py`` of this repository (= PC ``PT-v3m1``), a PC ``PT-v3m1``,
``PT-v3m2`` (Sonata, Concerto) or ``PT-v3m3`` (Utonia) backbone state dict (the released encoder
checkpoints store their constructor kwargs under ``"config"``), or a PC ``DefaultSegmentorV2``
checkpoint (``backbone.`` and ``seg_head.`` keys, e.g. the released ``model_best.pth``). PC m1
maps to the options of ``presets.PC_M1`` (conv stem, BatchNorm, shifted codes, the first decoder
block's CPE on the skip branch), PC m2 to ``presets.PC_M2`` (Linear stem, LayerNorm, re-encoded
codes, trunc-normal init), PC m3 to PC m2 plus ``rope_base`` (3D RoPE, ``modules.Rope3d``).

Names. A DDP ``module.`` prefix and a segmentor's ``backbone.`` prefix are taken off, then every
source key must match exactly one rule of ``RULES`` or ``DROPPED``, and every target key must be
filled once with a tensor of the target's shape; otherwise ``ConvertError`` lists every problem.
Dropped keys (Sonata's mask token) are reported. Sparse-conv weights are copied as they are
(spconv 2.x layout on both sides). A segmentor's ``seg_head.weight`` / ``seg_head.bias`` become the
model's ``seg_head`` (a Linear).

Pickles are read with ``weights_only=True``. Globals that are not tensors or containers raise,
except optimizer / scheduler classes, ``getattr`` and NumPy scalars, which PC checkpoints keep in
their training state: those load as inert placeholders and are never called.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from pointtransformerv3._util import suggest
from pointtransformerv3.presets import PC_M1, PC_M2

__all__ = [
    "DROPPED",
    "PC_DEFAULTS",
    "RULES",
    "ConvertError",
    "convert_state_dict",
    "is_native",
    "load_pc_checkpoint",
    "map_name",
    "pc_backbone_config",
    "pc_version",
]

PC_VERSIONS = ("PT-v3m1", "PT-v3m2", "PT-v3m3")  # PC's registry names of its PTv3 backbones


class ConvertError(ValueError):
    """A source checkpoint or PC config that cannot be converted exactly."""


# ---- names

_NORM = r"(weight|bias|running_mean|running_var|num_batches_tracked)"
_AFFINE = r"(weight|bias)"
_BLOCK = r"(enc|dec)\.\1(\d+)\.block(\d+)\."
_TO_BLOCK = r"\1.\2.blocks.\3."

# (pattern, replacement): fullmatch on a backbone key; every key matches exactly one rule.
RULES: tuple[tuple[str, str], ...] = (
    (r"embedding\.stem\.conv\.weight", "embedding.conv.weight"),  # m1: SubMConv3d k=5
    (rf"embedding\.stem\.linear\.{_AFFINE}", r"embedding.proj.\1"),  # m2: Linear
    (rf"embedding\.stem\.norm\.{_NORM}", r"embedding.norm.\1"),
    (rf"enc\.enc(\d+)\.down\.proj\.{_AFFINE}", r"enc.\1.down.proj.\2"),
    (rf"enc\.enc(\d+)\.down\.norm\.0\.{_NORM}", r"enc.\1.down.norm.\2"),
    (rf"dec\.dec(\d+)\.up\.proj\.0\.{_AFFINE}", r"dec.\1.up.proj.\2"),
    (rf"dec\.dec(\d+)\.up\.proj\.1\.{_NORM}", r"dec.\1.up.norm.\2"),
    (rf"dec\.dec(\d+)\.up\.proj_skip\.0\.{_AFFINE}", r"dec.\1.up.skip_proj.\2"),
    (rf"dec\.dec(\d+)\.up\.proj_skip\.1\.{_NORM}", r"dec.\1.up.skip_norm.\2"),
    (rf"{_BLOCK}cpe\.0\.{_AFFINE}", rf"{_TO_BLOCK}cpe.conv.\4"),
    (rf"{_BLOCK}cpe\.1\.{_AFFINE}", rf"{_TO_BLOCK}cpe.proj.\4"),
    (rf"{_BLOCK}cpe\.2\.{_AFFINE}", rf"{_TO_BLOCK}cpe.norm.\4"),
    (rf"{_BLOCK}(norm1|norm2)\.0\.{_AFFINE}", rf"{_TO_BLOCK}\4.\5"),
    (rf"{_BLOCK}attn\.(qkv|proj)\.{_AFFINE}", rf"{_TO_BLOCK}attn.\4.\5"),
    (rf"{_BLOCK}mlp\.0\.(fc1|fc2)\.{_AFFINE}", rf"{_TO_BLOCK}mlp.\4.\5"),
    (rf"{_BLOCK}(ls1|ls2)\.0\.gamma", rf"{_TO_BLOCK}\4.gamma"),  # m2 LayerScale
    (rf"{_BLOCK}attn\.rope\.inv_freq", rf"{_TO_BLOCK}attn.rope.inv_freq"),  # m3 RoPE
)
# (pattern, reason): backbone keys left out on purpose.
DROPPED: tuple[tuple[str, str], ...] = (
    (r"embedding\.mask_token", "Sonata pretraining mask token (not used for inference)"),
)
_RULES = tuple((re.compile(p), r) for p, r in RULES)
_DROPPED = tuple((re.compile(p), r) for p, r in DROPPED)
_HEAD = ("seg_head.weight", "seg_head.bias")


def map_name(key: str) -> str | None:
    """The new name of a PC backbone key, or None for a key in ``DROPPED``.

    Raises ConvertError when no rule or more than one rule matches."""
    mapped = [m.expand(r) for p, r in _RULES if (m := p.fullmatch(key))]
    dropped = [p for p, _ in _DROPPED if p.fullmatch(key)]
    if len(mapped) + len(dropped) != 1:
        what = "no rule" if not mapped and not dropped else "more than one rule"
        raise ConvertError(f"{what} matches PC key {key!r}")
    return mapped[0] if mapped else None


def is_native(state: Mapping[str, Any]) -> bool:
    """Whether ``state`` already uses this package's names (``enc.0.blocks.0...``) rather than
    Pointcept's (``enc.enc0.block0...``)."""
    keys = [k.removeprefix("module.") for k in state]
    return any(k.startswith(("enc.0.", "embedding.conv.", "embedding.proj.")) for k in keys)


def convert_state_dict(
    state: Mapping[str, Any], target: Mapping[str, Tensor]
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Map a PC state dict onto ``target`` (a model's ``state_dict()``, e.g. built on the meta
    device). Returns the new state dict in ``target``'s key order (floating tensors cast to the
    target dtype) and ``{source key: reason}`` of the dropped keys.

    The source is a PC backbone (bare keys) or a ``DefaultSegmentorV2`` (``backbone.`` and
    ``seg_head.``), optionally under a DDP ``module.`` prefix. ``seg_head.*`` maps to the
    target's ``seg_head.*`` when the target has a head, else it is dropped. Raises ConvertError
    listing every unmatched, duplicated, missing or mis-shaped key."""
    keys = list(state)
    if keys and all(k.startswith("module.") for k in keys):
        state = {k.removeprefix("module."): v for k, v in state.items()}
        keys = list(state)
    wrapped = any(k.startswith("backbone.") for k in keys)
    has_head = all(k in target for k in _HEAD)
    problems: list[str] = []
    out: dict[str, Tensor] = {}
    origin: dict[str, str] = {}
    claimed: set[str] = set()  # target keys some source key maps to (valid or not)
    dropped: dict[str, str] = {}
    for key in keys:
        if wrapped and not key.startswith(("backbone.", "seg_head.")):
            problems.append(f"{key!r}: expected a backbone. or seg_head. key in a segmentor")
            continue
        if key.startswith("seg_head."):  # a segmentor's or a linear probe's head
            if not has_head:
                dropped[key] = "segmentation head (the target has none; load(..., head=...))"
                continue
            name: str | None = key
        else:
            base = key.removeprefix("backbone.")
            try:
                name = map_name(base)
            except ConvertError as e:
                problems.append(str(e))
                continue
            if name is None:
                dropped[key] = next(r for p, r in _DROPPED if p.fullmatch(base))
                continue
        value = state[key]
        claimed.add(name)
        if not isinstance(value, Tensor):
            problems.append(f"{key!r}: expected a tensor, got {type(value).__name__}")
        elif name in origin:
            problems.append(f"{key!r} and {origin[name]!r} both map to {name!r}")
        elif name not in target:
            problems.append(f"{key!r} -> {name!r}: not in the target{suggest(name, target)}")
        elif tuple(value.shape) != tuple(target[name].shape):
            problems.append(
                f"{key!r} -> {name!r}: shape {tuple(value.shape)}, target "
                f"{tuple(target[name].shape)}"
            )
        elif value.is_floating_point() != target[name].is_floating_point() or (
            not value.is_floating_point() and value.dtype != target[name].dtype
        ):
            problems.append(
                f"{key!r} -> {name!r}: dtype {value.dtype}, target {target[name].dtype}"
            )
        else:
            origin[name] = key
            out[name] = value
    problems += [f"target key {k!r} is not filled" for k in target if k not in claimed]
    if problems:
        shown = "\n  ".join(problems[:40])
        more = f"\n  ... and {len(problems) - 40} more" if len(problems) > 40 else ""
        raise ConvertError(f"{len(problems)} problem(s) mapping the state dict:\n  {shown}{more}")
    return {k: out[k].detach().to(dtype=target[k].dtype) for k in target}, dropped


# ---- config

_PC_COMMON: dict[str, Any] = dict(
    in_channels=6,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    pre_norm=True,
    shuffle_orders=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    enc_mode=False,
)
# Constructor defaults: PT-v3m1 = this repository's original model.py (3229e9b); PT-v3m2 = PC
# point_transformer_v3m2_sonata.py / the Sonata repository; PT-v3m3 = the Utonia repository's
# model.py (rope_base 10; PC point_transformer_v3m3_utonia.py defaults to 10000).
PC_DEFAULTS: dict[str, dict[str, Any]] = {
    "PT-v3m1": {
        **_PC_COMMON,
        "cls_mode": False,
        "pdnorm_bn": False,
        "pdnorm_ln": False,
        "pdnorm_decouple": True,
        "pdnorm_adaptive": False,
        "pdnorm_affine": True,
        "pdnorm_conditions": ("ScanNet", "S3DIS", "Structured3D"),
    },
    "PT-v3m2": {
        **_PC_COMMON,
        "layer_scale": None,
        "traceable": False,
        "mask_token": False,
        "freeze_encoder": False,
    },
}
PC_DEFAULTS["PT-v3m3"] = {
    **PC_DEFAULTS["PT-v3m2"],
    "rope_base": 10,
    "shift_coords": None,
    "jitter_coords": None,
    "rescale_coords": None,
}
# PC option -> (the only value this package can represent, how to fix it)
_REQUIRED: dict[str, tuple[Any, str]] = {
    "qk_scale": (None, "the scale is head_dim ** -0.5; rescale the qkv weights instead"),
    "attn_drop": (0.0, "attention dropout is not implemented; use drop_path for regularization"),
    "proj_drop": (0.0, "projection dropout is not implemented; use drop_path for regularization"),
    "pre_norm": (True, "only the pre-norm block (the released recipe) is implemented"),
    "enable_rpe": (False, "relative position encoding is not implemented (use Pointcept)"),
    "pdnorm_bn": (False, "PDNorm (PPT) is not implemented (use Pointcept)"),
    "pdnorm_ln": (False, "PDNorm (PPT) is not implemented (use Pointcept)"),
    "shift_coords": (None, "RoPE coordinate augmentation (training only) is not implemented"),
    "jitter_coords": (None, "RoPE coordinate augmentation (training only) is not implemented"),
    "rescale_coords": (None, "RoPE coordinate augmentation (training only) is not implemented"),
}
_RENAMED = {
    "in_channels": "in_channels",
    "stride": "strides",
    "enc_depths": "enc_depths",
    "enc_channels": "enc_channels",
    "enc_num_head": "enc_num_heads",
    "enc_patch_size": "enc_patch_sizes",
    "dec_depths": "dec_depths",
    "dec_channels": "dec_channels",
    "dec_num_head": "dec_num_heads",
    "dec_patch_size": "dec_patch_sizes",
}


def pc_backbone_config(
    pc: Mapping[str, Any],
    version: str = "PT-v3m1",
    *,
    stale_cpe_input: bool = False,
    serialization_depth: int | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(config, runtime)`` of a PC backbone of PC ``version`` (``PC_VERSIONS``) from its
    constructor kwargs ``pc`` (PC's defaults fill missing keys; a ``"type"`` key, if present, must
    name the same version). ``config`` is a full ``Config`` dict: PC m1 gets ``presets.PC_M1``
    and ``legacy_cpe_stale_input=stale_cpe_input``, PC m2 and m3 ``presets.PC_M2`` (m3 with its
    ``rope_base``); the serialization depth is per batch like the originals (``"auto"`` for m1,
    ``"auto+1"`` for m2 and m3) unless
    ``serialization_depth`` is given. ``runtime`` holds the options that choose kernels, not
    weights: ``{"precision": "fp32"}`` for ``enable_flash=False`` (the original's exact path),
    else ``{}``."""
    if version not in PC_VERSIONS:
        raise ConvertError(f"unknown PC backbone {version!r}; use one of {PC_VERSIONS}")
    pc = dict(pc)
    pc_type = pc.pop("type", version)
    if pc_type != version:
        raise ConvertError(f"the PC config builds {pc_type!r}, not {version!r}")
    defaults = PC_DEFAULTS[version]
    unknown = [k for k in pc if k not in defaults]
    if unknown:
        raise TypeError(
            "; ".join(
                f"unknown PointTransformerV3 argument {k!r}{suggest(k, defaults)}" for k in unknown
            )
        )
    kw = {**defaults, **pc}
    bad = [
        f"{k}={kw[k]!r}: {why}"
        for k, (v, why) in _REQUIRED.items()
        if k in kw and kw[k] != v and not (k == "qk_scale" and _is_default_scale(kw))
    ]
    if bad:
        raise ConvertError("unsupported option(s): " + "; ".join(bad))
    if stale_cpe_input and version != "PT-v3m1":
        raise ConvertError(f"PC {version} has no stale CPE inputs (it postdates the 74c4fc1 fix)")
    order = (kw["order"],) if isinstance(kw["order"], str) else kw["order"]
    cfg: dict[str, Any] = {"orders": tuple(o.replace("_", "-") for o in order)}
    for pc_key, key in _RENAMED.items():
        value = kw[pc_key]
        cfg[key] = tuple(value) if isinstance(value, list | tuple) else value
    if kw["enc_mode"] or kw.get("cls_mode"):
        cfg.update(dec_depths=(), dec_channels=(), dec_num_heads=(), dec_patch_sizes=())
    cfg.update(
        shuffle_orders=bool(kw["shuffle_orders"]),
        mlp_ratio=float(kw["mlp_ratio"]),
        qkv_bias=bool(kw["qkv_bias"]),
        drop_path=float(kw["drop_path"]),
        patch_tail="overlap",
        rope_base=float(kw["rope_base"]) if kw.get("rope_base") else None,  # PC m3: 0/None = off
    )
    if version == "PT-v3m1":
        cfg.update(PC_M1, layer_scale=None, legacy_cpe_stale_input=stale_cpe_input)
        cfg["serialization_depth"] = "auto"
    else:
        scale = kw["layer_scale"]
        cfg.update(PC_M2, layer_scale=None if scale is None else float(scale))
        cfg["legacy_cpe_stale_input"] = False
        cfg["serialization_depth"] = "auto+1"
    if serialization_depth is not None:
        cfg["serialization_depth"] = serialization_depth
    runtime = {} if kw["enable_flash"] else {"precision": "fp32"}
    return cfg, runtime


def _is_default_scale(kw: Mapping[str, Any]) -> bool:
    """A ``qk_scale`` equal to head_dim ** -0.5 everywhere is the default scale."""
    scale = kw["qk_scale"]
    if not isinstance(scale, int | float):
        return False
    heads = [*zip(kw["enc_channels"], kw["enc_num_head"], strict=False)]
    heads += [*zip(kw["dec_channels"], kw["dec_num_head"], strict=False)]
    return all(abs(scale - (c // h) ** -0.5) < 1e-12 for c, h in heads)


def pc_version(pc_backbone: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    """The PC version of a backbone: its config's ``"type"``, else its stem in ``state`` (a PC
    state dict, optionally with ``module.`` / ``backbone.`` prefixes): PC m2's stem is a Linear
    (``embedding.stem.linear``), PC m1's a ``SubMConv3d`` (``embedding.stem.conv``); a Linear
    stem with ``rope_base`` in the config or RoPE buffers in ``state`` is PC m3 (Utonia)."""
    version = pc_backbone.get("type")
    if version is not None:
        if version not in PC_VERSIONS:
            raise ConvertError(
                f"the PC backbone type is {version!r}; only {', '.join(PC_VERSIONS)} convert"
            )
        return version
    stems = {k.removeprefix("module.").removeprefix("backbone.") for k in state}
    found = [
        v
        for v, key in (
            ("PT-v3m1", "embedding.stem.conv.weight"),
            ("PT-v3m2", "embedding.stem.linear.weight"),
        )
        if key in stems
    ]
    if len(found) != 1:
        raise ConvertError(
            "cannot tell the PC backbone version: the config has no 'type' and the state dict "
            "has no single stem (embedding.stem.conv / embedding.stem.linear)"
        )
    rope = "rope_base" in pc_backbone or any(k.endswith(".attn.rope.inv_freq") for k in stems)
    return "PT-v3m3" if found[0] == "PT-v3m2" and rope else found[0]


# ---- reading checkpoints


class _Placeholder:
    """Stands in for a training-state global of a PC checkpoint; never does anything."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __setstate__(self, state: Any) -> None:
        del state


def _inert(name: str) -> bool:
    return name == "builtins.getattr" or name.startswith(("torch.optim.", "numpy."))


def load_pc_checkpoint(path: str | Path, *, mmap: bool = True) -> dict[str, Any]:
    """``torch.load(path, weights_only=True, mmap=mmap)`` on the CPU. Optimizer / scheduler
    classes, ``getattr`` and NumPy globals (PC's training state) load as placeholders; any other
    global that torch does not allow raises ConvertError before anything is unpickled."""
    path = Path(path)
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    refused = sorted(name for name in unsafe if not _inert(name))
    if refused:
        raise ConvertError(f"{path}: refusing to unpickle globals {refused}")
    with torch.serialization.safe_globals([(_Placeholder, name) for name in unsafe]):
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if not isinstance(obj, dict):
        raise ConvertError(f"{path}: expected a dict checkpoint, got {type(obj).__name__}")
    return obj
