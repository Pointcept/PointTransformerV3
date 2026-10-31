"""``load``: a preset with its released weights, on a device, at a precision, warmed up; and
``set_backend``: the kernel backends of every later call."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from point_transformer_v3.convert import ConvertError, pc_backbone_config, pc_version
from point_transformer_v3.model import Config, PointTransformerV3, Precision, _check_precision
from point_transformer_v3.presets import PRESETS, Preset, get_preset
from point_transformer_v3.weights import fetch, read_checkpoint, state_dict_of

__all__ = ["load", "set_backend", "warmup"]

log = logging.getLogger(__name__)


def load(
    name_or_path: str | os.PathLike[str],
    *,
    device: str | torch.device = "auto",
    precision: Precision = "auto",
    warmup: bool | None = None,
    head: str | bool | None = None,
    weights: str | os.PathLike[str] | None = None,
    revision: str | None = None,
    compile: bool = False,
) -> PointTransformerV3:
    """A ready-to-run model: built on the meta device, its weights read (memory-mapped),
    converted and assigned, moved to ``device``, in eval mode.

    Args:
        name_or_path: a preset (``presets.PRESETS``: ``"ptv3_scannet"``, ``"sonata"``,
            ``"concerto"``, ``"utonia"``, ...; ``"ptv3"`` for random init), or a local checkpoint
            that carries its constructor kwargs (``"config"``, as ``sonata.pth``).
        device: ``"auto"`` (cuda, else mps, else cpu) or a device.
        precision: ``"auto"`` (fp16 autocast on CUDA, fp32 elsewhere), ``"fp32"``, ``"fp16"`` or
            ``"bf16"``. The Linear stem and the LayerNorm statistics stay fp32.
        warmup: run forwards on synthetic clouds (about 4k and 40k points) and ``embed`` on a
            scan-sized raw one first, so Triton's kernels are compiled and cuBLAS, the
            preprocessing and the allocator's pool are ready before the first real input
            (``warmup``). Default: True on CUDA, False elsewhere.
        head: a segmentation head of the preset (``preset.heads``), ``None`` for the preset's
            default (``ptv3_scannet``: ``"scannet20"``; the encoders: none) or ``False`` for
            none.
        weights: a local checkpoint to use instead of the preset's download.
        revision: a Hugging Face revision instead of the pinned one (the pinned sha256 is then
            not checked).
        compile: reserved; ``torch.compile`` of the whole model is not supported yet.

    Returns:
        ``PointTransformerV3`` with ``.preset``, ``.weights_info`` (files, sha256, licenses) and,
        with a head, ``.seg_head``.
    """
    if compile:
        raise NotImplementedError(
            "load(compile=True) is not supported yet: the eager path already runs fused Triton "
            "kernels; see docs/acceleration.md"
        )
    _check_precision(precision)
    dev = _device(device)
    t0 = time.perf_counter()
    preset: Preset | None = None
    path = Path(os.fspath(name_or_path)).expanduser()
    if str(name_or_path) in PRESETS or not path.suffix:
        preset = get_preset(str(name_or_path))
    elif not path.is_file():
        raise FileNotFoundError(f"{path} is not a file and not a preset name ({sorted(PRESETS)})")
    info: dict[str, Any] = {}
    ckpt = None
    if preset is not None:
        config = Config.from_dict(preset.config)
        if weights is not None:
            ckpt_path = Path(os.fspath(weights)).expanduser()
            info["backbone"] = {"file": str(ckpt_path), "source": "local"}
        elif preset.weights is not None:
            ckpt_path = fetch(preset.weights, revision=revision)
            info["backbone"] = _info(preset.weights, ckpt_path, revision)
        else:
            ckpt_path = None
        if ckpt_path is not None:
            ckpt = read_checkpoint(ckpt_path)
            _check_config(ckpt, config, preset.name)
    else:
        ckpt = read_checkpoint(path)
        kwargs = ckpt.get("config")
        if not isinstance(kwargs, dict):
            raise ConvertError(
                f"{path} has no 'config' (constructor kwargs); load it through its preset, e.g. "
                f"load('ptv3_scannet', weights={str(path)!r})"
            )
        version = pc_version(kwargs, state_dict_of(ckpt))
        values, _ = pc_backbone_config(kwargs, version)
        config = Config.from_dict(values)
        info["backbone"] = {"file": str(path), "source": "local"}

    with torch.device("meta"):
        model = PointTransformerV3(config=config, precision=precision)
    model.preset = preset.name if preset is not None else None
    head_name = _head_name(preset, head)
    if head_name is not None:
        assert preset is not None
        spec = preset.heads[head_name]
        with torch.device("meta"):
            model.attach_head(spec.num_classes, spec.in_channels)
    if ckpt is None:  # random init (preset "ptv3")
        model.to_empty(device="cpu")
        model._init_weights()
        if model.seg_head is not None:
            model.seg_head.reset_parameters()
    else:
        state = dict(state_dict_of(ckpt))
        if head_name is not None:
            spec = preset.heads[head_name]  # type: ignore[union-attr]
            if spec.weights is not None:
                head_path = fetch(spec.weights, revision=revision)
                info["head"] = _info(spec.weights, head_path, revision)
                head_state = state_dict_of(read_checkpoint(head_path))
                state = {k: v for k, v in state.items() if not k.startswith("seg_head.")}
                state.update({k: v for k, v in head_state.items() if k.startswith("seg_head.")})
        model.load_state_dict(state, strict=True, assign=True)
    model.to(dev)
    model.eval()
    model.weights_info = info
    log.info("load(%s): %.2f s on %s", name_or_path, time.perf_counter() - t0, dev)
    if warmup is None:
        warmup = dev.type == "cuda"
    if warmup:
        _warmup(model)
    return model


def _warmup(
    model: PointTransformerV3,
    sizes: tuple[int, ...] = (4096, 4100, 40000, 40004),
    points: int = 600_000,
) -> float:
    """Run inference on synthetic inputs, then synchronize, so the first real call runs at steady
    speed: forwards on grid-sampled clouds of ``sizes`` points (row counts that are and are not
    multiples of 16) compile every Triton specialization and start cuBLAS; then, for a preset,
    ``embed`` (``prepare``, forward, ``upcast``, map back) on a raw cloud of ``points`` points
    (about 2.3 per grid cell: a ScanNet-sized scan) runs the preprocessing kernels and grows the
    allocator's pool to a scan's size (on sample1 the first ``embed`` took 159 ms with forwards
    only, 69 ms with an 80k-point cloud, 38 ms = steady with this one). Returns the seconds it
    took (logged)."""
    device = next(model.parameters()).device
    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(0)
    with torch.inference_mode():
        for n in sizes:
            model(_synthetic(n, model.cfg.in_channels, gen, device))
    if model.preset is not None:
        spec = get_preset(model.preset).prepare
        if 3 * len(spec.feat) == model.cfg.in_channels:  # the preset's own columns
            model.embed(*_raw_cloud(points, spec.grid_size / spec.scale, gen))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.train(was_training)
    seconds = time.perf_counter() - t0
    log.info("warmup: %.2f s (%s points)", seconds, "/".join(map(str, sizes)))
    return seconds


def set_backend(
    attention: str | None = None, conv: str | None = None, layer_norm: str | None = None
) -> None:
    """Choose the kernel backends of every later call (also through the environment variables
    ``PTV3_ATTENTION_BACKEND``, ``PTV3_CONV_BACKEND``,
    ``PTV3_LAYER_NORM_BACKEND``): attention ``"auto" | "triton_d16" | "varlen" |
    "sdpa"``, conv ``"auto" | "triton" | "torch"``, layer_norm ``"auto" | "triton" | "torch"``.
    None leaves a backend as it is. Each choice is logged once per shape class."""
    from point_transformer_v3.ops.attention import set_attention_backend
    from point_transformer_v3.ops.norm import set_layer_norm_backend
    from point_transformer_v3.ops.sparse_conv import set_backend as set_conv_backend

    if attention is not None:
        set_attention_backend(attention)  # type: ignore[arg-type]
    if conv is not None:
        set_conv_backend(conv)  # type: ignore[arg-type]
    if layer_norm is not None:
        set_layer_norm_backend(layer_norm)  # type: ignore[arg-type]


warmup = _warmup


# ---- helpers


def _device(device: str | torch.device) -> torch.device:
    if str(device) != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _head_name(preset: Preset | None, head: str | bool | None) -> str | None:
    if head is False or preset is None:
        if isinstance(head, str):
            raise ValueError(f"head={head!r} needs a preset; attach one with model.attach_head")
        return None
    if head is None or head is True:
        name = preset.default_head
        if head is True and name is None:
            name = next(iter(preset.heads), None)
            if name is None:
                raise ValueError(f"preset {preset.name!r} has no heads")
        return name
    if head not in preset.heads:
        raise ValueError(
            f"preset {preset.name!r} has no head {head!r}; heads: {sorted(preset.heads) or 'none'}"
        )
    return head


def _info(source: Any, path: Path, revision: str | None) -> dict[str, Any]:
    return {
        "repo": source.repo,
        "file": source.file,
        "revision": revision or source.revision,
        "sha256": source.sha256 if revision in (None, source.revision) else None,
        "license": source.license,
        "path": str(path),
    }


def _check_config(ckpt: dict[str, Any], config: Config, name: str) -> None:
    """A checkpoint that carries constructor kwargs must describe the preset's architecture (the
    serialization depth rule and training-only switches aside)."""
    kwargs = ckpt.get("config")
    if not isinstance(kwargs, dict):
        return
    version = pc_version(kwargs, state_dict_of(ckpt))
    values, _ = pc_backbone_config(kwargs, version)
    mine = config.to_dict()
    skip = {"serialization_depth", "shuffle_orders", "drop_path"}
    diff = {k: (v, mine.get(k)) for k, v in values.items() if k not in skip and mine.get(k) != v}
    if diff:
        raise ConvertError(
            f"the checkpoint's architecture differs from preset {name!r}: "
            + ", ".join(f"{k}: checkpoint {a!r} vs preset {b!r}" for k, (a, b) in diff.items())
        )


def _raw_cloud(n: int, grid_size: float, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """``n`` raw points (CPU, as a user passes them) in a cube of ``n / 2`` grid cells of
    ``grid_size`` meters (about 2.3 points per occupied cell): coord, color in [0, 255], unit
    normals."""
    coord = torch.rand(n, 3, generator=gen) * (n / 2) ** (1 / 3) * grid_size
    normal = torch.nn.functional.normalize(torch.randn(n, 3, generator=gen), dim=1)
    return coord, torch.rand(n, 3, generator=gen) * 255, normal


def _synthetic(n: int, in_channels: int, gen: torch.Generator, device: torch.device) -> dict:
    """``n`` points on unique random grid cells in a cube about 3.5 m wide at 2 cm (sparse like
    a scan)."""
    side = max(8, round((n * 6) ** (1 / 3)))
    cells = torch.randperm(side**3, generator=gen)[:n]
    grid = torch.stack([cells // side**2, cells // side % side, cells % side], dim=1).int()
    feat = torch.randn(n, in_channels, generator=gen)
    return {
        "coord": (grid.float() * 0.02).to(device),
        "grid_coord": grid.to(device),
        "feat": feat.to(device),
        "offset": torch.tensor([n], device=device),
        "grid_size": 0.02,
    }
