"""Inference preprocessing: raw points -> a model input, and the inverse to map results back.

``prepare`` does what the Pointcept / Sonata / Utonia inference transforms do, in torch (CPU or
GPU):

0. Scale (``scale``, Utonia): coordinates times a fixed factor (Utonia's ``RandomScale`` with
   equal bounds; numpy multiplies in float64 and rounds to float32, and so does this).
1. Center (``center=True``): x and y to the middle of the bounding box, z to its floor
   (Pointcept's ``CenterShift(apply_z=True)``), in float32.
2. Grid sampling at ``grid_size`` meters (Pointcept's ``GridSample``): the lattice
   ``floor(coord / grid_size)`` minus its minimum, and one point per occupied grid cell
   (``pick="first"``: the lowest input row, deterministic; ``"random"``: a uniform draw, as
   Pointcept's training-mode ``GridSample``). The division runs in float32, as the original
   transforms did under the numpy 1.x they pin (``float32 array / np.array(grid_size)`` stays
   float32 there; numpy 2 promotes it to float64 and moves about 1 point in 10^5 to a
   neighboring cell). ``lattice="float64"`` takes the numpy 2 / Pointcept2 rule.
3. Features in the preset's order: ``coord`` (the centered coordinates of the picked point),
   ``color`` (``/ 255``, or ``/ 127.5 - 1`` for ``color_range="signed"``) and ``normal``.

It returns the model input (a ``Point``: ``coord``, ``grid_coord``, ``feat``, ``offset``,
``grid_size``) and ``inverse`` ``(N,)``: the row of every input point's grid cell, so
``feat[inverse]`` maps per-row results back to the input points.

Rows come out ordered by grid cell (x, y, z), not by Pointcept's hash order; the model is
row-permutation equivariant, so only the picked point of each cell can change results
(``pick``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from pointtransformerv3.ops.grid_sample import grid_sample, unique_grid
from pointtransformerv3.presets import PrepareSpec, get_preset
from pointtransformerv3.structure import Point

__all__ = ["collate", "prepare"]

_FEATS = ("coord", "color", "normal")


def prepare(
    coord: Any,
    color: Any = None,
    normal: Any = None,
    *,
    preset: str | None = None,
    grid_size: float | None = None,
    feat: Sequence[str] | None = None,
    color_range: str | None = None,
    center: bool | None = None,
    scale: float | None = None,
    pick: str = "first",
    generator: torch.Generator | None = None,
    lattice: str = "float32",
    device: str | torch.device | None = None,
) -> tuple[Point, Tensor]:
    """Turn one raw point cloud into a model input (module doc).

    Args:
        coord: ``(N, 3)`` coordinates in meters (numpy array, tensor or nested list).
        color: ``(N, 3)`` RGB in [0, 255], or None.
        normal: ``(N, 3)`` unit normals, or None.
        preset: take ``grid_size``, ``feat``, ``color_range``, ``center`` and ``scale`` from this
            preset (``presets.PRESETS[name].prepare``); explicit arguments override it. Without a
            preset the defaults are 0.02 m, ``("color", "normal")``, ``"unit"``, centering and 1.
        scale: multiply the coordinates by this first (Utonia: 0.5 for indoor scenes, 0.2 for
            outdoor LiDAR with ``center=False``; a larger scale gives finer features).
        feat: feature columns, from ``"coord"``, ``"color"``, ``"normal"``.
        pick: which point represents a grid cell: ``"first"`` or ``"random"`` (``"center"``, the
            point nearest the cell center, needs ``lattice="float64"``).
        generator: the torch generator of ``pick="random"`` (on ``device``).
        lattice: ``"float32"`` (the original transforms under numpy 1.x) or ``"float64"``.
        device: where to compute and return the tensors (default: coord's device, else CPU).

    Returns:
        ``(point, inverse)``: the model input and the ``(N,)`` int64 index of each input point's
        kept row (its grid cell).
    """
    spec = get_preset(preset).prepare if preset is not None else PrepareSpec()
    grid_size = spec.grid_size if grid_size is None else float(grid_size)
    feat = spec.feat if feat is None else tuple(feat)
    color_range = spec.color_range if color_range is None else color_range
    center = spec.center if center is None else center
    scale = spec.scale if scale is None else float(scale)
    bad = [k for k in feat if k not in _FEATS]
    if bad:
        raise ValueError(f"unknown feature columns {bad}; use some of {_FEATS}")
    if color_range not in ("unit", "signed"):
        raise ValueError(f"color_range must be 'unit' or 'signed', got {color_range!r}")
    if device is None:
        device = coord.device if isinstance(coord, Tensor) else torch.device("cpu")
    device = torch.device(device)
    coord = _rows("coord", coord, device)
    n = coord.shape[0]
    columns = {"color": color, "normal": normal}
    for key in feat:
        if key != "coord" and columns.get(key) is None:
            raise ValueError(
                f"this model's features are {feat}: pass {key}= (N, 3); prepare(..., "
                f"feat=...) changes the columns only for models trained that way"
            )
    color_t = None if color is None else _rows("color", color, device, n)
    normal_t = None if normal is None else _rows("normal", normal, device, n)
    if scale != 1.0:  # float64 as numpy (Utonia), except on MPS (no float64)
        coord = coord * scale if device.type == "mps" else (coord.double() * scale).float()
    if center and n:
        low, high = coord.amin(0), coord.amax(0)
        shift = torch.stack([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]])
        coord = coord - shift
    if lattice == "float64":
        sampled = grid_sample(coord, grid_size, pick=pick, generator=generator, return_inverse=True)  # type: ignore[arg-type]
    elif lattice == "float32":
        if pick not in ("first", "random"):
            raise ValueError(f'pick={pick!r} needs lattice="float64"; use "first" or "random"')
        # a device tensor divisor: CUDA turns division by a host scalar into a reciprocal multiply
        size = torch.full((), grid_size, dtype=torch.float32, device=device)
        cells = torch.floor(coord / size).long()
        if n:
            cells -= cells.amin(0)
        sampled = unique_grid(cells, pick=pick, generator=generator, return_inverse=True)  # type: ignore[arg-type]
    else:
        raise ValueError(f"lattice must be 'float32' or 'float64', got {lattice!r}")
    index = sampled.index
    picked = {"coord": coord.index_select(0, index)}
    if color_t is not None:
        c = color_t.index_select(0, index)
        picked["color"] = c / 255 if color_range == "unit" else c / 127.5 - 1
    if normal_t is not None:
        picked["normal"] = normal_t.index_select(0, index)
    point = Point(
        coord=picked["coord"],
        grid_coord=sampled.grid_coord,
        feat=torch.cat([picked[k] for k in feat], dim=1),
        offset=torch.tensor([index.shape[0]], device=device),
        grid_size=grid_size,
    )
    if sampled.inverse is None:  # pragma: no cover - return_inverse=True always sets it
        raise RuntimeError("grid sampling returned no inverse")
    return point, sampled.inverse


def collate(points: Sequence[Point]) -> Point:
    """Pack prepared ``Point``s into one batch (rows concatenated, ``offset`` = cumulative sample
    ends). Keys other than ``coord``, ``grid_coord``, ``feat`` and ``grid_size`` are dropped."""
    if not points:
        raise ValueError("collate needs at least one Point")
    if len(points) == 1:
        return points[0]
    sizes = {float(p["grid_size"]) for p in points if "grid_size" in p}
    if len(sizes) > 1:
        raise ValueError(f"cannot pack clouds of different grid sizes {sorted(sizes)}")
    device = points[0]["feat"].device
    counts = torch.tensor([p["feat"].shape[0] for p in points], device=device)
    out = Point(
        coord=torch.cat([p["coord"] for p in points]),
        grid_coord=torch.cat([p["grid_coord"] for p in points]),
        feat=torch.cat([p["feat"] for p in points]),
        offset=counts.cumsum(0),
    )
    if sizes:
        out["grid_size"] = sizes.pop()
    return out


def _rows(name: str, value: Any, device: torch.device, n: int | None = None) -> Tensor:
    t = value if isinstance(value, Tensor) else torch.as_tensor(value)
    t = t.to(device=device, dtype=torch.float32)
    if t.dim() != 2 or t.shape[1] != 3 or (n is not None and t.shape[0] != n):
        rows = "N" if n is None else str(n)
        raise ValueError(f"{name} must be ({rows}, 3), got {tuple(t.shape)}")
    return t
