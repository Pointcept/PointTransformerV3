"""Helpers shared by the demos: the demo samples, model loading, PCA and similarity colors, PLY and
PNG output, an optional open3d viewer.

Parts are adapted from the Utonia demos (github.com/Pointcept/Utonia, demo/ at 5fce241) and the
Concerto demos (github.com/Pointcept/Concerto, demo/ at 5b41117), Apache-2.0, which build on the
Sonata demos (Copyright (c) Meta Platforms, Inc. and affiliates): ``pca_color`` (their
``get_pca_color``), ``heat_color`` (the mapping of ``1_similarity.py``) and ``normalize_coord``
(Utonia's ``NormalizeCoord``). Changed: torch only (no matplotlib, open3d optional), one function
for the per-demo PCA variants, an exact PCA with fixed component signs instead of the seeded
``torch.pca_lowrank``. The Apache-2.0 text is in THIRD_PARTY_LICENSES.md next to this file.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

import point_transformer_v3 as ptv3

# Hugging Face dataset with the demo samples (``pointcept/demo``), pinned revision.
DEMO_REPO = "pointcept/demo"
DEMO_REVISION = "65420460fa65018c2a9763ccb588b077a7ca5ce7"
OUTPUT = Path(__file__).resolve().parent / "output"

UTONIA = ("utonia", "utonia_small", "utonia_tiny")
CONCERTO = ("concerto", "concerto_large", "concerto_siglip", "concerto_origin")
SONATA = ("sonata", "sonata_small")
ENCODERS = UTONIA + CONCERTO + SONATA

# PCA signs (``pca_color``) that reproduce the colors of the original indoor demos (Utonia
# 0_pca_indoor, Concerto 0_pca, Sonata 0_pca) on sample1; the other presets keep pca_color's own
# sign convention.
INDOOR_SIGNS = {
    "utonia": (-1, -1, -1, -1, 1, 1),
    "utonia_small": (-1, 1, -1, 1, -1, -1),
    "utonia_tiny": (-1, -1, -1, -1, -1, 1),
    "concerto_large": (1, 1, 1, -1, -1, 1),
    "sonata": (-1, 1, -1, 1, -1, -1),
}


def parser(
    doc: str | None,
    *,
    models: tuple[str, ...] = ENCODERS,
    model: str = "utonia",
    sample: str | None = "sample1",
) -> argparse.ArgumentParser:
    """The options every demo shares; a demo adds its own and calls ``parse_args``."""
    p = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default=model, choices=models, help=f"preset (default {model})")
    if sample is not None:
        p.add_argument(
            "--sample",
            default=sample,
            help=f"a pointcept/demo sample or an .npz with coord, color, normal (default {sample})",
        )
    p.add_argument("--wo_color", action="store_true", help="replace the colors with zeros")
    p.add_argument("--wo_normal", action="store_true", help="replace the normals with zeros")
    p.add_argument("--device", default="auto", help="auto, cuda, cpu, mps")
    p.add_argument("--precision", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    p.add_argument(
        "--show", action="store_true", help="open an open3d window (uv sync --extra demo)"
    )
    return p


def load_sample(name: str) -> dict[str, np.ndarray]:
    """A demo sample as float32 ``coord`` (m), ``color`` (0-255) and ``normal`` (when present),
    plus ``segment20`` (int64) for ScanNet scans. ``name`` is a local .npz or a pointcept/demo
    sample, fetched from Hugging Face at the pinned revision (``HF_HOME`` and offline mode
    apply)."""
    path = Path(name)
    if not path.is_file():
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        file = f"{name}.npz"
        if not isinstance(
            try_to_load_from_cache(DEMO_REPO, file, revision=DEMO_REVISION, repo_type="dataset"),
            str,
        ):
            print(f"downloading {file} from the {DEMO_REPO} dataset on Hugging Face ...")
        path = Path(
            hf_hub_download(DEMO_REPO, f"{name}.npz", repo_type="dataset", revision=DEMO_REVISION)
        )
    data = np.load(path)
    out = {k: data[k].astype(np.float32) for k in ("coord", "color", "normal") if k in data}
    if "segment20" in data:
        out["segment20"] = data["segment20"].astype(np.int64)
    return out


def drop_inputs(scan: dict[str, np.ndarray], opt: argparse.Namespace) -> dict[str, np.ndarray]:
    """``--wo_color`` / ``--wo_normal``: zeros instead of the colors or normals, as the Concerto
    and Utonia demos do (both models were trained to accept that)."""
    scan = dict(scan)
    for key, off in (("color", opt.wo_color), ("normal", opt.wo_normal)):
        if off or key not in scan:
            scan[key] = np.zeros_like(scan["coord"])
    return scan


def load_model(opt: argparse.Namespace, **kwargs: Any) -> ptv3.PointTransformerV3:
    return ptv3.load(opt.model, device=opt.device, precision=opt.precision, **kwargs)


def device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def prepare(model: ptv3.PointTransformerV3, scan: dict[str, np.ndarray], **kwargs: Any):
    """``ptv3.prepare`` with the model's preset, on the model's device (``scale``, ``center``,
    ``grid_size`` override the preset)."""
    return ptv3.prepare(
        scan["coord"],
        scan["color"],
        scan["normal"],
        preset=model.preset,
        device=device_of(model),
        **kwargs,
    )


def features(
    model: ptv3.PointTransformerV3, point: ptv3.Point, levels: int | None = 2
) -> torch.Tensor:
    """Per-row features of a prepared cloud (or batch): forward, then ``upcast``. ``levels=2``
    concatenates the three coarsest levels and copies them down (the PCA and similarity demos);
    ``None`` concatenates every level (the linear probe)."""
    with torch.inference_mode():
        return ptv3.upcast(model(point), levels=levels).feat.float()


def first_rows(inverse: torch.Tensor, rows: int) -> torch.Tensor:
    """The first input point of every grid-sampled row (the point ``prepare(pick="first")``
    kept), to place per-row results at input coordinates."""
    n = torch.arange(len(inverse), device=inverse.device)
    out = torch.full((rows,), len(inverse), dtype=torch.long, device=inverse.device)
    return out.scatter_reduce_(0, inverse, n, "amin")


def pca_color(
    feat: torch.Tensor,
    weights: tuple[float, ...] = (0.6, 0.4),
    *,
    brightness: float = 1.2,
    signs: Sequence[int] | None = None,
    channels: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """RGB in [0, 1] from the principal components of ``feat`` (the demos' ``get_pca_color``):
    component group ``i`` (components 3i to 3i+2) is weighted by ``weights[i]``, each channel is
    min-max scaled, times ``brightness`` and clipped; ``channels`` reorders the result.

    The PCA is exact (eigenvectors of the C x C covariance) and each component's sign is set by
    the data: the sign of the projection's skewness, or, where the projection is closer to
    symmetric, the side of the feature mean (whichever is larger relative to the component's
    standard deviation). So the colors do not depend on the device, the row order or a seed, and
    small changes of the features keep them. ``signs`` (+1 or -1 per component) flips components
    on top of that; the demos use it to reproduce the colors of the original demos."""
    feat = feat.float()
    k = 3 * len(weights)
    mean = feat.mean(0, keepdim=True)
    cov = torch.zeros(feat.shape[1], feat.shape[1], device=feat.device)
    for chunk in feat.split(1 << 18):  # centered in chunks: no second copy of a large feat
        chunk = chunk - mean
        cov += chunk.T @ chunk
    _, vecs = torch.linalg.eigh(cov.cpu().double())  # ascending eigenvalues
    v = vecs[:, -k:].flip(1).to(feat.device, feat.dtype)  # the k largest, largest first
    proj = feat @ v  # uncentered: the min-max scaling removes the offset
    offset = proj.mean(0)
    std = (proj - offset).pow(2).mean(0).sqrt().clamp(min=1e-12)
    skew = (proj - offset).pow(3).mean(0) / std**3
    side = offset / std  # where the feature mean lies along the component
    sign = torch.where(skew.abs() >= side.abs(), skew.sign(), side.sign())
    sign[sign == 0] = 1
    if signs is not None:
        if len(signs) != k:
            raise ValueError(f"signs needs {k} entries (3 per weight), got {len(signs)}")
        sign = sign * torch.tensor(signs, dtype=sign.dtype, device=sign.device)
    proj = proj * sign
    groups = [w * proj[:, 3 * i : 3 * i + 3] for i, w in enumerate(weights) if w]
    rgb = groups[0]
    for group in groups[1:]:
        rgb = rgb + group
    if channels is not None:
        rgb = rgb[:, list(channels)]
    low = rgb.min(dim=0, keepdim=True).values
    high = rgb.max(dim=0, keepdim=True).values
    rgb = (rgb - low) / (high - low).clamp(min=1e-6) * brightness
    return rgb.clamp(0.0, 1.0).cpu().numpy()


# ColorBrewer "Spectral" (11 classes), reversed: matplotlib's "Spectral_r" without matplotlib
_SPECTRAL_R = (
    np.array(
        [
            (94, 79, 162),
            (50, 136, 189),
            (102, 194, 165),
            (171, 221, 164),
            (230, 245, 152),
            (255, 255, 191),
            (254, 224, 139),
            (253, 174, 97),
            (244, 109, 67),
            (213, 62, 79),
            (158, 1, 66),
        ],
        dtype=np.float64,
    )
    / 255
)
_LUT = np.stack(
    [np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, 11), _SPECTRAL_R[:, c]) for c in range(3)],
    axis=1,
)


def spectral_r(x: np.ndarray) -> np.ndarray:
    """matplotlib's ``Spectral_r`` colormap (256 entries) for values in [0, 1]."""
    return _LUT[np.clip((np.asarray(x) * 256).astype(np.int64), 0, 255)]


def heat_color(
    sim: torch.Tensor,
    ref: torch.Tensor,
    oral: float = 0.02,
    highlight: float = 0.1,
    reject: float = 0.5,
) -> np.ndarray:
    """The similarity demos' heat map: cosine similarities ``sim`` mapped by sigmoids around the
    thresholds that the top ``oral`` / ``highlight`` fractions and the bottom ``reject``
    fraction of ``ref`` (the similarities to the whole scene) define; red = similar."""
    ranked = torch.sort(ref, descending=True).values
    n = ranked.numel()
    top, mid, low = ranked[int(n * oral)], ranked[int(n * highlight)], ranked[-int(n * reject)]
    x = sim - mid
    out = x.clone()
    out[x > 0] = torch.sigmoid(x[x > 0] / (top - mid))
    out[x < 0] = torch.sigmoid(x[x < 0] / (mid - low)) * 0.9
    return spectral_r(out.float().cpu().numpy())


def normalize_coord(coord: np.ndarray) -> np.ndarray:
    """Utonia's ``NormalizeCoord`` for single objects: centroid to the origin, farthest point at
    distance 1."""
    coord = coord - np.mean(coord, axis=0)
    return coord / np.max(np.sqrt(np.sum(coord**2, axis=1)))


def estimate_normals(coord: np.ndarray, k: int = 30) -> np.ndarray:
    """Unit normals for a cloud without them: open3d's ``estimate_normals`` (30 nearest
    neighbors, as the Utonia object demo) when open3d is installed, else the same plane fit in
    torch (signs may differ from open3d's). The torch fallback compares every point with every
    other (O(N^2) time): fine for objects of some 10k points, slow beyond 100k; install open3d
    for larger clouds."""
    try:
        import open3d as o3d  # pyright: ignore[reportMissingImports]
    except ImportError:
        o3d = None
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
        pcd.estimate_normals()
        return np.asarray(pcd.normals, dtype=np.float32)
    x = torch.from_numpy(coord).double()
    normals = []
    rows = max(1, min(4096, (1 << 26) // len(x)))  # a distance block of at most 512 MB
    for chunk in x.split(rows):
        idx = torch.cdist(chunk, x).topk(min(k, len(x)), largest=False).indices
        nb = x[idx] - x[idx].mean(1, keepdim=True)
        normals.append(torch.linalg.eigh(nb.transpose(1, 2) @ nb).eigenvectors[:, :, 0])
    return torch.cat(normals).float().numpy()


def write_ply(path: Path, coord: np.ndarray, rgb: np.ndarray) -> Path:
    """A binary little-endian PLY with float xyz and uchar rgb (``rgb`` in [0, 1])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(coord)
    rows = np.zeros(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    rows["x"], rows["y"], rows["z"] = coord[:, 0], coord[:, 1], coord[:, 2]
    rgb8 = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    rows["red"], rows["green"], rows["blue"] = rgb8[:, 0], rgb8[:, 1], rgb8[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rows.tobytes())
    return path


def write_png(
    path: Path,
    coord: np.ndarray,
    rgb: np.ndarray,
    *,
    elevation: float = 90.0,
    azimuth: float = 0.0,
    width: int = 1200,
    point_size: int = 2,
) -> Path:
    """A quick orthographic snapshot without a renderer: points splatted front to back on a
    white image. ``elevation`` 90 looks straight down; lower values tilt the view toward the
    horizon, ``azimuth`` turns it around z (degrees)."""
    xyz = np.asarray(coord, np.float64) - np.asarray(coord, np.float64).mean(0)
    a, e = np.radians(azimuth), np.radians(90.0 - elevation)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), np.sin(e)], [0, -np.sin(e), np.cos(e)]])
    view = xyz @ rz.T @ rx.T  # screen x, screen up, depth toward the viewer
    low, high = np.percentile(view[:, :2], [0.5, 99.5], axis=0)
    scale = (width - 2 * point_size) / max(high[0] - low[0], 1e-9)
    height = max(int((high[1] - low[1]) * scale) + 2 * point_size, 1)
    u = ((view[:, 0] - low[0]) * scale).astype(np.int64) + point_size
    v = height - 1 - point_size - ((view[:, 1] - low[1]) * scale).astype(np.int64)
    keep = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    order = np.argsort(view[keep, 2], kind="stable")[::-1]  # nearest first
    u, v, color = u[keep][order], v[keep][order], rgb[keep][order]
    image = np.full((height, width, 3), 255, np.uint8)
    for du in range(point_size):
        for dv in range(point_size):
            pix = np.clip(v + dv, 0, height - 1) * width + np.clip(u + du, 0, width - 1)
            pix, first = np.unique(pix, return_index=True)  # the nearest point of each pixel
            image.reshape(-1, 3)[pix] = (np.clip(color[first], 0, 1) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + row.tobytes() for row in image)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return path


def save(name: str, coord: np.ndarray, rgb: np.ndarray, **view: Any) -> None:
    """``demo/output/<name>.ply`` and a ``.png`` snapshot (``view``: ``write_png`` options)."""
    print("wrote", write_ply(OUTPUT / f"{name}.ply", coord, rgb))
    print("wrote", write_png(OUTPUT / f"{name}.png", coord, rgb, **view))


def show(
    clouds: Sequence[tuple[np.ndarray, np.ndarray]],
    lines: Sequence[tuple[np.ndarray, np.ndarray]] = (),
) -> None:
    """Open an open3d window with point clouds ``(coord, rgb)`` and black line segments."""
    try:
        import open3d as o3d  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("open3d is not installed (uv sync --extra demo); the PLY files were written")
        return
    geometries = []
    for coord, rgb in clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(coord, np.float64))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb, np.float64))
        geometries.append(pcd)
    for a, b in lines:
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.stack([a, b]).astype(np.float64))
        line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        line.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        geometries.append(line)
    o3d.visualization.draw_geometries(geometries)
