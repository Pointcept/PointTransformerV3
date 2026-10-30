"""Helpers shared by the demos: the demo point clouds, PCA colors, PLY output, an optional
open3d viewer. Written for this package (no code from the Sonata repository)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

# Hugging Face dataset with the demo scans (``pointcept/demo``), pinned revision.
DEMO_REPO = "pointcept/demo"
DEMO_REVISION = "65420460fa65018c2a9763ccb588b077a7ca5ce7"
OUTPUT = Path(__file__).resolve().parent / "output"


def args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sample", default="sample1", help="demo scan (pointcept/demo) or .npz")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto", choices=("auto", "fp32", "fp16", "bf16"))
    parser.add_argument("--show", action="store_true", help="open an open3d window")
    return parser.parse_args()


def load_sample(name: str = "sample1") -> dict[str, np.ndarray]:
    """A demo scan as float32 arrays: coord (m), color (0-255), normal, segment20 when present.
    Reuses an earlier download in ``~/.cache/sonata/data``; otherwise fetches it from Hugging
    Face (``HF_HOME`` and offline mode apply)."""
    path = Path(name)
    if not path.is_file():
        legacy = Path(os.path.expanduser(f"~/.cache/sonata/data/{name}.npz"))
        if legacy.is_file():
            path = legacy
        else:
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    DEMO_REPO, f"{name}.npz", repo_type="dataset", revision=DEMO_REVISION
                )
            )
    data = dict(np.load(path))
    out = {k: data[k].astype(np.float32) for k in ("coord", "color", "normal") if k in data}
    if "segment20" in data:
        out["segment20"] = data["segment20"].astype(np.int64)
    return out


def pca_color(feat: torch.Tensor, brightness: float = 1.2) -> np.ndarray:
    """The first three principal components of ``feat`` as RGB in [0, 1]."""
    feat = feat.float()
    _, _, v = torch.pca_lowrank(feat, q=6, niter=5, center=True)
    proj = (feat - feat.mean(0)) @ v[:, :3]
    low, high = proj.quantile(0.01, dim=0), proj.quantile(0.99, dim=0)
    rgb = ((proj - low) / (high - low)).clamp(0, 1) * brightness
    return rgb.clamp(0, 1).cpu().numpy()


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


def show(coord: np.ndarray, rgb: np.ndarray) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("open3d is not installed (uv sync --extra demo); the PLY file was written")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    o3d.visualization.draw_geometries([pcd])
