"""PCA of features on a point cloud lifted from a video (or a folder of images) by VGGT.

VGGT (github.com/facebookresearch/vggt) is not a dependency of this package and is not shipped
with it: install it yourself, with a torchvision that matches your torch (0.27 for torch 2.12).
Its code is under the VGGT License (Meta), the facebook/VGGT-1B weights it downloads (5 GB)
under CC BY-NC 4.0.

    uv run --with "vggt @ git+https://github.com/facebookresearch/vggt" \
        --with "torchvision==0.27.*" demo/8_pca_video.py --input my_video.mp4
    # --input my_frames/ for a folder of images; a video gives one frame per second
    uv run demo/8_pca_video.py --input demo/output/video/points.npz --model concerto_large

The lifted cloud (z up, scaled to 3 m height, ground on the xy-plane) is saved as
demo/output/video/points.npz and pcd.ply, so other models can run on it without VGGT. The PCA
colors go to demo/output/video/pca_<model>.ply and .png; --show opens a viewer. --tsdf fuses
the depth maps in a TSDF and uses its surface points (open3d: uv sync --extra demo).

Adapted from Utonia demo/8_pca_video.py and Concerto demo/4_pca_video.py (Apache-2.0, based on
the Sonata demo, Copyright (c) Meta Platforms, Inc. and affiliates; see
THIRD_PARTY_LICENSES.md): the steps (frames, VGGT depth and cameras, confidence filter, z up,
3 m height, ground alignment, PCA from the second component group) follow them; the code is
rewritten on VGGT's public API, without camtools, trimesh or scipy. Changed: the flags are
--input, --tsdf and --pointmap (the originals' --input_video, --if_TSDF and --prediction_mode);
the points come from VGGT's depth head and cameras (the originals convert the point map's
distances to depth and project that); normals come from the depth maps' image gradients
(the originals use open3d's estimate_normals); the cloud keeps its colors (the originals pass
colors in [0, 1] where 0-255 is expected, which makes them nearly black); --conf_thres drops
50% of the points by default (the originals 10%); --tsdf uses open3d's VoxelBlockGrid and its
surface points, which works on open3d 0.19 and 0.20 (the originals take the vertices of a
ScalableTSDFVolume mesh, which is empty on open3d 0.20); the PCA is exact (_common.pca_color).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from _common import (
    CONCERTO,
    OUTPUT,
    SONATA,
    UTONIA,
    device_of,
    drop_inputs,
    features,
    load_model,
    parser,
    pca_color,
    save,
    show,
)

import pointtransformerv3 as ptv3

IMAGES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
OUT = OUTPUT / "video"
NEEDS_VGGT = (
    "this demo needs VGGT, which pointtransformerv3 does not install: uv run --with "
    '"vggt @ git+https://github.com/facebookresearch/vggt" --with "torchvision==0.27.*" '
    "demo/8_pca_video.py ... (see the docstring), or --input <points.npz>"
)


def frames(source: Path, interval: float) -> list[Path]:
    """Image paths: the files of a folder, or one frame every ``interval`` seconds of a video."""
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGES)
    import cv2  # pyright: ignore[reportMissingImports]  # opencv-python, a VGGT dependency

    video = cv2.VideoCapture(str(source))
    step = max(int(video.get(cv2.CAP_PROP_FPS) * interval), 1)
    folder = OUT / "images"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    paths, count = [], 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        count += 1
        if count % step == 0:
            paths.append(folder / f"{len(paths):06d}.png")
            cv2.imwrite(str(paths[-1]), frame)
    return paths


def run_vggt(paths: list[Path], device: torch.device, pointmap: bool) -> dict[str, np.ndarray]:
    """VGGT on the frames: world points (S, H, W, 3), their confidence, colors, cameras."""
    try:
        from vggt.models.vggt import VGGT  # pyright: ignore[reportMissingImports]
        from vggt.utils.geometry import (  # pyright: ignore[reportMissingImports]
            unproject_depth_map_to_point_map,
        )
        from vggt.utils.load_fn import (  # pyright: ignore[reportMissingImports]
            load_and_preprocess_images,
        )
        from vggt.utils.pose_enc import (  # pyright: ignore[reportMissingImports]
            pose_encoding_to_extri_intri,
        )
    except ImportError as e:
        raise SystemExit(NEEDS_VGGT) from e
    print(f"VGGT on {len(paths)} frames (facebook/VGGT-1B, CC BY-NC 4.0) ...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    images = load_and_preprocess_images([str(p) for p in paths]).to(device)
    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    with torch.no_grad(), torch.autocast(device.type, dtype=amp, enabled=amp is not None):
        pred = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    out = {
        "depth": pred["depth"][0].float().cpu().numpy()[..., 0],  # (S, H, W)
        "extrinsic": extrinsic[0].float().cpu().numpy(),  # (S, 3, 4) world to camera
        "intrinsic": intrinsic[0].float().cpu().numpy(),
        "color": images.permute(0, 2, 3, 1).float().cpu().numpy(),  # (S, H, W, 3) in [0, 1]
    }
    if pointmap:
        out["points"] = pred["world_points"][0].float().cpu().numpy()
        out["conf"] = pred["world_points_conf"][0].float().cpu().numpy()
    else:
        out["points"] = unproject_depth_map_to_point_map(
            out["depth"][..., None], out["extrinsic"], out["intrinsic"]
        )
        out["conf"] = pred["depth_conf"][0].float().cpu().numpy()
    del model, pred
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def map_normals(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Normals of per-frame point maps (S, H, W, 3) from their image-space derivatives, turned
    toward the camera."""
    du = np.gradient(points, axis=2)
    dv = np.gradient(points, axis=1)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    r, t = extrinsic[:, :, :3], extrinsic[:, :, 3]
    centers = -np.einsum("sji,sj->si", r, t)  # camera centers in world coordinates
    away = np.sum(n * (points - centers[:, None, None]), axis=-1) > 0
    n[away] *= -1
    return n


def tsdf(vg: dict[str, np.ndarray], cell: float = 1 / 512) -> tuple[np.ndarray, ...]:
    """Fuse the depth maps in a TSDF (open3d's VoxelBlockGrid on grid cells of size ``cell``,
    truncated at 8 cells, as the originals) and return its surface points (the zero crossings,
    where a mesh's vertices would be) with their colors and normals."""
    try:
        import open3d as o3d  # pyright: ignore[reportMissingImports]
        import open3d.core as o3c  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise SystemExit("--tsdf needs open3d: uv sync --extra demo") from e

    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.uint16, o3c.uint16),
        attr_channels=(o3c.SizeVector([1]), o3c.SizeVector([1]), o3c.SizeVector([3])),
        voxel_size=cell,
        block_resolution=8,
        block_count=100_000,  # grows when needed
        device=o3c.Device("CPU:0"),
    )
    for depth, color, k, e in zip(
        vg["depth"], vg["color"], vg["intrinsic"], vg["extrinsic"], strict=True
    ):
        depth_mm = np.ascontiguousarray((depth * 1000).astype(np.uint16))
        rgb = np.ascontiguousarray((color * 255).astype(np.uint8))
        depth_image = o3d.t.geometry.Image(o3c.Tensor.from_numpy(depth_mm))
        rgb_image = o3d.t.geometry.Image(o3c.Tensor.from_numpy(rgb))
        k = o3c.Tensor.from_numpy(k.astype(np.float64))
        e = o3c.Tensor.from_numpy(np.vstack([e, [0, 0, 0, 1]]).astype(np.float64))
        blocks = grid.compute_unique_block_coordinates(depth_image, k, e, 1000.0, 10.0)
        grid.integrate(blocks, depth_image, rgb_image, k, k, e, 1000.0, 10.0)
    # surface points: extract_triangle_mesh segfaulted on large grids (open3d 0.19 and 0.20)
    surface = grid.extract_point_cloud(weight_threshold=1.0).to_legacy()
    return (
        np.asarray(surface.points),
        np.asarray(surface.colors),
        np.asarray(surface.normals),
    )


def ground_align(coord: np.ndarray, normal: np.ndarray, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Rotate the floor (a RANSAC plane among the lowest 20% of the points, within 40 degrees of
    horizontal) onto z = 0."""
    rng = np.random.default_rng(seed)
    low = coord[coord[:, 2] <= np.percentile(coord[:, 2], 20)]
    best, best_count = None, 0
    for _ in range(1000):
        a, b, c = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        n = n if n[2] >= 0 else -n
        if np.degrees(np.arccos(np.clip(n[2], -1, 1))) > 40:
            continue
        count = int((np.abs((low - a) @ n) < 0.01).sum())
        if count > best_count:
            best, best_count = (n, a), count
    if best is None:
        print("no ground plane found; the cloud stays as it is")
        return coord, normal
    n, a = best
    axis = np.cross(n, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(axis), n[2]
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + k + k @ k * ((1 - c) / s**2) if s > 1e-9 else np.eye(3)
    coord, normal = coord @ rot.T, normal @ rot.T
    inliers = np.abs((low - a) @ n) < 0.01
    coord[:, 2] -= np.mean(low[inliers] @ rot.T[:, 2])
    return coord, normal


def to_cloud(vg: dict[str, np.ndarray], conf_thres: float, fuse: bool) -> dict[str, np.ndarray]:
    """VGGT's output -> a z-up cloud 3 m high with colors (0-255) and normals."""
    coord = None
    if fuse:
        coord, color, normal = tsdf(vg)
        if not len(coord):
            print("the TSDF has no surface points, using the depth points")
            coord = None
    if coord is None:
        keep = vg["conf"] > 1e-5
        if conf_thres > 0:
            keep &= vg["conf"] >= np.percentile(vg["conf"], conf_thres)
        coord = vg["points"][keep]
        color = vg["color"][keep]
        normal = map_normals(vg["points"], vg["extrinsic"])[keep]
    # VGGT's world is the first camera's (x right, y down, z forward): turn it to z up
    to_z_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)
    coord, normal = coord @ to_z_up.T, normal @ to_z_up.T
    coord = (coord - coord.min(0)) * (3.0 / np.ptp(coord[:, 2]))  # 3 m high
    coord, normal = ground_align(coord, normal)
    return {
        "coord": coord.astype(np.float32),
        "color": np.asarray(np.clip(color, 0, 1) * 255, np.float32),
        "normal": normal.astype(np.float32),
    }


def lift(opt) -> dict[str, np.ndarray]:
    """Frames -> VGGT -> cloud, saved as demo/output/video/points.npz and pcd.ply."""
    if importlib.util.find_spec("vggt") is None:
        raise SystemExit(NEEDS_VGGT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opt.device != "auto":
        device = torch.device(opt.device)
    paths = frames(Path(opt.input), opt.frame_interval)
    if not paths:
        raise SystemExit(f"no frames in {opt.input}")
    out = to_cloud(run_vggt(paths, device, opt.pointmap), opt.conf_thres, opt.tsdf)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "points.npz", coord=out["coord"], color=out["color"], normal=out["normal"])
    save("video/pcd", out["coord"], out["color"] / 255)
    return out


if __name__ == "__main__":
    p = parser(__doc__, models=UTONIA + CONCERTO + SONATA, sample=None)
    p.add_argument("--input", required=True, help="a video, a folder of images or a points .npz")
    p.add_argument("--frame_interval", type=float, default=1.0, help="seconds between frames")
    p.add_argument(
        "--conf_thres",
        type=float,
        default=50.0,
        help="drop this percent of the points (confidence)",
    )
    p.add_argument("--pointmap", action="store_true", help="VGGT's point map instead of depth")
    p.add_argument("--tsdf", action="store_true", help="fuse the depth maps in a TSDF (open3d)")
    p.add_argument("--pca_start", type=int, default=1, help="first PCA component group")
    p.add_argument("--pca_brightness", type=float, default=1.2)
    opt = p.parse_args()
    if not Path(opt.input).exists():
        raise SystemExit(f"--input {opt.input} does not exist")
    if opt.input.endswith(".npz"):
        data = np.load(opt.input)
        cloud = {k: data[k].astype(np.float32) for k in ("coord", "color", "normal") if k in data}
    else:
        cloud = lift(opt)
    cloud = drop_inputs(cloud, opt)
    model = load_model(opt)
    scale = 1.0 if opt.model in UTONIA else None  # Utonia's video demo: scale 1 (finer)
    point, inverse = ptv3.prepare(
        cloud["coord"],
        cloud["color"],
        cloud["normal"],
        preset=model.preset,
        scale=scale,
        device=device_of(model),
    )
    feat = features(model, point)
    print(f"{opt.model}: {len(cloud['coord'])} points -> features {tuple(feat.shape)}")
    weights = (0.0,) * opt.pca_start + (1.0,)  # the originals' start group (0.6 + 0.4 of it)
    rgb = pca_color(feat, weights, brightness=opt.pca_brightness)
    rgb = rgb[inverse.cpu().numpy()]
    save(f"video/pca_{opt.model}", cloud["coord"], rgb, elevation=60)
    if opt.show:
        show([(cloud["coord"], rgb)])
