"""PCA of Utonia's features on a single object (a potted plant, 16k points without normals) and
on a copy with the y and z axes swapped, in one batch: the parts keep their colors when the
object lies on its side.

    uv run demo/6_pca_object.py                   # utonia; also utonia_small, utonia_tiny
    uv run demo/6_pca_object.py --normalize       # your own object: unit sphere first
    uv run demo/6_pca_object.py --sample my_object.npz --normalize

Normals come from open3d's estimate_normals when open3d is installed, else from the same plane
fit in torch. Writes demo/output/pca_object_<model>.ply and .png; --show opens a viewer.

Adapted from Utonia demo/6_pca_object.py (Apache-2.0, based on the Sonata demo, Copyright (c)
Meta Platforms, Inc. and affiliates; see THIRD_PARTY_LICENSES.md). Changed: ptv3.prepare /
collate / upcast instead of the original transforms; the swapped copy's normals are swapped too
(the original reused the first copy's normals); an exact PCA (_common.pca_color); headless
output.
"""

from __future__ import annotations

import numpy as np
from _common import (
    UTONIA,
    device_of,
    drop_inputs,
    estimate_normals,
    features,
    load_model,
    load_sample,
    normalize_coord,
    parser,
    pca_color,
    save,
    show,
)

import pointtransformerv3 as ptv3

if __name__ == "__main__":
    p = parser(__doc__, models=UTONIA, sample="sample3_object")
    p.add_argument("--scale", type=float, default=1.0, help="coordinate scale (default 1)")
    p.add_argument(
        "--normalize", action="store_true", help="center the object and scale it to the unit sphere"
    )
    opt = p.parse_args()
    model = load_model(opt)
    scan = load_sample(opt.sample)
    if opt.normalize:
        scan["coord"] = normalize_coord(scan["coord"]).astype(np.float32)
    if "normal" not in scan:
        scan["normal"] = estimate_normals(scan["coord"])
    swap = [0, 2, 1]
    copies = [scan, {k: v[:, swap] if k in ("coord", "normal") else v for k, v in scan.items()}]
    copies = [drop_inputs(c, opt) for c in copies]
    items = [
        ptv3.prepare(
            c["coord"],
            c["color"],
            c["normal"],
            preset=model.preset,
            scale=opt.scale,
            device=device_of(model),
        )
        for c in copies
    ]
    batch = ptv3.collate([point for point, _ in items])
    feat = features(model, batch)
    print(f"{opt.model}: {[len(c['coord']) for c in copies]} points -> {tuple(feat.shape)}")
    signs = (-1, 1, 1, 1, -1, 1) if opt.model == "utonia" else None  # the original's colors
    rgb = pca_color(feat, signs=signs)
    ends = batch.offset.tolist()
    xyz, colors = [], []
    for i, ((_, inverse), start, end) in enumerate(zip(items, [0, *ends[:-1]], ends, strict=True)):
        c = copies[i]["coord"]
        xyz.append(c - c.mean(0) + np.array([0, 1.3 * i * float(np.ptp(c)), 0], np.float32))
        colors.append(rgb[start:end][inverse.cpu().numpy()])
    save(
        f"pca_object_{opt.model}",
        np.concatenate(xyz),
        np.concatenate(colors),
        elevation=10,
        azimuth=90,
    )
    if opt.show:
        show(list(zip(xyz, colors, strict=True)))
