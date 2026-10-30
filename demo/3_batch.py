"""Several clouds in one forward: ``model.embed`` on lists packs them into one batch and returns
one feature tensor per cloud; the explicit form is ``prepare`` + ``collate`` + ``upcast``. The
PCA colors are computed over the batch, so the copies share one color scale.

    uv run demo/3_batch.py                   # two copies of sample1 with Utonia
    uv run demo/3_batch.py --rotate          # the second copy turned by 90 degrees
    uv run demo/3_batch.py --model sonata    # any encoder preset

Writes demo/output/batch_<model>.ply and .png; --show opens a viewer.

Adapted from Utonia and Concerto demo/3_batch_forward.py (Apache-2.0, based on the Sonata demo,
Copyright (c) Meta Platforms, Inc. and affiliates; see THIRD_PARTY_LICENSES.md). Changed: the
package's embed / prepare / collate, an optional rotated copy, an exact PCA (_common.pca_color).
"""

from __future__ import annotations

import time

import numpy as np
from _common import (
    INDOOR_SIGNS,
    device_of,
    drop_inputs,
    features,
    load_model,
    load_sample,
    parser,
    pca_color,
    save,
    show,
)

import pointtransformerv3 as ptv3

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--rotate", action="store_true", help="turn the second copy by 90 degrees")
    opt = p.parse_args()
    model = load_model(opt)
    scan = drop_inputs(load_sample(opt.sample), opt)
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32) if opt.rotate else np.eye(3)
    rot = rot.astype(np.float32)
    clouds = [scan["coord"], scan["coord"] @ rot.T]
    colors, normals = [scan["color"]] * 2, [scan["normal"], scan["normal"] @ rot.T]

    t0 = time.perf_counter()
    feats = model.embed(clouds, colors, normals)  # one forward, a list of (N, C) tensors
    print("embed(list):", [tuple(f.shape) for f in feats], f"{time.perf_counter() - t0:.2f} s")

    items = [
        ptv3.prepare(c, k, n, preset=model.preset, device=device_of(model))
        for c, k, n in zip(clouds, colors, normals, strict=True)
    ]
    batch = ptv3.collate([point for point, _ in items])
    feat = features(model, batch)  # (rows of both clouds, C); offset holds the sample ends
    print("prepare + collate: offsets", batch.offset.tolist(), "features", tuple(feat.shape))
    rgb = pca_color(feat, signs=INDOOR_SIGNS.get(opt.model))  # the colors of 0_pca
    ends = batch.offset.tolist()
    out_xyz, out_rgb, dx = [], [], 0.0
    for (_, inverse), start, end, xyz in zip(items, [0, *ends[:-1]], ends, clouds, strict=True):
        out_xyz.append(xyz - xyz.min(0) + np.array([dx, 0, 0], np.float32))
        out_rgb.append(rgb[start:end][inverse.cpu().numpy()])
        dx += float(np.ptp(xyz[:, 0])) + 1.0  # the next copy 1 m to the right
    save(f"batch_{opt.model}", np.concatenate(out_xyz), np.concatenate(out_rgb))
    if opt.show:
        show(list(zip(out_xyz, out_rgb, strict=True)))
