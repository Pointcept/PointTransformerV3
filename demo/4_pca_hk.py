"""PCA of Utonia's features on a city-scale aerial point cloud of Hong Kong (remote sensing,
40M points, a 1.5 GB download): coordinates scaled by 0.0015, so a grid cell is about 6.7 m.

    uv run demo/4_pca_hk.py                       # utonia; also --model utonia_small, utonia_tiny
    uv run demo/4_pca_hk.py --scale 0.003         # finer (more grid cells, more memory)

Writes demo/output/pca_hk_<model>.ply and .png at the grid-sampled points; --show opens a viewer.

Adapted from Utonia demo/4_pca_hk.py (Apache-2.0, based on the Sonata demo, Copyright (c) Meta
Platforms, Inc. and affiliates; see THIRD_PARTY_LICENSES.md). Changed: ptv3.prepare / upcast
instead of the original transforms, an exact PCA (_common.pca_color), headless output.
"""

from __future__ import annotations

import time

from _common import (
    UTONIA,
    drop_inputs,
    features,
    first_rows,
    load_model,
    load_sample,
    parser,
    pca_color,
    prepare,
    save,
    show,
)

if __name__ == "__main__":
    p = parser(__doc__, models=UTONIA, sample="sample5_hk")
    p.add_argument("--scale", type=float, default=0.0015, help="coordinate scale (default 0.0015)")
    opt = p.parse_args()
    model = load_model(opt)
    scan = drop_inputs(load_sample(opt.sample), opt)
    t0 = time.perf_counter()
    point, inverse = prepare(model, scan, scale=opt.scale)
    feat = features(model, point)
    print(
        f"{opt.model}: {len(scan['coord'])} points -> {len(feat)} grid-sampled -> features "
        f"{tuple(feat.shape)} in {time.perf_counter() - t0:.2f} s"
    )
    # the original's channel order; signs that reproduce its colors with utonia
    signs = (1, 1, -1, -1, 1, 1) if opt.model == "utonia" else None
    rgb = pca_color(feat, signs=signs, channels=(0, 2, 1))
    xyz = scan["coord"][first_rows(inverse, len(feat)).cpu().numpy()]
    save(f"pca_hk_{opt.model}", xyz, rgb, point_size=1)
    if opt.show:
        show([(xyz, rgb)])
