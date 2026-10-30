"""PCA of Utonia's features on a robot manipulation scene (a tabletop captured by RGB-D
cameras; the sample has no normals): coordinates scaled by 4, so a grid cell is 2.5 mm.

    uv run demo/5_pca_manipulation.py             # utonia; also utonia_small, utonia_tiny
    uv run demo/5_pca_manipulation.py --wo_color

Writes demo/output/pca_manipulation_<model>.ply and .png; --show opens a viewer.

Adapted from Utonia demo/5_pca_manipulation.py (Apache-2.0, based on the Sonata demo,
Copyright (c) Meta Platforms, Inc. and affiliates; see THIRD_PARTY_LICENSES.md). Changed:
ptv3.prepare / upcast instead of the original transforms, an exact PCA (_common.pca_color),
headless output.
"""

from __future__ import annotations

from _common import (
    UTONIA,
    drop_inputs,
    features,
    load_model,
    load_sample,
    parser,
    pca_color,
    prepare,
    save,
    show,
)

if __name__ == "__main__":
    p = parser(__doc__, models=UTONIA, sample="sample4_manipulation")
    p.add_argument("--scale", type=float, default=4.0, help="coordinate scale (default 4)")
    opt = p.parse_args()
    model = load_model(opt)
    scan = drop_inputs(load_sample(opt.sample), opt)
    point, inverse = prepare(model, scan, scale=opt.scale)
    feat = features(model, point)
    print(f"{opt.model}: {len(scan['coord'])} points -> features {tuple(feat.shape)}")
    signs = (-1, -1, -1, -1, 1, 1) if opt.model == "utonia" else None  # the original's colors
    rgb = pca_color(feat, signs=signs)[inverse.cpu().numpy()]
    save(f"pca_manipulation_{opt.model}", scan["coord"], rgb, elevation=45, azimuth=180)
    if opt.show:
        show([(scan["coord"], rgb)])
