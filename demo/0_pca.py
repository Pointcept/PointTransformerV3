"""PCA of an encoder's features on an indoor scan: the first principal components as colors.

    uv run demo/0_pca.py                          # Utonia on the ScanNet scan sample1
    uv run demo/0_pca.py --model concerto_large   # any encoder preset: sonata, concerto, ...
    uv run demo/0_pca.py --wo_color --wo_normal   # coordinates only (Concerto, Utonia)

Writes demo/output/pca_<model>.ply and .png; --show opens a viewer.

Adapted from Utonia demo/0_pca_indoor.py and Concerto demo/0_pca.py (Apache-2.0, based on the
Sonata demo, Copyright (c) Meta Platforms, Inc. and affiliates; see THIRD_PARTY_LICENSES.md).
Changed: one script for every encoder preset, ptv3.prepare / upcast instead of the original
transforms, an exact PCA (_common.pca_color) instead of the seeded low-rank one, headless output.
"""

from __future__ import annotations

import time

from _common import (
    INDOOR_SIGNS,
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
    opt = parser(__doc__).parse_args()
    model = load_model(opt)
    scan = drop_inputs(load_sample(opt.sample), opt)
    t0 = time.perf_counter()
    point, inverse = prepare(model, scan)
    feat = features(model, point)  # three coarsest levels, copied down to the grid rows
    print(
        f"{opt.model}: {len(scan['coord'])} points -> {len(point.feat)} grid-sampled -> "
        f"features {tuple(feat.shape)} in {time.perf_counter() - t0:.2f} s"
    )
    rgb = pca_color(feat, signs=INDOOR_SIGNS.get(opt.model))[inverse.cpu().numpy()]
    save(f"pca_{opt.model}", scan["coord"], rgb)
    if opt.show:
        show([(scan["coord"], rgb)])
