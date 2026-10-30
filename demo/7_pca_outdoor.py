"""PCA of features on outdoor LiDAR (a driving scene with the ego vehicle at the origin and the
road on the xy-plane): Utonia on an aggregated multi-frame sweep, or Concerto's outdoor
checkpoint on a single sweep. Coordinates are scaled by 0.2 and not centered.

    uv run demo/7_pca_outdoor.py                                  # utonia, utonia_small, utonia_tiny
    uv run demo/7_pca_outdoor.py --model concerto_large_outdoor   # Concerto's outdoor weights
    uv run demo/7_pca_outdoor.py --model concerto_large_outdoor --wo_color

Writes demo/output/pca_outdoor_<model>.ply and .png; --show opens a viewer.

concerto_large_outdoor is not a preset (yet): the demo loads its checkpoint from
Pointcept/Concerto (CC BY-NC 4.0) into the concerto_large architecture and applies Concerto's
outdoor transform (1 cm grid after scaling, x and y centered after grid sampling).

Adapted from Utonia demo/7_pca_outdoor.py and Concerto demo/0_pca_outdoor.py (Apache-2.0, based
on the Sonata demo, Copyright (c) Meta Platforms, Inc. and affiliates; see
THIRD_PARTY_LICENSES.md). Changed: one script for both, ptv3.prepare / upcast instead of the
original transforms, an exact PCA (_common.pca_color), headless output.
"""

from __future__ import annotations

import hashlib
import time

import torch
from _common import (
    UTONIA,
    drop_inputs,
    features,
    load_sample,
    parser,
    pca_color,
    prepare,
    save,
    show,
)

import pointtransformerv3 as ptv3

CONCERTO_OUTDOOR = ("Pointcept/Concerto", "concerto_large_outdoor.pth")
CONCERTO_REVISION = "efc130718a564eba0ce4899318f3c20b19d9c752"  # as the concerto presets
CONCERTO_SHA256 = "6ab48e07944d77cef9c550087e87824e01c3c8caa428bfac0102fd53e2b064b7"


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    p = parser(__doc__, models=(*UTONIA, "concerto_large_outdoor"), sample=None)
    p.add_argument(
        "--sample",
        help="default: sample2_outdoor_multiframe (Utonia), sample2_outdoor (Concerto)",
    )
    opt = p.parse_args()
    concerto = opt.model == "concerto_large_outdoor"
    sample = opt.sample or ("sample2_outdoor" if concerto else "sample2_outdoor_multiframe")
    if concerto:
        from huggingface_hub import hf_hub_download

        repo, file = CONCERTO_OUTDOOR
        path = hf_hub_download(repo, file, revision=CONCERTO_REVISION)
        if sha256_of(path) != CONCERTO_SHA256:
            raise SystemExit(f"{path} does not match the pinned sha256 {CONCERTO_SHA256}")
        print(f"{repo}/{file} is licensed CC-BY-NC-4.0 + research permission (NOTICE)")
        model = ptv3.load(
            "concerto_large", weights=path, device=opt.device, precision=opt.precision
        )
    else:
        model = ptv3.load(opt.model, device=opt.device, precision=opt.precision)
    scan = drop_inputs(load_sample(sample), opt)
    t0 = time.perf_counter()
    if concerto:
        point, inverse = prepare(model, scan, scale=0.2, grid_size=0.01, center=False)
        # Concerto's outdoor transform centers x and y after grid sampling (z stays)
        low, high = point.coord.amin(0), point.coord.amax(0)
        shift = torch.stack([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2] * 0])
        point["coord"] = point.coord - shift
        point.feat[:, :3] = point.coord  # the first feature columns are the coordinates
        weights = (0.2, 0.2, 0.1, 0.5)
        signs = (1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1)  # the original's colors
    else:
        point, inverse = prepare(model, scan, scale=0.2, center=False)
        weights = (0.4, 0.2, 0.0, 0.4)
        signs = (1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1) if opt.model == "utonia" else None
    feat = features(model, point)
    print(
        f"{opt.model}: {len(scan['coord'])} points -> {len(feat)} grid-sampled -> features "
        f"{tuple(feat.shape)} in {time.perf_counter() - t0:.2f} s"
    )
    rgb = pca_color(feat, weights, brightness=1.0, signs=signs)[inverse.cpu().numpy()]
    save(f"pca_outdoor_{opt.model}", scan["coord"], rgb, elevation=50, point_size=1)
    if opt.show:
        show([(scan["coord"], rgb)])
