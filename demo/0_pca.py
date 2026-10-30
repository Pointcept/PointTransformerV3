"""Sonata features of a scan, colored by their first three principal components.

uv run demo/0_pca.py            # writes demo/output/pca.ply (add --show for a viewer)
"""

from __future__ import annotations

import time

import torch
from _common import OUTPUT, args, load_sample, pca_color, show, write_ply

import pointtransformerv3 as ptv3

if __name__ == "__main__":
    opt = args(__doc__)
    model = ptv3.load("sonata", device=opt.device, precision=opt.precision)
    scan = load_sample(opt.sample)
    t0 = time.perf_counter()
    point, inverse = ptv3.prepare(
        scan["coord"],
        scan["color"],
        scan["normal"],
        preset="sonata",
        device=next(model.parameters()).device,
    )
    with torch.inference_mode():
        # levels=2: concatenate the three coarsest levels, then copy down (Sonata's PCA demo)
        feat = ptv3.upcast(model(point), levels=2).feat[inverse]
    print(
        f"{len(scan['coord'])} points -> {len(point.feat)} grid-sampled points -> features "
        f"{tuple(feat.shape)} in {time.perf_counter() - t0:.2f} s"
    )
    rgb = pca_color(feat)
    print("wrote", write_ply(OUTPUT / "pca.ply", scan["coord"], rgb))
    if opt.show:
        show(scan["coord"], rgb)
