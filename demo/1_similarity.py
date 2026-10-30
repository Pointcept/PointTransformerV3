"""Feature similarity: how similar every point is to one query point (cosine of Sonata
features, red = similar).

    uv run demo/1_similarity.py [--query 12345]
"""

from __future__ import annotations

import numpy as np
import torch.nn.functional as F
from _common import OUTPUT, args, load_sample, show, write_ply

import pointtransformerv3 as ptv3

if __name__ == "__main__":
    opt = args(__doc__)
    model = ptv3.load("sonata", device=opt.device, precision=opt.precision)
    scan = load_sample(opt.sample)
    feat = model.embed(scan["coord"], scan["color"], scan["normal"])  # (N, 1232)
    query = int(np.argmin(np.linalg.norm(scan["coord"] - scan["coord"].mean(0), axis=1)))
    sim = F.cosine_similarity(feat, feat[query : query + 1], dim=1)
    w = ((sim - sim.min()) / (sim.max() - sim.min())).cpu().numpy()[:, None]
    rgb = w * np.array([1.0, 0.1, 0.1]) + (1 - w) * np.array([0.15, 0.25, 0.9])
    print(f"query point {query}: {int((sim > 0.8).sum())} points with cosine > 0.8")
    print("wrote", write_ply(OUTPUT / "similarity.ply", scan["coord"], rgb))
    if opt.show:
        show(scan["coord"], rgb)
