"""Several clouds in one forward: ``model.embed`` on a list packs them (one model call) and
returns one feature tensor per cloud; the explicit form uses ``prepare`` + ``collate``.

    uv run demo/3_batch.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from _common import args, load_sample

import pointtransformerv3 as ptv3

if __name__ == "__main__":
    opt = args(__doc__)
    model = ptv3.load("sonata", device=opt.device, precision=opt.precision)
    scan = load_sample(opt.sample)
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32)  # a rotated copy
    clouds = [scan["coord"], scan["coord"] @ rot.T]
    colors, normals = [scan["color"]] * 2, [scan["normal"], scan["normal"] @ rot.T]
    t0 = time.perf_counter()
    feats = model.embed(clouds, colors, normals)
    print("embed(list):", [tuple(f.shape) for f in feats], f"{time.perf_counter() - t0:.2f} s")

    device = next(model.parameters()).device
    items = [
        ptv3.prepare(c, k, n, preset="sonata", device=device)
        for c, k, n in zip(clouds, colors, normals, strict=True)
    ]
    batch = ptv3.collate([p for p, _ in items])
    with torch.inference_mode():
        out = ptv3.upcast(model(batch))
    print("explicit: offsets", out.offset.tolist(), "features", tuple(out.feat.shape))
