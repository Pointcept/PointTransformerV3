"""ScanNet-20 semantic segmentation of an indoor scan: a linear probe on an encoder's features
(utonia, concerto_large, sonata) or the PTv3 ScanNet model (ptv3_scannet).

    uv run demo/2_sem_seg.py                          # Utonia + its linear probe on sample1
    uv run demo/2_sem_seg.py --model ptv3_scannet     # also concerto_large, sonata
    uv run demo/2_sem_seg.py --wo_color --wo_normal   # coordinates only (Utonia)

Writes demo/output/sem_seg_<model>.ply and .png and prints the mIoU against the scan's labels;
--show opens a viewer.

Adapted from Utonia and Concerto demo/2_sem_seg.py (Apache-2.0, based on the Sonata demo,
Copyright (c) Meta Platforms, Inc. and affiliates; see THIRD_PARTY_LICENSES.md). Changed: the
probes are the presets' ``scannet20`` heads, results on the input points, mIoU.
"""

from __future__ import annotations

import numpy as np
import torch
from _common import drop_inputs, load_model, load_sample, parser, save, show

import pointtransformerv3 as ptv3

# ScanNet-20 colors (the ScanNet benchmark palette)
PALETTE = (
    np.array(
        [
            (174, 199, 232),
            (152, 223, 138),
            (31, 119, 180),
            (255, 187, 120),
            (188, 189, 34),
            (140, 86, 75),
            (255, 152, 150),
            (214, 39, 40),
            (197, 176, 213),
            (148, 103, 189),
            (196, 156, 148),
            (23, 190, 207),
            (247, 182, 210),
            (219, 219, 141),
            (255, 127, 14),
            (158, 218, 229),
            (44, 160, 44),
            (112, 128, 144),
            (227, 119, 194),
            (82, 84, 163),
        ]
    )
    / 255
)
WITH_HEAD = tuple(n for n, p in ptv3.PRESETS.items() if "scannet20" in p.heads)

if __name__ == "__main__":
    opt = parser(__doc__, models=tuple(ptv3.PRESETS)).parse_args()
    if opt.model not in WITH_HEAD:
        raise SystemExit(
            f"preset {opt.model!r} has no ScanNet-20 head (no released linear probe); "
            f"presets with one: {', '.join(WITH_HEAD)}"
        )
    model = load_model(opt, head="scannet20")
    scan = drop_inputs(load_sample(opt.sample), opt)
    logits = model.predict(scan["coord"], scan["color"], scan["normal"])
    pred = logits.argmax(1).cpu()
    names = ptv3.PRESETS[opt.model].heads["scannet20"].class_names
    counts = torch.bincount(pred, minlength=20)
    print(
        f"{opt.model}: top classes",
        ", ".join(f"{names[i]} {int(counts[i])}" for i in counts.argsort(descending=True)[:5]),
    )
    if "segment20" in scan:
        gt = torch.from_numpy(scan["segment20"])
        valid = gt >= 0
        conf = torch.bincount(gt[valid] * 20 + pred[valid], minlength=400).view(20, 20).double()
        union = conf.sum(0) + conf.sum(1) - conf.diag()
        seen = conf.sum(1) > 0  # the classes in the labels
        miou = 100 * (conf.diag()[seen] / union[seen]).mean()
        print(f"mIoU over the {int(seen.sum())} classes in the scan's labels: {miou:.2f}")
    rgb = PALETTE[pred.numpy()]
    save(f"sem_seg_{opt.model}", scan["coord"], rgb)
    if opt.show:
        show([(scan["coord"], rgb)])
