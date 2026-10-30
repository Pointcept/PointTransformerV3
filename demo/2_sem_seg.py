"""ScanNet-20 semantic segmentation: the Sonata linear probe (default) or the PTv3 ScanNet model.

uv run demo/2_sem_seg.py [--model ptv3_scannet]
"""

from __future__ import annotations

import numpy as np
import torch
from _common import OUTPUT, args, load_sample, show, write_ply

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

if __name__ == "__main__":
    import sys

    name = "ptv3_scannet" if "--model" in sys.argv and "ptv3_scannet" in sys.argv else "sonata"
    sys.argv = [a for a in sys.argv if a not in ("--model", "ptv3_scannet", "sonata")]
    opt = args(__doc__)
    model = ptv3.load(name, head="scannet20", device=opt.device, precision=opt.precision)
    scan = load_sample(opt.sample)
    with torch.inference_mode():
        pred = model.predict(scan["coord"], scan["color"], scan["normal"]).argmax(1).cpu()
    names = ptv3.PRESETS[name].heads["scannet20"].class_names
    counts = torch.bincount(pred, minlength=20)
    print(
        "top classes:",
        ", ".join(f"{names[i]} {int(counts[i])}" for i in counts.argsort(descending=True)[:5]),
    )
    if "segment20" in scan:
        gt = torch.from_numpy(scan["segment20"])
        valid = gt >= 0
        conf = torch.bincount(gt[valid] * 20 + pred[valid], minlength=400).view(20, 20).double()
        union = conf.sum(0) + conf.sum(1) - conf.diag()
        seen = union > 0
        print(
            f"mIoU against the scan's labels: {100 * (conf.diag()[seen] / union[seen]).mean():.2f}"
        )
    rgb = PALETTE[pred.numpy()]
    print("wrote", write_ply(OUTPUT / f"sem_seg_{name}.ply", scan["coord"], rgb))
    if opt.show:
        show(scan["coord"], rgb)
