"""Write tests/golden/presets.pt: each preset's randomly initialized model (seed 0) on a tiny
seeded cloud, CPU fp32, projected to 8 channels. Regenerate only for an intended behavior
change, and say why in the commit:  .venv/bin/python -m tests.golden.make_golden"""

from __future__ import annotations

from pathlib import Path

import torch

import point_transformer_v3 as ptv3

PATH = Path(__file__).with_name("presets.pt")
PRESETS = ("ptv3", "ptv3_scannet", "sonata", "sonata_small", "utonia")


def golden_cloud() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    coord = torch.rand(1500, 3, generator=gen) * torch.tensor([2.0, 1.5, 1.0])
    color = torch.rand(1500, 3, generator=gen) * 255
    normal = torch.nn.functional.normalize(torch.randn(1500, 3, generator=gen), dim=1)
    return coord, color, normal


def outputs() -> dict[str, torch.Tensor]:
    ptv3.set_backend(attention="sdpa", conv="torch", layer_norm="torch")
    out = {}
    for name in PRESETS:
        torch.manual_seed(0)
        model = ptv3.PointTransformerV3.from_preset(name).eval()
        model.precision = "fp32"
        feat = model.embed(*golden_cloud())
        proj = torch.randn(feat.shape[1], 8, generator=torch.Generator().manual_seed(1))
        out[name] = feat @ proj
    ptv3.set_backend(attention="auto", conv="auto", layer_norm="auto")
    return out


if __name__ == "__main__":
    torch.save(outputs(), PATH)
    print("wrote", PATH)
