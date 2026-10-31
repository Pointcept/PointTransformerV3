"""Model inputs of the PTv3 parity cases (P1, P2), made once with ``point_transformer_v3.prepare``
(preset ``ptv3_scannet``, pick "first") so both implementations read identical tensors.

usage (this package's env, CPU; needs PTV3_PARITY_RAW): python -m tests.parity.make_inputs
"""

from __future__ import annotations

import numpy as np
import torch

import point_transformer_v3 as ptv3
from tests.parity.common import INPUTS, SCENES, env_path


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    raw_dir = env_path("PTV3_PARITY_RAW")
    cases = {}
    for name in SCENES:
        raw = np.load(raw_dir / f"{name}.npz")
        point, inverse = ptv3.prepare(
            raw["coord"], raw["color"], raw["normal"], preset="ptv3_scannet", pick="first"
        )
        cases[name] = dict(point, inverse=inverse, segment_full=torch.from_numpy(raw["segment20"]))
    gen = torch.Generator().manual_seed(7)  # a tiny synthetic cloud (several patches per level)
    coord = torch.rand(3000, 3, generator=gen) * torch.tensor([1.5, 1.0, 0.6])
    point, inverse = ptv3.prepare(
        coord,
        torch.rand(3000, 3, generator=gen) * 255,
        torch.nn.functional.normalize(torch.randn(3000, 3, generator=gen), dim=1),
        preset="ptv3_scannet",
    )
    cases["tiny"] = dict(point, inverse=inverse)
    parts = [cases[n] for n in SCENES]
    batch = ptv3.collate([ptv3.Point(p) for p in parts])
    cases["batch3"] = dict(batch, parts=torch.tensor([p["feat"].shape[0] for p in parts]))
    for name, case in cases.items():
        case = {k: v for k, v in case.items() if isinstance(v, torch.Tensor)}
        torch.save(case, INPUTS / f"ptv3_{name}.pt")
        print(name, {k: tuple(v.shape) for k, v in case.items()})


if __name__ == "__main__":
    main()
