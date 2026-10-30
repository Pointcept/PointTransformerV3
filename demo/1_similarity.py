"""Feature similarity between a local view and the whole scene: a query point on the arm of a
sofa in a crop of the scan, its cosine similarity to every point of the crop and of the full
scene (red = similar), and a line to the best match in the scene.

    uv run demo/1_similarity.py                          # Utonia on sample1
    uv run demo/1_similarity.py --model concerto_large   # any encoder preset
    uv run demo/1_similarity.py --query 1.19 2.30 0.64   # another query point (meters)

Writes demo/output/similarity_<model>.ply and .png (the crop is drawn next to the scene); --show
opens a viewer.

Adapted from Utonia and Concerto demo/1_similarity.py (Apache-2.0, based on the Sonata demo,
Copyright (c) Meta Platforms, Inc. and affiliates; see THIRD_PARTY_LICENSES.md). Changed: one
script for every encoder preset, the query given as a coordinate, results on the input points.
"""

from __future__ import annotations

import numpy as np
import torch.nn.functional as F
from _common import (
    drop_inputs,
    features,
    first_rows,
    heat_color,
    load_model,
    load_sample,
    parser,
    prepare,
    save,
    show,
)

# sample1: the crop around a sofa by the window and the sofa arm (the originals' local point 8706)
CROP_CENTER, QUERY = (0.48, 2.12, 0.67), (1.1914, 2.2988, 0.6401)
OFFSET = np.array([-3.5, 1.0, 0.0], np.float32)  # where the crop is drawn, next to the scene


def shift_of(coord: np.ndarray) -> np.ndarray:
    """The centering ``prepare`` applies: x, y to the box center, z to the floor."""
    low, high = coord.min(0), coord.max(0)
    return np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]], np.float32)


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--crop", type=float, nargs=3, help="center of the local view (meters)")
    p.add_argument("--query", type=float, nargs=3, help="query point (meters, in the local view)")
    p.add_argument("--size", type=int, default=16384, help="points in the local view")
    opt = p.parse_args()
    model = load_model(opt)
    scan = drop_inputs(load_sample(opt.sample), opt)
    coord = scan["coord"]
    default = opt.sample == "sample1"
    crop = np.array(opt.crop or (CROP_CENTER if default else coord.mean(0)))
    near = np.argsort(np.sum(np.square(coord - crop), axis=-1))[: opt.size]
    local = {k: v[near] for k, v in scan.items()}
    query = np.array(opt.query or (QUERY if default else crop))
    q_point = int(np.argmin(np.linalg.norm(local["coord"] - query, axis=1)))

    g_point, g_inverse = prepare(model, scan)
    l_point, l_inverse = prepare(model, local)
    g = F.normalize(features(model, g_point), dim=-1)
    lo = F.normalize(features(model, l_point), dim=-1)
    q = int(l_inverse[q_point])
    sim_local, sim_scene = lo @ lo[q], g @ lo[q]
    match = int(sim_scene.argmax())
    local_rgb = heat_color(sim_local, sim_scene)[l_inverse.cpu().numpy()]
    scene_rgb = heat_color(sim_scene, sim_scene)[g_inverse.cpu().numpy()]

    scene_xyz = coord - shift_of(coord)
    local_xyz = local["coord"] - shift_of(local["coord"]) + OFFSET
    g_rows = first_rows(g_inverse, len(g)).cpu().numpy()
    a, b = local_xyz[q_point], scene_xyz[g_rows[match]]
    q_xyz = local["coord"][q_point].astype(float).round(3).tolist()
    m_xyz = coord[g_rows[match]].astype(float).round(3).tolist()
    print(
        f"{opt.model}: query {q_xyz} -> best match in the scene {m_xyz} (cosine "
        f"{float(sim_scene[match]):.3f}); {int((sim_scene > 0.8).sum())} of {len(g)} rows above 0.8"
    )
    line = a + np.linspace(0, 1, 400, dtype=np.float32)[:, None] * (b - a)
    save(
        f"similarity_{opt.model}",
        np.concatenate([scene_xyz, local_xyz, line]),
        np.concatenate([scene_rgb, local_rgb, np.zeros_like(line)]),
    )
    if opt.show:
        show([(scene_xyz, scene_rgb), (local_xyz, local_rgb)], lines=[(a, b)])
