"""point_transformer_v3.prepare / collate: centering, grid-cell round trip, features per preset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import point_transformer_v3 as ptv3


def raw(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    coord = (rng.random((n, 3)) * [4.0, 3.0, 2.5] + [10.0, -3.0, 0.7]).astype(np.float32)
    color = rng.integers(0, 256, (n, 3)).astype(np.float32)
    normal = rng.normal(size=(n, 3)).astype(np.float32)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return coord, color, normal


def test_round_trip_every_point_maps_to_its_grid_cell():
    coord, color, normal = raw()
    point, inverse = ptv3.prepare(coord, color, normal, preset="sonata")
    low, high = coord.min(0), coord.max(0)
    shift = np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]], np.float32)
    centered = coord - shift
    cells = np.floor(centered / np.float32(0.02)).astype(np.int64)
    cells -= cells.min(0)
    assert inverse.shape == (len(coord),) and inverse.dtype == torch.int64
    assert np.array_equal(point.grid_coord[inverse].numpy(), cells)
    assert len(point.grid_coord) == len(np.unique(cells, axis=0))
    # the picked point of each cell is one of its own points, centered
    assert np.array_equal(
        np.floor(point.coord.numpy() / np.float32(0.02)).astype(np.int64)
        - np.floor(centered / np.float32(0.02)).astype(np.int64).min(0),
        point.grid_coord.numpy(),
    )
    assert (
        point.coord[:, 2].min() >= 0
        and abs(point.coord[:, 0].min() + point.coord[:, 0].max()) < 0.05
    )


def test_features_per_preset():
    coord, color, normal = raw(500)
    p, inv = ptv3.prepare(coord, color, normal, preset="sonata", pick="first")
    assert p.feat.shape[1] == 9
    torch.testing.assert_close(p.feat[:, :3], p.coord)
    first = torch.zeros(len(p.feat), dtype=torch.long).scatter_reduce(
        0, inv, torch.arange(len(coord)), "amin", include_self=False
    )
    torch.testing.assert_close(p.feat[:, 3:6], torch.from_numpy(color)[first] / 255)
    torch.testing.assert_close(p.feat[:, 6:], torch.from_numpy(normal)[first])
    q, _ = ptv3.prepare(coord, color, normal, preset="ptv3_scannet")
    assert q.feat.shape[1] == 6
    torch.testing.assert_close(q.feat[:, :3], torch.from_numpy(color)[first] / 127.5 - 1)
    with pytest.raises(ValueError, match="pass normal="):
        ptv3.prepare(coord, color, preset="sonata")
    r, _ = ptv3.prepare(coord, feat=("coord",), grid_size=0.05)
    assert r.feat.shape[1] == 3 and r.grid_size == 0.05


def test_random_pick_is_reproducible_and_float64_lattice():
    coord, color, normal = raw(3000)
    a, _ = ptv3.prepare(
        coord,
        color,
        normal,
        preset="sonata",
        pick="random",
        generator=torch.Generator().manual_seed(1),
    )
    b, _ = ptv3.prepare(
        coord,
        color,
        normal,
        preset="sonata",
        pick="random",
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(a.feat, b.feat)
    c, _ = ptv3.prepare(coord, color, normal, preset="sonata", lattice="float64", pick="center")
    assert abs(len(c.feat) - len(a.feat)) <= 3
    with pytest.raises(ValueError, match="lattice"):
        ptv3.prepare(coord, color, normal, preset="sonata", pick="center")


def test_collate_offsets():
    clouds = [ptv3.prepare(*raw(n, seed=n), preset="sonata")[0] for n in (300, 500)]
    batch = ptv3.collate(clouds)
    sizes = [len(c.feat) for c in clouds]
    assert batch.offset.tolist() == [sizes[0], sizes[0] + sizes[1]]
    assert batch.feat.shape == (sum(sizes), 9) and batch.grid_size == 0.02
    with pytest.raises(ValueError, match="grid sizes"):
        ptv3.collate([clouds[0], ptv3.prepare(*raw(100), preset="sonata", grid_size=0.05)[0]])


def test_scale_multiplies_in_float64_like_numpy_then_centers():
    """Utonia's fixed RandomScale: numpy multiplies float32 coordinates by a float64 array and
    rounds back to float32; prepare(scale=...) gives the same coordinates and grid cells."""
    coord, color, normal = raw()
    scaled = coord * np.array([0.2])  # float64, as RandomScale(scale=[0.2, 0.2])
    scaled = scaled.astype(np.float32)
    point, inverse = ptv3.prepare(coord, color, normal, preset="utonia", scale=0.2)
    ref, ref_inverse = ptv3.prepare(scaled, color, normal, preset="utonia", scale=1.0)
    assert torch.equal(point.coord, ref.coord) and torch.equal(point.grid_coord, ref.grid_coord)
    assert torch.equal(inverse, ref_inverse) and torch.equal(point.feat, ref.feat)
    default, _ = ptv3.prepare(coord, color, normal, preset="utonia")  # 0.5 at a 1 cm grid
    assert point.grid_size == default.grid_size == 0.01
    assert float(default.coord[:, 0].max() - default.coord[:, 0].min()) < 4.0 * 0.5 + 1e-3
