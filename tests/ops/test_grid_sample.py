# Adapted from Pointcept2 @ 771a949 (now tests/ops/test_grid_sample.py, renamed in 36bac04).
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from point_transformer_v3.ops.grid_sample import (
    GridSampled,
    _segment_bounds,
    grid_sample,
    lattice,
    unique_grid,
)
from tests.conftest import DEVICES

GRID = 0.02


# ---------------------------------------------------------------- numpy reference
def reference(coord: np.ndarray, grid_size: float, seq_lens: tuple[int, ...]) -> dict:
    """Brute-force grid sampling with numpy, cells ordered by (sequence, x, y, z)."""
    c64 = coord.astype(np.float64)
    grid = np.floor(c64 / grid_size).astype(np.int64)
    diff = c64 - (grid + 0.5) * grid_size
    diff = diff * diff
    dist = diff[:, 0] + diff[:, 1] + diff[:, 2]
    big = np.iinfo(np.int64).max
    parts = {"grid": [np.zeros((0, 3), np.int64)], "first": [], "center": [], "inverse": []}
    origins, lens = [], []
    start = 0
    for n in seq_lens:
        rows = np.arange(start, start + n)
        start += n
        origin = grid[rows].min(0) if n else np.zeros(3, np.int64)
        cells, inv = np.unique(grid[rows] - origin, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        first = np.full(len(cells), big)
        np.minimum.at(first, inv, rows)
        best = np.full(len(cells), np.inf)
        np.minimum.at(best, inv, dist[rows])
        tie = dist[rows] == best[inv]
        center = np.full(len(cells), big)
        np.minimum.at(center, inv[tie], rows[tie])
        parts["inverse"].append(inv + sum(lens))
        parts["grid"].append(cells)
        parts["first"].append(first)
        parts["center"].append(center)
        origins.append(origin)
        lens.append(len(cells))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    return {**out, "origin": np.stack(origins), "lens": tuple(lens)}


def random_cloud(seq_lens: tuple[int, ...], seed: int = 0, extent: float = 0.3) -> torch.Tensor:
    """Float32 points with many shared cells, negative coordinates and per-sequence offsets."""
    g = torch.Generator().manual_seed(seed)
    parts = []
    for i, n in enumerate(seq_lens):
        offset = torch.tensor([7.3 * i - 5.0, -3.1 * i, 0.9 * i])
        parts.append(torch.rand(n, 3, generator=g) * extent + offset)
    return torch.cat(parts) if parts else torch.zeros(0, 3)


def check_against_reference(v: GridSampled, coord: torch.Tensor, seq_lens, pick: str) -> None:
    ref = reference(coord.cpu().numpy(), GRID, seq_lens)
    assert v.seq_lens == ref["lens"]
    assert v.cu_seqlens.dtype == torch.int32 and v.grid_coord.dtype == torch.int32
    assert v.origin.dtype == torch.int32 and v.index.dtype == torch.int64
    np.testing.assert_array_equal(v.cu_seqlens.cpu().numpy(), np.cumsum((0, *ref["lens"])))
    np.testing.assert_array_equal(v.grid_coord.cpu().numpy(), ref["grid"])
    np.testing.assert_array_equal(v.origin.cpu().numpy(), ref["origin"])
    if v.inverse is not None:
        np.testing.assert_array_equal(v.inverse.cpu().numpy(), ref["inverse"])
    if pick in ("first", "center"):
        np.testing.assert_array_equal(v.index.cpu().numpy(), ref[pick])
    else:  # random: the representative lies in its cell
        inverse = torch.from_numpy(ref["inverse"]).to(v.index.device)
        cell = torch.arange(len(v.index), device=v.index.device)
        assert torch.equal(inverse[v.index], cell)


# ---------------------------------------------------------------- lattice
def test_lattice_is_float64_division():
    # multiples of the grid size in float32: float32 or reciprocal math moves some across a cell
    k = torch.arange(-3000, 3000, dtype=torch.float64)
    coord = (k[:, None] * torch.tensor([0.02, 0.05, 0.1], dtype=torch.float64)).float()
    for grid_size in (0.02, 0.05, 0.1, 0.03):
        expected = np.floor(coord.numpy() / np.array(grid_size)).astype(np.int64)
        got = lattice(coord, grid_size)
        assert got.dtype == torch.int64
        np.testing.assert_array_equal(got.numpy(), expected)
        float32 = torch.floor(coord / grid_size).long()
        if grid_size == 0.02:
            assert not torch.equal(float32, got)  # the reason for float64


@pytest.mark.gpu
def test_lattice_cuda_matches_cpu(cuda):
    k = torch.arange(-3000, 3000, dtype=torch.float64)
    coord = (k[:, None] * 0.02 + torch.tensor([0.0, 1e-9, -1e-9], dtype=torch.float64)).float()
    coord = torch.cat([coord, random_cloud((5000,), extent=20.0)])
    for grid_size in (0.02, 0.05, 0.03):
        assert torch.equal(lattice(coord.to(cuda), grid_size).cpu(), lattice(coord, grid_size))


# ------------------------------------------------------------- grid_sample vs numpy
@pytest.mark.parametrize("pick", ["first", "center", "random"])
@pytest.mark.parametrize("seq_lens", [(2000,), (700, 0, 1500, 1), (0,), (1,)])
def test_grid_sample_matches_reference_cpu(pick, seq_lens):
    coord = random_cloud(seq_lens)
    g = torch.Generator().manual_seed(1)
    v = grid_sample(coord, GRID, seq_lens=seq_lens, pick=pick, generator=g, return_inverse=True)
    check_against_reference(v, coord, seq_lens, pick)
    assert v.rank is None


@pytest.mark.gpu
@pytest.mark.parametrize("pick", ["first", "center", "random"])
@pytest.mark.parametrize("seq_lens", [(2000,), (700, 0, 1500, 1)])
def test_grid_sample_matches_reference_cuda(cuda, pick, seq_lens):
    coord = random_cloud(seq_lens).to(cuda)
    g = torch.Generator(device=cuda).manual_seed(1)
    v = grid_sample(coord, GRID, seq_lens=seq_lens, pick=pick, generator=g, return_inverse=True)
    assert v.index.device == coord.device and v.cu_seqlens.device == coord.device
    check_against_reference(v, coord, seq_lens, pick)


@pytest.mark.parametrize("device", DEVICES)
def test_negative_cells_in_a_pack(device):
    # every cell of the second sequence is negative, its maximum cell is exactly -1
    cells = torch.tensor([[-1, -1, -1], [-2, -1, -3], [-1, -4, -1], [-2, -1, -3]])
    coord = torch.cat([random_cloud((50,)), ((cells.double() + 0.5) * GRID).float()])
    v = grid_sample(coord.to(device), GRID, seq_lens=(50, 4), pick="first", return_inverse=True)
    check_against_reference(v, coord, (50, 4), "first")
    assert v.origin[1].tolist() == [-2, -4, -3]


@pytest.mark.parametrize("device", DEVICES)
def test_segment_bounds(device):
    g = torch.Generator().manual_seed(0)
    seq_lens = (700, 0, 3, 1, 300)
    grid = torch.randint(-50, 50, (sum(seq_lens), 3), generator=g)
    grid[700:703] = torch.tensor([[-1, -7, -1], [-3, -1, -2], [-1, -2, -9]])
    batch = torch.repeat_interleave(torch.arange(len(seq_lens)), torch.tensor(seq_lens))
    low, high = _segment_bounds(grid.to(device), batch.to(device), len(seq_lens))
    for b, rows in enumerate(grid.split(seq_lens)):
        lo, hi = (rows.amin(0), rows.amax(0)) if len(rows) else (torch.zeros(3),) * 2
        assert low[b].tolist() == lo.tolist() and high[b].tolist() == hi.tolist()


def test_seq_lens_none_is_one_sequence():
    coord = random_cloud((1500,))
    a = grid_sample(coord, GRID, pick="first")
    b = grid_sample(coord, GRID, seq_lens=[1500], pick="first")
    assert a.seq_lens == b.seq_lens and torch.equal(a.index, b.index)


def test_center_ties_go_to_the_lowest_row():
    point = torch.tensor([[0.013, 0.007, 0.011]])
    coord = torch.cat([torch.tensor([[0.001, 0.001, 0.001]]), point, point, point])
    v = grid_sample(coord, GRID, pick="center")
    assert v.index.tolist() == [1]


def test_rank_follows_input_order_within_a_cell():
    coord = random_cloud((3000,))
    v = grid_sample(coord, GRID, pick="first", return_inverse=True, return_rank=True)
    assert v.inverse is not None and v.rank is not None
    order = torch.argsort(v.inverse, stable=True)  # rows grouped by cell, input order inside
    counts = torch.bincount(v.inverse, minlength=len(v.index))
    starts = torch.cumsum(counts, 0) - counts
    expected = torch.empty_like(v.rank)
    expected[order] = torch.arange(len(order)) - starts[v.inverse[order]]
    assert torch.equal(v.rank, expected)
    assert torch.equal(v.rank[v.index], torch.zeros_like(v.index))  # "first" has rank 0


def test_int64_key_fallback_for_large_extents():
    # 3 axes spanning ~60000 cells: B * extent product overflows int32, so the CPU sorts int64
    g = torch.Generator().manual_seed(0)
    cells = torch.randint(0, 3, (4000, 3), generator=g) * 30000
    coord = ((cells.double() + 0.5) * GRID).float()
    coord = torch.cat([coord, coord[:100]])  # duplicates
    v = grid_sample(coord, GRID, pick="first", return_inverse=True)
    check_against_reference(v, coord, (len(coord),), "first")
    assert v.grid_coord.max() == 60000


@pytest.mark.parametrize("device", DEVICES)
def test_range_overflow_raises(device):
    coord = torch.tensor([[0.0, 0.0, 0.0], [70000 * GRID, 0.0, 0.0]], device=device)
    with pytest.raises(ValueError, match=r"grid must lie in \[0, 65536\)"):
        grid_sample(coord, GRID)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"coord": torch.zeros(4, 2)}, TypeError, "coord must be an"),
        ({"coord": torch.zeros(4, 3, dtype=torch.int32)}, TypeError, "coord must be an"),
        ({"grid_size": 0.0}, ValueError, "positive and finite"),
        ({"grid_size": float("nan")}, ValueError, "positive and finite"),
        ({"grid_size": "0.02"}, TypeError, "grid_size must be a float"),
        ({"pick": "middle"}, ValueError, "pick must be one of"),
        ({"seq_lens": (3,)}, ValueError, "summing to 4"),
        ({"seq_lens": (5, -1)}, ValueError, "non-negative"),
        ({"seq_lens": ()}, ValueError, "non-negative"),
    ],
)
def test_grid_sample_validation(kwargs, error, match):
    args = {"coord": torch.zeros(4, 3), "grid_size": GRID, **kwargs}
    with pytest.raises(error, match=match):
        grid_sample(args.pop("coord"), args.pop("grid_size"), **args)


# ---------------------------------------------------------------- random pick
def test_random_pick_is_reproducible():
    coord = random_cloud((4000, 3000))

    def draw(seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return grid_sample(coord, GRID, seq_lens=(4000, 3000), generator=g).index

    assert torch.equal(draw(3), draw(3))
    assert not torch.equal(draw(3), draw(4))


@pytest.mark.gpu
def test_random_pick_is_reproducible_cuda(cuda):
    coord = random_cloud((4000, 3000)).to(cuda)

    def draw(seed: int) -> torch.Tensor:
        g = torch.Generator(device=cuda).manual_seed(seed)
        return grid_sample(coord, GRID, seq_lens=(4000, 3000), generator=g).index

    assert torch.equal(draw(3), draw(3))
    assert not torch.equal(draw(3), draw(4))


def test_random_pick_is_uniform_within_a_cell():
    # cells holding 1..7 points; every rank of a cell must be drawn equally often
    sizes = torch.arange(1, 8).repeat(40)  # 280 cells
    cells = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
    grid = torch.stack([cells % 16, cells // 16, torch.zeros_like(cells)], 1)
    g = torch.Generator().manual_seed(0)
    coord = (
        (grid.double() + 0.1 + 0.8 * torch.rand(grid.shape, generator=g, dtype=torch.float64))
        * GRID
    ).float()
    perm = torch.randperm(len(coord), generator=g)
    coord = coord[perm]
    first = grid_sample(coord, GRID, pick="first", return_inverse=True, return_rank=True)
    assert first.inverse is not None and first.rank is not None
    count = torch.bincount(first.inverse)
    hits = torch.zeros(8, 8)  # [count, rank]
    trials = 400
    for _ in range(trials):
        index = grid_sample(coord, GRID, pick="random", generator=g).index
        hits.index_put_((count, first.rank[index]), torch.ones(len(index)), accumulate=True)
    for c in range(2, 8):
        observed = hits[c, :c]
        expected = trials * 40 / c
        chi2 = ((observed - expected) ** 2 / expected).sum().item()
        assert chi2 < 30.0, (c, observed.tolist())  # df <= 6: p < 1e-4 if truly uniform
    assert hits[1, 0] == trials * 40


# ---------------------------------------------------------------- sync budget
@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode")
@pytest.mark.parametrize("pick", ["first", "center", "random"])
def test_cuda_grid_sample_syncs_once(cuda, pick):
    coord = random_cloud((3000, 0, 2000)).to(cuda)
    g = torch.Generator(device=cuda).manual_seed(0)
    grid_sample(coord, GRID, seq_lens=(3000, 0, 2000), pick=pick, generator=g)  # warm up
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            grid_sample(
                coord,
                GRID,
                seq_lens=(3000, 0, 2000),
                pick=pick,
                generator=g,
                return_inverse=True,
                return_rank=True,
            )
        finally:
            torch.cuda.set_sync_debug_mode(0)
    syncs = [w for w in caught if "called a synchronizing" in str(w.message)]
    assert len(syncs) == 1, [str(w.message) for w in syncs]


# ---------------------------------------------------------------- unique_grid
@pytest.mark.parametrize("device", DEVICES)
def test_unique_grid(device):
    grid = torch.tensor(
        [[5, 1, 0], [2, 3, 4], [5, 1, 0], [2, 3, 4], [9, 9, 9], [0, 0, 0], [0, 0, 0]],
        dtype=torch.int32,
        device=device,
    )
    v = unique_grid(grid, seq_lens=(5, 2), return_inverse=True)
    assert v.seq_lens == (3, 1)
    assert v.grid_coord.tolist() == [[2, 3, 4], [5, 1, 0], [9, 9, 9], [0, 0, 0]]
    assert v.index.tolist() == [1, 0, 4, 5]
    assert v.inverse is not None and v.inverse.tolist() == [1, 0, 1, 0, 2, 3, 3]
    assert v.origin.tolist() == [[0, 0, 0], [0, 0, 0]] and v.origin.dtype == torch.int32
    gen = torch.Generator(device=device).manual_seed(0)
    r = unique_grid(grid, seq_lens=(5, 2), pick="random", generator=gen)
    assert torch.equal(grid[r.index], v.grid_coord)


@pytest.mark.parametrize("device", DEVICES)
def test_unique_grid_rejects_out_of_range(device):
    with pytest.raises(ValueError, match="grid must lie in"):
        unique_grid(torch.tensor([[0, -1, 0]], device=device))
    with pytest.raises(ValueError, match="grid must lie in"):
        unique_grid(torch.tensor([[0, 1 << 16, 0]], device=device))
    with pytest.raises(ValueError, match="no coordinates"):
        unique_grid(torch.zeros(1, 3, dtype=torch.int64), pick="center")
    with pytest.raises(TypeError, match="integer tensor"):
        unique_grid(torch.zeros(1, 3))
