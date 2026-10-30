# Adapted from Pointcept2 tests/ops/test_kmap.py @ 771a949.
"""Kernel offsets and the submanifold kernel map (ops/neighbors.py)."""

from __future__ import annotations

import itertools

import pytest
import torch

from pointtransformerv3.ops.neighbors import build_kmap, kernel_offsets
from tests.conftest import edge_case_pack, scannet_pack


def brute_kmap(grid: torch.Tensor, batch: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Dictionary reference: (N, K) int32 neighbor rows, -1 when missing."""
    rows = {(int(b), *map(int, g)): i for i, (g, b) in enumerate(zip(grid, batch, strict=True))}
    offsets = kernel_offsets(kernel_size).tolist()
    out = [
        [rows.get((int(b), g[0] + dx, g[1] + dy, g[2] + dz), -1) for dx, dy, dz in offsets]
        for g, b in zip(grid.tolist(), batch.tolist(), strict=True)
    ]
    return torch.tensor(out, dtype=torch.int32).view(len(out), kernel_size**3)


def test_kernel_offsets_order_matches_spconv_weight():
    for k in (1, 3, 5):
        offs = kernel_offsets(k)
        r = k // 2
        assert offs.dtype == torch.int64 and offs.shape == (k**3, 3)
        assert offs.tolist() == [list(p) for p in itertools.product(range(-r, r + 1), repeat=3)]
        assert torch.equal(offs.flip(0), -offs)


def test_kernel_offsets_rejects_bad_sizes():
    with pytest.raises(ValueError, match="odd"):
        kernel_offsets(4)
    with pytest.raises(ValueError, match="odd"):
        kernel_offsets(0)
    with pytest.raises(TypeError, match="int"):
        kernel_offsets(3.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
def test_build_kmap_matches_brute_force_on_edge_cases(kernel_size):
    grid, batch = edge_case_pack()
    kmap = build_kmap(grid, batch, kernel_size)
    assert kmap.dtype == torch.int32 and kmap.shape == (len(grid), kernel_size**3)
    assert torch.equal(kmap, brute_kmap(grid, batch, kernel_size))
    center = kernel_size**3 // 2
    assert torch.equal(kmap[:, center], torch.arange(len(grid), dtype=torch.int32))


@pytest.mark.parametrize("kernel_size", [3, 5])
def test_build_kmap_random_pack_and_symmetry(synthetic_pack, kernel_size):
    pack = synthetic_pack([300, 1, 0, 450], extent=10, seed=kernel_size)
    kmap = build_kmap(pack.grid_coord, pack.batch, kernel_size)
    assert torch.equal(kmap, brute_kmap(pack.grid_coord, pack.batch, kernel_size))
    # kmap[p, k] == q  <=>  kmap[q, K-1-k] == p  (the conv backward depends on this)
    num = kernel_size**3
    p, k = torch.nonzero(kmap >= 0, as_tuple=True)
    q = kmap[p, k].long()
    assert torch.equal(kmap[q, num - 1 - k].long(), p)


def test_build_kmap_is_row_permutation_equivariant(synthetic_pack):
    pack = synthetic_pack([200, 120], extent=8, seed=3)
    perm = torch.randperm(len(pack.grid_coord), generator=torch.Generator().manual_seed(0))
    kmap = build_kmap(pack.grid_coord, pack.batch, 3)
    kmap_perm = build_kmap(pack.grid_coord[perm], pack.batch[perm], 3)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(len(perm))
    expected = torch.where(kmap[perm] >= 0, inverse[kmap[perm].clamp(min=0).long()].int(), -1)
    assert torch.equal(kmap_perm, expected)


def test_build_kmap_empty_and_dtypes():
    empty = build_kmap(torch.zeros(0, 3, dtype=torch.int32), torch.zeros(0, dtype=torch.int64), 3)
    assert empty.shape == (0, 27) and empty.dtype == torch.int32
    grid, batch = edge_case_pack()
    assert torch.equal(build_kmap(grid.long(), batch.int(), 3), build_kmap(grid, batch, 3))


def test_build_kmap_validates_inputs(monkeypatch):
    grid, batch = edge_case_pack()
    with pytest.raises(ValueError, match="odd"):
        build_kmap(grid, batch, 2)
    with pytest.raises(TypeError, match=r"\(N, 3\)"):
        build_kmap(grid[:, :2], batch, 3)
    with pytest.raises(TypeError, match="integer"):
        build_kmap(grid.float(), batch, 3)
    with pytest.raises(TypeError, match="batch"):
        build_kmap(grid, batch[:-1], 3)
    # value checks need a sync and run only under PTV3_DEBUG=1
    dup = torch.cat([grid, grid[:1]])
    dup_batch = torch.cat([batch, batch[:1]])
    build_kmap(dup, dup_batch, 3)
    monkeypatch.setenv("PTV3_DEBUG", "1")
    with pytest.raises(ValueError, match="unique"):
        build_kmap(dup, dup_batch, 3)
    with pytest.raises(ValueError, match="grid values"):
        build_kmap(grid, batch, 7)  # the top corner leaves no room for a margin of 3
    with pytest.raises(ValueError, match="non-negative"):
        build_kmap(grid, batch - 1, 3)


@pytest.mark.gpu
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_build_kmap_cuda_matches_cpu_without_sync(synthetic_pack, kernel_size):
    pack = synthetic_pack([3000, 1, 0, 2500], extent=20, seed=1)
    expected = build_kmap(pack.grid_coord, pack.batch, kernel_size)
    grid, batch = pack.grid_coord.cuda(), pack.batch.cuda()
    build_kmap(grid, batch, kernel_size)  # warm-up (lazy CUDA init may sync)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        kmap = build_kmap(grid, batch, kernel_size)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(kmap.cpu(), expected)


@pytest.mark.data
@pytest.mark.gpu
def test_build_kmap_scannet_cuda_matches_cpu(scannet_scene):
    grid, batch = scannet_pack(scannet_scene, [0, 10])
    for kernel_size in (3, 5):
        kmap = build_kmap(grid.cuda(), batch.cuda(), kernel_size).cpu()
        assert torch.equal(kmap, build_kmap(grid, batch, kernel_size))
        p, k = torch.nonzero(kmap >= 0, as_tuple=True)
        assert torch.equal(kmap[kmap[p, k].long(), kernel_size**3 - 1 - k].long(), p)
        # a surface scan at 2 cm: most cells have several of their 26 neighbors
        assert 4 < (kmap >= 0).sum(1).float().mean() < kernel_size**3
