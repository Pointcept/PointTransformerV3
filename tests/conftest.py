# Adapted from Pointcept2 tests/conftest.py @ 771a949.
"""Shared test helpers: CUDA skips, synthetic and edge-case packs, ``PointGrid`` builders, real
ScanNet scenes, CPU thread limits.

torch is imported lazily so tests that do not need it stay fast.
Plain helpers are imported with ``from tests.conftest import ...``.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

if TYPE_CHECKING:
    import numpy as np
    import torch
    from torch import Tensor

    from pointtransformerv3.structure import PointGrid

# Several agents and CI jobs share the 32 cores: cap intra-op threads unless the caller set them.
# Must run before torch/numpy are imported (they read these at start-up).
TEST_THREADS = os.environ.get("PTV3_TEST_THREADS", "8")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, TEST_THREADS)

# Processed ScanNet scenes (optional, ``data`` marker): a directory with ``index.json``.
SCANNET = Path(os.environ["PTV3_SCANNET"]).expanduser() if os.environ.get("PTV3_SCANNET") else None

# Device parameters. Tests that need CUDA carry the ``gpu`` marker: it skips them without CUDA
# (``pytest_runtest_setup``) and selects them with ``-m gpu``.
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("gpu") is not None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")


@pytest.fixture
def cuda() -> torch.device:
    """The CUDA device; skips the test when CUDA is unavailable."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch.device("cuda")


class SyntheticPack(NamedTuple):
    """B samples of unique grid cells concatenated along dim 0 (T rows in all)."""

    coord: Tensor  # (T, 3) float32, meters; lattice(coord, grid_size) == grid_coord
    grid_coord: Tensor  # (T, 3) int32, per-sample minimum 0 when the sample is not empty
    batch: Tensor  # (T,) int64 sample index
    cu_seqlens: Tensor  # (B + 1,) int32
    seq_lens: tuple[int, ...]
    grid_size: float


def make_synthetic_pack(
    seq_lens: tuple[int, ...] | list[int],
    extent: int = 32,
    seed: int = 0,
    device: torch.device | str = "cpu",
    grid_size: float = 0.02,
) -> SyntheticPack:
    """A pack of ``len(seq_lens)`` samples whose grid cells are distinct within each sample.

    Each sample draws ``n`` distinct cells of an ``extent ** 3`` cube (rows in random order),
    re-based to a minimum of 0, and places one point strictly inside each cell, so grid sampling
    ``coord`` at ``grid_size`` gives back ``grid_coord``. Deterministic in ``seed`` (CPU RNG).
    """
    import torch

    seq_lens = tuple(int(n) for n in seq_lens)
    cells = extent**3
    if any(n < 0 or n > cells for n in seq_lens):
        raise ValueError(f"every seq_len must be in [0, extent**3 = {cells}], got {seq_lens}")
    gen = torch.Generator().manual_seed(seed)
    grids = []
    for n in seq_lens:
        flat = _distinct_cells(n, cells, gen)
        grid = torch.stack([flat // extent**2, flat // extent % extent, flat % extent], 1)
        if n:
            grid = grid - grid.min(0).values
        grids.append(grid)
    grid = torch.cat(grids).int()
    jitter = 0.05 + 0.9 * torch.rand(grid.shape, generator=gen, dtype=torch.float64)
    coord = ((grid.double() + jitter) * grid_size).float()
    batch = torch.repeat_interleave(torch.arange(len(seq_lens)), torch.tensor(seq_lens))
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(seq_lens).cumsum(0)
    return SyntheticPack(
        coord.to(device), grid.to(device), batch.to(device), cu.to(device), seq_lens, grid_size
    )


def _distinct_cells(n: int, cells: int, gen: torch.Generator) -> Tensor:
    """(n,) int64 distinct values in [0, cells), in random order."""
    import torch

    if cells <= 1 << 22:
        return torch.randperm(cells, generator=gen)[:n]
    picked = torch.empty(0, dtype=torch.int64)
    while picked.numel() < n:
        draw = torch.randint(cells, (2 * n,), generator=gen)
        picked = torch.cat([picked, draw]).unique()
    return picked[torch.randperm(picked.numel(), generator=gen)[:n]]


@pytest.fixture
def synthetic_pack() -> Callable[..., SyntheticPack]:
    """Factory: ``synthetic_pack(seq_lens, extent=32, seed=0, device="cpu", grid_size=0.02)``."""
    return make_synthetic_pack


@pytest.fixture
def scannet_scene(request: pytest.FixtureRequest) -> Callable[..., dict[str, np.ndarray]]:
    """Loader ``scannet_scene(i, split="train")`` for processed ScanNet scenes (a dict of numpy
    arrays: coord f32, color u8, normal f32, segment20 i16, ...). Needs ``@pytest.mark.data``;
    skips unless ``PTV3_SCANNET`` names a directory with ``index.json``."""
    if request.node.get_closest_marker("data") is None:
        pytest.fail("tests that use scannet_scene must be marked @pytest.mark.data")
    if SCANNET is None or not (SCANNET / "index.json").is_file():
        pytest.skip(f"processed ScanNet not found (PTV3_SCANNET={SCANNET})")
    pytest.skip("processed ScanNet scenes are read by tests/parity only in this package")


def point_grid(pack: SyntheticPack, depth: int = 15, **kwargs: Any) -> PointGrid:
    """Level 0 ``PointGrid`` of a synthetic pack."""
    from pointtransformerv3.structure import PointGrid

    return PointGrid(
        coord=pack.coord,
        grid_coord=pack.grid_coord,
        cu_seqlens=pack.cu_seqlens,
        seq_lens=pack.seq_lens,
        grid_size=pack.grid_size,
        depth=depth,
        **kwargs,
    )


def cell_grid(grid_coord: Tensor, seq_lens: Sequence[int], depth: int = 16) -> PointGrid:
    """Level 0 ``PointGrid`` of bare grid cells (rows sample-contiguous, ``coord`` at the cell
    centers of a 0.02 m lattice): the ``SparseGrid`` of conv tests and benchmarks."""
    from pointtransformerv3.ops.seqlens import cu_seqlens_of
    from pointtransformerv3.structure import PointGrid

    seq_lens = tuple(int(n) for n in seq_lens)
    return PointGrid(
        coord=(grid_coord.float() + 0.5) * 0.02,
        grid_coord=grid_coord,
        cu_seqlens=cu_seqlens_of(seq_lens, grid_coord.device),
        seq_lens=seq_lens,
        grid_size=0.02,
        depth=depth,
    )


def regroup(grid_coord: Tensor, batch: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """The rows of a ``(grid, batch)`` pack sorted by sample (stable) and the sample sizes, the
    layout ``cell_grid`` needs (packs keep samples contiguous)."""
    import torch

    order = torch.sort(batch, stable=True).indices
    return grid_coord[order], tuple(torch.bincount(batch.cpu()).tolist())


def edge_case_pack() -> tuple[Tensor, Tensor]:
    """(grid (N, 3) int32, batch (N,) int64): isolated points, a single-point sample, a dense
    block, an empty sample and samples whose coordinates sit at both ends of the key range
    (2**16 - 2 * 2 - 1 is the top for k=5)."""
    import torch

    top = (1 << 16) - 5
    samples = [
        [[0, 0, 0], [5, 5, 5], [10, 0, 3]],  # isolated
        [[7, 7, 7]],  # single point
        [list(p) for p in itertools.product(range(3), range(2), range(3))],  # dense block
        [],  # empty sample
        [[top, top, top], [top - 1, top, top], [0, top, 0]],  # far corner
        [[0, 0, 0], [0, 0, 1], [1, 0, 0]],  # same coords as sample 0: must not connect
    ]
    grid = torch.tensor([p for s in samples for p in s], dtype=torch.int32)
    batch = torch.repeat_interleave(
        torch.arange(len(samples)), torch.tensor(list(map(len, samples)))
    )
    return grid, batch


def scannet_pack(
    load: Callable[..., dict[str, np.ndarray]], scenes: list[int], grid_size: float = 0.02
) -> tuple[Tensor, Tensor]:
    """(grid (N, 3) int32, batch (N,) int64): the unique ``floor(coord / grid_size)`` cells of
    several ScanNet scenes (``load`` is the ``scannet_scene`` fixture), rows shuffled."""
    import numpy as np
    import torch

    grids, batches = [], []
    for b, i in enumerate(scenes):
        coord = load(i)["coord"].astype(np.float64)
        g = np.floor(coord / grid_size).astype(np.int64)
        g = np.unique(g - g.min(0), axis=0)
        grids.append(g)
        batches.append(np.full(len(g), b))
    grid = torch.from_numpy(np.concatenate(grids)).int()
    batch = torch.from_numpy(np.concatenate(batches))
    perm = torch.randperm(len(grid), generator=torch.Generator().manual_seed(0))
    return grid[perm], batch[perm]
