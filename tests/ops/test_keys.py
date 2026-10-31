# Adapted from Pointcept2 tests/ops/test_keys.py @ 771a949.
from __future__ import annotations

import pytest
import torch

from point_transformer_v3.ops.keys import (
    AXIS_BITS,
    BATCH_SHIFT,
    check_key_range,
    pack_keys,
    unpack_keys,
)


def random_grid(n: int, high: int, batch_size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.randint(0, high, (n, 3), generator=g, dtype=torch.int32)
    batch = torch.randint(0, batch_size, (n,), generator=g)
    return grid, batch


@pytest.mark.parametrize("margin", [0, 1, 2])
def test_round_trip_at_the_limits(margin):
    high = (1 << AXIS_BITS) - 2 * margin
    grid, batch = random_grid(5000, high, 1 << 15)
    corners = torch.tensor([[0, 0, 0], [high - 1, high - 1, high - 1]], dtype=torch.int32)
    grid = torch.cat([grid, corners])
    batch = torch.cat([batch, torch.tensor([0, (1 << 15) - 1])])
    check_key_range(grid, (1 << 15) - 1, margin)  # the spec keeps batch_size < 2**15
    keys = pack_keys(grid, batch, margin)
    assert keys.dtype == torch.int64 and bool((keys >= 0).all())
    out_grid, out_batch = unpack_keys(keys, margin)
    assert out_grid.dtype == torch.int32 and out_batch.dtype == torch.int64
    assert torch.equal(out_grid, grid) and torch.equal(out_batch, batch)


def test_layout_and_margin_offsets():
    grid = torch.tensor([[1, 2, 3]])
    key = pack_keys(grid, torch.tensor([4]), margin=1)
    assert key.item() == (4 << BATCH_SHIFT) | (2 << 32) | (3 << 16) | 4
    # a neighbor offset of -margin stays in its field
    neighbor = pack_keys(grid - 1, torch.tensor([4]), margin=1)
    assert unpack_keys(neighbor, margin=1)[0].tolist() == [[0, 1, 2]]


def test_sorting_keys_is_lexicographic():
    grid, batch = random_grid(4000, 50, 3, seed=1)
    order = torch.argsort(pack_keys(grid, batch), stable=True)
    rows = torch.cat([batch[:, None], grid.long()], 1)[order].tolist()
    assert rows == sorted(rows)


@pytest.mark.parametrize(
    ("grid", "batch_size", "margin", "match"),
    [
        ([[0, -1, 0]], 1, 0, r"\[0, 65536\) for margin 0, got \[-1, 0\]"),
        ([[0, 65536, 0]], 1, 0, "got \\[0, 65536\\]"),
        ([[0, 65534, 0]], 1, 1, "\\[0, 65534\\) for margin 1"),
        ([[0, 0, 0]], 1 << 15, 0, "batch_size must be in"),
    ],
)
def test_check_key_range_rejects(grid, batch_size, margin, match):
    with pytest.raises(ValueError, match=match):
        check_key_range(torch.tensor(grid), batch_size, margin)


def test_check_key_range_accepts_empty_and_limits():
    check_key_range(torch.zeros(0, 3, dtype=torch.int32), 0, 3)
    check_key_range(torch.tensor([[65533, 0, 0]]), (1 << 15) - 1, 1)


def test_type_errors():
    with pytest.raises(TypeError, match="grid must be an"):
        pack_keys(torch.zeros(3, 3), torch.zeros(3, dtype=torch.int64))
    with pytest.raises(TypeError, match="grid must be an"):
        pack_keys(torch.zeros(3, 2, dtype=torch.int64), torch.zeros(3, dtype=torch.int64))
    with pytest.raises(TypeError, match="batch must be"):
        pack_keys(torch.zeros(3, 3, dtype=torch.int64), torch.zeros(4, dtype=torch.int64))
    with pytest.raises(TypeError, match="keys must be"):
        unpack_keys(torch.zeros(3, dtype=torch.int32))


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode")
def test_cuda_round_trip_without_sync(cuda):
    grid, batch = random_grid(10000, 1 << AXIS_BITS, 1 << 15)
    expected = pack_keys(grid, batch, 0)
    grid, batch = grid.to(cuda), batch.to(cuda)
    torch.cuda.set_sync_debug_mode("error")
    try:
        keys = pack_keys(grid, batch)
        out_grid, out_batch = unpack_keys(keys)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert torch.equal(keys.cpu(), expected)
    assert torch.equal(out_grid, grid) and torch.equal(out_batch, batch)
