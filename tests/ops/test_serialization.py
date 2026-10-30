# Adapted from Pointcept2 tests/ops/test_serialization.py @ 771a949.
"""Serialization: bit-exact codes of the original PTv3 serialization (stored in
tests/golden/serialization.npz: random and edge rows at every depth, exhaustive grids at depths
1-6, arbitrary int32 values), the curve properties m1 relies on (bijection, unit steps, prefix
under ``>> 3``, Hilbert's dependence on ``depth % 3``), input dtypes and layouts, torch.compile /
torch.export, and sync-freedom on CUDA."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from pointtransformerv3.ops import serialization
from pointtransformerv3.ops.serialization import MAX_DEPTH, ORDERS, decode, encode, serialize
from tests.conftest import DEVICES

DEPTHS = range(1, MAX_DEPTH + 1)
FULL_DEPTHS = range(1, 7)  # exhaustive grids: 8**6 = 262,144 cells at most
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "serialization.npz"
GOLDEN_HEAD = 64  # rows per (order, depth) whose codes are stored, not only hashed


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    """The original code's outputs (tests/golden/make_serialization.py): the random inputs
    ``grid`` (512, 3), ``batch`` (512,) and ``wild`` (256, 3), the codes ``head`` (orders, depths,
    64) of the first rows of ``grid``, and ``sha256``: the digest of every compared code array."""
    with np.load(GOLDEN) as f:
        data: dict[str, Any] = {
            k: torch.from_numpy(f[k]) for k in ("grid", "batch", "wild", "head")
        }
        data["sha256"] = dict(zip(f["keys"].tolist(), f["sha256"].tolist(), strict=True))
    return data


def sha256(code: Tensor) -> str:
    """Digest of an int64 code array (little-endian bytes, row order)."""
    return hashlib.sha256(code.cpu().numpy().astype("<i8").tobytes()).hexdigest()


def low_bits(grid: Tensor, depth: int) -> Tensor:
    """``grid`` reduced to its low ``depth`` bits: cells in [0, 2**depth)."""
    return grid & ((1 << depth) - 1)


def edge_rows(depth: int) -> Tensor:
    """(R, 3) int32 cells at the edges of [0, 2**depth): every triple of {0, 1, 2, m - 1, m,
    2**(d-1) - 1, 2**(d-1)} (m = 2**d - 1, the 8 corners among them) and, per bit k, the one-hot,
    low-ones and all-but-one values on each axis."""
    m = (1 << depth) - 1
    half = 1 << depth - 1
    values = sorted({v for v in (0, 1, 2, m - 1, m, half - 1, half) if 0 <= v <= m})
    rows = list(itertools.product(values, repeat=3))
    for k in range(depth):
        for v in (1 << k, (1 << k) - 1, m ^ (1 << k)):
            rows += [(v, 0, 0), (0, v, 0), (0, 0, v), (v, v, v), (v, m ^ v, v)]
    return torch.tensor(rows, dtype=torch.int32)


def random_grid(n: int, depth: int, device: str = "cpu", seed: int = 0) -> Tensor:
    """(n, 3) int32 cells in [0, 2**depth): ``edge_rows(depth)`` first (as many as fit), then
    uniform random cells."""
    gen = torch.Generator().manual_seed(seed)
    grid = torch.randint(0, 1 << depth, (n, 3), generator=gen, dtype=torch.int32)
    edges = edge_rows(depth)[:n]
    grid[: len(edges)] = edges
    return grid.to(device)


def random_batch(n: int, batch_size: int, device: str = "cpu", seed: int = 1) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, batch_size, (n,), generator=gen).to(device)


def full_grid(depth: int, device: str = "cpu") -> Tensor:
    """(8**depth, 3) int32: every cell of [0, 2**depth)**3, the origin first."""
    r = torch.arange(1 << depth, dtype=torch.int32)
    return torch.cartesian_prod(r, r, r).to(device)


def wild_head() -> Tensor:
    """(216, 3) int32: every triple of the extremes -2**31, -1, 2**31 - 1 and +-2**16 (around
    MAX_DEPTH)."""
    extremes = (-(2**31), -65536, -1, 65535, 65536, 2**31 - 1)
    return torch.tensor(list(itertools.product(extremes, repeat=3)), dtype=torch.int32)


def wild_grid(n: int, device: str = "cpu", seed: int = 2) -> Tensor:
    """(n + 216, 3) int32 with arbitrary values: ``wild_head()``, then uniform random int32
    values."""
    gen = torch.Generator().manual_seed(seed)
    tail = torch.randint(-(2**31), 2**31, (n, 3), generator=gen, dtype=torch.int64).int()
    return torch.cat([wild_head(), tail]).to(device)


def max_batch(n: int, depth: int, device: str = "cpu", seed: int = 3) -> Tensor:
    """(n,) sample indices spread over the whole 63 - 3 * depth bits above the code, the largest
    (2**(63 - 3 * depth) - 1) and 0 first."""
    gen = torch.Generator().manual_seed(seed)
    top = 1 << 63 - 3 * depth
    batch = torch.randint(0, top, (n,), generator=gen)
    batch[:2] = torch.tensor([top - 1, 0])
    return batch.to(device)


def unit_steps(cells: Tensor) -> bool:
    """Whether consecutive rows of ``cells`` are face neighbors (one unit step on one axis)."""
    return bool(((cells[1:].long() - cells[:-1].long()).abs().sum(1) == 1).all())


# ---- the original code's codes (tests/golden/serialization.npz)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", ORDERS)
def test_encode_matches_original(golden, order, device):
    """``edge_rows(depth)`` and the stored random rows at every depth."""
    for depth in DEPTHS:
        rows = low_bits(golden["grid"], depth)
        grid = torch.cat([edge_rows(depth), rows]).to(device)
        code = encode(grid, depth, order)
        assert code.dtype == torch.int64 and code.device == grid.device
        head = code[len(grid) - len(rows) :][:GOLDEN_HEAD].cpu()
        expected = golden["head"][ORDERS.index(order), depth - 1]
        assert torch.equal(head, expected), f"depth={depth}"
        assert sha256(code) == golden["sha256"][f"encode/{order}/{depth}"], f"depth={depth}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", ORDERS)
def test_serialize_matches_original(golden, order, device):
    batch = golden["batch"].to(device)
    for depth in DEPTHS:
        grid = low_bits(golden["grid"], depth).to(device)
        key = serialize(grid, batch, depth, order, batch_size=7)
        assert sha256(key) == golden["sha256"][f"serialize/{order}/{depth}"], f"depth={depth}"


@pytest.mark.parametrize("device", DEVICES)
def test_full_grids_match_original(golden, device):
    """Every cell of the cube at depths 1-6."""
    for depth, order in itertools.product(FULL_DEPTHS, ORDERS):
        code = encode(full_grid(depth, device), depth, order)
        assert sha256(code) == golden["sha256"][f"full/{order}/{depth}"], f"depth={depth} {order}"


def test_arbitrary_int32_values_match_original(golden):
    """Bits above ``depth`` are ignored, as in the original: negative values, the int32 extremes
    and values >= 2**16 encode like their low ``depth`` bits."""
    grid = torch.cat([wild_head(), golden["wild"]])
    for depth, order in itertools.product(DEPTHS, ORDERS):
        code = encode(grid, depth, order)
        assert sha256(code) == golden["sha256"][f"wild/{order}/{depth}"], f"depth={depth} {order}"
        assert torch.equal(encode(low_bits(grid, depth), depth, order), code)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", ORDERS)
def test_decode_round_trip(order, device):
    for depth in DEPTHS:
        grid = random_grid(2048, depth, device, seed=depth)
        batch = random_batch(2048, 5, device)
        code = serialize(grid, batch, depth, order)
        out_grid, out_batch = decode(code, depth, order)
        assert out_grid.dtype == torch.int32 and out_batch.dtype == torch.int64
        assert torch.equal(out_grid, grid) and torch.equal(out_batch, batch), f"depth={depth}"


# ---- curve properties


@pytest.mark.parametrize("device", DEVICES)
def test_full_grid_codes_are_a_bijection(device):
    """At depths 1-6 every order maps the cube one-to-one onto [0, 8**depth) with the origin at
    code 0, ``decode`` inverts it, and the Hilbert orders visit the cells in unit steps."""
    for depth in FULL_DEPTHS:
        grid = full_grid(depth, device)
        cells = torch.arange(8**depth, device=device)
        for order in ORDERS:
            code = encode(grid, depth, order)
            path = torch.argsort(code)
            assert torch.equal(code[path], cells), f"depth={depth} {order}"
            assert code[0] == 0, f"depth={depth} {order}"
            assert torch.equal(decode(cells, depth, order)[0], grid[path]), f"depth={depth}"
            if order.startswith("hilbert"):
                assert unit_steps(grid[path]), f"depth={depth} {order}"


@pytest.mark.parametrize("device", DEVICES)
def test_hilbert_windows_are_unit_step_walks(device):
    """At depths 7-16, 2**14 consecutive Hilbert codes (at the start, a random offset and the end
    of the curve) decode to a unit-step walk from the origin that encodes back to the same codes."""
    window = 1 << 14
    gen = torch.Generator().manual_seed(7)
    for depth in range(7, MAX_DEPTH + 1):
        last = 8**depth - window
        middle = int(torch.randint(0, last, (1,), generator=gen))
        for start, order in itertools.product((0, middle, last), ("hilbert", "hilbert-trans")):
            codes = torch.arange(start, start + window, device=device)
            cells, _ = decode(codes, depth, order)
            assert unit_steps(cells), f"depth={depth} {order} start={start}"
            assert torch.equal(encode(cells, depth, order), codes), f"depth={depth} {order}"
            if start == 0:
                assert not cells[0].any(), f"depth={depth} {order}: code 0 is not the origin"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", ORDERS)
def test_prefix_property(order, device):
    """m1 pooling: the parent cell's code at depth - 1 is the child's code shifted by 3 bits."""
    for depth in range(2, MAX_DEPTH + 1):
        grid = random_grid(2048, depth, device, seed=depth)
        batch = random_batch(2048, 4, device)
        child = serialize(grid, batch, depth, order)
        parent = serialize(grid >> 1, batch, depth - 1, order)
        assert torch.equal(child >> 3, parent), f"depth={depth}"


@pytest.mark.parametrize("device", DEVICES)
def test_hilbert_order_depends_on_depth_mod_3(device):
    """For cells inside [0, 2**d), the Hilbert order at depth d equals the one at depth 16 exactly
    when d = 16 (mod 3); z-order codes do not depend on the depth at all."""
    for depth in DEPTHS:
        grid = random_grid(4096, depth, device, seed=depth)
        for order in ("hilbert", "hilbert-trans"):
            code = encode(grid, depth, order)
            same = torch.equal(
                torch.argsort(code, stable=True),
                torch.argsort(encode(grid, MAX_DEPTH, order), stable=True),
            )
            assert same == (depth % 3 == MAX_DEPTH % 3), f"depth={depth} {order}"
            if depth + 3 <= MAX_DEPTH:
                assert torch.equal(code, encode(grid, depth + 3, order))
        for order in ("z", "z-trans"):
            assert torch.equal(encode(grid, depth, order), encode(grid, MAX_DEPTH, order))


# ---- tracing and sync-freedom


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype")
def test_cuda_path_is_sync_free_from_a_cold_cache():
    grid = random_grid(100_000, 15, "cuda")
    batch = random_batch(100_000, 2, "cuda")
    torch.cuda.synchronize()
    serialization._TABLE_CACHE.clear()  # the first call also copies the tables to the device
    torch.cuda.set_sync_debug_mode("error")
    try:
        codes = [serialize(grid, batch, 15, order, batch_size=2) for order in ORDERS]
        grids = [decode(code, 15, order)[0] for code, order in zip(codes, ORDERS, strict=True)]
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert all(torch.equal(out, grid) for out in grids)


def every_order(grid: Tensor, batch: Tensor, depth: int = 12) -> list[Tensor]:
    """The keys of all four orders, then each one decoded."""
    codes = [serialize(grid, batch, depth, order) for order in ORDERS]
    return codes + [decode(c, depth, o)[0] for c, o in zip(codes, ORDERS, strict=True)]


@pytest.mark.parametrize("device", DEVICES)
def test_compile_matches_eager(monkeypatch, device):
    """torch.compile(fullgraph=True) from a cold table cache: dynamo builds the real tables
    eagerly while it traces (no FakeTensor reaches ``pin_memory`` or the cache) and bakes them
    into the graph. CPU runs AOTAutograd (the FakeTensor tracing; Inductor's C++ compile takes
    about 4 s), CUDA runs Inductor; both bit-exact with eager. GPU safety: no compile-time
    autotuning (CLAUDE.md); set through ``inductor_config.patch`` because pytest's monkeypatch
    deletes torch config entries on undo."""
    import torch._inductor.config as inductor_config

    monkeypatch.delenv("PTV3_DEBUG", raising=False)  # the debug checks read the device: eager only
    grid, batch = random_grid(5000, 12, device, seed=5), random_batch(5000, 3, device)
    expected = every_order(grid, batch)
    graphs: list[torch.fx.GraphModule] = []

    def record(gm: torch.fx.GraphModule, example_inputs: list[Tensor]):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    serialization._TABLE_CACHE.clear()
    patch = {"triton.autotune_at_compile_time": False, "max_autotune": False, "compile_threads": 1}
    try:
        with inductor_config.patch(patch):
            backend = "inductor" if device == "cuda" else "aot_eager"
            got = torch.compile(every_order, backend=backend, fullgraph=True)(grid, batch)
        for a, b in zip(got, expected, strict=True):
            assert torch.equal(a, b)
        assert all(type(t) is Tensor for t in serialization._TABLE_CACHE[grid.device])
        torch._dynamo.reset()
        torch.compile(every_order, backend=record, fullgraph=True)(grid, batch)
        nodes = list(graphs[0].graph.nodes)
        assert sum(n.op == "get_attr" for n in nodes) == 4  # the tables, as graph constants
        assert not any("arange" in str(n.target) for n in nodes)  # nothing rebuilds them
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("device", DEVICES)
def test_export_keeps_the_table_cache_real(monkeypatch, device):
    """Non-strict torch.export runs the code on FakeTensors: the lookup tables built then must not
    enter the process-wide cache, or later eager calls get FakeTensors. Export with a cold cache
    (the program builds the tables), then eager, then again with the warm cache (the program
    holds the cached tables as constants and builds nothing)."""
    monkeypatch.delenv("PTV3_DEBUG", raising=False)  # the debug checks read the device: eager only

    class Serialize(torch.nn.Module):
        def forward(self, grid, batch):
            code = serialize(grid, batch, 10, "hilbert")
            return code, decode(code, 10, "hilbert-trans")[0]

    grid, batch = random_grid(300, 10, device), random_batch(300, 3, device)
    expected = Serialize()(grid, batch)
    serialization._TABLE_CACHE.clear()
    for warm in (False, True):
        exported = torch.export.export(Serialize(), (grid, batch), strict=False)
        for got, want in zip(exported.module()(grid, batch), expected, strict=True):
            assert torch.equal(got, want)
        building = [n for n in exported.graph.nodes if "arange" in str(n.target)]
        assert bool(building) != warm
        code = serialize(grid, batch, 10, "hilbert")
        assert type(code) is Tensor and torch.equal(code, expected[0])
        tables = serialization._TABLE_CACHE[grid.device]
        assert all(type(t) is Tensor for t in tables)
        if warm:
            constants = [c for c in exported.constants.values() if isinstance(c, Tensor)]
            assert all(any(c.data_ptr() == t.data_ptr() for c in constants) for t in tables)


# ---- inputs and validation


@pytest.mark.parametrize("device", DEVICES)
def test_largest_batch_prefix_fits_63_bits(device):
    """Sample indices up to 2**(63 - 3 * depth) - 1 sit above the code: keys stay non-negative,
    keep the code in their low bits and decode back."""
    for depth in DEPTHS:
        grid = random_grid(1024, depth, device, seed=depth)
        batch = max_batch(1024, depth, device, seed=depth)
        for order in ORDERS:
            key = serialize(grid, batch, depth, order)
            assert bool((key >= 0).all()), f"depth={depth} {order}"
            assert torch.equal(key, batch << 3 * depth | encode(grid, depth, order))
            out_grid, out_batch = decode(key, depth, order)
            assert torch.equal(out_grid, grid) and torch.equal(out_batch, batch), f"depth={depth}"


@pytest.mark.parametrize("device", DEVICES)
def test_integer_dtypes_layouts_and_sizes(device):
    """uint8 / int8 / int16 / int64 grids, non-contiguous and column-major layouts, N = 1 and
    N = 0 give the codes of the int32 grid."""
    grid = random_grid(512, 7, device)
    wide = random_grid(512, 16, device, seed=4)
    holder = torch.cat([wide, wide[:, :1]], 1)  # (N, 4): wide as a non-contiguous view
    small = wild_grid(256, device)
    for order in ORDERS:
        expected = encode(grid, 7, order)
        for dtype in (torch.uint8, torch.int16, torch.int64):
            assert torch.equal(encode(grid.to(dtype), 7, order), expected)
        low = (small.long() & 127).int()  # int8 wraps the values: their low 7 bits stay
        assert torch.equal(encode(small.to(torch.int8), 7, order), encode(low, 7, order))
        expected = encode(wide, 16, order)
        assert torch.equal(encode(holder[:, :3], 16, order), expected)
        assert torch.equal(encode(wide.t().contiguous().t(), 16, order), expected)
        assert torch.equal(encode(wide[:1].clone(), 16, order), expected[:1])
        one = serialize(wide[:1], torch.ones(1, dtype=torch.int64, device=device), 16, order)
        assert torch.equal(decode(one, 16, order)[0], wide[:1])
        empty = encode(torch.zeros((0, 3), dtype=torch.int32, device=device), 7, order)
        assert empty.shape == (0,) and empty.dtype == torch.int64
        key = serialize(wide[:0], torch.zeros(0, dtype=torch.int64, device=device), 16, order)
        assert key.shape == (0,) and key.dtype == torch.int64
        out_grid, out_batch = decode(empty, 7, order)
        assert out_grid.shape == (0, 3) and out_batch.shape == (0,)


def test_validation_errors():
    grid = random_grid(16, 4)
    batch = torch.zeros(16, dtype=torch.int64)
    with pytest.raises(TypeError, match="integer tensor"):
        encode(grid.float(), 4, "z")
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        encode(grid[:, :2], 4, "z")
    with pytest.raises(TypeError, match="depth must be an int"):
        encode(grid, True, "z")  # type: ignore[arg-type]
    for depth in (0, MAX_DEPTH + 1):
        with pytest.raises(ValueError, match=r"depth must be in 1\.\.16"):
            encode(grid, depth, "z")
    with pytest.raises(ValueError, match="did you mean 'hilbert'"):
        encode(grid, 4, "hilbret")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="batch must be an integer tensor"):
        serialize(grid, batch.float(), 4, "z")
    with pytest.raises(ValueError, match=r"batch must have shape \(16,\)"):
        serialize(grid, batch[:8], 4, "z")
    with pytest.raises(TypeError, match="1-D int64"):
        decode(batch.int(), 4, "z")


def test_batch_size_must_fit_above_the_code():
    grid = random_grid(16, 16)
    batch = torch.zeros(16, dtype=torch.int64)
    serialize(grid, batch, 16, "hilbert", batch_size=2**15 - 1)  # 48 + 15 bits
    with pytest.raises(ValueError, match="int64 keys hold 63"):
        serialize(grid, batch, 16, "hilbert", batch_size=2**15)
    serialize(grid, batch, 15, "hilbert", batch_size=2**18 - 1)
    with pytest.raises(ValueError, match="positive int"):
        serialize(grid, batch, 16, "z", batch_size=0)


def test_debug_mode_checks_values(monkeypatch):
    grid = random_grid(16, 4)
    batch = torch.zeros(16, dtype=torch.int64)
    encode(grid | 16, 4, "z")  # out of range: high bits are dropped silently, as in Pointcept
    monkeypatch.setenv("PTV3_DEBUG", "1")
    encode(grid, 4, "z")
    with pytest.raises(ValueError, match=r"must lie in \[0, 2\*\*4\)"):
        encode(grid | 16, 4, "hilbert")
    with pytest.raises(ValueError, match=r"must lie in \[0, 2\*\*4\)"):
        encode(grid - 1, 4, "z")
    with pytest.raises(ValueError, match="batch indices must be >= 0"):
        serialize(grid, batch - 1, 4, "z")
    with pytest.raises(ValueError, match="int64 keys hold 63"):
        serialize(random_grid(16, 16), batch + 2**15 - 1, 16, "z")
