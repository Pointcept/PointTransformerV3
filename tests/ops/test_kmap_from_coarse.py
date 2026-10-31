# Adapted from Pointcept2 tests/ops/test_kmap_from_coarse.py @ 771a949.
"""Kernel maps from the stride-2 coarse level (ops/neighbors.py: kmap_from_coarse).

Every case builds a stride-2 hierarchy and checks, level by level from the top, that the map built
from the coarse level's 3^3 map equals ``build_kmap`` (k = 1, 3, 5), with the coarse maps
themselves chained top-down from a single ``build_kmap`` at the coarsest level. On CUDA that is
the Triton path (I3); its kernels also run in the Triton interpreter on CPU (a subprocess running
this file), with corrupt inputs that must give a wrong map but no out-of-range access, and direct
op calls with shapes, dtypes, devices or kernel sizes the kernels cannot take, which the op must
reject before it launches (never tried on CUDA):

    TRITON_INTERPRET=1 python -m tests.ops.test_kmap_from_coarse
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from point_transformer_v3.ops import neighbors
from point_transformer_v3.ops.keys import pack_keys, unpack_keys
from point_transformer_v3.ops.neighbors import build_kmap, kmap_from_coarse
from tests.conftest import DEVICES, cell_grid, edge_case_pack, regroup
from tests.ops.test_kmap import brute_kmap


def coarse_level(grid: torch.Tensor, batch: torch.Tensor):
    """Stride-2 coarse level, built without PointGrid: (coarse grid, coarse batch, inverse)."""
    keys, inverse = torch.unique(pack_keys(grid >> 1, batch), return_inverse=True)
    coarse_grid, coarse_batch = unpack_keys(keys)
    return coarse_grid, coarse_batch, inverse


def check_hierarchy(grid: torch.Tensor, batch: torch.Tensor, num_levels: int = 5) -> int:
    """Assert the top-down maps equal build_kmap on every level; returns the levels checked."""
    levels = [(grid, batch)]
    inverses = []
    for _ in range(num_levels - 1):
        coarse_grid, coarse_batch, inverse = coarse_level(*levels[-1])
        levels.append((coarse_grid, coarse_batch))
        inverses.append(inverse)
    coarse_kmap = build_kmap(*levels[-1], 3)
    for i in range(num_levels - 2, -1, -1):
        g, b = levels[i]
        for k in (1, 3, 5):
            got = kmap_from_coarse(coarse_kmap, inverses[i], g, k)
            assert got.dtype == torch.int32 and got.shape == (len(g), k**3)
            assert torch.equal(got, build_kmap(g, b, k)), (i, k)
        coarse_kmap = kmap_from_coarse(coarse_kmap, inverses[i], g, 3)  # chain downwards
    return num_levels


def shuffled(grid: torch.Tensor, batch: torch.Tensor, seed: int = 0):
    perm = torch.randperm(len(grid), generator=torch.Generator().manual_seed(seed))
    return grid[perm].contiguous(), batch[perm].contiguous()


def cases() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """(grid int32 (N, 3), batch int64 (N,)) edge cases from the verification report."""
    gen = torch.Generator().manual_seed(0)
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    out["edge pack (isolated, single, empty, corner 2**16-5)"] = edge_case_pack()
    top = (1 << 16) - 5  # build_kmap's limit for k=5
    near = torch.randint(0, 6, (300, 3), generator=gen).unique(dim=0)
    far = (top - torch.randint(0, 6, (300, 3), generator=gen)).unique(dim=0)
    spanning = torch.cat([near, far])  # one sample at both ends of the key range
    out["far apart samples"] = shuffled(
        torch.cat([spanning, far, near]).int(),
        torch.repeat_interleave(
            torch.arange(3), torch.tensor([len(spanning), len(far), len(near)])
        ),
    )
    sizes = torch.randint(0, 6, (400,), generator=gen) * (torch.rand(400, generator=gen) < 0.4)
    grids = [torch.randperm(27, generator=gen)[: int(n)] for n in sizes]  # cells of a 3^3 block
    cells = torch.cat(grids)
    out["400 samples, many empty, overlapping"] = shuffled(
        torch.stack([cells // 9, cells // 3 % 3, cells % 3], 1).int(),
        torch.repeat_interleave(torch.arange(400), sizes),
    )
    spaced = torch.randperm(20**3, generator=gen)[:60]
    spaced = torch.stack([spaced // 400, spaced // 20 % 20, spaced % 20], 1) * 7
    out["60 isolated points"] = (spaced.int(), torch.zeros(60, dtype=torch.int64))
    block = torch.tensor(list(itertools.product(range(9), repeat=3))) + torch.tensor([1, 3, 5])
    out["shuffled 9^3 block at an odd offset"] = shuffled(
        block.int(), torch.zeros(len(block), dtype=torch.int64)
    )
    straddle = torch.tensor([[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2], [2, 2, 2], [3, 3, 3]])
    out["pairs straddling parent cells"] = (straddle.int(), torch.zeros(6, dtype=torch.int64))
    plane = torch.tensor([[x, y, 4] for x in range(12) for y in range(12)])
    line = torch.tensor([[5, 5, z] for z in range(12) if z != 4])
    out["plane plus line"] = shuffled(
        torch.cat([plane, line]).int(), torch.zeros(len(plane) + len(line), dtype=torch.int64)
    )
    random = [torch.randperm(30**3, generator=gen)[:n] for n in (4000, 1, 2500, 3000)]
    cells = torch.cat(random)
    out["random samples"] = shuffled(
        torch.stack([cells // 900, cells // 30 % 30, cells % 30], 1).int(),
        torch.repeat_interleave(torch.arange(4), torch.tensor([4000, 1, 2500, 3000])),
    )
    return out


CASES = cases()
REPO = Path(__file__).resolve().parents[2]


def triton_kmap(coarse_kmap, inverse, grid, k, path="eager"):
    """The Triton path on ``grid``'s device (CPU = interpreter): the eager launcher or the op."""
    from point_transformer_v3.ops import _neighbors_triton

    if path == "op":
        return torch.ops.point_transformer_v3.kmap_from_coarse(coarse_kmap, inverse, grid, k)
    return _neighbors_triton.launch(coarse_kmap, inverse, grid, k)


def corrupt_cases(grid, batch, near_only):
    """(name, coarse_kmap, inverse) with out-of-range entries: "near" ones just outside [0, M)
    (allowed on CUDA, where every access is masked), "far" ones (interpreter only)."""
    coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)
    coarse_kmap = build_kmap(coarse_grid, coarse_batch, 3)
    m = len(coarse_grid)
    bad = [m, m + 3, -2, -7] + ([] if near_only else [2**30, -(2**31)])
    gen = torch.Generator().manual_seed(1)
    out = []
    for name, table in (("inverse", inverse), ("coarse_kmap", coarse_kmap)):
        table = table.clone()
        flat = table.view(-1)
        idx = torch.randint(0, flat.numel(), (max(1, flat.numel() // 10),), generator=gen)
        flat[idx] = torch.tensor(bad, dtype=flat.dtype)[
            torch.randint(0, len(bad), idx.shape, generator=gen)
        ]
        out.append(
            (
                name,
                coarse_kmap if name == "inverse" else table,
                table if name == "inverse" else inverse,
            )
        )
    return out


def bad_op_calls(coarse_kmap, inverse, grid):
    """(name, op arguments, error, message) of direct ``point_transformer_v3::kmap_from_coarse`` calls the
    op body and the fake impl must reject: launched, each would index past the coarse map, the
    child table or a short input."""
    m, n = coarse_kmap.shape[0], grid.shape[0]
    wide = torch.full((m, 125), -1, dtype=torch.int32, device=coarse_kmap.device)
    return [
        ("kernel_size 7", (coarse_kmap, inverse, grid, 7), ValueError, "1, 3 or 5"),
        ("kernel_size 2", (coarse_kmap, inverse, grid, 2), ValueError, "1, 3 or 5"),
        ("coarse (M, 125)", (wide, inverse, grid, 3), TypeError, r"coarse_kmap .*\(M, 27\)"),
        ("coarse (M, 9)", (coarse_kmap[:, :9], inverse, grid, 3), TypeError, "coarse_kmap"),
        ("coarse flat", (coarse_kmap.reshape(-1), inverse, grid, 3), TypeError, "coarse_kmap"),
        ("coarse int64", (coarse_kmap.long(), inverse, grid, 3), TypeError, "coarse_kmap"),
        ("grid (N, 2)", (coarse_kmap, inverse, grid[:, :2], 3), TypeError, r"grid .*\(N, 3\)"),
        ("grid float", (coarse_kmap, inverse, grid.float(), 3), TypeError, "grid"),
        ("short inverse", (coarse_kmap, inverse[1:], grid, 3), TypeError, rf"inverse .*\({n},\)"),
        ("inverse (N, 1)", (coarse_kmap, inverse[:, None], grid, 3), TypeError, "inverse"),
        ("float inverse", (coarse_kmap, inverse.float(), grid, 3), TypeError, "inverse"),
        ("coarse on meta", (coarse_kmap.to("meta"), inverse, grid, 3), ValueError, "a device"),
    ]


def check_op_rejects_bad_calls() -> bool:
    """Interpreter: every ``bad_op_calls`` case raises its error before a launch (without the
    checks, the interpreter would run the kernels on them)."""
    from point_transformer_v3.ops import _neighbors_triton  # noqa: F401  (registers the op)

    grid, batch = CASES["random samples"]
    coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)
    ok = True
    for name, args, error, message in bad_op_calls(build_kmap(coarse_grid, coarse_batch, 3),
                                                   inverse, grid):  # fmt: skip
        try:
            torch.ops.point_transformer_v3.kmap_from_coarse(*args)
            good, got = False, "no error"
        except error as e:
            good, got = re.search(message, str(e)) is not None, str(e)
        print(f"{'OK ' if good else 'BAD'} cpu op rejects {name}: {got}", flush=True)
        ok = ok and good
    return ok


def check_triton(device="cpu", near_only=False) -> bool:
    """The Triton path against build_kmap on every case, level and k, both call paths; corrupt
    inputs: no crash, and entries that are not -1 point to rows in [0, N)."""
    ok = True
    for name, (grid, batch) in CASES.items():
        grid, batch = grid.to(device), batch.to(device)
        coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)
        coarse_kmap = build_kmap(coarse_grid, coarse_batch, 3)
        for k in (1, 3, 5):
            for path in ("eager", "op"):
                got = triton_kmap(coarse_kmap, inverse, grid, k, path)
                good = torch.equal(got, build_kmap(grid, batch, k))
                print(f"{'OK ' if good else 'BAD'} {device} {path:5s} k={k} {name}", flush=True)
                ok = ok and good
        for what, bad_kmap, bad_inverse in corrupt_cases(grid.cpu(), batch.cpu(), near_only):
            got = triton_kmap(bad_kmap.to(device), bad_inverse.to(device), grid, 3)
            good = bool(((got >= -1) & (got < len(grid))).all())
            print(f"{'OK ' if good else 'BAD'} {device} corrupt {what} {name}", flush=True)
            ok = ok and good
    return ok


def interpret_main() -> int:
    if os.environ.get("TRITON_INTERPRET") != "1":
        print("set TRITON_INTERPRET=1 before running this module (CPU interpreter)")
        return 2
    ok = check_triton("cpu")
    ok = check_op_rejects_bad_calls() and ok
    print("ALL OK" if ok else "FAILURES", flush=True)
    return 0 if ok else 1


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton is not installed")
def test_triton_kernels_in_interpreter():
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tests.ops.test_kmap_from_coarse"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0 and "ALL OK" in result.stdout, result.stdout + result.stderr


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton is not installed")
def test_op_checks_inputs():
    """A direct ``torch.ops.point_transformer_v3.kmap_from_coarse`` call is checked before any launch (CPU
    tensors, so no kernel could run), and so is its fake impl (FakeTensorMode: tracing and
    export), which still gives the ``(N, K)`` int32 shape of valid inputs."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from point_transformer_v3.ops import _neighbors_triton  # noqa: F401  (registers the op)

    grid, batch = CASES["random samples"]
    coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)
    coarse_kmap = build_kmap(coarse_grid, coarse_batch, 3)
    for name, args, error, message in bad_op_calls(coarse_kmap, inverse, grid):
        with pytest.raises(error, match=message):
            torch.ops.point_transformer_v3.kmap_from_coarse(*args)
            pytest.fail(name)
    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(t) for t in (coarse_kmap, inverse, grid)]
        out = torch.ops.point_transformer_v3.kmap_from_coarse(*fake, 5)
        assert out.shape == (len(grid), 125) and out.dtype == torch.int32
        for name, args, error, message in bad_op_calls(*fake):
            with pytest.raises(error, match=message):
                torch.ops.point_transformer_v3.kmap_from_coarse(*args)
                pytest.fail(f"fake: {name}")


@pytest.mark.gpu
def test_triton_on_cuda():
    """The Triton path on CUDA: equal to build_kmap through both call paths, and to the torch
    path bit for bit; near out-of-range inputs stay inside the tensors."""
    from point_transformer_v3.ops.sparse_conv import triton_impl

    if not triton_impl.available(min_capability=triton_impl.TRITON_MIN_CAPABILITY):
        pytest.skip("triton unavailable")
    assert check_triton("cuda", near_only=True)
    grid, batch = (t.cuda() for t in CASES["random samples"])
    coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)
    coarse_kmap = build_kmap(coarse_grid, coarse_batch, 3)
    for k in (3, 5):
        torch_path = neighbors._from_coarse_torch(coarse_kmap, inverse, grid, k)
        assert torch.equal(kmap_from_coarse(coarse_kmap, inverse, grid, k), torch_path)


@pytest.mark.gpu
def test_triton_compile_and_export():
    """D15: the op traces (fullgraph compile, strict and non-strict export) and computes the same
    map."""
    import torch._inductor.config as inductor_config

    grid, batch = (t.cuda() for t in CASES["random samples"])
    coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)
    inputs = (build_kmap(coarse_grid, coarse_batch, 3), inverse, grid)
    expected = build_kmap(grid, batch, 5)

    class FromCoarse(torch.nn.Module):
        def forward(self, coarse_kmap, inverse, grid):
            return kmap_from_coarse(coarse_kmap, inverse, grid, 5)

    for strict in (False, True):
        exported = torch.export.export(FromCoarse(), inputs, strict=strict)
        targets = {str(n.target) for n in exported.graph.nodes if n.op == "call_function"}
        assert "point_transformer_v3.kmap_from_coarse.default" in targets, (strict, targets)
        assert torch.equal(exported.module()(*inputs), expected)
    torch._dynamo.reset()
    patch = {  # fx_graph_cache off: see tests/ops/test_norm_ops.py
        "triton.autotune_at_compile_time": False,
        "max_autotune": False,
        "compile_threads": 1,
        "fx_graph_cache": False,
    }
    try:
        with inductor_config.patch(patch):
            compiled = torch.compile(FromCoarse(), fullgraph=True, dynamic=True)
            assert torch.equal(compiled(*inputs), expected)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(CASES))
def test_equals_build_kmap_on_every_level(name, device):
    grid, batch = CASES[name]
    check_hierarchy(grid.to(device), batch.to(device))


@pytest.mark.parametrize("device", DEVICES)
def test_point_grid_hierarchy(synthetic_pack, device):
    """On PointGrid.downsample(2) levels: the coarse level's kmap and pool inverse. The reference
    is build_kmap itself, not PointGrid.kmap, so the test stays independent of how PointGrid
    builds its maps."""
    pack = synthetic_pack([3000, 1, 0, 2000], extent=24, seed=1)
    edge, edge_batch = edge_case_pack()
    grid_coord = torch.cat([pack.grid_coord, edge])
    batch = torch.cat([pack.batch, edge_batch + 4])
    fine = cell_grid(*regroup(grid_coord.to(device), batch.to(device)))
    for _ in range(4):
        coarse = fine.downsample(2)
        assert coarse.pool is not None
        coarse_kmap = coarse.kmap(3)
        assert torch.equal(coarse_kmap, build_kmap(coarse.grid_coord, coarse.batch_index(), 3))
        for k in (3, 5):
            got = kmap_from_coarse(coarse_kmap, coarse.pool.inverse, fine.grid_coord, k)
            expected = build_kmap(fine.grid_coord, fine.batch_index(), k)
            assert torch.equal(got, expected), (fine.level, k)
        fine = coarse


def test_beyond_the_key_range_of_build_kmap():
    """Coordinates up to 2**16 - 1 (build_kmap's packed keys stop at 2**16 - 2r): brute force."""
    grid = torch.tensor(
        [[65535, 65535, 65535], [65534, 65535, 65535], [65533, 65534, 65535], [0, 0, 0], [1, 0, 0]]
    ).int()
    batch = torch.zeros(5, dtype=torch.int64)
    coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)
    coarse_kmap = build_kmap(coarse_grid, coarse_batch, 3)
    for k in (3, 5):
        assert torch.equal(
            kmap_from_coarse(coarse_kmap, inverse, grid, k), brute_kmap(grid, batch, k)
        )


def test_export_keeps_the_lookup_table_cache_real(monkeypatch):
    """Non-strict torch.export runs the code on FakeTensors: the lookup tables built then must not
    enter the process-wide cache, or later eager calls get FakeTensors. Export with a cold cache,
    then eager; again with a warm cache."""
    monkeypatch.delenv("PTV3_DEBUG", raising=False)  # the debug checks read the device: eager only

    class FromCoarse(torch.nn.Module):
        def forward(self, coarse_kmap, inverse, grid):
            return kmap_from_coarse(coarse_kmap, inverse, grid, 5)

    grid, batch = CASES["random samples"]
    coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)
    inputs = (build_kmap(coarse_grid, coarse_batch, 3), inverse, grid)
    expected = build_kmap(grid, batch, 5)
    neighbors._octree_luts.cache_clear()
    try:
        for _ in range(2):  # cold cache, then warm
            exported = torch.export.export(FromCoarse(), inputs)
            assert torch.equal(exported.module()(*inputs), expected)
            eager = kmap_from_coarse(*inputs, 5)
            assert type(eager) is torch.Tensor and torch.equal(eager, expected)
            luts = neighbors._octree_luts(5, grid.device)
            assert all(type(t) is torch.Tensor for t in luts)
    finally:
        neighbors._octree_luts.cache_clear()


def test_debug_checks_reject_a_wrong_coarse_level(monkeypatch):
    """Inputs with unique (coarse row, slot) addresses that are still not the stride-2 pooling of
    the rows give a wrong, symmetric map; PTV3_DEBUG=1 catches them by geometry."""
    monkeypatch.delenv("PTV3_DEBUG", raising=False)
    gen = torch.Generator().manual_seed(0)
    cells = (torch.randint(0, 40, (40, 3), generator=gen) * 4).unique(dim=0)
    pairs = torch.cat([cells, cells + torch.tensor([3, 0, 0])]).int()  # (4a, 4a + 3): no clash
    sparse = cell_grid(pairs, [len(pairs)])
    stride4 = sparse.downsample(4)  # inverse to grid >> 2: one coarse row spans 2 parent cells
    assert stride4.pool is not None
    grid, batch = CASES["random samples"]
    shifted_grid, shifted_batch, shifted_inverse = coarse_level(grid + 1, batch)  # other origin
    fine = cell_grid(*regroup(grid, batch))
    z, hilbert = fine.downsample(2, order="z"), fine.downsample(2, order="hilbert")
    assert z.pool is not None and z.num_rows == hilbert.num_rows
    cases = {
        "different parent cells": (stride4.kmap(3), stride4.pool.inverse, sparse),
        "different parent cells ": (
            build_kmap(shifted_grid, shifted_batch, 3),
            shifted_inverse,
            grid,
        ),
        "wrong offsets": (hilbert.kmap(3), z.pool.inverse, fine),  # two orders' levels
    }
    for message, (coarse_kmap, inverse, rows) in cases.items():
        g = rows if isinstance(rows, torch.Tensor) else rows.grid_coord
        b = batch if isinstance(rows, torch.Tensor) else rows.batch_index()
        assert not torch.equal(kmap_from_coarse(coarse_kmap, inverse, g, 3), build_kmap(g, b, 3))
        monkeypatch.setenv("PTV3_DEBUG", "1")
        with pytest.raises(ValueError, match=message.strip()):
            kmap_from_coarse(coarse_kmap, inverse, g, 3)
        monkeypatch.delenv("PTV3_DEBUG")


@pytest.mark.gpu
def test_peak_memory_on_cuda():
    """At most about two (N, K) index buffers live at once (build_kmap: about one)."""
    gen = torch.Generator().manual_seed(3)
    cells = torch.randperm(80**3, generator=gen)[:200_000]
    grid = torch.stack([cells // 6400, cells // 80 % 80, cells % 80], 1).int().cuda()
    batch = torch.zeros(len(grid), dtype=torch.int64, device="cuda")
    coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)
    coarse_kmap = build_kmap(coarse_grid, coarse_batch, 3)
    kmap_from_coarse(coarse_kmap, inverse, grid, 5)  # warm the lookup tables and the allocator
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    kmap = kmap_from_coarse(coarse_kmap, inverse, grid, 5)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    assert peak < 2.5 * kmap.numel() * kmap.element_size(), peak / (kmap.numel() * 4)


@pytest.mark.gpu
def test_sync_free_on_cuda():
    grid, batch = CASES["random samples"]
    grid, batch = grid.cuda(), batch.cuda()
    coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)  # unique syncs: outside
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        coarse_kmap = build_kmap(coarse_grid, coarse_batch, 3)
        k3 = kmap_from_coarse(coarse_kmap, inverse, grid, 3)
        k5 = kmap_from_coarse(coarse_kmap, inverse, grid, 5)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(k3, build_kmap(grid, batch, 3)) and torch.equal(
        k5, build_kmap(grid, batch, 5)
    )


def test_validation_and_debug_checks(monkeypatch):
    grid, batch = CASES["random samples"]
    coarse_grid, coarse_batch, inverse = coarse_level(grid, batch)
    coarse_kmap = build_kmap(coarse_grid, coarse_batch, 3)
    with pytest.raises(ValueError, match="1, 3 or 5"):
        kmap_from_coarse(coarse_kmap, inverse, grid, 7)
    with pytest.raises(TypeError, match="coarse_kmap"):
        kmap_from_coarse(coarse_kmap.long(), inverse, grid, 3)
    with pytest.raises(TypeError, match="coarse_kmap"):
        kmap_from_coarse(build_kmap(coarse_grid, coarse_batch, 5), inverse, grid, 3)
    with pytest.raises(TypeError, match="inverse"):
        kmap_from_coarse(coarse_kmap, inverse[1:], grid, 3)
    with pytest.raises(TypeError, match="grid"):
        kmap_from_coarse(coarse_kmap, inverse, grid.float(), 3)
    empty = kmap_from_coarse(coarse_kmap, inverse[:0], grid[:0], 5)
    assert empty.shape == (0, 125) and empty.dtype == torch.int32

    monkeypatch.setenv("PTV3_DEBUG", "1")
    kmap_from_coarse(coarse_kmap, inverse, grid, 3)  # valid inputs pass the checks
    bad = inverse.clone()
    bad[0] = len(coarse_grid)
    with pytest.raises(ValueError, match="inverse must lie in"):
        kmap_from_coarse(coarse_kmap, bad, grid, 3)
    duplicated = torch.cat([grid, grid[:1]])
    with pytest.raises(ValueError, match="share a coarse row and child slot"):
        kmap_from_coarse(coarse_kmap, torch.cat([inverse, inverse[:1]]), duplicated, 3)


if __name__ == "__main__":
    sys.exit(interpret_main())
