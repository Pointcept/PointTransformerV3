# Adapted from Pointcept2 tests/ops/test_pooling.py @ 771a949.
"""Pooling index and segment reductions (ops/pooling.py): brute-force references, gradients,
packing invariance, the padded-gather path, determinism, CPU/CUDA parity, sync-free and
compile/export capture."""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from point_transformer_v3.ops.pooling import REDUCES, PoolIndex, build_pool_index, pool, unpool
from point_transformer_v3.structure import PointGrid
from tests.conftest import DEVICES, point_grid


def index_from_clusters(cluster: Tensor) -> PoolIndex:
    """PoolIndex of an arbitrary (N,) cluster assignment (ids 0..M-1, all used), the PC way."""
    counts = torch.bincount(cluster)
    sort = torch.argsort(cluster, stable=True)
    ptr = F.pad(counts.cumsum(0), (1, 0))
    return PoolIndex(inverse=cluster, sort=sort, ptr=ptr, head=sort[ptr[:-1]])


def reference_pool(feat: Tensor, cluster: Tensor, reduce: str) -> Tensor:
    m = int(cluster.max()) + 1 if cluster.numel() else 0
    out = feat.new_zeros(m, *feat.shape[1:])
    index = cluster.view(-1, *([1] * (feat.dim() - 1))).expand_as(feat)
    amax = {"max": "amax", "min": "amin", "sum": "sum", "mean": "mean"}[reduce]
    return out.scatter_reduce(0, index, feat, amax, include_self=False)


# ---- PoolIndex and build_pool_index


def test_pool_index_validation():
    i = torch.arange(4)
    with pytest.raises(TypeError, match="int64"):
        PoolIndex(inverse=i.int(), sort=i, ptr=torch.tensor([0, 4]), head=i[:1])
    with pytest.raises(ValueError, match="shapes disagree"):
        PoolIndex(inverse=i, sort=i[:3], ptr=torch.tensor([0, 4]), head=i[:1])
    with pytest.raises(ValueError, match="shapes disagree"):
        PoolIndex(inverse=i, sort=i, ptr=torch.tensor([0, 2, 4]), head=i[:1])
    index = PoolIndex(
        inverse=torch.zeros(4, dtype=torch.long), sort=i, ptr=torch.tensor([0, 4]), head=i[:1]
    )
    assert (index.num_fine, index.num_coarse, index.grouped) == (4, 1, False)


def test_build_pool_index_hand_example():
    # rows 0..6 of two samples (4 + 3); keys given in sorted order through `sort`
    sort = torch.tensor([2, 0, 3, 1, 6, 4, 5])
    keys = torch.tensor([10, 10, 11, 12, 1 << 40, (1 << 40) + 5, (1 << 40) + 5])
    cu = torch.tensor([0, 4, 7], dtype=torch.int32)
    index, cu_c, seq_lens = build_pool_index(keys, sort, cu)
    assert seq_lens == (3, 2)
    assert cu_c.dtype == torch.int32 and cu_c.tolist() == [0, 3, 5]
    assert index.ptr.tolist() == [0, 2, 3, 4, 5, 7]
    assert index.head.tolist() == [2, 3, 1, 6, 4]
    assert index.inverse.tolist() == [0, 2, 0, 1, 4, 4, 3]
    assert torch.equal(index.sort, sort) and not index.grouped


def test_build_pool_index_grouped_and_empty_samples():
    keys = torch.tensor([3, 3, 3, 7, 9, 9])
    cu = torch.tensor([0, 0, 4, 4, 6, 6], dtype=torch.int32)  # empty samples 0, 2, 4
    index, cu_c, seq_lens = build_pool_index(keys, torch.arange(6), cu, grouped=True)
    assert seq_lens == (0, 2, 0, 1, 0) and cu_c.tolist() == [0, 0, 2, 2, 3, 3]
    assert index.grouped and index.ptr.tolist() == [0, 3, 4, 6]
    assert index.inverse.tolist() == [0, 0, 0, 1, 2, 2] and index.head.tolist() == [0, 3, 4]
    empty, _, seq_e = build_pool_index(
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.zeros(3).int()
    )
    assert seq_e == (0, 0) and empty.ptr.tolist() == [0] and empty.num_coarse == 0


def test_build_pool_index_argument_checks(monkeypatch):
    keys, sort, cu = torch.tensor([1, 1, 2]), torch.tensor([2, 0, 1]), torch.tensor([0, 3]).int()
    with pytest.raises(TypeError, match="keys"):
        build_pool_index(keys.int(), sort, cu)
    with pytest.raises(TypeError, match="sort"):
        build_pool_index(keys, sort[:2], cu)
    with pytest.raises(TypeError, match="cu_seqlens"):
        build_pool_index(keys, sort, cu.long())
    monkeypatch.setenv("PTV3_DEBUG", "1")
    with pytest.raises(ValueError, match="permutation"):
        build_pool_index(keys, torch.tensor([0, 0, 1]), cu)
    with pytest.raises(ValueError, match="contiguous"):
        build_pool_index(torch.tensor([1, 2, 1]), sort, cu)
    with pytest.raises(ValueError, match="spans two samples"):
        build_pool_index(torch.tensor([1, 1, 2]), sort, torch.tensor([0, 1, 3]).int())
    with pytest.raises(ValueError, match="grouped"):
        build_pool_index(keys, sort, cu, grouped=True)
    build_pool_index(keys, sort, cu)  # valid input passes the debug checks
    build_pool_index(keys, torch.arange(3), cu, grouped=True)


# ---- pool / unpool values and gradients


@pytest.mark.parametrize("reduce", ["max", "mean", "min", "sum"])
@pytest.mark.parametrize("device", DEVICES)
def test_pool_matches_scatter_reduce(device, reduce):
    gen = torch.Generator().manual_seed(1)
    cluster = torch.randperm(50, generator=gen)[torch.randint(50, (400,), generator=gen)] % 37
    cluster = torch.unique(cluster, return_inverse=True)[1]  # ids 0..M-1, all used
    index = index_from_clusters(cluster.to(device))
    for shape in ((400, 7), (400, 2, 8)):
        feat = torch.randn(shape, generator=gen, dtype=torch.float64).to(device)
        torch.testing.assert_close(
            pool(feat, index, reduce), reference_pool(feat, cluster.to(device), reduce)
        )


@pytest.mark.parametrize("reduce", ["max", "mean", "sum"])
def test_pool_gradcheck(reduce):
    gen = torch.Generator().manual_seed(2)
    cluster = torch.unique(torch.randint(9, (40,), generator=gen), return_inverse=True)[1]
    index = index_from_clusters(cluster)
    feat = torch.randn(40, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: pool(x, index, reduce), (feat,))


@pytest.mark.parametrize("grouped", [False, True])
def test_unpool_value_and_gradcheck(grouped):
    gen = torch.Generator().manual_seed(3)
    cluster = torch.unique(torch.randint(9, (40,), generator=gen), return_inverse=True)[1]
    if grouped:
        cluster = cluster.sort().values
    index = index_from_clusters(cluster)
    if grouped:
        index = PoolIndex(index.inverse, index.sort, index.ptr, index.head, grouped=True)
        assert torch.equal(index.sort, torch.arange(40))
    feat = torch.randn(index.num_coarse, 2, 3, generator=gen, dtype=torch.float64)
    feat.requires_grad_()
    assert torch.equal(unpool(feat, index), feat[cluster])
    assert torch.autograd.gradcheck(lambda x: unpool(x, index), (feat,))


def test_row_count_and_reduce_checks():
    index = index_from_clusters(torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match="3 fine rows"):
        pool(torch.zeros(4, 2), index)
    with pytest.raises(ValueError, match="2 coarse rows"):
        unpool(torch.zeros(3, 2), index)
    with pytest.raises(ValueError, match="reduce"):
        pool(torch.zeros(3, 2), index, "median")  # pyright: ignore[reportArgumentType]


# ---- on real level indices


@pytest.mark.parametrize("device", DEVICES)
def test_downsample_pool_unpool_against_brute_force(synthetic_pack, device):
    pack = synthetic_pack([700, 0, 1, 1300], extent=24, seed=4, device=device)
    grid = point_grid(pack)
    for level, order in enumerate(("z", "hilbert", "hilbert")):
        coarse = grid.downsample(2, order=order)
        index = coarse.pool
        assert index is not None and index.grouped == (level == 2)  # rows already along hilbert
        parent_key = grid.batch_index() << 48 | _xyz(grid.grid_coord.long() >> 1)
        coarse_key = coarse.batch_index() << 48 | _xyz(coarse.grid_coord.long())
        assert torch.equal(coarse_key[index.inverse], parent_key)
        feat = torch.randn(grid.num_rows, 5, device=device, dtype=torch.float64)
        for reduce in ("max", "mean"):
            torch.testing.assert_close(
                pool(feat, index, reduce), reference_pool(feat, index.inverse, reduce)
            )
        up = torch.randn(coarse.num_rows, 5, device=device, dtype=torch.float64)
        assert torch.equal(unpool(up, index), up[index.inverse])
        grid = coarse


def _xyz(g: Tensor) -> Tensor:
    return g[:, 0] << 32 | g[:, 1] << 16 | g[:, 2]


@pytest.mark.parametrize("device", DEVICES)
def test_pooling_packing_invariance(synthetic_pack, device):
    """pool/unpool on pack [A, B] == cat of A and B pooled alone (bitwise, every reduce)."""
    a = synthetic_pack([900], extent=20, seed=5, device=device)
    b = synthetic_pack([600], extent=20, seed=6, device=device)
    ab = PointGrid(
        coord=torch.cat([a.coord, b.coord]),
        grid_coord=torch.cat([a.grid_coord, b.grid_coord]),
        cu_seqlens=torch.tensor([0, 900, 1500], dtype=torch.int32, device=device),
        seq_lens=(900, 600),
        grid_size=a.grid_size,
        depth=15,
    )
    gen = torch.Generator().manual_seed(7)
    feat = torch.randn(1500, 16, generator=gen).to(device, torch.bfloat16)
    grids = [ab, point_grid(a), point_grid(b)]
    feats = [feat, feat[:900], feat[900:]]
    for _ in range(3):
        coarse = [g.downsample(2) for g in grids]
        index = [index_of(c) for c in coarse]
        assert coarse[0].seq_lens == coarse[1].seq_lens + coarse[2].seq_lens
        for name in ("grid_coord", "coord"):
            joint = getattr(coarse[0], name)
            assert torch.equal(joint, torch.cat([getattr(c, name) for c in coarse[1:]])), name
        for reduce in REDUCES:
            pooled = [pool(f, i, reduce) for f, i in zip(feats, index, strict=True)]
            assert torch.equal(pooled[0], torch.cat(pooled[1:])), reduce
        feats = [pool(f, i, "max") for f, i in zip(feats, index, strict=True)]
        up = [unpool(f, i) for f, i in zip(feats, index, strict=True)]
        assert torch.equal(up[0], torch.cat(up[1:]))
        grids = coarse


def index_of(grid: PointGrid) -> PoolIndex:
    assert grid.pool is not None
    return grid.pool


# ---- CUDA paths


@pytest.mark.gpu
@pytest.mark.parametrize("channels", [8, 32, 64, 128, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cuda_padded_gather_is_exact(cuda, synthetic_pack, channels, dtype):
    """Rows of 16..128 bytes take the padded-stride gather; values and grads equal plain torch."""
    pack = synthetic_pack([3000, 2000], extent=32, seed=8, device=cuda)
    index = point_grid(pack).downsample(2).pool
    assert index is not None
    feat = torch.randn(5000, channels, device=cuda).to(dtype).requires_grad_()
    ref = feat.detach().clone().requires_grad_()
    out = pool(feat, index, "max")
    want = torch.segment_reduce(ref[index.sort], "max", offsets=index.ptr, axis=0)
    assert torch.equal(out, want)
    grad = torch.randn_like(out)
    assert torch.equal(*torch.autograd.grad([out, want], [feat, ref], [grad, grad]))
    coarse = torch.randn(index.num_coarse, channels, device=cuda).to(dtype).requires_grad_()
    assert torch.equal(unpool(coarse, index), coarse[index.inverse])


@pytest.mark.gpu
def test_cuda_backward_is_deterministic_and_matches_cpu(cuda, synthetic_pack):
    pack = synthetic_pack([4000, 3000], extent=40, seed=9, device=cuda)
    grid = point_grid(pack)
    l1 = grid.downsample(2)
    l2 = l1.downsample(2)  # grouped along "z"
    gen = torch.Generator().manual_seed(10)
    feat0 = torch.randn(grid.num_rows, 32, generator=gen)
    grads = []
    for device in ("cuda", "cuda", "cpu"):
        i1, i2 = (lvl.pool for lvl in (l1, l2))
        assert i1 is not None and i2 is not None
        if device == "cpu":
            i1, i2 = (_to(i, "cpu") for i in (i1, i2))
        feat = feat0.to(device, torch.bfloat16).requires_grad_()
        x1 = pool(feat, i1, "max")
        x2 = pool(x1, i2, "mean")
        out = unpool(unpool(x2, i2) + x1, i1)
        (grad,) = torch.autograd.grad(out.float().square().sum(), feat)
        grads.append((out.detach().cpu(), grad.cpu()))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.equal(grads[0][1], grads[1][1])
    torch.testing.assert_close(grads[0][0], grads[2][0])
    torch.testing.assert_close(grads[0][1], grads[2][1], rtol=2e-2, atol=2e-2)


def _to(index: PoolIndex, device: str) -> PoolIndex:
    return PoolIndex(
        index.inverse.to(device),
        index.sort.to(device),
        index.ptr.to(device),
        index.head.to(device),
        index.grouped,
    )


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode")
def test_cuda_pool_unpool_are_sync_free(cuda, synthetic_pack):
    pack = synthetic_pack([2000, 1500], extent=32, seed=11, device=cuda)
    l1 = point_grid(pack).downsample(2)
    l2 = l1.downsample(2)
    i1, i2 = l1.pool, l2.pool
    assert i1 is not None and i2 is not None
    feat = torch.randn(3500, 32, device=cuda, dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Synchronization debug mode")
        torch.cuda.set_sync_debug_mode("error")
    try:
        x1 = pool(feat, i1, "max")
        out = unpool(unpool(pool(x1, i2, "mean"), i2) + x1, i1)
        out.float().sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode(0)


# ---- capture (D15: the feature network must compile and export)


def test_pool_unpool_compile_fullgraph(synthetic_pack):
    pack = synthetic_pack([300, 200], extent=12, seed=12)
    l1 = point_grid(pack).downsample(2)
    l2 = l1.downsample(2)
    i1, i2 = l1.pool, l2.pool
    assert i1 is not None and i2 is not None

    def features(x: Tensor, a: PoolIndex, b: PoolIndex) -> Tensor:
        y = pool(x, a, "max")
        return unpool(unpool(pool(y, b, "mean"), b) + y, a)

    x = torch.randn(500, 8, requires_grad=True)
    compiled = torch.compile(features, fullgraph=True, backend="eager")
    out = compiled(x, i1, i2)
    assert torch.equal(out, features(x, i1, i2))
    (grad,) = torch.autograd.grad(out.sum(), x)
    (want,) = torch.autograd.grad(features(x, i1, i2).sum(), x)
    assert torch.equal(grad, want)


def test_pool_unpool_export(synthetic_pack):
    pack = synthetic_pack([300, 200], extent=12, seed=13)
    index = point_grid(pack).downsample(2).pool
    assert index is not None

    class Features(torch.nn.Module):
        def forward(self, x, inverse, sort, ptr, head):
            i = PoolIndex(inverse=inverse, sort=sort, ptr=ptr, head=head)
            return unpool(pool(x, i, "max"), i) + x

    n, m = torch.export.Dim("n", min=2), torch.export.Dim("m", min=2)
    args = (torch.randn(500, 8), index.inverse, index.sort, index.ptr, index.head)
    shapes = ({0: n}, {0: n}, {0: n}, {0: m + 1}, {0: m})
    program = torch.export.export(Features(), args, dynamic_shapes=shapes)
    assert torch.equal(program.module()(*args), Features()(*args))


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("reduce", ["max", "mean", "min", "sum"])
def test_mps_scatter_path_matches_segment_reduce(reduce, grouped):
    """``pool`` on MPS uses ``_pool_scatter`` (no ``segment_reduce`` there); run it on the CPU."""
    from point_transformer_v3.ops.pooling import _pool_scatter

    gen = torch.Generator().manual_seed(5)
    cluster = torch.unique(torch.randint(60, (500,), generator=gen), return_inverse=True)[1]
    if grouped:
        cluster = cluster.sort().values
    index = index_from_clusters(cluster)
    if grouped:
        index = PoolIndex(index.inverse, index.sort, index.ptr, index.head, grouped=True)
    feat = torch.randn(500, 2, 6, generator=gen).round(decimals=1)  # ties for max / min
    feat.requires_grad_()
    want = pool(feat, index, reduce)
    got = _pool_scatter(feat, index, reduce)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    grad = torch.randn(want.shape, generator=gen)
    (g_got,) = torch.autograd.grad(got, feat, grad)
    if reduce in ("max", "min"):  # ties share evenly (segment_reduce does so only for g > 0)
        tie = feat.detach() == want.detach()[cluster]
        ties = torch.zeros_like(want).index_add_(0, cluster, tie.float())
        g_want = tie * (grad / ties)[cluster]
    else:
        (g_want,) = torch.autograd.grad(want, feat, grad)
    torch.testing.assert_close(g_got, g_want, rtol=1e-6, atol=1e-6)
    half = _pool_scatter(feat.detach().bfloat16(), index, reduce)
    assert half.dtype == torch.bfloat16
    torch.testing.assert_close(half.float(), want.detach(), rtol=2e-2, atol=2e-2)
