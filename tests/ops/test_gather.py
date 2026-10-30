# Adapted from Pointcept2 tests/ops/test_gather.py @ 771a949.
"""Row gathers (ops/gather.py): ``gather_rows`` and ``permute_rows`` exact against ``x[index]``
(values and grads), the slow-path predicate, the padded source, sync-freedom, capture, and timing
checks on CUDA that the workarounds still beat plain indexing."""

from __future__ import annotations

import statistics
import warnings

import pytest
import torch

from pointtransformerv3.ops.gather import (
    gather_rows,
    gather_source,
    permute_rows,
    slow_gather,
    strided_index,
)
from tests.conftest import DEVICES

DTYPES = [torch.bfloat16, torch.float16, torch.float32, torch.float64]


def make(n: int, shape: tuple[int, ...], dtype: torch.dtype, device, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, *shape, generator=gen).to(device, dtype)


def indices(n: int, device, seed: int = 1) -> dict[str, torch.Tensor]:
    """Permutation, repeats (unpool-like), many repeats (> 4 reads per row: the source-copy
    path), empty."""
    gen = torch.Generator().manual_seed(seed)
    return {
        "perm": torch.randperm(n, generator=gen).to(device),
        "repeats": torch.randint(n, (2 * n + 3,), generator=gen).to(device),
        "many": torch.randint(n, (5 * n,), generator=gen).to(device),
        "empty": torch.zeros(0, dtype=torch.int64, device=device),
    }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(8,), (16,), (32,), (64,), (128,), (3,), (2, 16), (256,)])
def test_gather_rows_equals_indexing(device, dtype, shape):
    x = make(300, shape, dtype, device)
    for name, index in indices(300, device).items():
        for idx in (index, index.int()):
            out = gather_rows(x, idx)
            assert out.dtype == x.dtype and out.is_contiguous(), name
            assert torch.equal(out, x[index]), name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("channels", [16, 32, 64, 128])
def test_gather_rows_grad_equals_indexing(device, channels):
    """Same values and bitwise the same (deterministic) gradient as autograd's ``x[index]``."""
    for name, index in indices(500, device).items():
        for idx in (index, index.int()):
            x = make(500, (channels,), torch.bfloat16, device).requires_grad_()
            ref = x.detach().clone().requires_grad_()
            out, want = gather_rows(x, idx), ref[index]
            assert torch.equal(out, want), name
            grad = make(index.shape[0], (channels,), torch.bfloat16, device, seed=2)
            (gx,) = torch.autograd.grad(out, x, grad)
            (gr,) = torch.autograd.grad(want, ref, grad)
            assert torch.equal(gx, gr), name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(32,), (3, 3, 16), (3,), (256,)])
def test_permute_rows_equals_indexing(device, shape):
    """Values and bitwise the same gradient as autograd's ``x[perm]`` (index_copy_ backward)."""
    perm = indices(400, device)["perm"]
    for idx in (perm, perm.int()):
        x = make(400, shape, torch.bfloat16, device).requires_grad_()
        ref = x.detach().clone().requires_grad_()
        out, want = permute_rows(x, idx), ref[perm]
        assert out.is_contiguous() and torch.equal(out, want)
        grad = make(400, shape, torch.bfloat16, device, seed=2)
        (gx,) = torch.autograd.grad(out, x, grad)
        (gr,) = torch.autograd.grad(want, ref, grad)
        assert torch.equal(gx, gr)
        with torch.no_grad():
            assert torch.equal(permute_rows(x, idx), want)
    view = make(400, (96,), torch.bfloat16, device)[:, 32:64]  # qkv-style view
    assert torch.equal(permute_rows(view, perm), view[perm])
    empty = torch.zeros(0, 4, device=device, requires_grad=True)
    assert permute_rows(empty, perm[:0]).shape == (0, 4)


def test_permute_rows_validation(monkeypatch):
    x = torch.zeros(5, 4)
    with pytest.raises(ValueError, match="one entry per row"):
        permute_rows(x, torch.arange(4))
    with pytest.raises(TypeError, match="int64 or int32"):
        permute_rows(x, torch.arange(5.0))
    monkeypatch.setenv("PTV3_DEBUG", "1")
    with pytest.raises(ValueError, match="permutation"):
        permute_rows(x, torch.tensor([0, 1, 1, 3, 4]))
    assert torch.equal(permute_rows(x, torch.tensor([4, 3, 2, 1, 0])), x.flip(0))


@pytest.mark.parametrize("device", DEVICES)
def test_gather_rows_non_contiguous_sources(device):
    qkv = make(400, (96,), torch.bfloat16, device)
    q = qkv[:, :32]  # row stride 96 elements: the attention q/k/v views
    heads = qkv.view(400, 3, 2, 16)[:, 0]  # (N, H, D) view
    index = indices(400, device)["perm"]
    assert torch.equal(gather_rows(q, index), q[index])
    assert torch.equal(gather_rows(heads, index), heads[index])
    assert torch.equal(gather_rows(qkv, strided_index(index)), qkv[index])


@pytest.mark.parametrize("device", DEVICES)
def test_gather_source(device):
    for dtype in (torch.bfloat16, torch.float32):
        for shape in [(32,), (2, 16), (3,), (256,)]:
            x = make(50, shape, dtype, device)
            src = gather_source(x, zero_rows=2)
            assert src.shape == (52, *shape) and src.dtype == dtype
            assert torch.equal(src[:50], x) and not src[50:].any()
            row = x[0].numel()
            row_bytes = row * x.element_size()
            padded = device == "cuda" and row_bytes % 16 == 0 and row_bytes <= 256
            assert src.stride(0) == row + int(padded), (dtype, shape)
            index = torch.tensor([3, 50, 0, 51, 49], device=device)
            assert torch.equal(src.index_select(0, index), src[index])
    assert gather_source(torch.zeros(0, 4)).shape == (0, 4)
    with pytest.raises(ValueError, match="zero_rows"):
        gather_source(torch.zeros(2, 4), zero_rows=-1)


def test_gather_source_traces_without_tensor_constants():
    """Review F3: the zero rows are a ``zero_()``, not an assignment of a CPU scalar that every
    captured graph would carry as a tensor constant (Inductor built a CPU kernel for it)."""
    from torch.fx.experimental.proxy_tensor import make_fx

    graph = make_fx(lambda x: gather_source(x, zero_rows=2))(torch.randn(5, 4)).graph
    constants = [n.target for n in graph.nodes if n.op == "get_attr"]
    assert not constants, constants
    assert any("zero" in str(n.target) for n in graph.nodes if n.op == "call_function")


def test_strided_index():
    index = torch.arange(10)
    s = strided_index(index)
    assert s.stride() == (2,) and torch.equal(s, index)
    assert strided_index(s) is s
    with pytest.raises(ValueError, match="1-D"):
        strided_index(index.view(2, 5))


@pytest.mark.gpu
def test_slow_gather_predicate(cuda):
    index = torch.arange(4, device=cuda)
    bf16 = torch.zeros(4, 32, device=cuda, dtype=torch.bfloat16)
    assert slow_gather(bf16, index)  # 64-byte rows, contiguous index
    assert not slow_gather(bf16.cpu(), index.cpu())
    assert not slow_gather(bf16, strided_index(index))  # already off the vectorized kernel
    assert not slow_gather(gather_source(bf16), index)  # padded row stride
    assert not slow_gather(torch.zeros(4, 3, device=cuda), index)  # 12-byte rows
    assert not slow_gather(torch.zeros(4, 256, device=cuda, dtype=torch.bfloat16), index)  # 512 B
    assert slow_gather(torch.zeros(4, 64, device=cuda), index)  # fp32, 256 B
    assert not slow_gather(bf16, index[:0])
    qkv = torch.zeros(4, 96, device=cuda, dtype=torch.bfloat16)
    assert slow_gather(qkv[:, 32:64], index)  # view: 192-byte row stride, pointer +64 B
    assert slow_gather(qkv.view(4, 3, 2, 16)[:, 1], index)  # (N, H, D) rows
    assert not slow_gather(qkv[:, 4:36], index)  # pointer +8 B: not 16-byte aligned
    assert not slow_gather(qkv.view(4, 3, 32)[:, :, :16], index)  # rows not one dense block


def test_gather_rows_validation(monkeypatch):
    x = torch.zeros(5, 4)
    with pytest.raises(TypeError, match="int64 or int32"):
        gather_rows(x, torch.zeros(3))
    with pytest.raises(TypeError, match="1-D"):
        gather_rows(x, torch.zeros(3, 1, dtype=torch.int64))
    with pytest.raises(TypeError, match="leading row"):
        gather_rows(torch.tensor(1.0), torch.zeros(1, dtype=torch.int64))
    monkeypatch.setenv("PTV3_DEBUG", "1")
    with pytest.raises(ValueError, match=r"\[0, 5\)"):
        gather_rows(x, torch.tensor([0, 5]))
    with pytest.raises(ValueError, match=r"\[0, 5\)"):
        gather_rows(x, torch.tensor([-1, 2]))


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode")
def test_gather_rows_is_sync_free(cuda):
    x = make(2000, (32,), torch.bfloat16, cuda).requires_grad_()
    perm, many = indices(2000, cuda)["perm"], indices(2000, cuda)["many"]
    gather_rows(x, perm).sum().backward()  # warm-up (lazy init may sync)
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Synchronization debug mode")
        torch.cuda.set_sync_debug_mode("error")
    try:
        out = gather_rows(x, perm) + gather_rows(x, many)[: perm.shape[0]] + permute_rows(x, perm)
        out.float().sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode(0)


@pytest.mark.parametrize("device", DEVICES)
def test_gather_rows_compile_fullgraph(device):
    """Under ``torch.compile`` the plain op is traced (no strided-index copies in the graph)."""
    x = make(300, (32,), torch.bfloat16, device).requires_grad_()
    index = indices(300, device)["repeats"]
    compiled = torch.compile(gather_rows, fullgraph=True, backend="aot_eager")
    out = compiled(x, index)
    assert torch.equal(out, x[index])
    (grad,) = torch.autograd.grad(out.float().sum(), x)
    (want,) = torch.autograd.grad(x[index].float().sum(), x)
    assert torch.equal(grad, want)
    perm = indices(300, device)["perm"]
    out = torch.compile(permute_rows, fullgraph=True, backend="aot_eager")(x, perm)
    assert torch.equal(out, x[perm])
    (grad,) = torch.autograd.grad(out.float().sum(), x)
    (want,) = torch.autograd.grad(x[perm].float().sum(), x)
    assert torch.equal(grad, want)


def test_gather_rows_export():
    class Gather(torch.nn.Module):
        def forward(self, x, index):
            return gather_rows(x, index) * 2

    n, m = torch.export.Dim("n", min=2), torch.export.Dim("m", min=2)
    args = (torch.randn(300, 32), indices(300, "cpu")["repeats"])
    program = torch.export.export(Gather(), args, dynamic_shapes=({0: n}, {0: m}))
    assert torch.equal(program.module()(*args), Gather()(*args))


# ---- the reason this module exists


def _median_ms(fns: list, repeat: int = 15) -> list[float]:
    """Interleaved CUDA-event medians of each fn (one warm-up call each)."""
    for fn in fns:
        fn()
    times: list[list[float]] = [[] for _ in fns]
    for _ in range(repeat):
        for fn, t in zip(fns, times, strict=True):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            t.append(start.elapsed_time(end))
    return [statistics.median(t) for t in times]


# (case, dtype, channels): plain/ours ratios measured at 0.25-0.6 (module doc); the bf16 C=128
# permutation (0.93) is left out, it is too close to 1 to assert on a shared GPU.
BENCH_CASES = [
    ("perm", torch.bfloat16, 32),
    ("perm", torch.bfloat16, 64),
    ("perm", torch.float32, 16),
    ("unpool", torch.bfloat16, 32),
    ("unpool", torch.bfloat16, 64),
    ("unpool", torch.bfloat16, 128),
    ("unpool", torch.float32, 32),
    ("kmap", torch.bfloat16, 32),
    ("kmap", torch.bfloat16, 64),
    ("kmap", torch.bfloat16, 128),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("case", "dtype", "channels"), BENCH_CASES)
def test_gather_beats_plain_indexing(cuda, case, dtype, channels, record_property):
    """torch 2.12's vectorized gather takes ~0.48 ms per 1M rows of 16..256 bytes on sm_120. If a
    torch upgrade fixes it, this fails and ``ops/gather.py`` can go. 1M-row permutation, unpool-like
    gather of 1M rows from 420k (``gather_rows``), and a k=3 kernel-map gather of 2.7M rows from
    100k with an int32 index from a ``gather_source`` copy (the conv path, copy included)."""
    gen = torch.Generator(cuda).manual_seed(0)
    if case == "kmap":
        x = torch.randn(100_000, channels, device=cuda, generator=gen).to(dtype)
        index = torch.randint(100_001, (2_700_000,), device=cuda, generator=gen).int()

        def ours() -> torch.Tensor:
            return gather_source(x, zero_rows=1).index_select(0, index)

        def plain() -> torch.Tensor:
            return torch.cat([x, x.new_zeros(1, channels)]).index_select(0, index)

    else:
        n = 420_000 if case == "unpool" else 1_000_000
        x = torch.randn(n, channels, device=cuda, generator=gen).to(dtype)
        if case == "unpool":
            index = torch.randint(n, (1_000_000,), device=cuda, generator=gen).sort().values
        else:
            index = torch.randperm(n, device=cuda, generator=gen)

        def ours() -> torch.Tensor:
            return gather_rows(x, index)

        def plain() -> torch.Tensor:
            return x[index]

    assert torch.equal(ours(), plain())
    fast, slow = _median_ms([ours, plain])
    record_property("ms", {"ours": round(fast, 3), "plain": round(slow, 3)})
    print(f"{case} {dtype} C={channels}: ours {fast:.3f} ms, plain {slow:.3f} ms")
    assert fast < 0.9 * slow, f"{case}: ours {fast:.3f} ms vs plain {slow:.3f} ms"


@pytest.mark.gpu
def test_permute_rows_backward_beats_indexing(cuda):
    """fwd+bwd of a 1M-row permutation of bf16 qkv rows (C=48: 288 B, plain kernel forward):
    ``permute_rows``' index_copy_ backward against autograd's sorted index_put_ (measured 0.95 vs
    2.17 ms)."""
    gen = torch.Generator(cuda).manual_seed(0)
    x = torch.randn(1_000_000, 3, 3, 16, device=cuda, generator=gen).bfloat16()
    x.requires_grad_()
    perm = torch.randperm(1_000_000, device=cuda, generator=gen)
    grad = torch.randn(x.shape, device=cuda, generator=gen).bfloat16()
    ours, plain = _median_ms(
        [
            lambda: torch.autograd.grad(permute_rows(x, perm), x, grad),
            lambda: torch.autograd.grad(x[perm], x, grad),
        ]
    )
    print(f"permute qkv C=48: ours {ours:.3f} ms, x[perm] {plain:.3f} ms")
    assert ours < 0.8 * plain, f"permute_rows {ours:.3f} ms vs x[perm] {plain:.3f} ms"
