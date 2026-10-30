# Adapted from Pointcept2 tests/ops/test_serialized_attention.py @ 771a949.
"""serialized_attention and the triton_d16 backend (ops/attention.py, ops/_attention_triton.py).

CPU: the kernels in the Triton interpreter (a subprocess running this file: ``TRITON_INTERPRET=1``
must be set before triton is imported) in fp32 against an fp64 reference built from
``build_patches``' packed layout (not from the range table), both call paths (the eager
``SerializedAttentionFunction`` and the op), small and the released tile configs; the patch table
against the packed layout; the backend table's choices, forcing and errors; the CPU path equal to
the permute + SDPA path bit for bit; the ``PTV3_DEBUG`` checks of rows and table. CUDA (``gpu``):
the same cases in bf16 and fp16 on tiny inputs, eager vs op bit for bit, a bitwise repeatable
backward, more patches than CUDA's 65535 grid rows, ``auto`` against ``varlen``, sync-freedom,
packing invariance (C15), fallback warnings, and torch.compile / torch.export keeping the op (D15).

    TRITON_INTERPRET=1 python -m tests.ops.test_serialized_attention
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import os
import subprocess
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pointtransformerv3.ops import attention
from pointtransformerv3.ops.attention import (
    get_attention_backend,
    serialized_attention,
    set_attention_backend,
    varlen_attention,
)
from pointtransformerv3.ops.gather import gather_rows, permute_rows
from pointtransformerv3.ops.patches import PatchLayout, build_patches

REPO = Path(__file__).resolve().parents[2]
# (sample sizes, patch size, tail): overlap and shrink tails, an empty sample, samples shorter
# than a patch, a tail starting inside a tile, one short sample
CASES = [
    ((70, 20, 0, 32, 45), 32, "overlap"),
    ((70, 20, 0, 32, 45), 32, "shrink"),
    ((100, 33), 32, "overlap"),
    ((5,), 32, "overlap"),
]
# max error over the reference's largest magnitude: (out, gradients)
TOLERANCES = {
    torch.float32: (5e-6, 5e-6),
    torch.bfloat16: (1e-2, 1.5e-2),
    torch.float16: (1.5e-3, 2e-3),
}


def make(seq_lens, heads: int, dim: int, dtype: torch.dtype, device="cpu", seed: int = 0):
    """qkv ``(T, 3C)`` rows of the level, an upstream gradient ``(T, C)`` and a random row order."""
    gen = torch.Generator().manual_seed(seed)
    t, c = sum(seq_lens), heads * dim
    qkv = (torch.randn(t, 3 * c, generator=gen) * 1.5).to(device, dtype)
    grad = torch.randn(t, c, generator=gen).to(device, dtype)
    order = torch.randperm(t, generator=gen).to(device)
    return qkv, grad, order


def reference(x: Tensor, order: Tensor, layout: PatchLayout, heads: int, scale: float):
    """fp64 out (rows of the level) per patch with plain softmax, from the packed layout; x is
    the fp64 CPU qkv."""
    t, c3 = x.shape
    c = c3 // 3
    rows = order.cpu()
    q_rows, kv_rows = rows[layout.q_pos.cpu()], rows[layout.kv_pos.cpu()]
    q = x[q_rows, :c].unflatten(1, (heads, -1))
    k = x[kv_rows, c : 2 * c].unflatten(1, (heads, -1))
    v = x[kv_rows, 2 * c :].unflatten(1, (heads, -1))
    outs = []
    cu_q, cu_k = layout.cu_q.tolist(), layout.cu_k.tolist()
    for i in range(len(cu_q) - 1):
        qi, ki, vi = q[cu_q[i] : cu_q[i + 1]], k[cu_k[i] : cu_k[i + 1]], v[cu_k[i] : cu_k[i + 1]]
        a = (torch.einsum("qhd,khd->hqk", qi, ki) * scale).softmax(-1)
        outs.append(torch.einsum("hqk,khd->qhd", a, vi))
    out_pos = torch.cat(outs).flatten(1) if outs else x.new_zeros(0, c)
    return torch.zeros(t, c, dtype=torch.float64).index_copy(0, q_rows, out_pos)


def reference_grads(qkv, grad, order, layout, heads, scale) -> tuple[Tensor, Tensor]:
    x = qkv.detach().double().cpu().requires_grad_()
    out = reference(x, order, layout, heads, scale)
    (dx,) = torch.autograd.grad(out, x, grad.double().cpu())
    return out.detach(), dx


def rel(a: Tensor, b: Tensor) -> float:
    if b.numel() == 0:
        return 0.0
    err = (a.detach().double().cpu() - b).abs().max()
    return float(err / b.abs().max().clamp(min=1e-6))


def run_case(seq_lens, k, tail, heads, dim, dtype, device, path, configs=None) -> bool:
    """One case through ``path`` ("eager" Function or "op"); prints one line, returns ok."""
    from pointtransformerv3.ops import _attention_triton as kernels

    saved = kernels._FWD, kernels._DQ, kernels._DKDV
    if configs is not None:
        kernels._FWD, kernels._DQ, kernels._DKDV = configs
    try:
        qkv, grad, order = make(seq_lens, heads, dim, dtype, device, seed=sum(seq_lens) + dim)
        layout = build_patches(seq_lens, k, tail, device)
        scale = dim**-0.5
        x = qkv.clone().requires_grad_()
        args = (x, order, layout.table, layout.max_q, heads, scale)
        if path == "eager":
            out = kernels.SerializedAttentionFunction.apply(*args)
        else:
            out = torch.ops.pointtransformerv3.serialized_attention(*args)[0]
        (dx,) = torch.autograd.grad(out, x, grad)
    finally:
        kernels._FWD, kernels._DQ, kernels._DKDV = saved
    ref_out, ref_dx = reference_grads(qkv, grad, order, layout, heads, scale)
    c = heads * dim
    errs = {"out": rel(out, ref_out)}
    errs |= {n: rel(dx[:, i * c : (i + 1) * c], ref_dx[:, i * c : (i + 1) * c])
             for i, n in enumerate(("dq", "dk", "dv"))}  # fmt: skip
    tol_out, tol_grad = TOLERANCES[dtype]
    ok = errs["out"] <= tol_out and all(errs[n] <= tol_grad for n in ("dq", "dk", "dv"))
    ok = ok and out.dtype == dtype and dx.dtype == dtype
    report = " ".join(f"{n}={e:.1e}" for n, e in errs.items())
    name = str(dtype).removeprefix("torch.")
    print(f"{'OK ' if ok else 'BAD'} {device} {name:8s} {path:5s} {seq_lens} K={k} {tail:7s} "
          f"H{heads} D{dim} {report}", flush=True)  # fmt: skip
    return ok


def small_configs():
    from pointtransformerv3.ops._attention_triton import LaunchConfig

    small = LaunchConfig(16, 16, 4, 1)
    return small, small, small


def interpret_main() -> int:
    if os.environ.get("TRITON_INTERPRET") != "1":
        print("set TRITON_INTERPRET=1 before running this module (CPU interpreter)")
        return 2
    small = small_configs()
    results = [run_case(lens, k, tail, 2, 16, torch.float32, "cpu", path, small)
               for lens, k, tail in CASES for path in ("eager", "op")]  # fmt: skip
    results += [run_case(CASES[0][0], 32, "overlap", 2, d, torch.float32, "cpu", "eager", small)
                for d in (32, 64)]  # fmt: skip
    # padded head blocks (Utonia: 18 in 32), through both paths and the released tiles
    results += [run_case(lens, k, tail, 3, 18, torch.float32, "cpu", path, small)
                for lens, k, tail in CASES[:2] for path in ("eager", "op")]  # fmt: skip
    results += [run_case(CASES[0][0], 32, "overlap", 2, d, torch.float32, "cpu", "eager", small)
                for d in (6, 24, 40)]  # fmt: skip
    results += [run_case((300, 140), 128, "overlap", 2, 18, torch.float32, "cpu", "eager")]
    # the released tiles: an overlap tail whose keys start inside the previous patch's tile
    results += [run_case((300, 140), 128, tail, 2, 16, torch.float32, "cpu", "eager")
                for tail in ("overlap", "shrink")]  # fmt: skip
    print("ALL OK" if all(results) else "FAILURES", flush=True)
    return 0 if all(results) else 1


@pytest.fixture(autouse=True)
def restore_state() -> Iterator[None]:
    saved = attention._backend
    yield
    attention._backend = saved
    attention._noted.clear()


def old_path(x, qkv, order, inverse, layout, heads, scale, in_order=False) -> Tensor:
    """SerializedAttention's attention before R1: permute, qkv, gather tails, varlen_attention,
    permute back."""
    xs = x if in_order else permute_rows(x, order)
    q, k, v = qkv(xs).unflatten(-1, (3, heads, -1)).unbind(1)
    if layout.kv_pos is not layout.q_pos:
        k, v = gather_rows(k, layout.kv_pos), gather_rows(v, layout.kv_pos)
    cu_q, cu_k, max_q, max_k = layout.cu_q, layout.cu_k, layout.max_q, layout.max_k
    out = varlen_attention(q, k, v, cu_q, cu_k, max_q, max_k, scale=scale).flatten(1)
    return out if in_order else permute_rows(out, inverse)


def inputs(seq_lens, k, tail, heads, dim, device, seed=0):
    """x (N, C), a qkv Linear, order and inverse, the layout."""
    gen = torch.Generator().manual_seed(seed)
    c = heads * dim
    n = sum(seq_lens)
    x = torch.randn(n, c, generator=gen).to(device)
    torch.manual_seed(seed)
    qkv = nn.Linear(c, 3 * c).to(device)
    order = torch.randperm(n, generator=gen).to(device)
    inverse = torch.empty_like(order).scatter_(0, order, torch.arange(n, device=device))
    return x, qkv, order, inverse, build_patches(seq_lens, k, tail, device)


# ---------------------------------------------------------------- CPU


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton is not installed")
def test_kernels_in_interpreter():
    """Every case in the Triton interpreter (CPU, fp32) against fp64, through both call paths."""
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tests.ops.test_serialized_attention"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0 and "ALL OK" in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("tail", ["overlap", "shrink"])
@pytest.mark.parametrize("k", [1, 4, 7, 32])
def test_table_matches_the_packed_layout(k, tail):
    """``PatchLayout.table`` holds the packed layout's runs as ranges (and the sentinel)."""
    gen = torch.Generator().manual_seed(k)
    seq_lens = (*torch.randint(0, 3 * k + 2, (9,), generator=gen).tolist(), 0, k, k + 1)
    layout = build_patches(seq_lens, k, tail)
    t = sum(seq_lens)
    tab = layout.table
    assert tab.dtype == torch.int32 and tab.shape == (layout.cu_q.shape[0], 4)
    assert tab[-1].tolist() == [t, 0, t, 0]
    q_start, q_len, k_start, k_len = tab[:-1].unbind(1)
    assert torch.equal(q_start, layout.cu_q[:-1]) and torch.equal(q_len, layout.cu_q.diff())
    assert torch.equal(k_len, layout.cu_k.diff())
    keys = [torch.arange(s, s + n) for s, n in zip(k_start.tolist(), k_len.tolist(), strict=True)]
    assert torch.equal(torch.cat(keys) if keys else torch.empty(0, dtype=torch.long), layout.kv_pos)
    attention._check_rows_and_table(torch.randperm(t), tab, layout.max_q)  # passes its checks


def test_cpu_takes_sdpa_bitwise_equal_to_the_old_path(caplog):
    """D22: on the CPU ``auto`` takes padded SDPA through the same permute + gather path as before
    R1, bit for bit (fp32 and bf16 autocast), logged once; the row-order skip too."""
    set_attention_backend("auto")
    x, qkv, order, inverse, layout = inputs((70, 20, 0, 45), 16, "overlap", 3, 16, "cpu")
    args = (x, qkv, order, inverse, layout)
    kwargs = {"num_heads": 3, "head_dim": 16, "scale": 0.25}
    with caplog.at_level(logging.INFO, logger=attention.__name__):
        got = [serialized_attention(*args, **kwargs) for _ in range(2)]
    assert torch.equal(got[0], old_path(*args, 3, 0.25)) and torch.equal(got[0], got[1])
    messages = [r.getMessage() for r in caplog.records if "serialized_attention" in r.getMessage()]
    assert len(messages) == 1 and "sdpa backend" in messages[0], messages
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = serialized_attention(*args, **kwargs)
        assert got.dtype == torch.bfloat16 and torch.equal(got, old_path(*args, 3, 0.25))
    ident = torch.arange(x.shape[0])
    got = serialized_attention(x, qkv, ident, ident, layout, **kwargs, in_order=True)
    assert torch.equal(got, old_path(x, qkv, ident, ident, layout, 3, 0.25, in_order=True))


def test_forcing_and_errors():
    x, qkv, order, inverse, layout = inputs((40,), 16, "overlap", 2, 16, "cpu")
    kwargs = {"num_heads": 2, "head_dim": 16, "scale": 0.25}
    with pytest.raises(ValueError, match="unknown attention backend"):
        set_attention_backend("flash")  # pyright: ignore[reportArgumentType]
    for name in ("triton_d16", "varlen"):
        set_attention_backend(name)  # pyright: ignore[reportArgumentType]
        assert get_attention_backend() == name
        with pytest.raises(ValueError, match=f"'{name}' needs CUDA"):
            serialized_attention(x, qkv, order, inverse, layout, **kwargs)
    set_attention_backend("triton_d16")
    q = torch.randn(4, 1, 8)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="takes packed q, k, v"):
        varlen_attention(q, q, q, cu, cu, 4, 4, scale=1.0)
    set_attention_backend("sdpa")
    assert serialized_attention(x, qkv, order, inverse, layout, **kwargs).shape == (40, 32)
    with pytest.raises(TypeError, match="int64"):
        serialized_attention(x, qkv, order.int(), inverse, layout, **kwargs)
    with pytest.raises(ValueError, match="cover 40 positions but x has 39"):
        serialized_attention(x[:39], qkv, order[:39], inverse[:39], layout, **kwargs)
    with pytest.raises(TypeError, match="head_dim must be a positive host int"):
        serialized_attention(x, qkv, order, inverse, layout, num_heads=2, head_dim=0, scale=1.0)


def test_debug_checks_reject_bad_rows_and_tables():
    """PTV3_DEBUG=1 checks (run before any launch): rows must be a permutation, the table's runs
    must tile the positions, and keys may reach back only into the previous patch."""
    layout = build_patches((10, 7), 4, "overlap")
    good = torch.randperm(17)
    check = attention._check_rows_and_table
    check(good, layout.table, layout.max_q)
    bad_rows = good.clone()
    bad_rows[3] = 17
    with pytest.raises(ValueError, match="permutation"):
        check(bad_rows, layout.table, layout.max_q)
    with pytest.raises(ValueError, match="permutation"):
        check(torch.zeros(17, dtype=torch.long), layout.table, layout.max_q)
    with pytest.raises(ValueError, match="max_q=3"):
        check(good, layout.table, 3)
    with pytest.raises(ValueError, match=r"order with unit stride, got stride \(2,\)"):
        check(torch.stack([good, good], 1)[:, 0], layout.table, layout.max_q)
    with pytest.raises(ValueError, match=r"table with unit stride, got stride \(8, 1\)"):
        check(good, torch.cat([layout.table, layout.table], 1)[:, :4], layout.max_q)
    for row, col, value, match in [
        (-1, 0, 16, "sentinel"),
        (1, 1, 3, "tile"),
        (2, 2, 0, "outside its own and the previous"),  # a tail reaching back two patches
        (0, 3, 6, "outside its own"),  # keys past the patch's own end
        (3, 3, 0, "no keys"),
    ]:
        table = layout.table.clone()
        table[row, col] = value
        with pytest.raises(ValueError, match=match):
            check(good, table, layout.max_q)


def test_table_metadata_is_checked_on_every_call(monkeypatch):
    """The patch table's dtype, shape and row count and max_q are host metadata, checked without
    PTV3_DEBUG (the kernels would read an int64 table as int32 pairs)."""
    monkeypatch.delenv("PTV3_DEBUG", raising=False)
    set_attention_backend("sdpa")
    x, qkv, order, inverse, layout = inputs((40, 9), 16, "overlap", 2, 16, "cpu")
    kwargs = {"num_heads": 2, "head_dim": 16, "scale": 0.25}
    other = build_patches((49,), 7, "overlap")  # 7 patches, layout has 4
    for table, error, match in [
        (layout.table.long(), TypeError, r"\(P \+ 1, 4\) int32, got torch.int64"),
        (layout.table[:, :3], TypeError, r"\(P \+ 1, 4\) int32, got torch.int32 \(5, 3\)"),
        (other.table, ValueError, "has 8 rows but cu_q describes 4 patches"),
    ]:
        bad = dataclasses.replace(layout, table=table)
        with pytest.raises(error, match=match):
            serialized_attention(x, qkv, order, inverse, bad, **kwargs)
    bad = dataclasses.replace(layout, max_q=torch.tensor(16))
    with pytest.raises(TypeError, match="max_q must be a host int"):
        serialized_attention(x, qkv, order, inverse, bad, **kwargs)


def test_empty_level_keeps_the_graph():
    """T = 0 on the packed backends: an empty output inside q's, k's and v's autograd graph, so
    a backward runs (zero gradients)."""
    set_attention_backend("sdpa")
    x, qkv, order, inverse, layout = inputs((0,), 16, "overlap", 2, 16, "cpu")
    x.requires_grad_()
    out = serialized_attention(x, qkv, order, inverse, layout, num_heads=2, head_dim=16, scale=0.25)
    assert out.shape == (0, 32)
    out.sum().backward()
    assert x.grad is not None and x.grad.shape == (0, 32)
    assert qkv.weight.grad is not None and not qkv.weight.grad.any()
    q = torch.randn(0, 2, 8, requires_grad=True)
    k, v = (torch.randn(3, 2, 8, requires_grad=True) for _ in range(2))
    cu_q, cu_k = torch.tensor([0, 0], dtype=torch.int32), torch.tensor([0, 3], dtype=torch.int32)
    out = varlen_attention(q, k, v, cu_q, cu_k, 0, 3, scale=1.0)
    grads = torch.autograd.grad(out.sum(), (q, k, v))
    assert out.shape == (0, 2, 8)
    for g, t in zip(grads, (q, k, v), strict=True):
        assert g.shape == t.shape and not g.any()


# ---------------------------------------------------------------- CUDA


@pytest.fixture
def triton_cuda() -> None:
    from pointtransformerv3.ops._triton import MIN_CAPABILITY, toolchain_problem

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    problem = toolchain_problem(torch.device("cuda"), min_capability=MIN_CAPABILITY)
    if problem is not None:
        pytest.skip(f"triton unavailable: {problem}")
    set_attention_backend("auto")
    attention._choose(torch.device("cuda"), torch.bfloat16, 16, "serialized_attention", False)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tiny_cuda_against_fp64(triton_cuda, dtype):
    small = small_configs()
    results = [run_case(lens, k, tail, 2, 16, dtype, "cuda", path, small)
               for lens, k, tail in CASES for path in ("eager", "op")]  # fmt: skip
    results += [run_case(lens, k, tail, 2, 16, dtype, "cuda", "eager") for lens, k, tail in CASES]
    results += [run_case(CASES[0][0], 32, "overlap", 2, d, dtype, "cuda", "eager", small)
                for d in (32, 64)]  # fmt: skip
    results += [run_case(CASES[0][0], 32, "overlap", 3, d, dtype, "cuda", path, small)
                for d in (18, 24, 40) for path in ("eager", "op")]  # fmt: skip
    lens = (3000, 1024, 517, 2100)  # the released patch size and tiles on tiny inputs
    heads_dims = ((4, 16), (2, 32), (1, 64), (3, 18))
    results += [run_case(lens, 1024, tail, h, d, dtype, "cuda", "eager")
                for tail in ("overlap", "shrink") for h, d in heads_dims]  # fmt: skip
    assert all(results)


def d16_step(qkv: Tensor, grad: Tensor, order: Tensor, layout: PatchLayout, heads: int,
             path: str = "eager") -> tuple[Tensor, Tensor]:  # fmt: skip
    from pointtransformerv3.ops import _attention_triton as kernels

    x = qkv.clone().requires_grad_()
    args = (x, order, layout.table, layout.max_q, heads, (qkv.shape[1] // 3 // heads) ** -0.5)
    if path == "eager":
        out = kernels.SerializedAttentionFunction.apply(*args)
    else:
        out = torch.ops.pointtransformerv3.serialized_attention(*args)[0]
    (dx,) = torch.autograd.grad(out, x, grad)
    return out, dx


@pytest.mark.gpu
def test_eager_is_the_op_deterministic_and_sync_free(triton_cuda):
    """I5: the eager Function and the op give the same bits; the backward has one writer per
    row, so two runs give the same bits; no host sync."""
    seq_lens = (5000, 3100, 900, 2048)
    qkv, grad, order = make(seq_lens, 4, 16, torch.bfloat16, "cuda", seed=1)
    layout = build_patches(seq_lens, 1024, "overlap", "cuda")
    first = d16_step(qkv, grad, order, layout, 4)
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Synchronization debug mode")
        torch.cuda.set_sync_debug_mode("error")
    try:
        second = d16_step(qkv, grad, order, layout, 4)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    op = d16_step(qkv, grad, order, layout, 4, "op")
    for a, b, c in zip(first, second, op, strict=True):
        assert torch.equal(a, b) and torch.equal(a, c)


@pytest.mark.gpu
def test_more_patches_than_a_grid_axis_holds(triton_cuda):
    """70k patches (CUDA caps grid axes 1 and 2 at 65535): the patch index shares axis 0."""
    seq_lens = (3,) * 35_000
    qkv, grad, order = make(seq_lens, 1, 16, torch.bfloat16, "cuda", seed=2)
    layout = build_patches(seq_lens, 2, "overlap", "cuda")
    assert layout.table.shape[0] - 1 == 70_000
    out, dx = d16_step(qkv, grad, order, layout, 1)
    # reference: fp32 padded SDPA on the packed layout (a per-patch loop would take minutes)
    x = qkv.float().requires_grad_()
    q, k, v = x[order].unflatten(1, (3, 1, 16)).unbind(1)
    k, v = k[layout.kv_pos], v[layout.kv_pos]
    set_attention_backend("sdpa")
    ref = varlen_attention(q, k, v, layout.cu_q, layout.cu_k, 2, 2, scale=0.25).flatten(1)
    ref = torch.empty_like(ref).index_copy(0, order, ref)
    (ref_dx,) = torch.autograd.grad(ref, x, grad.float())
    assert rel(out, ref.double().cpu()) <= 1e-2 and rel(dx, ref_dx.double().cpu()) <= 1.5e-2


def cosine(a: Tensor, b: Tensor) -> float:
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


@pytest.mark.gpu
@pytest.mark.parametrize("tail", ["overlap", "shrink"])
def test_auto_takes_triton_d16_close_to_varlen(triton_cuda, tail, caplog):
    """Under bf16 autocast ``auto`` takes triton_d16 (logged once); outputs and the gradients of
    x and the qkv weight agree with the varlen path to bf16 rounding (D29)."""
    from pointtransformerv3.ops import _attention_triton as kernels

    x, qkv, order, inverse, layout = inputs((5000, 3100, 900, 2048), 1024, tail, 4, 16, "cuda")
    grad = torch.randn(x.shape[0], 64, device="cuda")
    kwargs = {"num_heads": 4, "head_dim": 16, "scale": 0.25}
    calls = []
    real = kernels.SerializedAttentionFunction.apply

    def spy(*args):
        calls.append(args[0].shape)
        return real(*args)

    def run(name: str) -> tuple[Tensor, ...]:
        set_attention_backend(name)  # pyright: ignore[reportArgumentType]
        xi = x.clone().requires_grad_()
        qkv.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = serialized_attention(xi, qkv, order, inverse, layout, **kwargs)
        out.backward(grad.to(out.dtype))
        assert qkv.weight.grad is not None and xi.grad is not None
        return out.detach(), xi.grad, qkv.weight.grad.clone()

    kernels.SerializedAttentionFunction.apply = spy
    try:
        with caplog.at_level(logging.INFO, logger=attention.__name__):
            got = run("auto")
    finally:
        kernels.SerializedAttentionFunction.apply = real
    assert calls == [(x.shape[0], 192)]
    assert any("triton_d16 backend" in r.getMessage() for r in caplog.records)
    want = run("varlen")
    assert got[0].dtype == torch.bfloat16
    for a, b in zip(got, want, strict=True):
        assert cosine(a, b) >= 0.9999 and rel(a, b.double().cpu()) <= 3e-2


@pytest.mark.gpu
def test_packing_invariance(triton_cuda):
    """C15: patches are independent, so a pack's output and gradients are the concatenation of
    its parts' bit for bit."""
    lens_a, lens_b = (3000, 517), (2100, 40)
    qkv, grad, _ = make(lens_a + lens_b, 2, 16, torch.bfloat16, "cuda", seed=3)
    n_a = sum(lens_a)
    order_a = torch.randperm(n_a, device="cuda")
    order_b = torch.randperm(sum(lens_b), device="cuda")

    def patches(lens: tuple[int, ...]) -> PatchLayout:
        return build_patches(lens, 1024, "overlap", "cuda")

    both = d16_step(qkv, grad, torch.cat([order_a, order_b + n_a]), patches(lens_a + lens_b), 2)
    part_a = d16_step(qkv[:n_a], grad[:n_a], order_a, patches(lens_a), 2)
    part_b = d16_step(qkv[n_a:], grad[n_a:], order_b, patches(lens_b), 2)
    for whole, a, b in zip(both, part_a, part_b, strict=True):
        assert torch.equal(whole, torch.cat([a, b]))


@pytest.mark.gpu
def test_strided_order_and_table(triton_cuda, monkeypatch):
    """The kernels read the order and the table with unit stride; serialized_attention makes
    strided views contiguous (equal bits), also under PTV3_DEBUG=1. The views' neighbors are
    valid rows and table entries, so a regression gives wrong values, never an out-of-range
    read."""
    x, qkv, order, inverse, layout = inputs((3000, 517, 1024), 1024, "overlap", 2, 16, "cuda")
    kwargs = {"num_heads": 2, "head_dim": 16, "scale": 0.25}
    strided_order = torch.stack([order, order.flip(0)], 1)[:, 0]
    table = torch.cat([layout.table, layout.table.flip(0)], 1)[:, :4]
    assert strided_order.stride() == (2,) and table.stride() == (8, 1)
    strided = dataclasses.replace(layout, table=table)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        want = serialized_attention(x, qkv, order, inverse, layout, **kwargs)
        for debug in ("0", "1"):
            monkeypatch.setenv("PTV3_DEBUG", debug)
            for o, lay in ((strided_order, layout), (order, strided), (strided_order, strided)):
                assert torch.equal(serialized_attention(x, qkv, o, inverse, lay, **kwargs), want)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["triton_d16", "varlen", "sdpa"])
def test_empty_level_backward_on_cuda(triton_cuda, backend):
    """T = 0 on every CUDA backend: forward and backward run, zero gradients."""
    set_attention_backend(backend)
    x, qkv, order, inverse, layout = inputs((0,), 16, "overlap", 2, 16, "cuda")
    x.requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = serialized_attention(x, qkv, order, inverse, layout, num_heads=2, head_dim=16,
                                   scale=0.25)  # fmt: skip
    assert out.shape == (0, 32) and out.dtype == torch.bfloat16
    out.float().sum().backward()
    assert qkv.weight.grad is not None and not qkv.weight.grad.any()


@pytest.mark.gpu
def test_cuda_fallbacks_warn_once(triton_cuda, monkeypatch):
    """C18: on CUDA, ``auto`` gives a call it cannot run on triton_d16 to the next backend with
    one warning naming the reason: head_dim 72 -> varlen, fp32 -> padded SDPA, a GPU below
    sm_80 (simulated) -> padded SDPA; forcing triton_d16 there raises."""
    monkeypatch.setattr(attention, "_noted", set())
    cases = [(1, 72, torch.bfloat16, "varlen_attn", "head_dim <= 64, got 72"),
             (2, 16, None, "padded SDPA", "needs bf16/fp16 inputs, got torch.float32")]  # fmt: skip
    for heads, dim, autocast_dtype, label, reason in cases:
        x, qkv, order, inverse, layout = inputs((300, 90), 64, "overlap", heads, dim, "cuda")
        kwargs = {"num_heads": heads, "head_dim": dim, "scale": dim**-0.5}
        with torch.autocast("cuda", dtype=autocast_dtype or torch.float16,
                            enabled=autocast_dtype is not None):  # fmt: skip
            with pytest.warns(UserWarning, match=f"falls back to {label} on CUDA") as caught:
                out = serialized_attention(x, qkv, order, inverse, layout, **kwargs)
            assert reason in str(caught[0].message)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                again = serialized_attention(x, qkv, order, inverse, layout, **kwargs)
            assert torch.equal(out, again)
            assert torch.equal(out, old_path(x, qkv, order, inverse, layout, heads, dim**-0.5))
    old = "cuda:0 (Some GPU) is sm_75, but the triton kernels need compute capability 8.0+"
    monkeypatch.setattr(attention, "toolchain_problem", lambda device, min_capability: old)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: (7, 5))
    monkeypatch.setattr(attention, "_capability_problems", {})
    x, qkv, order, inverse, layout = inputs((300, 90), 64, "overlap", 2, 16, "cuda")
    kwargs = {"num_heads": 2, "head_dim": 16, "scale": 0.25}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with pytest.warns(UserWarning, match="falls back to padded SDPA") as caught:
            serialized_attention(x, qkv, order, inverse, layout, **kwargs)
        assert "triton_d16 cannot run here" in str(caught[0].message)
        assert "varlen needs compute capability 8.0+" in str(caught[0].message)
        set_attention_backend("triton_d16")
        with pytest.raises(ValueError, match=r"'triton_d16' cannot run here.*sm_75"):
            serialized_attention(x, qkv, order, inverse, layout, **kwargs)


class AttentionBlock(nn.Module):
    """qkv -> serialized_attention -> proj on one level, the order and patches as inputs."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.qkv, self.proj = nn.Linear(channels, 3 * channels), nn.Linear(channels, channels)
        self.heads, self.head_dim = heads, channels // heads

    def forward(self, x: Tensor, order: Tensor, inverse: Tensor, layout: PatchLayout) -> Tensor:
        d = self.head_dim
        out = serialized_attention(
            x, self.qkv, order, inverse, layout, num_heads=self.heads, head_dim=d, scale=d**-0.5
        )
        return self.proj(out)


@pytest.mark.gpu
def test_compile_and_export_keep_the_op(triton_cuda):
    """D15: torch.compile (fullgraph, dynamic rows, compile-time autotune off) of an attention
    block matches eager bit for bit, forward and backward, on two levels; strict and non-strict
    torch.export record ``pointtransformerv3.serialized_attention`` and match eager."""
    import torch._inductor.config as inductor_config

    torch.manual_seed(0)
    block = AttentionBlock(64, 4).cuda()
    levels = []
    for lens in ((5000, 3100, 900), (1500, 700)):
        x, _, order, inverse, layout = inputs(lens, 1024, "overlap", 4, 16, "cuda", seed=len(lens))
        levels.append((x, order, inverse, layout))

    def step(fn, x, order, inverse, layout) -> tuple[Tensor, ...]:
        block.zero_grad()
        xi = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(xi, order, inverse, layout)
        out.float().square().mean().backward()
        assert xi.grad is not None and block.qkv.weight.grad is not None
        return out.detach(), xi.grad, block.qkv.weight.grad.clone()

    torch._dynamo.reset()
    patch = {
        "triton.autotune_at_compile_time": False,
        "max_autotune": False,
        "compile_threads": 1,
        "fx_graph_cache": False,
    }
    try:
        with inductor_config.patch(patch):
            compiled = torch.compile(block, fullgraph=True, dynamic=True)
            for level in levels:
                got, want = step(compiled, *level), step(block, *level)
                for a, b in zip(got, want, strict=True):
                    assert torch.equal(a, b)
    finally:
        torch._dynamo.reset()

    x, order, inverse, layout = levels[0]

    class Exported(nn.Module):  # the layout's tensors as inputs
        def __init__(self) -> None:
            super().__init__()
            self.block = block

        def forward(self, x, order, inverse, q_pos, kv_pos, cu_q, cu_k, table):
            lay = PatchLayout(q_pos, kv_pos, cu_q, cu_k, table, layout.max_q, layout.max_k)
            return self.block(x, order, inverse, lay)

    tensors = (layout.q_pos, layout.kv_pos, layout.cu_q, layout.cu_k, layout.table)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        want = block(x, order, inverse, layout)
        for strict in (False, True):
            exported = torch.export.export(Exported(), (x, order, inverse, *tensors), strict=strict)
            targets = {str(n.target) for n in exported.graph.nodes if n.op == "call_function"}
            assert "pointtransformerv3.serialized_attention.default" in targets, (strict, targets)
            assert torch.equal(exported.module()(x, order, inverse, *tensors), want)
        # strict export with unbounded dynamic sizes: no size check may add a guard on T
        t, tk, p1 = (torch.export.Dim(n, min=2) for n in ("T", "Tk", "P1"))
        shapes = ({0: t}, {0: t}, {0: t}, {0: t}, {0: tk}, {0: p1}, {0: p1}, {0: p1})
        args = (x, order, inverse, *tensors)
        exported = torch.export.export(Exported(), args, dynamic_shapes=shapes, strict=True)
        assert torch.equal(exported.module()(*args), want)


if __name__ == "__main__":
    sys.exit(interpret_main())
