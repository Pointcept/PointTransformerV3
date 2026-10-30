# Adapted from Pointcept2 tests/ops/test_norm_ops.py @ 771a949.
"""ops/norm.py: the Triton LayerNorm (I1) and its torch fallback.

CPU: the kernels in the Triton interpreter (a subprocess running this file, since
``TRITON_INTERPRET=1`` must be set before triton is imported) against fp64, both call paths (the
eager ``LayerNormFunction`` and the ``triton_op``); the backend choice and the CPU path. CUDA
(``gpu``): the same cases on tiny inputs, eager vs op bit for bit, accuracy against fp64 and
torch's kernel at PTv3 widths, determinism, sync-freedom, packing invariance (C15), and
torch.compile / torch.export keeping the op (D15).

    TRITON_INTERPRET=1 python -m tests.ops.test_norm_ops

No index in these kernels comes from data (every address is row * C + column), so no input can
make them read outside their tensors.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from pointtransformerv3.ops import norm as ln_module
from pointtransformerv3.ops.norm import layer_norm, set_layer_norm_backend

REPO = Path(__file__).resolve().parents[2]
EPS = 1e-5
# (N, C): empty, one row, C below the 16-wide tile, odd widths, the PTv3 widths, the generic
# config (1000, 4096 = the widest Triton row)
CASES = [(0, 32), (1, 5), (7, 3), (33, 48), (300, 32), (65, 130), (40, 64), (20, 512), (9, 1000),
         (3, 4096)]  # fmt: skip
# max error relative to the reference's largest magnitude: (out and dx, dweight and dbias)
TOLERANCES = {
    torch.float32: (4e-6, 4e-6),
    torch.bfloat16: (8e-3, 1e-4),
    torch.float16: (1e-3, 1e-4),
}


def make(n: int, c: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """fp32 rows with a non-zero mean and per-channel scales, an upstream gradient, and affine
    parameters away from (1, 0)."""
    gen = torch.Generator().manual_seed(seed)
    scale = 0.5 + 2 * torch.rand(c, generator=gen)
    x = torch.randn(n, c, generator=gen) * scale + 0.7
    grad = torch.randn(n, c, generator=gen)
    weight = 1 + 0.3 * torch.randn(c, generator=gen)
    bias = 0.2 * torch.randn(c, generator=gen)
    return x, grad, weight, bias


def reference(x, grad, weight, bias) -> tuple[torch.Tensor, ...]:
    """fp64 (out, dx, dweight, dbias) of the given (already rounded) inputs."""
    x, w, b = (t.double().requires_grad_() for t in (x, weight, bias))
    out = F.layer_norm(x, (x.shape[1],), w, b, EPS)
    return (out.detach(), *torch.autograd.grad(out, (x, w, b), grad.double()))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max abs error over the reference's largest magnitude (at least 1e-6, for references that
    are 0 up to fp64 rounding)."""
    if b.numel() == 0:
        return 0.0
    err = (a.detach().double().cpu() - b.cpu()).abs().max()
    return float(err / b.abs().max().clamp(min=1e-6))


def run(n, c, dtype=torch.float32, device="cpu", path="eager", out_dtype=None) -> bool:
    """One case through the Triton launchers (``path``: "eager" = ``LayerNormFunction``, "op" =
    ``torch.ops.pointtransformerv3.layer_norm``); prints one line. Features in ``dtype``, fp32 weight and
    bias, output in ``out_dtype`` (default ``dtype``)."""
    from pointtransformerv3.ops import _norm_triton

    out_dtype = out_dtype or dtype
    x, grad, weight, bias = make(n, c, seed=n * 7 + c)
    x, grad = x.to(dtype), grad.to(out_dtype)
    expected = reference(x, grad, weight, bias)
    x_in = x.to(device).requires_grad_()
    w_in, b_in = weight.to(device).requires_grad_(), bias.to(device).requires_grad_()
    args = (x_in, w_in, b_in, EPS, out_dtype)
    if path == "eager":
        out = _norm_triton.LayerNormFunction.apply(*args)
    else:
        out = torch.ops.pointtransformerv3.layer_norm(*args)[0]
    grads = torch.autograd.grad(out, (x_in, w_in, b_in), grad.to(device))
    tol_out, tol_wb = TOLERANCES[out_dtype if out_dtype != torch.float32 else dtype]
    errs = {"out": rel(out, expected[0]), "dx": rel(grads[0], expected[1])}
    errs |= {"dw": rel(grads[1], expected[2]), "db": rel(grads[2], expected[3])}
    gates = {"out": tol_out, "dx": TOLERANCES[dtype][0], "dw": tol_wb, "db": tol_wb}
    ok = all(e <= gates[k] for k, e in errs.items())
    ok = ok and out.dtype == out_dtype and grads[0].dtype == dtype
    ok = ok and grads[1].dtype == grads[2].dtype == torch.float32
    report = " ".join(f"{k}={e:.1e}" for k, e in errs.items())
    tag = f"{device} {str(dtype).removeprefix('torch.')}->{str(out_dtype).removeprefix('torch.')}"
    print(f"{'OK ' if ok else 'BAD'} {tag:18s} {path:5s} N={n:4d} C={c:4d} {report}", flush=True)
    return ok


def interpret_main() -> int:
    if os.environ.get("TRITON_INTERPRET") != "1":
        print("set TRITON_INTERPRET=1 before running this module (CPU interpreter)")
        return 2
    results = [run(n, c, path=path) for n, c in CASES for path in ("eager", "op")]
    results += [run(n, c, torch.bfloat16) for n, c in CASES[:7]]
    results += [run(n, c, torch.float32, out_dtype=torch.bfloat16) for n, c in CASES[2:6]]
    print("ALL OK" if all(results) else "FAILURES", flush=True)
    return 0 if all(results) else 1


@pytest.fixture(autouse=True)
def restore_state() -> Iterator[None]:
    saved = ln_module._backend
    yield
    ln_module._backend = saved
    ln_module._noted.clear()


# ---------------------------------------------------------------- CPU


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton is not installed")
def test_kernels_in_interpreter():
    """Every case in the Triton interpreter (CPU) against fp64, through both call paths, fp32,
    bf16 and an fp32 input with a bf16 output."""
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tests.ops.test_norm_ops"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0 and "ALL OK" in result.stdout, result.stdout + result.stderr


def test_cpu_takes_torch_and_logs_once(caplog):
    """Off CUDA ``auto`` takes torch's kernel (fp32 for a bf16 input, as before I1), logged once
    per (device, dtype); a forced triton raises there; unknown names are rejected."""
    x, _, weight, bias = make(50, 24)
    set_layer_norm_backend("auto")
    with caplog.at_level(logging.INFO, logger=ln_module.__name__):
        for _ in range(2):
            y = layer_norm(x.bfloat16(), (24,), weight, bias, EPS)
    assert torch.equal(y, F.layer_norm(x.bfloat16().float(), (24,), weight, bias, EPS).bfloat16())
    assert [r.getMessage() for r in caplog.records] == [
        "layer_norm on cpu bfloat16: torch backend (auto: only option for bfloat16 on cpu)"
    ]
    set_layer_norm_backend("triton")
    with pytest.raises(ValueError, match="triton layer_norm needs fp32/fp16/bf16 CUDA rows"):
        layer_norm(x, (24,), weight, bias, EPS)
    with pytest.raises(ValueError, match="unknown layer_norm backend 'cuda'"):
        set_layer_norm_backend("cuda")  # type: ignore[arg-type]


def test_fp64_and_other_shapes_take_torch():
    """fp64, a 2-D normalized shape and a missing affine use torch's kernel, also on CPU."""
    x, _, weight, bias = make(20, 8)
    x = x.double()
    want = F.layer_norm(x, (8,), weight.double(), bias.double(), EPS)
    assert torch.equal(layer_norm(x, (8,), weight, bias, EPS), want)
    x2 = x.view(20, 2, 4).float()
    w2, b2 = weight.view(2, 4), bias.view(2, 4)
    assert torch.equal(layer_norm(x2, (2, 4), w2, b2, EPS), F.layer_norm(x2, (2, 4), w2, b2, EPS))
    want = F.layer_norm(x2, (4,), None, None, EPS)
    assert torch.equal(layer_norm(x2, (4,), None, None, EPS), want)


# ---------------------------------------------------------------- CUDA


@pytest.fixture
def triton_cuda() -> None:
    from pointtransformerv3.ops.sparse_conv import triton_impl

    problem = triton_impl.toolchain_problem(min_capability=triton_impl.TRITON_MIN_CAPABILITY)
    if problem is not None:
        pytest.skip(f"triton unavailable: {problem}")
    set_layer_norm_backend("auto")
    ln_module._choose(torch.device("cuda"), torch.float32, torch.float32)  # loads the kernels


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_tiny_cuda_against_fp64(triton_cuda, dtype):
    results = [run(n, c, dtype, "cuda", path) for n, c in CASES for path in ("eager", "op")]
    if dtype == torch.float32:
        results += [run(n, c, dtype, "cuda", out_dtype=torch.bfloat16) for n, c in CASES]
    assert all(results)


def ptv3_rows(n: int, c: int, dtype: torch.dtype, seed: int = 0) -> tuple[torch.Tensor, ...]:
    x, grad, weight, bias = make(n, c, seed)
    return x.to("cuda", dtype), grad.to("cuda", dtype), weight.cuda(), bias.cuda()


def step(x, weight, bias, grad, out_dtype=None) -> tuple[torch.Tensor, ...]:
    """(out, dx, dweight, dbias) of one fwd+bwd through ``layer_norm``."""
    x = x.detach().clone().requires_grad_()
    w, b = weight.detach().clone().requires_grad_(), bias.detach().clone().requires_grad_()
    out = layer_norm(x, (x.shape[-1],), w, b, EPS, out_dtype=out_dtype)
    return (out.detach(), *torch.autograd.grad(out, (x, w, b), grad.to(out.dtype)))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("c", [32, 64, 128, 256, 512])
def test_accuracy_at_ptv3_widths(triton_cuda, dtype, c):
    """Against fp64 of the same (rounded) inputs: bf16 outputs within one rounding of the exact
    value, fp32 within 4e-6; against torch's kernel (the pre-I1 path) within the same bounds."""
    x, grad, weight, bias = ptv3_rows(20_000, c, dtype, seed=c)
    got = step(x, weight, bias, grad)
    expected = reference(x.cpu(), grad.cpu(), weight.cpu(), bias.cpu())
    out_tol, wb_tol = TOLERANCES[dtype]
    tolerances = (out_tol, out_tol, wb_tol, wb_tol)
    for name, a, b, tol in zip(("out", "dx", "dw", "db"), got, expected, tolerances, strict=True):
        assert rel(a, b) <= tol, (name, rel(a, b))
    if dtype == torch.bfloat16:  # elementwise: one bf16 rounding of the exact output
        exact = expected[0].to("cuda")
        assert ((got[0].double() - exact).abs() <= exact.abs() * 2**-8 + 1e-6).all()
    set_layer_norm_backend("torch")
    old = step(x, weight, bias, grad)
    for name, a, b in zip(("out", "dx", "dw", "db"), got, old, strict=True):
        assert rel(a, b) <= 2 * (out_tol if name in ("out", "dx") else 4e-3), (name, rel(a, b))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eager_is_the_op_bitwise_deterministic_and_sync_free(triton_cuda, monkeypatch, dtype):
    """I5: the eager Function and the op give the same bits; two runs give the same bits (fixed
    backward grid); no host sync."""
    x, grad, weight, bias = ptv3_rows(30_000, 64, dtype, seed=3)
    first = step(x, weight, bias, grad)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        second = step(x, weight, bias, grad)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    monkeypatch.setattr(ln_module, "tracing", lambda x: True)
    op = step(x, weight, bias, grad)
    for a, b, c in zip(first, second, op, strict=True):
        assert torch.equal(a, b) and torch.equal(a, c)


@pytest.mark.gpu
def test_bad_arguments_get_torchs_errors(triton_cuda):
    """Shapes or devices that do not fit take torch's kernel, which raises (no launch reads past
    a weight of the wrong size)."""
    x, _, weight, bias = ptv3_rows(100, 48, torch.bfloat16)
    with pytest.raises(RuntimeError):
        layer_norm(x, (48,), weight[:40], bias[:40], EPS)
    with pytest.raises(RuntimeError):
        layer_norm(x, (40,), weight, bias, EPS)
    with pytest.raises(RuntimeError):
        layer_norm(x, (48,), weight.cpu(), bias.cpu(), EPS)


@pytest.mark.gpu
def test_packing_invariance_and_shapes(triton_cuda):
    """C15: rows are normalized independently, so a pack's output is the concatenation of its
    parts' bit for bit; (..., C) inputs and non-contiguous inputs work."""
    x, _, weight, bias = ptv3_rows(5000, 48, torch.bfloat16, seed=5)

    def norm(t: torch.Tensor) -> torch.Tensor:
        return layer_norm(t, (48,), weight, bias, EPS)

    both = norm(x)
    assert torch.equal(both, torch.cat([norm(x[:1234]), norm(x[1234:])]))
    assert torch.equal(norm(x.view(50, 100, 48)), both.view(50, 100, 48))
    wide = torch.cat([x, x], 1)[:, ::2]  # non-contiguous rows
    assert torch.equal(norm(wide), norm(wide.contiguous()))


@pytest.mark.gpu
def test_compile_and_export_keep_the_op(triton_cuda):
    """D15: torch.compile (fullgraph, dynamic rows, compile-time autotune off) matches eager, and
    strict / non-strict torch.export record ``pointtransformerv3.layer_norm``."""
    import torch._inductor.config as inductor_config

    x, grad, weight, bias = ptv3_rows(3000, 64, torch.bfloat16, seed=6)

    def fn(t: torch.Tensor) -> torch.Tensor:
        return layer_norm(t, (64,), weight, bias, EPS)

    torch._dynamo.reset()
    # fx_graph_cache off: the on-disk graph cache does not key user-defined Triton kernels by their
    # source (an edited kernel was served from a graph compiled before the edit)
    patch = {
        "triton.autotune_at_compile_time": False,
        "max_autotune": False,
        "compile_threads": 1,
        "fx_graph_cache": False,
    }
    try:
        with inductor_config.patch(patch):
            compiled = torch.compile(fn, fullgraph=True, dynamic=True)
            for rows in (3000, 1700):
                t = x[:rows].clone().requires_grad_()
                got = compiled(t)
                (gx,) = torch.autograd.grad(got, t, grad[:rows])
                t2 = x[:rows].clone().requires_grad_()
                want = fn(t2)
                (wx,) = torch.autograd.grad(want, t2, grad[:rows])
                assert torch.equal(got, want) and torch.equal(gx, wx)
    finally:
        torch._dynamo.reset()

    class Norm(torch.nn.Module):
        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return layer_norm(t, (64,), weight, bias, EPS)

    for strict in (False, True):
        exported = torch.export.export(Norm(), (x,), strict=strict)
        targets = {str(n.target) for n in exported.graph.nodes if n.op == "call_function"}
        assert "pointtransformerv3.layer_norm.default" in targets, (strict, targets)
        assert torch.equal(exported.module()(x), fn(x))


if __name__ == "__main__":
    sys.exit(interpret_main())
