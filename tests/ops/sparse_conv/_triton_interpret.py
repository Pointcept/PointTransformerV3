# Adapted from Pointcept2 tests/ops/sparse_conv/_triton_interpret.py @ 771a949.
"""Tiny checks of the triton sparse conv kernels in the Triton interpreter on CPU.

Run by ``test_triton_conv.py::test_kernels_in_interpreter`` in a subprocess, because
``TRITON_INTERPRET=1`` must be set before triton is imported:

    TRITON_INTERPRET=1 python -m tests.ops.sparse_conv._triton_interpret

Every case compares out, grad_x, grad_weight and grad_bias of ``triton_impl.subm_conv3d`` (fp32;
the eager ``SubmConv3dFunction``, and the ``triton_op`` that traced graphs run) with an fp64
reference that gathers through the kernel map and differentiates by autograd, so the reference
input gradient is a scatter and does not assume a symmetric map.

Corrupt kernel maps: the kernels must treat entries outside ``[0, N)`` as missing (forward and
weight gradient; the input gradient needs a symmetric map). Features and gradients sit in the
middle of buffers padded on both sides with ``SENTINEL`` rows, so the "near" corrupt entries
(``NEAR``: just below 0 and at or above N) stay inside the allocation and a broken range guard
shows up as a mismatch, never as a fault. Only these run on CUDA (``test_tiny_cuda_against_fp64``).
The "far" entries (``FAR``: 2**30, -2**31) point outside any allocation and run in the interpreter
only, where a broken guard crashes this subprocess instead of the GPU. The bias cases pass a
strided bias (every other element of a ``SENTINEL``-filled buffer) and an expanded one (stride 0,
the first element of a ``SENTINEL``-filled buffer), so a kernel that reads the bias at offset c
without the op making it contiguous sees sentinels; the expanded one runs in the interpreter only.
Exits 1 on any failure; every line is flushed, so a crash still shows the cases before it.
"""

from __future__ import annotations

import os
import sys

import torch

from pointtransformerv3.ops import _tracing
from pointtransformerv3.ops.neighbors import build_kmap

# (name, sample sizes, extent, kernel size, C_in, C_out, bias): channel counts that are not
# multiples of the tiles, ragged samples, the k=5 stem, a single point and an empty input
CASES = [
    ("k3 3->5 odd", [30, 12], 5, 3, 3, 5, True),
    ("k3 16->24 3 samples", [20, 15, 7], 4, 3, 16, 24, True),
    ("k3 40->70 ragged", [24, 9], 5, 3, 40, 70, True),
    ("k5 6->32 stem", [28], 5, 5, 6, 32, False),
    ("k3 single point", [1], 4, 3, 16, 16, True),
    ("k3 N=0", [0], 4, 3, 8, 8, True),
]
CORRUPT = [CASES[0], CASES[2], CASES[3]]  # k3 odd, k3 ragged, k5 stem
BIAS_CASES = [(CASES[0], "strided"), (CASES[2], "strided"), (CASES[1], "expanded")]
TOLERANCES = {torch.float32: 2e-6, torch.bfloat16: 1e-2, torch.float16: 2e-3}
PAD = 64  # sentinel rows before and after every feature / gradient buffer
SENTINEL = 1e3  # exact in fp32/fp16/bf16; far above the data, so any read of it is a mismatch


def near(n: int) -> list[int]:
    """Out-of-range kmap entries that still address the padded buffers (safe on CUDA)."""
    return [n, n + 3, n + PAD - 1, -7, -PAD]


FAR = [2**30, -(2**31)]  # outside every allocation: interpreter (CPU) only


def random_grid(seq_lens: list[int], extent: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct grid cells per sample, rows shuffled: (grid (N, 3) int32, batch (N,) int64)."""
    gen = torch.Generator().manual_seed(seed)
    grids, batches = [], []
    for b, n in enumerate(seq_lens):
        cells = torch.randperm(extent**3, generator=gen)[:n]
        grids.append(torch.stack([cells // extent**2, cells // extent % extent, cells % extent], 1))
        batches.append(torch.full((n,), b, dtype=torch.int64))
    grid, batch = torch.cat(grids).int(), torch.cat(batches)
    perm = torch.randperm(len(grid), generator=gen)
    return grid[perm], batch[perm]


def reference(x, weight, bias, kmap, grad):
    """fp64 (out, grad_x, grad_weight, grad_bias); kmap entries outside [0, N) are missing."""
    x = x.double().requires_grad_()
    w = weight.double().flatten(1, 3).requires_grad_()
    b = None if bias is None else bias.double().requires_grad_()
    km = kmap.long()
    valid = (km >= 0) & (km < len(x))
    gathered = torch.where(valid[..., None], x[km.clamp(0, max(len(x) - 1, 0))], 0.0)
    out = torch.einsum("nvc,ovc->no", gathered, w)
    if b is not None:
        out = out + b
    grads = torch.autograd.grad(out, [t for t in (x, w, b) if t is not None], grad.double())
    return out.detach(), *grads


def padded(t: torch.Tensor) -> torch.Tensor:
    """``t`` as the middle rows of a buffer with ``PAD`` ``SENTINEL`` rows on each side (a
    contiguous view: the ops keep it, so the kernels see the padded allocation)."""
    buf = t.new_full((len(t) + 2 * PAD, *t.shape[1:]), SENTINEL)
    buf[PAD : PAD + len(t)] = t
    return buf[PAD : PAD + len(t)]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    if b.numel() == 0:
        return 0.0
    return float((a.double().cpu() - b).abs().max() / b.abs().max().clamp(min=1e-30))


def run(
    case,
    corrupt: str | None = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    bias_layout: str = "contiguous",
    path: str = "eager",
) -> bool:
    """One case on ``device`` with features in ``dtype`` (fp32 weights); prints a line.

    ``corrupt``: None, "near" (entries inside the padded buffers) or "far" (CPU only).
    ``bias_layout``: "contiguous", "strided" (every other element of a sentinel-filled buffer) or
    "expanded" (the first element of a sentinel-filled buffer, stride 0; CPU only). Both non-
    contiguous layouts leave sentinels where a stride-1 read of the bias would land.
    ``path``: "eager" (the default call) or "op" (the ``triton_op``, as while tracing)."""
    from pointtransformerv3.ops.sparse_conv import triton_impl

    triton_impl.tracing = (lambda x: True) if path == "op" else _tracing.tracing

    if device != "cpu" and (corrupt == "far" or bias_layout == "expanded"):
        raise ValueError("far kmap entries and expanded biases are for the interpreter only")
    name, seq_lens, extent, k, c_in, c_out, use_bias = case
    grid, batch = random_grid(seq_lens, extent, seed=sum(map(ord, name)))
    kmap = build_kmap(grid, batch, k)
    n = len(grid)
    if corrupt is not None and n:
        gen = torch.Generator().manual_seed(1)
        bad = torch.tensor(near(n) if corrupt == "near" else near(n) + FAR, dtype=torch.int32)
        idx = torch.randint(0, kmap.numel(), (max(1, kmap.numel() // 10),), generator=gen)
        kmap.view(-1)[idx] = bad[torch.randint(0, len(bad), idx.shape, generator=gen)]
        if device != "cpu" and not ((kmap >= -PAD) & (kmap < n + PAD)).all():
            raise ValueError("a CUDA kmap entry would address memory outside the padded buffers")
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(n, c_in, generator=gen)
    weight = torch.randn(c_out, k, k, k, c_in, generator=gen) / (c_in * k**3) ** 0.5
    bias = torch.randn(c_out, generator=gen) if use_bias else None
    grad = torch.randn(n, c_out, generator=gen)
    if bias is not None and bias_layout == "expanded":
        bias = bias[:1].expand(c_out).contiguous()  # one value: what an expanded bias holds
    expected = reference(x, weight, bias, kmap, grad)

    x_in = padded(x.to(device, dtype)).requires_grad_(corrupt is None)
    w_in = weight.to(device).requires_grad_()
    b_in = None
    if bias is not None:
        if bias_layout == "strided":
            buf = torch.full((2 * c_out,), SENTINEL)
            buf[::2] = bias
            b_in = buf.to(device)[::2]
        elif bias_layout == "expanded":
            # stride 0 over buf[0]; buf[1:] are sentinels, so a stride-1 read of it mismatches
            buf = torch.full((c_out,), SENTINEL)
            buf[0] = bias[0]
            b_in = buf.to(device)[:1].expand(c_out)
        else:
            b_in = bias.to(device)
        b_in.requires_grad_()
    out = triton_impl.subm_conv3d(x_in, w_in, b_in, kmap.to(device))
    out.backward(padded(grad.to(device, dtype)))
    assert w_in.grad is not None
    errs = {"out": rel(out.detach(), expected[0]), "dw": rel(w_in.grad.flatten(1, 3), expected[2])}
    if corrupt is None:
        assert x_in.grad is not None
        errs["dx"] = rel(x_in.grad, expected[1])
    if b_in is not None:
        assert b_in.grad is not None
        errs["db"] = rel(b_in.grad, expected[-1])
    ok = all(e <= TOLERANCES[dtype] for e in errs.values())
    ok = ok and out.dtype == dtype and w_in.grad.dtype == torch.float32
    report = " ".join(f"{key}={e:.1e}" for key, e in errs.items())
    tag = f"{device} {str(dtype).removeprefix('torch.'):8s} {name:20s}"
    extra = " ".join(
        t for t in (corrupt, bias_layout, path) if t not in (None, "contiguous", "eager")
    )
    print(f"{'OK ' if ok else 'BAD'} {tag} {extra:8s} N={n:3d} {report}", flush=True)
    return ok


def main() -> int:
    if os.environ.get("TRITON_INTERPRET") != "1":
        print("set TRITON_INTERPRET=1 before running this module (CPU interpreter)")
        return 2
    results = [run(case) for case in CASES]
    results += [run(case, path="op") for case in CASES]
    results += [run(case, bias_layout=layout) for case, layout in BIAS_CASES]
    results += [run(case, "near") for case in CORRUPT]  # mismatch, not a crash, on a bad guard
    results += [run(case, "far") for case in CORRUPT]
    print("ALL OK" if all(results) else "FAILURES", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
