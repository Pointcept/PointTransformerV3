# Vendored from Pointcept2 src/pointcept2/ops/sparse_conv/torch_impl.py @ 771a949 (simplified for point_transformer_v3).
"""Pure-torch submanifold sparse conv: im2col gather + GEMM with a deterministic backward.

The CPU (and MPS) implementation, kept first-class for CPU inference (D22), and the CUDA fallback
when Triton cannot run.

Forward: ``out = gather(x, kmap) @ W``, where the gather builds ``(rows, K * C_in)`` patches.
Backward uses the symmetry of the kernel map (``kmap[p, k] == q`` iff ``kmap[q, K-1-k] == p``):
``grad_x = gather(grad_out, kmap) @ W_flipped`` is again a gather + GEMM with no atomics, and the
weight gradient reuses the same gathered rows, ``x^T @ gather(grad_out, kmap)``. When x needs no
gradient (the stem), the weight gradient gathers the narrower of x and grad_out instead. The
weight gradient accumulates in fp32.

Gathers (from the ocnn-pytorch port, research/sparse_conv/ocnn_port/README.md, section 2): the
source is x after one zero row, so a -1 ("no neighbor") entry becomes row 0 by adding 1 to the
map, the cheapest sentinel mapping (no mask, no compare); its row stride avoids torch's slow
vectorized gather on CUDA (``ops/gather.py``). Rows are gathered in chunks into one buffer reused
across the chunks of a call (``index_select(out=)``), x is saved contiguous for backward (aligned
GEMM operands), and chunks of narrow rows (the k=5 stem's 6 input channels) are smaller. On the
CPU, chunks stay small enough for malloc to reuse the heap (no page faults per call) and for the
gathered patches to stay in cache for the GEMM.

Chunk count and tracing: eagerly the number of chunks depends on N. While torch.compile /
torch.export trace (or ``make_fx`` with a symbolic N) the conv runs as one chunk, so no graph
specializes on N (a compiled model recompiled for every new pack before: review F2); Inductor then
schedules the whole-N buffers itself.

A sparse-matrix formulation (one GEMM to ``(N, K * C_out)``, then a cuSPARSE SpMM with the 0/1
kernel-map matrix) was measured as an alternative and was no faster once the gather avoids
torch's slow vectorized kernel (``ops.gather.gather_source``): on a 704k-point ScanNet pack with
shuffled rows, bf16 fwd+bwd, k=3, gather vs SpMM: C=32 5.4 vs 5.9 ms, C=64 10.9 vs 10.8 ms,
C=256 51 vs 49 ms. It also needs an (N, K * C_out) buffer and a CSR matrix per dtype.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

__all__ = ["subm_conv3d"]

# Bytes of gathered patches per chunk, by device type; a conv's peak extra memory is about 3x this.
# CUDA: 128 MiB (2**26 bf16 elements) for C -> C convs, 64 MiB for narrow rows (C < 16, the k=5
# stem: 4.7 vs 5.0 ms fwd+bwd at 1M, ocnn port). CPU: see research/integrate/phase2/B.md.
_CHUNK_BYTES = 128 << 20
_NARROW_CHUNK_BYTES = 64 << 20
_CPU_CHUNK_BYTES = 16 << 20
_NARROW_CHANNELS = 16
_ALIGN = 16  # bytes: rows whose stride is a multiple of this take torch's slow CUDA gather
_MAX_SLOW_ROW_BYTES = 256  # above this the vectorized gather is as fast (ops/gather.py)


def subm_conv3d(x: Tensor, weight: Tensor, bias: Tensor | None, kmap: Tensor) -> Tensor:
    """Submanifold conv ``out[p] = bias + sum_k W[:, k] @ x[kmap[p, k]]`` in pure torch.

    Shapes:
        x: ``(N, C_in)`` float. weight: ``(C_out, k, k, k, C_in)``. bias: ``(C_out,)`` or None.
        kmap: ``(N, k**3)`` int32 from ``build_kmap`` (-1 = no neighbor).
        returns: ``(N, C_out)`` in the autocast dtype when autocast is on for x's device, else in
        x's dtype; products accumulate in fp32 and the weight/bias gradients keep their dtype.
    """
    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
    else:
        dtype = x.dtype
    with torch.autocast(device_type, enabled=False):
        return _SubMConv3d.apply(x, weight, bias, kmap, dtype)


class _SubMConv3d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: Tensor, weight: Tensor, bias: Tensor | None, kmap: Tensor, dtype: torch.dtype
    ) -> Tensor:
        c_out = weight.shape[0]
        x_cast = x.to(dtype).contiguous()  # saved: aligned GEMM operand of the weight gradient
        w = weight.to(dtype).reshape(c_out, -1).t()  # (K * C_in, C_out), rows in (k, c) order
        b = None if bias is None else bias.to(dtype)
        if _whole(kmap):
            out = _linear(_im2col(_source(x_cast), kmap), w, b)
        else:
            out = x_cast.new_empty(x.shape[0], c_out)
            for rows, patches in _chunks(_source(x_cast), kmap):
                _linear(patches, w, b, out=out[rows])
        ctx.save_for_backward(x_cast, weight, kmap)
        ctx.x_dtype = x.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        return out

    # once_differentiable's annotation makes ctx positional-only, which pyright flags.
    @staticmethod
    @once_differentiable
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any, *grad_outputs: Tensor
    ) -> tuple[Tensor | None, ...]:
        (grad_out,) = grad_outputs
        x, weight, kmap = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        grad_out = grad_out.to(x.dtype).contiguous()
        acc_dtype = torch.promote_types(x.dtype, torch.float32)
        c_out, c_in = weight.shape[0], weight.shape[-1]

        grad_x = grad_w = grad_b = None
        if need_b:
            grad_b = grad_out.sum(0, dtype=acc_dtype).to(ctx.bias_dtype)
        if need_x or (need_w and c_out < c_in):
            grad_x, grad_w = _grads_from_grad_out(grad_out, x, weight, kmap, need_x, need_w)
        elif need_w:
            # Weight gradient only, from x patches: (C_out, N) @ (N, K * C_in) = weight layout.
            acc = grad_out.new_zeros(c_out, weight[0].numel(), dtype=acc_dtype)
            whole = _whole(kmap)
            for rows, patches in _chunks(_source(x), kmap):
                acc = _accumulate_mm(acc, grad_out[rows].t(), patches, whole)
            grad_w = acc
        if grad_x is not None:
            grad_x = grad_x.to(ctx.x_dtype)
        if grad_w is not None:
            grad_w = grad_w.reshape(weight.shape).to(weight.dtype)
        return grad_x, grad_w, grad_b, None, None


def _grads_from_grad_out(
    grad_out: Tensor, x: Tensor, weight: Tensor, kmap: Tensor, need_x: bool, need_w: bool
) -> tuple[Tensor | None, Tensor | None]:
    """Input and weight gradients from one gather of grad_out patches ``G (rows, K * C_out)``.

    ``grad_x[q] = sum_j grad_out[kmap[q, j]] @ W[:, K-1-j]^T`` and the weight gradient of offset k
    is block ``K-1-k`` of ``x^T @ G``.
    """
    c_out, c_in = weight.shape[0], weight.shape[-1]
    num_offsets = kmap.shape[1]
    acc_dtype = torch.promote_types(x.dtype, torch.float32)
    # (K * C_out, C_in), block j = W[:, K-1-j] of shape (C_out, C_in)
    w_flip = weight.to(x.dtype).view(c_out, num_offsets, c_in).flip(1).transpose(0, 1)
    w_flip = w_flip.reshape(-1, c_in)
    acc = x.new_zeros(c_in, num_offsets * c_out, dtype=acc_dtype) if need_w else None
    grad_x = None
    if _whole(kmap):
        patches = _im2col(_source(grad_out), kmap)
        if need_x:
            grad_x = patches @ w_flip
        if acc is not None:
            acc = _accumulate_mm(acc, x.t(), patches, True)
    else:
        grad_x = x.new_empty(x.shape) if need_x else None
        for rows, patches in _chunks(_source(grad_out), kmap):
            if grad_x is not None:
                torch.mm(patches, w_flip, out=grad_x[rows])
            if acc is not None:
                acc = _accumulate_mm(acc, x[rows].t(), patches, False)
    grad_w = None
    if acc is not None:  # (C_in, K, C_out) with reversed offsets -> (C_out, K, C_in)
        grad_w = acc.view(c_in, num_offsets, c_out).flip(1).permute(2, 1, 0)
    return grad_x, grad_w


# ---- gathers


def _whole(kmap: Tensor) -> bool:
    """One chunk: while torch.compile / torch.export trace, or with a symbolic row count (make_fx).
    The chunk count then never depends on N."""
    return torch.compiler.is_compiling() or isinstance(kmap.shape[0], torch.SymInt)


def _source(x: Tensor) -> Tensor:
    """``(N + 1, C)`` copy of x after one zero row: ``kmap + 1`` indexes it (-1 -> the zero row).

    On CUDA, rows of 16..256 bytes with a 16-byte-multiple stride take torch's slow vectorized
    gather (``ops/gather.py``), so the copy is a view of a buffer one element wider per row.
    """
    n, c = x.shape
    row_bytes = c * x.element_size()
    pad = int(x.is_cuda and 0 < row_bytes <= _MAX_SLOW_ROW_BYTES and row_bytes % _ALIGN == 0)
    src = x.new_empty(n + 1, c + pad)[:, :c]
    src[0].zero_()
    src[1:].copy_(x)
    return src


def _im2col(src: Tensor, kmap: Tensor) -> Tensor:
    """``(N, K * C)`` patches of every row in one gather (the traced path)."""
    index = (kmap + 1).view(-1)
    return src.index_select(0, index).view(kmap.shape[0], -1)


def _chunks(src: Tensor, kmap: Tensor) -> Iterator[tuple[slice, Tensor]]:
    """``(rows, patches)`` of consecutive row chunks: ``patches (R, K * C)`` holds the rows of
    ``src`` (``_source``) at ``kmap[rows] + 1``. Eagerly the chunks are gathered into one buffer
    reused across chunks, so each yielded tensor is overwritten by the next; traced, one chunk."""
    if _whole(kmap):
        yield slice(None), _im2col(src, kmap)
        return
    n, num_offsets = kmap.shape
    c = src.shape[1]
    step = max(1, _chunk_bytes(src) // max(1, num_offsets * c * src.element_size()))
    step = min(step, max(n, 1))
    patch_buffer = src.new_empty(step * num_offsets * c)
    index_buffer = kmap.new_empty(step * num_offsets)
    for start in range(0, n, step):
        stop = min(start + step, n)
        size = (stop - start) * num_offsets
        index = torch.add(kmap[start:stop].reshape(-1), 1, out=index_buffer[:size])
        patches = patch_buffer[: size * c].view(size, c)
        torch.index_select(src, 0, index, out=patches)
        yield slice(start, stop), patches.view(stop - start, num_offsets * c)


def _chunk_bytes(src: Tensor) -> int:
    if src.device.type == "cpu":
        return _CPU_CHUNK_BYTES
    return _NARROW_CHUNK_BYTES if src.shape[1] < _NARROW_CHANNELS else _CHUNK_BYTES


# ---- GEMMs


def _linear(patches: Tensor, w: Tensor, b: Tensor | None, out: Tensor | None = None) -> Tensor:
    """``patches @ w (+ b)``, into ``out`` when given."""
    if b is None:
        return torch.mm(patches, w) if out is None else torch.mm(patches, w, out=out)
    return torch.addmm(b, patches, w) if out is None else torch.addmm(b, patches, w, out=out)


def _accumulate_mm(acc: Tensor, a: Tensor, b: Tensor, traced: bool) -> Tensor:
    """``acc + a @ b`` accumulated in acc's dtype (fp32 when a and b are half precision): in place
    eagerly, out of place when ``traced`` (``addmm(out_dtype=, out=)`` fails on FakeTensors under
    torch 2.12, m1-as-built open issue 4)."""
    if a.dtype == acc.dtype:
        return acc + a @ b if traced else acc.addmm_(a, b)
    if acc.is_cuda:
        if traced:
            return acc + torch.mm(a, b, out_dtype=acc.dtype)
        return torch.addmm(acc, a, b, out_dtype=acc.dtype, out=acc)
    # no mixed-dtype GEMM off CUDA: each chunk's product is rounded once
    return acc + a @ b if traced else acc.add_(a @ b)
