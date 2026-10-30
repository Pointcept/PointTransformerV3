# Vendored from Pointcept2 src/pointcept2/ops/_attention_triton.py @ 771a949 (changed: do_not_specialize on row counts, so a warm-up compiles every variant the model needs; padded head blocks for head dims that are not powers of two).
"""Triton kernels and ``triton_op``s of the ``triton_d16`` attention backend (imports triton;
``ops.attention`` imports this module the first time it chooses the backend).

Attention inside the patches of a serialized sequence, read through the serialization: the qkv
Linear runs on the level's rows as they are, the kernels read each patch's q / k / v rows through
``rows`` (serialized position -> row, ``Serialized.order``) and write ``out`` and ``dq | dk | dv``
back to the same rows. No permutation, key gather, head_dim padding or dq|dk|dv concatenation runs
outside the kernels (R1, research/accel/r2/reports/r1-attention-d16.md). The structure is
FlashAttention-2's (and the Triton tutorial's) online softmax:

- ``_fwd_kernel``: one program per (query block of a patch, head); loops over the patch's keys and
  stores ``out`` and the fp32 log-sum-exp (base 2) per (head, position).
- ``_dq_kernel`` (backward, first): one program per (query block, head); recomputes P from the
  log-sum-exp and stores dq and ``delta = rowsum(dO * O)`` (fp32) for the next kernel.
- ``_dkdv_kernel`` (backward, second): one program per (key block of the rows a patch owns, head).
  A patch owns its query positions; a key row also serves the next patch when that is an
  "overlap" tail whose key range starts inside this patch, so the program loops over its own
  patch's queries and then over the tail's. Every row has exactly one writer: no atomics, and the
  backward is bitwise reproducible.

The patch table (``PatchLayout.table``, ``ops.patches``) has one int32 row ``(q_start, q_len,
k_start, k_len)`` per patch in serialized positions and a sentinel row ``(T, 0, T, 0)``, which
``_dkdv_kernel`` reads as the "next patch" of the last one. Programs are ``(patch * NB + block,
head)``: CUDA caps grid axes 1 and 2 at 65535, so the patch index shares axis 0 with the query
block (``NB = cdiv(max_q, BLOCK)``); a block past its patch's end exits. Row and stat addresses
are int64 (``row * stride + column``, ``head * T + position``); position offsets into ``rows`` and
the table (``q_start + offs_m``, ``own_start + offs_n``) are int32, which ``T < 2**31`` keeps in
range (lanes past a run's end are masked). ``rows`` and the table are read with unit stride and
trusted here: ``ops.attention`` makes them contiguous and checks the table's dtype and shape on
every call, and their values under ``PTV3_DEBUG=1`` on eager calls.

Head dims: a head is one block of ``DP`` lanes, ``DP = max(16, next_power_of_2(head_dim))``
(``tl.dot`` and ``tl.arange`` need powers of two of at least 16). For a head_dim below DP (Utonia's
18 -> 32) the lanes past head_dim are masked on every load and store (zeros in q, k, v, so scores
and outputs are exact) and read column 0 of their head, so no address leaves the head's row
segment; for a power-of-two head_dim the code is the unpadded kernel (``_cols`` / ``_ok`` branch
on constexprs).

Launch configs are fixed (bounded offline sweeps of 24 configs per kernel on the 1M pack's L0
shapes, bf16, head_dim 16, RTX PRO 6000: research/accel/r2/r1-attention-d16/logs/sweep_*), never
autotuned. Eager calls launch the kernels from ``SerializedAttentionFunction``; traced graphs
(torch.compile, torch.export, AOTInductor) record ``pointtransformerv3::serialized_attention`` and
its backward op (D15, I5).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.library import triton_op

from pointtransformerv3.ops._triton import device_guard, raw, traceable

__all__ = [
    "SerializedAttentionFunction",
    "serialized_attention_backward",
    "serialized_attention_op",
]

LOG2E = math.log2(math.e)


class LaunchConfig(NamedTuple):
    block_m: int  # query rows per tile
    block_n: int  # key rows per tile
    num_warps: int
    num_stages: int


# Fixed configs (research/accel/r2/r1-attention-d16/bench.py sweep); no runtime autotune. In
# _DKDV block_n is the program's key block and block_m the query tile of its inner loop.
_FWD = LaunchConfig(128, 64, 4, 2)
_DQ = LaunchConfig(128, 64, 4, 2)
_DKDV = LaunchConfig(64, 128, 4, 2)


def head_block(head_dim: int) -> int:
    """The lanes of one head in the kernels: a power of two of at least 16."""
    return max(16, int(triton.next_power_of_2(head_dim)))


@triton.jit
def _cols(D: tl.constexpr, DP: tl.constexpr):
    """Column offsets of a head block: ``arange(DP)``, padded lanes clamped to column 0."""
    offs = tl.arange(0, DP)
    if D < DP:
        offs = tl.where(offs < D, offs, 0)
    return offs


@triton.jit
def _ok(row_ok, D: tl.constexpr, DP: tl.constexpr):
    """The mask of a (rows, head block) tile: the rows, and with padding the head_dim lanes."""
    if D < DP:
        return row_ok[:, None] & (tl.arange(0, DP) < D)[None, :]
    else:
        return row_ok[:, None]


@triton.jit(do_not_specialize=["T", "NB"])
def _fwd_kernel(QKV, ROWS, TAB, OUT, LSE, T, NB, stride_qkv, stride_out, qk_scale,
                H: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr):  # fmt: skip
    p = (tl.program_id(0) // NB).to(tl.int64)
    mb = tl.program_id(0) % NB
    h = tl.program_id(1)
    qk_scale = tl.cast(qk_scale, tl.float32)  # Inductor passes Python floats as fp64
    q_start = tl.load(TAB + 4 * p)
    q_len = tl.load(TAB + 4 * p + 1)
    if mb * BM < q_len:
        k_start = tl.load(TAB + 4 * p + 2)
        k_len = tl.load(TAB + 4 * p + 3)
        C: tl.constexpr = H * D
        offs_m = mb * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_d = _cols(D, DP)
        q_ok = offs_m < q_len
        q_row = tl.load(ROWS + q_start + offs_m, mask=q_ok, other=0)
        q = tl.load(QKV + q_row[:, None] * stride_qkv + h * D + offs_d[None, :],
                    mask=_ok(q_ok, D, DP), other=0.0)  # fmt: skip
        m_i = tl.full([BM], float("-inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, DP], tl.float32)
        for n in range(0, k_len, BN):
            k_ok = n + offs_n < k_len
            k_row = tl.load(ROWS + k_start + n + offs_n, mask=k_ok, other=0)
            kp = QKV + k_row[:, None] * stride_qkv + h * D + offs_d[None, :]
            k = tl.load(kp + C, mask=_ok(k_ok, D, DP), other=0.0)
            v = tl.load(kp + 2 * C, mask=_ok(k_ok, D, DP), other=0.0)
            s = tl.dot(q, tl.trans(k)) * qk_scale
            s = tl.where(k_ok[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.math.exp2(m_i - m_new)
            pr = tl.math.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(pr, 1)
            acc = acc * alpha[:, None] + tl.dot(pr.to(v.dtype), v)
            m_i = m_new
        acc = acc / l_i[:, None]
        tl.store(OUT + q_row[:, None] * stride_out + h * D + offs_d[None, :],
                 acc.to(OUT.dtype.element_ty), mask=_ok(q_ok, D, DP))  # fmt: skip
        tl.store(LSE + h.to(tl.int64) * T + q_start + offs_m, m_i + tl.math.log2(l_i), mask=q_ok)


@triton.jit(do_not_specialize=["T", "NB"])
def _dq_kernel(QKV, ROWS, TAB, OUT, DOUT, LSE, DELTA, DQKV, T, NB, stride_qkv, stride_out,
               stride_dout, stride_dqkv, qk_scale, sm_scale,
               H: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, BM: tl.constexpr,
               BN: tl.constexpr):  # fmt: skip
    p = (tl.program_id(0) // NB).to(tl.int64)
    mb = tl.program_id(0) % NB
    h = tl.program_id(1)
    qk_scale = tl.cast(qk_scale, tl.float32)
    sm_scale = tl.cast(sm_scale, tl.float32)
    q_start = tl.load(TAB + 4 * p)
    q_len = tl.load(TAB + 4 * p + 1)
    if mb * BM < q_len:
        k_start = tl.load(TAB + 4 * p + 2)
        k_len = tl.load(TAB + 4 * p + 3)
        C: tl.constexpr = H * D
        offs_m = mb * BM + tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_d = _cols(D, DP)
        q_ok = offs_m < q_len
        q_row = tl.load(ROWS + q_start + offs_m, mask=q_ok, other=0)
        col = h * D + offs_d[None, :]
        qd_ok = _ok(q_ok, D, DP)
        q = tl.load(QKV + q_row[:, None] * stride_qkv + col, mask=qd_ok, other=0.0)
        do = tl.load(DOUT + q_row[:, None] * stride_dout + col, mask=qd_ok, other=0.0)
        o = tl.load(OUT + q_row[:, None] * stride_out + col, mask=qd_ok, other=0.0)
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
        stat = h.to(tl.int64) * T + q_start + offs_m
        tl.store(DELTA + stat, delta, mask=q_ok)
        lse = tl.load(LSE + stat, mask=q_ok, other=0.0)
        dq = tl.zeros([BM, DP], tl.float32)
        for n in range(0, k_len, BN):
            k_ok = n + offs_n < k_len
            k_row = tl.load(ROWS + k_start + n + offs_n, mask=k_ok, other=0)
            kp = QKV + k_row[:, None] * stride_qkv + col
            k = tl.load(kp + C, mask=_ok(k_ok, D, DP), other=0.0)
            v = tl.load(kp + 2 * C, mask=_ok(k_ok, D, DP), other=0.0)
            s = tl.dot(q, tl.trans(k)) * qk_scale
            pr = tl.where(k_ok[None, :], tl.math.exp2(s - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = pr * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k)
        dq = (dq * sm_scale).to(DQKV.dtype.element_ty)
        tl.store(DQKV + q_row[:, None] * stride_dqkv + col, dq, mask=qd_ok)


@triton.jit
def _dkdv_patch(dk, dv, k, v, k_ok, QKV, ROWS, DOUT, LSE, DELTA, q_start, q_len, h, T, stride_qkv,
                stride_dout, qk_scale, D: tl.constexpr, DP: tl.constexpr,
                BM: tl.constexpr):  # fmt: skip
    """Add one patch's queries to the (BN, DP) dk, dv of a key block (k_ok: the block's rows that
    are keys of this patch)."""
    offs_d = _cols(D, DP)
    col = h * D + offs_d[None, :]
    for m in range(0, q_len, BM):
        offs_m = m + tl.arange(0, BM)
        q_ok = offs_m < q_len
        q_row = tl.load(ROWS + q_start + offs_m, mask=q_ok, other=0)
        qd_ok = _ok(q_ok, D, DP)
        q = tl.load(QKV + q_row[:, None] * stride_qkv + col, mask=qd_ok, other=0.0)
        do = tl.load(DOUT + q_row[:, None] * stride_dout + col, mask=qd_ok, other=0.0)
        stat = h.to(tl.int64) * T + q_start + offs_m
        delta = tl.load(DELTA + stat, mask=q_ok, other=0.0)
        lse = tl.load(LSE + stat, mask=q_ok, other=0.0)
        pt = tl.math.exp2(tl.dot(k, tl.trans(q)) * qk_scale - lse[None, :])  # (BN, BM) = P^T
        pt = tl.where(k_ok[:, None] & q_ok[None, :], pt, 0.0)
        dv += tl.dot(pt.to(do.dtype), do)
        dpt = tl.dot(v, tl.trans(do))
        dk += tl.dot((pt * (dpt - delta[None, :])).to(q.dtype), q)
    return dk, dv


@triton.jit(do_not_specialize=["T", "NB"])
def _dkdv_kernel(QKV, ROWS, TAB, DOUT, LSE, DELTA, DQKV, T, NB, stride_qkv, stride_dout,
                 stride_dqkv, qk_scale, sm_scale,
                 H: tl.constexpr, D: tl.constexpr, DP: tl.constexpr, BM: tl.constexpr,
                 BN: tl.constexpr):  # fmt: skip
    p = (tl.program_id(0) // NB).to(tl.int64)
    nb = tl.program_id(0) % NB
    h = tl.program_id(1)
    qk_scale = tl.cast(qk_scale, tl.float32)
    sm_scale = tl.cast(sm_scale, tl.float32)
    own_start = tl.load(TAB + 4 * p)
    own_len = tl.load(TAB + 4 * p + 1)
    if nb * BN < own_len:
        C: tl.constexpr = H * D
        offs_n = nb * BN + tl.arange(0, BN)
        offs_d = _cols(D, DP)
        row_ok = offs_n < own_len
        pos = own_start + offs_n
        k_row = tl.load(ROWS + pos, mask=row_ok, other=0)
        kp = QKV + k_row[:, None] * stride_qkv + h * D + offs_d[None, :]
        kd_ok = _ok(row_ok, D, DP)
        k = tl.load(kp + C, mask=kd_ok, other=0.0)
        v = tl.load(kp + 2 * C, mask=kd_ok, other=0.0)
        dk = tl.zeros([BN, DP], tl.float32)
        dv = tl.zeros([BN, DP], tl.float32)
        k_start = tl.load(TAB + 4 * p + 2)
        k_ok = row_ok & (pos >= k_start) & (pos < k_start + tl.load(TAB + 4 * p + 3))
        dk, dv = _dkdv_patch(dk, dv, k, v, k_ok, QKV, ROWS, DOUT, LSE, DELTA, own_start, own_len,
                             h, T, stride_qkv, stride_dout, qk_scale, D, DP, BM)  # fmt: skip
        # the next patch (the sentinel after the last one) is an "overlap" tail whose keys start
        # inside these rows: its queries also read them
        next_k_start = tl.load(TAB + 4 * p + 6)
        if next_k_start < own_start + own_len:
            k_ok = row_ok & (pos >= next_k_start) & (pos < next_k_start + tl.load(TAB + 4 * p + 7))
            dk, dv = _dkdv_patch(dk, dv, k, v, k_ok, QKV, ROWS, DOUT, LSE, DELTA,
                                 tl.load(TAB + 4 * p + 4), tl.load(TAB + 4 * p + 5), h, T,
                                 stride_qkv, stride_dout, qk_scale, D, DP, BM)  # fmt: skip
        dp = DQKV + k_row[:, None] * stride_dqkv + h * D + offs_d[None, :]
        tl.store(dp + C, (dk * sm_scale).to(DQKV.dtype.element_ty), mask=kd_ok)
        tl.store(dp + 2 * C, dv.to(DQKV.dtype.element_ty), mask=kd_ok)


# ---- ops (traced graphs) and the eager Function


@triton_op("pointtransformerv3::serialized_attention", mutates_args=())
def serialized_attention_op(
    qkv: Tensor, rows: Tensor, table: Tensor, max_q: int, num_heads: int, scale: float
) -> tuple[Tensor, Tensor]:
    """``(out, lse)``: softmax attention of every patch, out in qkv's dtype on the rows of the
    level, lse the fp32 log2-sum-exp of the scaled scores per (head, serialized position).

    Shapes:
        qkv: ``(T, 3C)`` bf16/fp16, ``[q | k | v]`` each head-major, unit column stride.
        rows: ``(T,)`` int64 contiguous, serialized position -> row (a permutation of ``[0, T)``).
        table: ``(P + 1, 4)`` int32 contiguous ``PatchLayout.table``; ``max_q`` >= every q_len.
        returns: out ``(T, C)``; lse ``(num_heads, T)`` fp32.
    """
    return _forward(qkv, rows, table, max_q, num_heads, scale, traceable)


@serialized_attention_op.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _forward_fake(
    qkv: Tensor, rows: Tensor, table: Tensor, max_q: int, num_heads: int, scale: float
) -> tuple[Tensor, Tensor]:
    t = qkv.shape[0]
    return qkv.new_empty(t, qkv.shape[1] // 3), qkv.new_empty(num_heads, t, dtype=torch.float32)


@triton_op("pointtransformerv3::serialized_attention_backward", mutates_args=())
def serialized_attention_backward(
    grad_out: Tensor,
    qkv: Tensor,
    out: Tensor,
    lse: Tensor,
    rows: Tensor,
    table: Tensor,
    max_q: int,
    num_heads: int,
    scale: float,
) -> Tensor:
    """``dqkv`` ``(T, 3C)`` in qkv's dtype, on the rows of the level (every row written once)."""
    args = (grad_out, qkv, out, lse, rows, table, max_q, num_heads, scale)
    return _backward_launch(*args, traceable)


@serialized_attention_backward.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _backward_fake(
    grad_out: Tensor,
    qkv: Tensor,
    out: Tensor,
    lse: Tensor,
    rows: Tensor,
    table: Tensor,
    max_q: int,
    num_heads: int,
    scale: float,
) -> Tensor:
    return torch.empty_like(qkv)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: tuple[Tensor, Tensor]) -> None:
    qkv, rows, table, max_q, num_heads, scale = inputs
    ctx.save_for_backward(qkv, *output, rows, table)
    ctx.args = (max_q, num_heads, scale)


def _backward(
    ctx: Any,
    grad_out: Tensor,
    *_lse_grad: Any,
    launch: Callable[..., Tensor] = serialized_attention_backward,
) -> tuple[Tensor | None, ...]:
    """Gradients of ``(qkv, rows, table, max_q, num_heads, scale)``: through the op (traced graphs
    keep it) or through the launcher (``SerializedAttentionFunction``)."""
    qkv, out, lse, rows, table = ctx.saved_tensors
    grad_qkv = launch(grad_out.contiguous(), qkv, out, lse, rows, table, *ctx.args)
    return grad_qkv, None, None, None, None, None


serialized_attention_op.register_autograd(  # pyright: ignore[reportFunctionMemberAccess]
    _backward, setup_context=_setup_context
)


class SerializedAttentionFunction(torch.autograd.Function):
    """Eager fast path of ``pointtransformerv3::serialized_attention``: the op's kernels launched directly
    (``ops._tracing.tracing`` picks it). Arguments as the op; returns out only."""

    @staticmethod
    def forward(
        ctx: Any, qkv: Tensor, rows: Tensor, table: Tensor, max_q: int, num_heads: int, scale: float
    ) -> Tensor:
        out, lse = _forward(qkv, rows, table, max_q, num_heads, scale)
        _setup_context(ctx, (qkv, rows, table, max_q, num_heads, scale), (out, lse))
        return out

    # once_differentiable's annotation makes ctx positional-only, which pyright flags.
    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out: Tensor) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _backward(ctx, grad_out, launch=_backward_launch)


# ---- launchers: the op bodies (wrap=traceable) and the eager Function (wrap=raw)


def _forward(
    qkv: Tensor,
    rows: Tensor,
    table: Tensor,
    max_q: int,
    num_heads: int,
    scale: float,
    wrap: Callable[[Any], Any] = raw,
) -> tuple[Tensor, Tensor]:
    t, c3 = qkv.shape
    c = c3 // 3
    out = qkv.new_empty(t, c)
    lse = qkv.new_empty(num_heads, t, dtype=torch.float32)
    num_patches = table.shape[0] - 1
    if t == 0 or num_patches == 0:
        return out, lse
    cfg = _FWD
    nb = triton.cdiv(max_q, cfg.block_m)
    with device_guard(qkv):
        wrap(_fwd_kernel)[(num_patches * nb, num_heads)](
            qkv, rows, table, out, lse, t, nb, qkv.stride(0), out.stride(0), scale * LOG2E,
            H=num_heads, D=c // num_heads, DP=head_block(c // num_heads), BM=cfg.block_m,
            BN=cfg.block_n,
            num_warps=cfg.num_warps, num_stages=cfg.num_stages,
        )  # fmt: skip
    return out, lse


def _backward_launch(
    grad_out: Tensor,
    qkv: Tensor,
    out: Tensor,
    lse: Tensor,
    rows: Tensor,
    table: Tensor,
    max_q: int,
    num_heads: int,
    scale: float,
    wrap: Callable[[Any], Any] = raw,
) -> Tensor:
    t, c3 = qkv.shape
    d = c3 // 3 // num_heads
    grad_qkv = torch.empty_like(qkv)
    num_patches = table.shape[0] - 1
    if t == 0 or num_patches == 0:
        return grad_qkv
    delta = torch.empty_like(lse)  # rowsum(dO * O): written by _dq_kernel, read by _dkdv_kernel
    dq, dkdv = _DQ, _DKDV
    nb_q, nb_k = triton.cdiv(max_q, dq.block_m), triton.cdiv(max_q, dkdv.block_n)
    with device_guard(qkv):
        wrap(_dq_kernel)[(num_patches * nb_q, num_heads)](
            qkv, rows, table, out, grad_out, lse, delta, grad_qkv, t, nb_q, qkv.stride(0),
            out.stride(0), grad_out.stride(0), grad_qkv.stride(0), scale * LOG2E, scale,
            H=num_heads, D=d, DP=head_block(d), BM=dq.block_m, BN=dq.block_n,
            num_warps=dq.num_warps,
            num_stages=dq.num_stages,
        )  # fmt: skip
        wrap(_dkdv_kernel)[(num_patches * nb_k, num_heads)](
            qkv, rows, table, grad_out, lse, delta, grad_qkv, t, nb_k, qkv.stride(0),
            grad_out.stride(0), grad_qkv.stride(0), scale * LOG2E, scale,
            H=num_heads, D=d, DP=head_block(d), BM=dkdv.block_m, BN=dkdv.block_n,
            num_warps=dkdv.num_warps,
            num_stages=dkdv.num_stages,
        )  # fmt: skip
    return grad_qkv
