# Vendored from Pointcept2 src/pointcept2/ops/sparse_conv/_triton_kernels.py @ 771a949 (simplified for pointtransformerv3).
"""Triton implicit-GEMM kernels of the ``triton`` sparse conv backend: forward, dgrad, wgrad.

Adapted from ocnn-pytorch (https://github.com/octree-nn/ocnn-pytorch, commit 7521c22; MIT,
Copyright (c) 2022 Peng-Shuai Wang), whose ``ocnn/nn/kernels`` are themselves adapted from
FlexGEMM (https://github.com/JeffreyXiang/FlexGEMM; MIT, Copyright (c) 2025 Jianfeng Xiang).
Both license texts are in ``THIRD_PARTY_LICENSES.md`` next to this file.

Source map (paths relative to ``ocnn/nn/kernels/`` at that commit):

- ``fwd_kernel``   <- ``conv_fwd_implicit_gemm.py`` L15-85: a (B1 rows x B2 C_out) tile, a K loop
  over ``V * ceil(C_in / BK)`` steps, ``% N`` / ``% C_out`` wrap instead of row masks, a masked
  gather of missing neighbors, an fp32 accumulator.
- ``dgrad_kernel`` <- ``conv_bwd_implicit_gemm.py`` L15-75: the input gradient as a gather on the
  mirrored kernel-map column ``V - 1 - v`` (no transposed rulebook, no atomics).
- ``wgrad_kernel`` <- ``conv_bwd_implicit_gemm_splitk.py`` L94-184 and its heuristics L86-91:
  ``BV x BCi`` column packing (a small C_in still fills a 64-128 wide tile) and split-K over the
  rows into fp32 partials that the caller sums in a fixed order.

Changes against ocnn, all deliberate:

1. Index safety. A neighbor is used only when ``0 <= nbr < N`` (ocnn: ``nbr != -1``), and row,
   kernel-map and partial-buffer addresses are computed in int64, so a corrupt kernel map can never
   address memory outside the tensors (on WSL2 + WDDM an illegal address resets the whole GPU).
2. Weight dtype. The weight tile is cast to the input dtype after the load, so the op keeps the
   fp32 master weight; the weight gradient is reduced and returned in fp32 (ocnn rounds it to
   the input dtype). The bias is added to the fp32 accumulator before the downcast.
3. ``Ci``, ``Co`` and ``V`` are ``tl.constexpr`` (one compile per conv shape); ``N`` and ``SPLITK``
   are runtime arguments marked ``do_not_specialize``, so one binary serves every N and split.
4. No ``triton.autotune`` and no tuning keyed on N: launch configs are a pinned function of the GPU
   and the conv shape (``triton_impl``), so results are bitwise identical across processes and
   traced graphs carry no guard on N. Inductor never benchmarks these kernels with random indices.
5. ``tl.dot`` always uses ``input_precision="ieee"``: fp32 inputs get IEEE fp32 products (no
   TF32), bf16/fp16 inputs accumulate in fp32.
6. No forward/dgrad split-K and no L2 swizzle groups (neither paid off at Pointcept2's sizes).
"""

from __future__ import annotations

import triton
import triton.language as tl

__all__ = ["dgrad_kernel", "fwd_kernel", "wgrad_kernel"]


@triton.jit(do_not_specialize=["N"])
def fwd_kernel(
    x,  # (N, Ci) compute dtype
    w,  # (Co, V, Ci) any float dtype, cast to x's dtype per tile
    bias,  # (Co,) any float dtype (ignored unless HAS_BIAS)
    kmap,  # (N, V) int32, -1 = no neighbor
    out,  # (N, Co) x's dtype
    N,
    Ci: tl.constexpr,
    Co: tl.constexpr,
    V: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    B1: tl.constexpr,  # rows per tile
    B2: tl.constexpr,  # output channels per tile
    BK: tl.constexpr,  # input channels per K step
):
    pid = tl.program_id(axis=0)
    num_co: tl.constexpr = (Co + B2 - 1) // B2
    pid_co = pid % num_co
    pid_n = pid // num_co
    num_k: tl.constexpr = (Ci + BK - 1) // BK

    rows = pid_n * B1 + tl.arange(0, B1)
    rows_in = (rows % N).to(tl.int64)  # ocnn L46: wrap instead of masking the row loads
    cols = pid_co * B2 + tl.arange(0, B2)
    cols_in = cols % Co
    offs_k = tl.arange(0, BK)
    kmap_row = kmap + rows_in * V
    acc = tl.zeros((B1, B2), dtype=tl.float32)
    for k in range(num_k * V):
        v = k // num_k
        bk = k % num_k
        nbr = tl.load(kmap_row + v)
        valid = (nbr >= 0) & (nbr < N)
        ci = bk * BK + offs_k
        k_mask = ci < Ci
        a = tl.load(
            x + nbr.to(tl.int64)[:, None] * Ci + ci[None, :],
            mask=valid[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            w + cols_in[None, :] * (V * Ci) + v * Ci + ci[:, None], mask=k_mask[:, None], other=0.0
        ).to(a.dtype)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    if HAS_BIAS:
        acc += tl.load(bias + cols_in).to(tl.float32)[None, :]
    mask = (rows < N)[:, None] & (cols < Co)[None, :]
    tl.store(
        out + rows.to(tl.int64)[:, None] * Co + cols[None, :],
        acc.to(out.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=["N"])
def dgrad_kernel(
    grad_out,  # (N, Co) compute dtype
    w,  # (Co, V, Ci) any float dtype
    kmap,  # (N, V) int32, symmetric: kmap[p, v] == q  <=>  kmap[q, V-1-v] == p
    grad_x,  # (N, Ci) grad_out's dtype
    N,
    Ci: tl.constexpr,
    Co: tl.constexpr,
    V: tl.constexpr,
    B1: tl.constexpr,  # rows per tile
    B2: tl.constexpr,  # input channels per tile
    BK: tl.constexpr,  # output channels per K step
):
    pid = tl.program_id(axis=0)
    num_ci: tl.constexpr = (Ci + B2 - 1) // B2
    pid_ci = pid % num_ci
    pid_n = pid // num_ci
    num_k: tl.constexpr = (Co + BK - 1) // BK

    rows = pid_n * B1 + tl.arange(0, B1)
    rows_in = (rows % N).to(tl.int64)
    cols = pid_ci * B2 + tl.arange(0, B2)
    cols_in = cols % Ci
    offs_k = tl.arange(0, BK)
    kmap_row = kmap + rows_in * V
    acc = tl.zeros((B1, B2), dtype=tl.float32)
    for k in range(num_k * V):
        v = k // num_k
        bk = k % num_k
        nbr = tl.load(kmap_row + (V - 1 - v))  # ocnn L56: the mirrored offset
        valid = (nbr >= 0) & (nbr < N)
        co = bk * BK + offs_k
        k_mask = co < Co
        a = tl.load(
            grad_out + nbr.to(tl.int64)[:, None] * Co + co[None, :],
            mask=valid[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            w + co[:, None] * (V * Ci) + v * Ci + cols_in[None, :], mask=k_mask[:, None], other=0.0
        ).to(a.dtype)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    mask = (rows < N)[:, None] & (cols < Ci)[None, :]
    tl.store(
        grad_x + rows.to(tl.int64)[:, None] * Ci + cols[None, :],
        acc.to(grad_x.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=["N", "SPLITK"])
def wgrad_kernel(
    grad_out,  # (N, Co) compute dtype
    x,  # (N, Ci) compute dtype
    kmap,  # (N, V) int32
    partial,  # (SPLITK, Co, V, Ci) fp32
    N,
    Ci: tl.constexpr,
    Co: tl.constexpr,
    V: tl.constexpr,
    B1: tl.constexpr,  # output channels per tile
    B2: tl.constexpr,  # packed (offset, input channel) columns per tile = BV * BCi
    BK: tl.constexpr,  # rows per reduction step
    BV: tl.constexpr,  # kernel offsets per tile
    BCi: tl.constexpr,  # input channels per tile
    SPLITK,  # runtime: another split count needs no recompile
):
    pid_co = tl.program_id(axis=0)
    pid_vci = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)
    num_ci: tl.constexpr = (Ci + BCi - 1) // BCi
    pid_v = pid_vci // num_ci
    pid_ci = pid_vci % num_ci

    num_k = tl.cdiv(N, BK)
    k_start = tl.cdiv(num_k * pid_k, SPLITK)  # ocnn L131-132: a contiguous slice of row blocks
    k_end = tl.cdiv(num_k * (pid_k + 1), SPLITK)

    co = pid_co * B1 + tl.arange(0, B1)
    co_in = co % Co
    vv = pid_v * BV + tl.arange(0, BV)
    vv_in = vv % V
    ci = pid_ci * BCi + tl.arange(0, BCi)
    ci_in = ci % Ci
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((B1, BV * BCi), dtype=tl.float32)
    for k in range(k_start, k_end):
        rows = k * BK + offs_k
        row_ok = rows < N
        rows64 = rows.to(tl.int64)
        nbr = tl.load(kmap + rows64[:, None] * V + vv_in[None, :], mask=row_ok[:, None], other=-1)
        valid = (nbr >= 0) & (nbr < N)  # (BK, BV)
        g = tl.load(
            grad_out + rows64[None, :] * Co + co_in[:, None], mask=row_ok[None, :], other=0.0
        )  # (B1, BK) = grad_out^T tile
        a = tl.load(
            x + nbr.to(tl.int64)[:, :, None] * Ci + ci_in[None, None, :],
            mask=valid[:, :, None],
            other=0.0,
        ).reshape(BK, BV * BCi)
        acc = tl.dot(g, a, acc, input_precision="ieee")

    col = (vv[:, None] * Ci + ci[None, :]).reshape(BV * BCi)
    col_ok = ((vv < V)[:, None] & (ci < Ci)[None, :]).reshape(BV * BCi)
    base = partial + pid_k.to(tl.int64) * (Co * V * Ci)
    ptr = base + co.to(tl.int64)[:, None] * (V * Ci) + col[None, :]
    tl.store(ptr, acc, mask=(co < Co)[:, None] & col_ok[None, :])
