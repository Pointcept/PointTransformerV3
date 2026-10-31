# Vendored from Pointcept2 src/pointcept2/ops/sparse_conv/_triton_ops.py @ 771a949 (simplified for point_transformer_v3).
"""``torch.library.triton_op``s of the triton sparse conv backend (imports triton).

``point_transformer_v3::subm_conv3d`` launches the forward kernel and has a fake impl and an
autograd formula built from ``point_transformer_v3::subm_conv3d_dgrad`` and
``point_transformer_v3::subm_conv3d_wgrad``. Under torch.compile, torch.export and AOTInductor the
kernels are captured as user-defined Triton kernels, so a traced graph keeps no Python body. The
op bodies read shapes and the launch configs of ``triton_impl`` only: no host sync, no branch on
N. Eager launches run under a device guard for the inputs' device (Triton launches on the current
device and its current stream). ``triton_impl`` imports this module on first use; call the ops
through ``triton_impl.subm_conv3d`` (autocast, input checks).

Eager calls take ``SubmConv3dFunction`` instead (I5): a plain ``torch.autograd.Function`` that
launches the same kernels with the same configs (bitwise equal results) without the custom-op
dispatch and autograd: 356 -> 235 us of host time per bf16 conv forward + backward
(research/accel/integrate_a/host_conv.py). The op bodies and the Function share the launchers
below (``wrap`` is ``ops._triton.traceable``, i.e. ``wrap_triton``, in the op bodies).

Adapted from ocnn-pytorch (commit 7521c22; MIT, Copyright (c) 2022 Peng-Shuai Wang): the
forward / dgrad / wgrad split of ``ocnn/nn/octree_conv_t.py`` L27-52, with kernels adapted from
FlexGEMM (MIT, Copyright (c) 2025 Jianfeng Xiang). Licenses in ``THIRD_PARTY_LICENSES.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import triton
from torch import Tensor
from torch.autograd.function import once_differentiable
from torch.library import triton_op

from point_transformer_v3.ops._triton import device_guard, raw, traceable
from point_transformer_v3.ops.sparse_conv._triton_kernels import (
    dgrad_kernel,
    fwd_kernel,
    wgrad_kernel,
)
from point_transformer_v3.ops.sparse_conv.triton_impl import gemm_config, wgrad_config

__all__ = ["SubmConv3dFunction", "conv_dgrad", "conv_fwd", "conv_wgrad"]


@triton_op("point_transformer_v3::subm_conv3d", mutates_args=())
def conv_fwd(x: Tensor, weight: Tensor, bias: Tensor | None, kmap: Tensor) -> Tensor:
    """``out[p] = bias + sum_v weight[:, v] @ x[kmap[p, v]]``, accumulated in fp32.

    Shapes:
        x: ``(N, C_in)`` fp32/fp16/bf16, the compute dtype. weight: ``(C_out, V, C_in)`` any float
        dtype, cast to x's dtype. bias: ``(C_out,)`` or None, added in fp32. kmap: ``(N, V)``
        int32, entries outside ``[0, N)`` are missing. returns: ``(N, C_out)`` in x's dtype.
    """
    _check("subm_conv3d", weight, kmap, x=(x, weight.shape[-1]))
    _check_bias(bias, weight, x)
    return _fwd(x, weight, bias, kmap, traceable)


@conv_fwd.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _conv_fwd_fake(x: Tensor, weight: Tensor, bias: Tensor | None, kmap: Tensor) -> Tensor:
    _check("subm_conv3d", weight, kmap, x=(x, weight.shape[-1]))
    _check_bias(bias, weight, x)
    return x.new_empty(x.shape[0], weight.shape[0])


@triton_op("point_transformer_v3::subm_conv3d_dgrad", mutates_args=())
def conv_dgrad(grad_out: Tensor, weight: Tensor, kmap: Tensor) -> Tensor:
    """Input gradient ``grad_x[q] = sum_v grad_out[kmap[q, V-1-v]] @ weight[:, v]``.

    Valid for a symmetric kernel map (``kmap[p, v] == q`` iff ``kmap[q, V-1-v] == p``).

    Shapes:
        grad_out: ``(N, C_out)`` compute dtype. weight: ``(C_out, V, C_in)``. kmap: ``(N, V)``
        int32. returns: ``(N, C_in)`` in grad_out's dtype.
    """
    _check("subm_conv3d_dgrad", weight, kmap, grad_out=(grad_out, weight.shape[0]))
    return _dgrad(grad_out, weight, kmap, traceable)


@conv_dgrad.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _conv_dgrad_fake(grad_out: Tensor, weight: Tensor, kmap: Tensor) -> Tensor:
    return grad_out.new_empty(grad_out.shape[0], weight.shape[2])


@triton_op("point_transformer_v3::subm_conv3d_wgrad", mutates_args=())
def conv_wgrad(grad_out: Tensor, x: Tensor, kmap: Tensor) -> Tensor:
    """fp32 weight gradient ``sum_p grad_out[p]^T x[kmap[p, v]]`` per offset v: ``SPLITK`` fp32
    partials over contiguous row ranges, then one ``sum(0)`` (fixed order, reproducible).

    Shapes:
        grad_out: ``(N, C_out)`` and x: ``(N, C_in)`` in the compute dtype. kmap: ``(N, V)`` int32.
        returns: ``(C_out, V, C_in)`` fp32.
    """
    c_out, c_in = grad_out.shape[-1], x.shape[-1]
    _check("subm_conv3d_wgrad", None, kmap, grad_out=(grad_out, c_out), x=(x, c_in))
    if grad_out.dtype != x.dtype:
        raise TypeError(f"subm_conv3d_wgrad: grad_out is {grad_out.dtype} but x is {x.dtype}")
    return _wgrad(grad_out, x, kmap, traceable)


@conv_wgrad.register_fake  # pyright: ignore[reportFunctionMemberAccess]
def _conv_wgrad_fake(grad_out: Tensor, x: Tensor, kmap: Tensor) -> Tensor:
    return grad_out.new_empty(grad_out.shape[1], kmap.shape[1], x.shape[1], dtype=torch.float32)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    x, weight, bias, kmap = inputs
    ctx.save_for_backward(x, weight, kmap)
    ctx.bias_dtype = None if bias is None else bias.dtype


def _backward(ctx: Any, grad_out: Tensor, dgrad: Any = conv_dgrad, wgrad: Any = conv_wgrad) -> Any:
    """Gradients of ``(x, weight, bias, kmap)``: through the ops (traced graphs keep them), or
    through the launchers (``SubmConv3dFunction``)."""
    x, weight, kmap = ctx.saved_tensors
    need_x, need_w, need_b = ctx.needs_input_grad[:3]
    grad = grad_out.to(x.dtype)
    grad_x = dgrad(grad, weight, kmap) if need_x else None
    grad_w = wgrad(grad, x, kmap).to(weight.dtype) if need_w else None
    grad_b = None
    if ctx.bias_dtype is not None and need_b:
        grad_b = grad.sum(0, dtype=torch.float32).to(ctx.bias_dtype)
    return grad_x, grad_w, grad_b, None


conv_fwd.register_autograd(  # pyright: ignore[reportFunctionMemberAccess]
    _backward, setup_context=_setup_context
)


class SubmConv3dFunction(torch.autograd.Function):
    """Eager fast path of ``point_transformer_v3::subm_conv3d``: the op's kernels and configs
    launched directly (``ops._tracing.tracing`` picks it). Arguments as ``conv_fwd``, already
    checked by ``triton_impl.subm_conv3d``."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, weight: Tensor, bias: Tensor | None, kmap: Tensor) -> Tensor:
        _setup_context(ctx, (x, weight, bias, kmap), None)
        return _fwd(x, weight, bias, kmap)

    # once_differentiable's annotation makes ctx positional-only, which pyright flags.
    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out: Tensor) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _backward(ctx, grad_out, _dgrad, _wgrad)


# ---- launchers: the op bodies (wrap=traceable) and the eager Function (wrap=raw)


def _fwd(
    x: Tensor, weight: Tensor, bias: Tensor | None, kmap: Tensor, wrap: Callable[[Any], Any] = raw
) -> Tensor:
    n, c_in = x.shape
    c_out, num_offsets, _ = weight.shape
    x = x.contiguous()
    w = weight.to(x.dtype).contiguous()
    cfg = gemm_config("fwd", c_in, c_out, num_offsets, x.dtype, x.device)
    out = x.new_empty(n, c_out)
    grid = (triton.cdiv(n, cfg.B1) * triton.cdiv(c_out, cfg.B2),)
    with device_guard(x):
        wrap(fwd_kernel)[grid](
            x,
            w,
            w if bias is None else bias.contiguous(),  # the kernel reads bias[c] at offset c
            kmap.contiguous(),
            out,
            n,
            Ci=c_in,
            Co=c_out,
            V=num_offsets,
            HAS_BIAS=bias is not None,
            B1=cfg.B1,
            B2=cfg.B2,
            BK=cfg.BK,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    return out


def _dgrad(
    grad_out: Tensor, weight: Tensor, kmap: Tensor, wrap: Callable[[Any], Any] = raw
) -> Tensor:
    n, c_out = grad_out.shape
    _, num_offsets, c_in = weight.shape
    grad_out = grad_out.contiguous()
    w = weight.to(grad_out.dtype).contiguous()
    cfg = gemm_config("dgrad", c_in, c_out, num_offsets, grad_out.dtype, grad_out.device)
    grad_x = grad_out.new_empty(n, c_in)
    grid = (triton.cdiv(n, cfg.B1) * triton.cdiv(c_in, cfg.B2),)
    with device_guard(grad_out):
        wrap(dgrad_kernel)[grid](
            grad_out,
            w,
            kmap.contiguous(),
            grad_x,
            n,
            Ci=c_in,
            Co=c_out,
            V=num_offsets,
            B1=cfg.B1,
            B2=cfg.B2,
            BK=cfg.BK,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    return grad_x


def _wgrad(grad_out: Tensor, x: Tensor, kmap: Tensor, wrap: Callable[[Any], Any] = raw) -> Tensor:
    n, c_out = grad_out.shape
    c_in, num_offsets = x.shape[1], kmap.shape[1]
    grad_out = grad_out.contiguous()
    x = x.contiguous()
    cfg = wgrad_config(c_in, c_out, num_offsets, x.dtype, x.device)
    partial = grad_out.new_empty(cfg.SPLITK, c_out, num_offsets, c_in, dtype=torch.float32)
    grid = (
        triton.cdiv(c_out, cfg.B1),
        triton.cdiv(num_offsets, cfg.BV) * triton.cdiv(c_in, cfg.BCi),
        cfg.SPLITK,
    )
    with device_guard(x):
        wrap(wgrad_kernel)[grid](
            grad_out,
            x,
            kmap.contiguous(),
            partial,
            n,
            Ci=c_in,
            Co=c_out,
            V=num_offsets,
            B1=cfg.B1,
            B2=cfg.B2,
            BK=cfg.BK,
            BV=cfg.BV,
            BCi=cfg.BCi,
            SPLITK=cfg.SPLITK,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    return partial.sum(0)


def _check_bias(bias: Tensor | None, weight: Tensor, x: Tensor) -> None:
    if bias is not None and (
        not bias.dtype.is_floating_point
        or bias.shape != (weight.shape[0],)
        or bias.device != x.device
    ):
        raise ValueError(
            f"subm_conv3d: bias must be a float ({weight.shape[0]},) tensor on {x.device} or "
            f"None, got {bias.dtype} {tuple(bias.shape)} on {bias.device}"
        )


def _check(name: str, weight: Tensor | None, kmap: Tensor, **feats: tuple[Tensor, int]) -> None:
    """Ranks, shapes, dtypes and devices (no device read, safe under tracing). ``feats`` maps each
    feature argument to ``(tensor, channels)``; every feature tensor has one row per kmap row (the
    kernels read ``kmap`` for each of them)."""
    if kmap.dtype != torch.int32 or kmap.dim() != 2:
        raise TypeError(
            f"{name}: kmap must be a 2-D int32 tensor (build_kmap), got {kmap.dtype} "
            f"{tuple(kmap.shape)}"
        )
    if weight is not None and (
        weight.dim() != 3 or not weight.dtype.is_floating_point or weight.shape[1] != kmap.shape[1]
    ):
        raise ValueError(
            f"{name}: weight must be a float (C_out, {kmap.shape[1]}, C_in) tensor, got "
            f"{weight.dtype} {tuple(weight.shape)}"
        )
    tensors = [kmap] if weight is None else [kmap, weight]
    for arg, (t, channels) in feats.items():
        if not t.dtype.is_floating_point or t.dim() != 2 or t.shape[1] != channels:
            raise ValueError(
                f"{name}: {arg} must be a float (N, {channels}) tensor, got {t.dtype} "
                f"{tuple(t.shape)}"
            )
        if t.shape[0] != kmap.shape[0]:
            raise ValueError(f"{name}: {arg} has {t.shape[0]} rows but kmap has {kmap.shape[0]}")
        tensors.append(t)
    if len({t.device for t in tensors}) > 1:
        raise ValueError(f"{name}: inputs on several devices: {[str(t.device) for t in tensors]}")
