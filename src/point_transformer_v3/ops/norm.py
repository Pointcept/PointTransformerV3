# Vendored from Pointcept2 src/pointcept2/ops/norm.py @ 771a949 (simplified for point_transformer_v3).
"""Layer normalization over the last dim in the activation dtype, with fp32 statistics (C10).

``layer_norm`` is the functional behind ``models.layers.norm.LayerNorm`` (built by ``make_norm``,
C11). Backends, chosen per call:

- ``triton`` (``_norm_triton``): CUDA, fp32/fp16/bf16 input and output, a 1-D normalized
  shape with weight and bias, C <= 4096. One row-tile forward kernel (fp32 mean and variance,
  ``rsqrt(var + eps)``, fp32 weight and bias, the output rounded once) and one persistent backward
  kernel (input gradient in the input dtype, fp32 weight and bias gradients reduced in a fixed
  order: bitwise reproducible). Eager calls launch the kernels from a plain autograd Function;
  traced graphs (torch.compile / torch.export) record ``point_transformer_v3::layer_norm`` (D15).
- ``torch``: ``F.layer_norm`` with autocast off, everywhere else (CPU, MPS, fp64, other shapes,
  and CUDA without a working Triton). On CUDA it runs torch's kernel in the input dtype with the
  weight and bias cast to it; on other devices a bf16/fp16 input is computed in fp32 (torch's CPU
  kernel sums the weight and bias gradients in bf16: cosine 0.9995 against fp32 at 8k rows).

``auto`` (the default, or env ``PTV3_LAYER_NORM_BACKEND``) takes ``triton`` where it applies;
``set_layer_norm_backend`` overrides it. Each choice is logged once per (device, dtype); on CUDA a
fallback to ``torch`` under ``auto`` warns once (C18). The output is the input dtype unless
``out_dtype`` says otherwise (the module passes the autocast dtype for an fp32 input).

Against torch's kernel the Triton path differs in rounding only (I1, D29): bf16 outputs within
1 ulp of fp64 math, fp32 outputs and gradients within ~1e-6 relative; the weight and bias stay
fp32 inside the kernel instead of being rounded to bf16 first, and their gradients are summed in
fp32. 1M x 32 bf16 rows (RTX PRO 6000): forward + backward about 5x faster than torch's kernel
(research/accel/r1/reports/norm-mlp-fusion.md; integration numbers in
research/accel/integrate_a/report.md).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp.autocast_mode import is_autocast_available

from point_transformer_v3.ops._tracing import assume_constant_result, tracing
from point_transformer_v3.ops._triton import (
    TRITON_MIN_CAPABILITY,
    available,
    skip_warning_frames_in,
    toolchain_problem,
    warn_at_caller,
)

__all__ = ["LayerNormBackend", "get_layer_norm_backend", "layer_norm", "set_layer_norm_backend"]

LayerNormBackend = Literal["auto", "torch", "triton"]

_BACKENDS: tuple[str, ...] = ("auto", "torch", "triton")
_ENV = "PTV3_LAYER_NORM_BACKEND"
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_LOW_PRECISION = (torch.bfloat16, torch.float16)
_MAX_CHANNELS = 4096  # one row tile per program; wider rows take the torch path
# No tl.dot: any GPU Triton compiles for (Triton's floor, as the depthwise conv).
_MIN_CAPABILITY = TRITON_MIN_CAPABILITY

log = logging.getLogger(__name__)
skip_warning_frames_in(__file__)  # the fallback warning names the caller's line
_backend: str | None = None
_noted: set[tuple[str, str, str, str]] = set()
_function: Any = None  # _norm_triton.LayerNormFunction once the kernels are loaded


def set_layer_norm_backend(name: LayerNormBackend) -> None:
    """Choose the backend of every later ``layer_norm`` call (overrides the env var). Graphs
    already traced keep the backend they were traced with."""
    global _backend
    _backend = _check_backend(name, "set_layer_norm_backend")


def get_layer_norm_backend() -> str:
    """The configured backend: ``set_layer_norm_backend``, else env ``PTV3_LAYER_NORM_BACKEND``,
    else ``"auto"``."""
    global _backend
    if _backend is None:
        _backend = _check_backend(os.environ.get(_ENV, "auto"), _ENV)
    return _backend


def layer_norm(
    x: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
    *,
    out_dtype: torch.dtype | None = None,
) -> Tensor:
    """``F.layer_norm`` with fp32 statistics that keeps the activation dtype (module doc).

    Shapes:
        x: ``(..., *normalized_shape)`` float.
        weight, bias: ``normalized_shape`` or None.
        returns: x's shape in ``out_dtype`` (default x's dtype). Gradients come back in the
        dtypes of x, weight and bias.

    Sync-free; no autotuning (fixed launch configs per channel count).
    """
    out_dtype = x.dtype if out_dtype is None else out_dtype
    c = x.shape[-1] if x.dim() else 0
    if (  # anything else, mismatched shapes and devices included, goes to torch (and its errors)
        tuple(normalized_shape) == (c,)
        and weight is not None
        and bias is not None
        and weight.shape == bias.shape == (c,)
        and 0 < c <= _MAX_CHANNELS
        and weight.device == x.device == bias.device
        and _choose(x.device, x.dtype, out_dtype) == "triton"
    ):
        rows = x.reshape(-1, c).contiguous()
        args = (rows, weight.contiguous(), bias.contiguous(), eps, out_dtype)
        if tracing(rows):  # traced graphs keep the op (D15)
            return torch.ops.point_transformer_v3.layer_norm(*args)[0].view(x.shape)
        return _function.apply(*args).view(x.shape)  # I5: the kernels without the op dispatch
    return _torch_layer_norm(x, normalized_shape, weight, bias, eps, out_dtype)


def _torch_layer_norm(
    x: Tensor,
    normalized_shape: tuple[int, ...],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
    out_dtype: torch.dtype,
) -> Tensor:
    """torch's kernel with autocast off: in the input dtype on CUDA (weight and bias cast to it),
    in fp32 for a low-precision input elsewhere."""
    device = x.device.type
    if x.dtype in _LOW_PRECISION and device != "cuda":
        x = x.float()
    weight = None if weight is None else weight.to(x.dtype)
    bias = None if bias is None else bias.to(x.dtype)
    if is_autocast_available(device) and torch.is_autocast_enabled(device):
        with torch.autocast(device, enabled=False):
            y = F.layer_norm(x, normalized_shape, weight, bias, eps)
    else:
        y = F.layer_norm(x, normalized_shape, weight, bias, eps)
    return y.to(out_dtype)


@assume_constant_result
def _choose(device: torch.device, dtype: torch.dtype, out_dtype: torch.dtype) -> str:
    """``"triton"`` or ``"torch"`` for ``dtype`` rows on ``device``; logs each choice once. Runs
    eagerly even while torch.compile / torch.export trace (its result is baked into the graph), so
    the toolchain probe and logging never enter a graph."""
    global _function
    name = get_layer_norm_backend()
    on_cuda = device.type == "cuda" and dtype in _DTYPES and out_dtype in _DTYPES
    if name == "torch":
        choice, reason = "torch", "forced"
    elif not on_cuda:
        if name == "triton":
            raise ValueError(
                f"the triton layer_norm needs fp32/fp16/bf16 CUDA rows, got {dtype} on {device}; "
                "use backend 'torch' or 'auto'"
            )
        choice, reason = "torch", f"only option for {str(dtype).removeprefix('torch.')} on {device}"
    elif name == "triton" or available(device, min_capability=_MIN_CAPABILITY):
        problem = toolchain_problem(device, min_capability=_MIN_CAPABILITY)
        if problem is not None:  # forced
            raise RuntimeError(
                f"the triton layer_norm backend cannot run here: {problem}. Use backend 'auto' "
                "or 'torch' instead."
            )
        choice, reason = "triton", "forced" if name == "triton" else "fastest on CUDA"
    else:
        problem = toolchain_problem(device, min_capability=_MIN_CAPABILITY)
        choice, reason = "torch", f"triton unavailable: {problem}"
    key = (name, str(device), str(dtype), choice)
    if key not in _noted:
        _noted.add(key)
        log.info(
            "layer_norm on %s %s: %s backend (%s: %s)",
            device,
            str(dtype).removeprefix("torch."),
            choice,
            name,
            reason,
        )
        if on_cuda and choice == "torch" and name != "torch":
            warn_at_caller(
                f"layer_norm falls back to torch's slower kernel on CUDA ({reason}). Fix the "
                "Triton toolchain for the fast path."
            )
    if choice == "triton" and _function is None:
        from point_transformer_v3.ops import _norm_triton  # imports triton, registers the ops

        _function = _norm_triton.LayerNormFunction
    return choice


def _check_backend(name: str, source: str) -> str:
    if name not in _BACKENDS:
        raise ValueError(f"{source}: unknown layer_norm backend {name!r}; choose from {_BACKENDS}")
    return name
