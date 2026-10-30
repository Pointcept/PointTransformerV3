# Vendored from Pointcept2 src/pointcept2/models/layers/norm.py @ 771a949 (simplified for pointtransformerv3).
"""Norms that run in the activation dtype (C10) and the one place norms are built (C11).

Under CUDA autocast torch runs ``layer_norm`` in fp32: a bf16 activation is upcast, normalized and
leaves the norm in fp32, and the next Linear casts it back to bf16. ``LayerNorm`` and
``BatchNorm1d`` keep the activation dtype instead, with fp32 statistics inside the kernel. Both
run with autocast off:

- ``LayerNorm``: ``ops.norm.layer_norm``. On CUDA a Triton kernel (I1): fp32 mean and
  variance, fp32 weight and bias, the output rounded once to the activation dtype; fp32 weight and
  bias gradients. Elsewhere (CPU, MPS, fp64) torch's kernel: in fp32 for a bf16/fp16 input
  (torch's CPU kernel sums the weight and bias gradients in bf16: cosine 0.9995 against fp32 at 8k
  rows). See that module for the backends and their numerics.
- ``BatchNorm1d``: torch's mixed-dtype kernel (bf16/fp16 input, fp32 weight, bias and running
  statistics). Batch statistics accumulate in fp32, the fp32 running statistics update in place
  and the weight and bias gradients come out in fp32.

An fp32 input under autocast is normalized in fp32 and the output is cast to the autocast dtype,
so no fp32 activation leaves a norm. Outside autocast the output has the input dtype; fp32 math
is that of ``nn.LayerNorm`` / ``nn.BatchNorm1d`` (bitwise off CUDA; on CUDA the Triton LayerNorm
is within ~1e-6 of torch's kernel). Parameters, buffers and state_dict keys are those of the
torch classes.

Measured on an RTX PRO 6000 (sm_120), torch 2.12.1, 1M x 64 bf16 rows, forward + backward under
bf16 autocast, ms, medians of 30 interleaved runs (``research/m2/p1/bench_norm.py``), torch's
``nn.LayerNorm`` (fp32 output, cast again by the Linear) -> ``LayerNorm`` with torch's kernel in
bf16 (before the Triton kernel):

    LayerNorm + Linear(64, 64) / (64, 192) / (64, 256)    3.81 -> 2.52 / 4.54 -> 3.22 / 4.82 -> 3.51
    LayerNorm alone                                        2.93 -> 1.83

Against an fp32 reference (fp32 input, parameters and upstream gradient) the bf16 outputs and the
input, weight and bias gradients of both norms have cosine >= 0.99999.
"""

from __future__ import annotations

from typing import Literal, get_args

import torch
from torch import Tensor, nn
from torch.amp.autocast_mode import is_autocast_available

from pointtransformerv3.ops.norm import layer_norm
from pointtransformerv3.ops.sparse_conv.functional import skip_warning_frames_in

__all__ = ["BN_EPS", "BN_MOMENTUM", "LN_EPS", "BatchNorm1d", "LayerNorm", "NormKind", "make_norm"]

NormKind = Literal["ln", "bn"]

# PTv3 constants (PC point_transformer_v3m1_base.py: BatchNorm1d(eps=1e-3, momentum=0.01),
# nn.LayerNorm defaults).
LN_EPS = 1e-5
BN_EPS = 1e-3
BN_MOMENTUM = 0.01

skip_warning_frames_in(__file__)  # a fallback warning names the model line, not this file


def make_norm(kind: NormKind, channels: int) -> LayerNorm | BatchNorm1d:
    """``"ln"``: ``LayerNorm(channels, eps=1e-5)``; ``"bn"``: ``BatchNorm1d(channels, eps=1e-3,
    momentum=0.01)``. Every norm of a model is built here (C11)."""
    if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
        raise ValueError(f"channels must be a positive int, got {channels!r}")
    if kind == "ln":
        return LayerNorm(channels, eps=LN_EPS)
    if kind == "bn":
        return BatchNorm1d(channels, eps=BN_EPS, momentum=BN_MOMENTUM)
    raise ValueError(f"unknown norm kind {kind!r}; use one of {get_args(NormKind)}")


class LayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` that keeps the activation dtype under autocast (see the module doc)."""

    def forward(self, input: Tensor) -> Tensor:
        """
        Shapes:
            input: ``(..., *normalized_shape)`` float.
            returns: input's shape and dtype (for an fp32 input under autocast: the autocast
                dtype).
        """
        out_dtype = _out_dtype(input, _autocast_dtype(input))
        return layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps, out_dtype=out_dtype
        )


class BatchNorm1d(nn.BatchNorm1d):
    """``nn.BatchNorm1d`` that keeps the activation dtype under autocast (see the module doc)."""

    def forward(self, input: Tensor) -> Tensor:
        """
        Shapes:
            input: ``(N, C)`` or ``(N, C, L)`` float.
            returns: input's shape and dtype (for an fp32 input under autocast: the autocast
                dtype).
        """
        autocast = _autocast_dtype(input)
        if autocast is None:
            return super().forward(input)
        with torch.autocast(input.device.type, enabled=False):
            y = super().forward(input)
        return y.to(_out_dtype(input, autocast))


def _autocast_dtype(x: Tensor) -> torch.dtype | None:
    """The autocast dtype of x's device type, or None when autocast is off there (or absent, as
    on the meta device)."""
    device = x.device.type
    if is_autocast_available(device) and torch.is_autocast_enabled(device):
        return torch.get_autocast_dtype(device)
    return None


def _out_dtype(x: Tensor, autocast: torch.dtype | None) -> torch.dtype:
    """A norm's output dtype: x's, or the autocast dtype for an fp32 x under autocast."""
    return autocast if autocast is not None and x.dtype == torch.float32 else x.dtype
