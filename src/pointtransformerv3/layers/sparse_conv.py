# Vendored from Pointcept2 src/pointcept2/models/layers/sparse_conv.py @ 771a949 (simplified for pointtransformerv3).
"""Submanifold sparse 3-D convolution module ``SubMConv3d``. It holds the parameters and calls
the functional op of ``pointtransformerv3.ops.sparse_conv``; the backend is chosen there
(``set_backend``), per call.

Parameters use the spconv 2.x layout ``(C_out, k, k, k, C_in)``, so switching backends never
touches the weights and checkpoints trained with spconv load as they are.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from pointtransformerv3.ops.sparse_conv import SparseGrid, subm_conv3d
from pointtransformerv3.ops.sparse_conv.functional import skip_warning_frames_in

# Fallback warnings skip this file's forward() frames and name the model line that called it.
skip_warning_frames_in(__file__)

__all__ = ["SubMConv3d"]


class SubMConv3d(nn.Module):
    """Submanifold 3-D convolution: stride 1, outputs on the input points, cubic odd kernel.

    Computes ``out[p] = bias + sum_k weight[:, k] @ x[kmap[p, k]]`` (cross-correlation, as spconv).
    The backend is chosen globally (``set_backend``) and resolved per call.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True
    ) -> None:
        super().__init__()
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
            raise TypeError(f"kernel_size must be an int, got {kernel_size!r}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        k = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, k, k, k, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """spconv's default init: weight and bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)),
        ``fan_in = C_in * k**3`` (kaiming-uniform with a = sqrt(5), drawn in spconv's order)."""
        fan_in = self.in_channels * self.kernel_size**3
        gain = nn.init.calculate_gain("leaky_relu", math.sqrt(5))
        bound = math.sqrt(3.0) * (gain / math.sqrt(fan_in))
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                bias_bound = 1 / math.sqrt(fan_in)
                self.bias.uniform_(-bias_bound, bias_bound)

    def forward(self, x: Tensor, grid: SparseGrid) -> Tensor:
        """
        Shapes:
            x: ``(N, C_in)`` float, row i at ``grid.grid_coord[i]``.
            returns: ``(N, C_out)`` in the autocast dtype under autocast, else in x's dtype.
        """
        return subm_conv3d(x, self.weight, self.bias, grid)

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"bias={self.bias is not None}"
        )
