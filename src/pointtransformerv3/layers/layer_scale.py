# Vendored from Pointcept2 src/pointcept2/models/layers/layer_scale.py @ 771a949 (simplified for pointtransformerv3).
"""Per-channel residual branch scale (CaiT LayerScale)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["LayerScale"]


class LayerScale(nn.Module):
    """``x * gamma`` with a learnable ``gamma`` of shape ``(channels,)`` set to ``init_value``
    (PC m2 / timm ``LayerScale``, parameter name ``gamma``).

    The product follows torch's type promotion, as in PC: under autocast a bf16 branch times the
    fp32 ``gamma`` is fp32, so a residual stream it is added to stays fp32. With small gammas
    (1e-5) a bf16 stream would round most branch updates away.
    """

    def __init__(self, channels: int, init_value: float = 1e-5) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError(f"channels must be a positive int, got {channels!r}")
        self.init_value = float(init_value)
        self.gamma = nn.Parameter(torch.empty(channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.gamma, self.init_value)

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x: ``(..., channels)`` float.
            returns: x's shape, in ``torch.promote_types(x.dtype, gamma.dtype)``.
        """
        return x * self.gamma

    def extra_repr(self) -> str:
        return f"{self.gamma.shape[0]}, init_value={self.init_value:g}"
