# Vendored from Pointcept2 src/pointcept2/models/layers/mlp.py @ 771a949 (simplified for pointtransformerv3).
"""Two-layer perceptron of transformer blocks."""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["Mlp"]


class Mlp(nn.Module):
    """``fc2(GELU(fc1(x)))`` with the exact (erf) GELU: PC's ``MLP`` without dropout, same
    parameter names (``fc1``, ``fc2``)."""

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int | None = None
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        for name, value in (
            ("in_channels", in_channels),
            ("hidden_channels", hidden_channels),
            ("out_channels", out_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x: ``(..., in_channels)`` float.
            returns: ``(..., out_channels)`` (in the autocast dtype under autocast).
        """
        return self.fc2(self.act(self.fc1(x)))
