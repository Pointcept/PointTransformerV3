# Vendored from Pointcept2 src/pointcept2/models/layers/drop_path.py @ 771a949 (simplified for point_transformer_v3).
"""Stochastic depth over packed rows."""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["DropPath"]


class DropPath(nn.Module):
    """Drop a residual branch per row (token) with probability ``drop_prob`` in training and scale
    the kept rows by ``1 / (1 - drop_prob)``; the identity in eval or when ``drop_prob == 0``.

    The same math and RNG draws as timm's ``drop_path`` (PC's ``DropPath``): one Bernoulli draw
    per row from torch's global RNG (the engine seeds it per step), made in x's dtype on x's device,
    so equal seeds give bitwise equal outputs. In a packed batch a row is a token, so tokens of one
    sample are dropped independently.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if isinstance(drop_prob, bool) or not isinstance(drop_prob, int | float):
            raise TypeError(f"drop_prob must be a float, got {drop_prob!r}")
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            x: ``(N, ...)`` float, one row per token.
            returns: x's shape and dtype; rows are 0 or ``x[i] / (1 - drop_prob)``.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            mask.div_(keep_prob)
        return x * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"
