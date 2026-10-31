"""Submanifold sparse 3-D convolution with switchable backends (pure torch or Triton).

Functional op ``subm_conv3d`` with weight ``(C_out, k, k, k, C_in)``, the spconv 2.x layout, so
checkpoints trained with spconv load as they are. The module holding the parameters
(``SubMConv3d``) lives in ``point_transformer_v3.layers.sparse_conv``.
"""

from point_transformer_v3.ops.sparse_conv.functional import (
    SparseConvBackend,
    SparseGrid,
    get_backend,
    set_backend,
    subm_conv3d,
)

__all__ = ["SparseConvBackend", "SparseGrid", "get_backend", "set_backend", "subm_conv3d"]
