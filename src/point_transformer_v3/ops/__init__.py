"""Functional ops and their kernels: tensors in, tensors out (no ``nn.Module``).

Keys, grid sampling, serialization, kernel maps, attention patches, pooling, row gathers, attention
(serialized and varlen), layer norm and submanifold sparse conv. Each op chooses its backend
inside; the kernels are private files next to the op that uses them (imported on first use, so
importing this package does not import triton). The backend switches stay in their modules
(``ops.sparse_conv.set_backend``, ``ops.attention.set_attention_backend``,
``ops.norm.set_layer_norm_backend``); ``point_transformer_v3.set_backend`` sets all three.

Vendored from Pointcept2 ``src/pointcept2/ops`` @ 771a949 (spconv and depthwise conv removed).
"""

from point_transformer_v3.ops.attention import (
    AttentionBackend,
    get_attention_backend,
    serialized_attention,
    set_attention_backend,
    varlen_attention,
)
from point_transformer_v3.ops.gather import gather_rows, permute_rows
from point_transformer_v3.ops.grid_sample import GridSampled, Pick, grid_sample, lattice
from point_transformer_v3.ops.keys import AXIS_BITS, BATCH_SHIFT, pack_keys, unpack_keys
from point_transformer_v3.ops.neighbors import (
    build_kmap,
    check_kmap,
    kernel_offsets,
    kmap_from_coarse,
)
from point_transformer_v3.ops.norm import (
    LayerNormBackend,
    get_layer_norm_backend,
    layer_norm,
    set_layer_norm_backend,
)
from point_transformer_v3.ops.patches import PatchLayout, Tail, build_patches
from point_transformer_v3.ops.pooling import (
    REDUCES,
    PoolIndex,
    Reduce,
    build_pool_index,
    pool,
    unpool,
)
from point_transformer_v3.ops.seqlens import cu_seqlens_of
from point_transformer_v3.ops.serialization import (
    MAX_DEPTH,
    ORDERS,
    Order,
    decode,
    encode,
    serialize,
)
from point_transformer_v3.ops.sparse_conv import SparseConvBackend, SparseGrid, subm_conv3d

__all__ = [
    "AXIS_BITS",
    "BATCH_SHIFT",
    "MAX_DEPTH",
    "ORDERS",
    "REDUCES",
    "AttentionBackend",
    "GridSampled",
    "LayerNormBackend",
    "Order",
    "PatchLayout",
    "Pick",
    "PoolIndex",
    "Reduce",
    "SparseConvBackend",
    "SparseGrid",
    "Tail",
    "build_kmap",
    "build_patches",
    "build_pool_index",
    "check_kmap",
    "cu_seqlens_of",
    "decode",
    "encode",
    "gather_rows",
    "get_attention_backend",
    "get_layer_norm_backend",
    "grid_sample",
    "kernel_offsets",
    "kmap_from_coarse",
    "lattice",
    "layer_norm",
    "pack_keys",
    "permute_rows",
    "pool",
    "serialize",
    "serialized_attention",
    "set_attention_backend",
    "set_layer_norm_backend",
    "subm_conv3d",
    "unpack_keys",
    "unpool",
    "varlen_attention",
]
