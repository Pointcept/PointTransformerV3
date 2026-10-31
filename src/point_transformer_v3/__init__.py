"""Point Transformer V3 in pure PyTorch + Triton: PTv3, Sonata, Concerto and Utonia weights, one
model class (``PRESETS``).

Quick start::

    import point_transformer_v3 as ptv3

    model = ptv3.load("sonata")                     # downloads the weights once (CC BY-NC 4.0)
    feat = model.embed(coord, color, normal)        # (N, 1232) per input point

    point, inverse = ptv3.prepare(coord, color, normal, preset="sonata", device="cuda")
    out = ptv3.upcast(model(point))                 # level-0 features of the kept points
    feat = out.feat[inverse]                        # back to the input points

Exports are lazy: ``import point_transformer_v3`` imports neither torch, triton nor huggingface_hub;
the first attribute access imports what it needs (torch for the model, triton only when a CUDA
kernel first runs, huggingface_hub only when weights are downloaded).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.2"

_EXPORTS = {
    "load": "point_transformer_v3.api",
    "warmup": "point_transformer_v3.api",
    "set_backend": "point_transformer_v3.api",
    "PointTransformerV3": "point_transformer_v3.model",
    "Config": "point_transformer_v3.model",
    "Point": "point_transformer_v3.structure",
    "prepare": "point_transformer_v3.preprocess",
    "collate": "point_transformer_v3.preprocess",
    "upcast": "point_transformer_v3.heads",
    "PRESETS": "point_transformer_v3.presets",
    "get_preset": "point_transformer_v3.presets",
}

__all__ = [
    "PRESETS",
    "Config",
    "Point",
    "PointTransformerV3",
    "__version__",
    "collate",
    "get_preset",
    "load",
    "prepare",
    "set_backend",
    "upcast",
    "warmup",
]

if TYPE_CHECKING:  # static analyzers see the lazy exports
    from point_transformer_v3.api import load, set_backend, warmup
    from point_transformer_v3.heads import upcast
    from point_transformer_v3.model import Config, PointTransformerV3
    from point_transformer_v3.preprocess import collate, prepare
    from point_transformer_v3.presets import PRESETS, get_preset
    from point_transformer_v3.structure import Point


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'point_transformer_v3' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
