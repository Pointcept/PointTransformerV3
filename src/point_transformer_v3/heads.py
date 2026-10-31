"""Heads and feature upcasting.

``upcast`` walks an encoder output (a ``Point`` with ``pooling_parent`` / ``pooling_inverse``) back
to level 0, concatenating the features of the levels it passes (Sonata's demos), so every level-0
row gets a multi-scale feature: 48 + 96 + 192 + 384 + 512 = 1232 channels for ``sonata``. A
decoder output is already at level 0 and is returned as it is.

Segmentation heads are a plain ``nn.Linear`` on the model (``model.seg_head``, state-dict keys
``seg_head.weight`` / ``seg_head.bias``, as Pointcept's ``DefaultSegmentorV2`` and Sonata's linear
probe), applied to the upcast features.
"""

from __future__ import annotations

import torch

from point_transformer_v3.structure import Point

__all__ = ["upcast"]


def upcast(point: Point, *, levels: int | None = None) -> Point:
    """The level-0 ``Point`` of an encoder output with multi-scale features.

    Args:
        point: a model output. Not modified.
        levels: how many coarse-to-fine steps concatenate (``parent.feat = cat([parent.feat,
            feat[inverse]])``); the remaining steps only copy the features down
            (``parent.feat = feat[inverse]``). None (default) concatenates every level (Sonata's
            linear probe: 1232 channels); 2 is Sonata's PCA / similarity demos (1088 channels).

    Returns:
        A new level-0 ``Point`` (the original keys, e.g. a ``prepare`` inverse, are kept) whose
        ``feat`` is ``(N0, C)``. A decoder output (no ``pooling_parent``) is returned unchanged.
    """
    if "pooling_parent" not in point:
        return point
    feat = point["feat"]
    step = 0
    while "pooling_parent" in point:
        parent = point["pooling_parent"]
        inverse = point["pooling_inverse"]
        up = feat[inverse]
        concat = levels is None or step < levels
        feat = torch.cat([parent["feat"], up], dim=-1) if concat else up
        point = parent
        step += 1
    out = Point(point)
    out["feat"] = feat
    return out
