# Vendored from Pointcept2 src/pointcept2/ops/_tracing.py @ 771a949 (simplified for pointtransformerv3).
"""Compiler markers that cost nothing at import time; the eager-or-traced test of Triton ops.

``torch.compiler.assume_constant_result`` imports ``torch._dynamo`` when it decorates, and
``torch._dynamo`` imports ``torch._inductor`` and triton (about 0.6 s, D22: importing a module must
not import triton). ``assume_constant_result`` here sets the same marker without that import.

``tracing`` decides how an op with Triton kernels runs a call: eagerly through a plain
``torch.autograd.Function`` that launches the kernels itself, or through its
``torch.library.triton_op`` while a graph is being traced (I5: the op's dispatch and custom-op
autograd cost about 120 us of host time per conv forward + backward and 85 us per LayerNorm).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import torch
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

__all__ = ["CONSTANT_RESULT_MARKER", "assume_constant_result", "tracing"]

F = TypeVar("F", bound=Callable[..., Any])

# What torch._dynamo.decorators.assume_constant_result sets (torch 2.12) and what dynamo reads
# (torch/_dynamo/variables/functions.py); tests/ops/test_tracing.py checks the two still agree.
CONSTANT_RESULT_MARKER = "_dynamo_marked_constant"


def assume_constant_result(fn: F) -> F:
    """``torch.compiler.assume_constant_result`` without importing ``torch._dynamo``.

    torch.compile / torch.export call ``fn`` eagerly while they trace (its arguments must be
    constants) and bake its result into the graph, with no guard on what it read: use it for
    choices that are fixed for the process (backend names, device capabilities).
    """
    setattr(fn, CONSTANT_RESULT_MARKER, True)
    return fn


def tracing(x: torch.Tensor) -> bool:
    """Whether a call on ``x`` is being traced rather than run: torch.compile or strict
    torch.export (dynamo), a dispatch mode such as the fake and proxy modes of non-strict
    torch.export, make_fx and AOTAutograd, or a tensor subclass input. Ops with Triton kernels
    take their ``triton_op`` then, so the graph holds the op; otherwise they launch the same
    kernels from a plain ``autograd.Function`` (same configs, bitwise equal results)."""
    return (
        torch.compiler.is_compiling() or type(x) is not torch.Tensor or is_in_torch_dispatch_mode()
    )
