# Vendored from Pointcept2 src/pointcept2/ops/sparse_conv/functional.py @ 771a949 (simplified for point_transformer_v3).
"""Submanifold sparse 3-D convolution: the functional entry point and the global backend switch.

Backends:

- ``torch`` (``torch_impl``): gather + GEMM in pure torch on any device; always available. It is
  the CPU (and MPS) implementation, kept first-class for CPU inference, and the CUDA fallback.
- ``triton`` (``triton_impl``): Triton implicit-GEMM kernels on CUDA over the grid's kernel map.
  Forcing it checks that Triton can launch here (triton, CUDA, compute capability 8.0+ on every
  visible GPU, Triton's cached CUDA driver helper) and raises ``RuntimeError`` naming what is
  missing.
- ``auto`` (the default, or env ``PTV3_CONV_BACKEND``): per call, ``triton`` for
  fp32/fp16/bf16 features on a CUDA device where the triton backend can run
  (``triton_impl.available(device)``: checked per device, so mixed GPU types work), ``torch``
  everywhere else. On CUDA without a working Triton it warns once, naming the missing piece: the
  torch path is 2-5x slower there.

The configured name (``set_backend``, the env var, a per-call ``backend``; a forced ``triton``
probes the toolchain) and the choice are resolved outside traced graphs
(``assume_constant_result``), so torch.compile / torch.export bake in the backend configured when
they traced and never trace the env lookup or the toolchain probe, also when the first conv call of
a process is the one being traced. Each resolved choice is logged once per (backend, choice, kernel
size, C_in), never while compiling (dynamo cannot trace ``logging``). The slow-fallback warning
names the first caller outside torch and this package (``warn_at_caller``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Hashable
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from point_transformer_v3.ops._tracing import assume_constant_result
from point_transformer_v3.ops._triton import skip_warning_frames_in, warn_at_caller
from point_transformer_v3.ops.sparse_conv import torch_impl, triton_impl

__all__ = [
    "SparseConvBackend",
    "SparseGrid",
    "get_backend",
    "set_backend",
    "subm_conv3d",
]

SparseConvBackend = Literal["auto", "torch", "triton"]

_BACKENDS: tuple[str, ...] = ("auto", "torch", "triton")
_ENV = "PTV3_CONV_BACKEND"
_CUDA_DTYPES = (torch.float32, torch.float16, torch.bfloat16)  # the triton kernels


log = logging.getLogger(__name__)
_backend: str | None = None
_noted: set[tuple[str, int, int, str]] = set()
# A fallback warning skips torch's frames and this package's (subm_conv3d, the depthwise conv) to
# reach the user's line. Module wrappers in higher layers add their own file with
# ``skip_warning_frames_in`` (re-exported here from ``ops._triton``; ops never import or name them).
skip_warning_frames_in(os.path.dirname(__file__) + os.sep)


class SparseGrid(Protocol):
    """Geometry a submanifold conv runs on: unique grid cells of B samples at one resolution.

    Implemented by ``PointGrid``; any object with these members works.
    """

    grid_coord: Tensor  # (N, 3) int32
    cu_seqlens: Tensor  # (B + 1,) int32
    seq_lens: tuple[int, ...]
    cache: dict[Hashable, Any]  # cached index tables (kernel maps per kernel size)

    def batch_index(self) -> Tensor:
        """(N,) int64 sample index of each row."""
        ...

    def kmap(self, kernel_size: int) -> Tensor:
        """(N, kernel_size**3) int32 ``build_kmap`` of the grid, cached."""
        ...


def set_backend(name: SparseConvBackend) -> None:
    """Choose the backend of every later eager ``subm_conv3d`` call (overrides the env var).

    Graphs that torch.compile / torch.export already traced keep the backend they were traced
    with (the name is resolved in ``_configured``, an ``assume_constant_result`` function, so no
    guard and no recompile): call ``torch._dynamo.reset()`` after switching to retrace them.
    ``"triton"`` raises ``RuntimeError`` when Triton cannot run here (``triton_impl.require``).
    """
    global _backend
    _backend = _check_backend(name, "set_backend")


def get_backend() -> str:
    """The configured backend: ``set_backend``, else env ``PTV3_CONV_BACKEND``, else ``"auto"``."""
    global _backend
    if _backend is None:
        _backend = _check_backend(os.environ.get(_ENV, "auto"), _ENV)
    return _backend


def subm_conv3d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    grid: SparseGrid,
    *,
    backend: SparseConvBackend | None = None,
) -> Tensor:
    """Submanifold 3-D cross-correlation (no kernel flip) on the points of ``grid``.

    ``out[p] = bias + sum_k weight[:, k] @ x[kmap[p, k]]`` over the ``K = k**3`` offsets of
    ``kernel_offsets(k)``, with missing neighbors contributing zero.

    Shapes:
        x: ``(N, C_in)`` float, row i at ``grid.grid_coord[i]``.
        weight: ``(C_out, k, k, k, C_in)``, the spconv 2.x layout; k odd.
        bias: ``(C_out,)`` or None.
        returns: ``(N, C_out)`` in the autocast dtype under autocast, else in x's dtype.

    ``backend`` overrides the global choice for this call. An empty input always takes the torch
    path.
    """
    kernel_size = _check_args(x, weight, bias, grid)
    name = _configured(backend)
    if x.shape[0]:
        choice = _choose(name, x.device, x.dtype)
        _note(name, choice, kernel_size, weight.shape[-1], x)
        if choice == "triton":
            return triton_impl.subm_conv3d(x, weight, bias, grid.kmap(kernel_size))
    return torch_impl.subm_conv3d(x, weight, bias, grid.kmap(kernel_size))


@assume_constant_result
def _configured(backend: str | None) -> str:
    """The backend name of a call: ``backend`` validated, else ``get_backend()`` (``set_backend``,
    else the env var). A forced ``triton`` probes the toolchain here. Runs eagerly even while
    torch.compile / torch.export trace (its result is baked into the graph), so a cold capture
    never traces the env lookup or the probe (``importlib.util.find_spec``, the driver helper)."""
    return get_backend() if backend is None else _check_backend(backend, "backend")


@assume_constant_result
def _choose(name: str, device: torch.device, dtype: torch.dtype) -> str:
    """The backend of a call: ``"torch"`` or ``"triton"``. Raises for a forced backend that
    cannot take these features. Runs eagerly even while torch.compile / torch.export trace
    (its result is baked into the graph), so the toolchain probe never enters a graph."""
    on_cuda = device.type == "cuda" and dtype in _CUDA_DTYPES
    if name == "torch":
        return "torch"
    if name == "triton" and not on_cuda:
        raise ValueError(
            f"the {name} backend needs fp32/fp16/bf16 CUDA features, got {dtype} on {device}; "
            "use backend 'torch' or 'auto'"
        )
    if name == "triton":
        triton_impl.require(device)  # registers the ops; set_backend checked the toolchain
        return "triton"
    if not on_cuda:
        return "torch"
    if triton_impl.available(device):
        triton_impl.require(device)  # registers the ops and logs the toolchain once
        return "triton"
    return "torch"


def _note(name: str, choice: str, kernel_size: int, in_channels: int, x: Tensor) -> None:
    """Log each (backend, choice, kernel_size, in_channels) once; warn once when CUDA features
    fall back to the torch path under ``auto``. Skipped while compiling (dynamo cannot trace
    logging; the eager calls before or after log the same choice)."""
    if torch.compiler.is_compiling():
        return
    key = (name, kernel_size, in_channels, choice)
    if key in _noted:
        return
    _noted.add(key)
    reason, warn = _reason(name, choice, kernel_size, x)
    log.info(
        "subm_conv3d k=%d C_in=%d: %s backend (%s: %s)",
        kernel_size,
        in_channels,
        choice,
        name,
        reason,
    )
    if warn:
        warn_at_caller(
            f"subm_conv3d k={kernel_size} on {x.device} falls back to the slow torch backend "
            f"({reason}); the triton backend is 2-5x faster on CUDA. Fix the Triton toolchain "
            "for the fast path."
        )


def _reason(name: str, choice: str, kernel_size: int, x: Tensor) -> tuple[str, bool]:
    """Why ``auto`` (or a forced backend) took ``choice``, and whether that is a slow fallback."""
    if name != "auto":
        return "forced", False
    if x.device.type != "cuda":
        return f"only option on {x.device.type}", False
    if x.dtype not in _CUDA_DTYPES:
        return f"{str(x.dtype).removeprefix('torch.')} features: torch only", False
    if choice == "triton":
        return "fastest on CUDA", False
    return f"triton unavailable: {triton_impl.toolchain_problem(x.device)}", True


def _check_backend(name: str, source: str) -> str:
    if name not in _BACKENDS:
        raise ValueError(f"{source}: unknown sparse conv backend {name!r}; choose from {_BACKENDS}")
    if name == "triton":
        try:
            triton_impl.require()  # RuntimeError naming the missing piece
        except RuntimeError as e:
            if source == _ENV:
                raise RuntimeError(
                    f"{_ENV}={name}: {e} Unset {_ENV} or set it to one of those."
                ) from e
            call = f"set_backend({name!r})" if source == "set_backend" else f"{source}={name!r}"
            raise RuntimeError(f"{call}: {e}") from e
    return name


def _check_args(x: Tensor, weight: Tensor, bias: Tensor | None, grid: SparseGrid) -> int:
    """Validate shapes, dtypes and devices (no sync); return the kernel size."""
    if weight.dim() != 5 or not weight.shape[1] == weight.shape[2] == weight.shape[3]:
        raise ValueError(
            f"weight must be (C_out, k, k, k, C_in) (spconv layout), got {tuple(weight.shape)}"
        )
    c_out, kernel_size, c_in = weight.shape[0], weight.shape[1], weight.shape[4]
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel size must be odd for a submanifold conv, got {kernel_size}")
    if not x.is_floating_point() or x.dim() != 2 or x.shape[1] != c_in:
        raise ValueError(f"x must be a float (N, {c_in}) tensor, got {x.dtype} {tuple(x.shape)}")
    if x.shape[0] != grid.grid_coord.shape[0]:
        raise ValueError(
            f"x has {x.shape[0]} rows but the grid has {grid.grid_coord.shape[0]} points"
        )
    if bias is not None and bias.shape != (c_out,):
        raise ValueError(f"bias must have shape ({c_out},), got {tuple(bias.shape)}")
    for name, t in (("weight", weight), ("bias", bias), ("grid.grid_coord", grid.grid_coord)):
        if t is not None and t.device != x.device:
            raise ValueError(f"x is on {x.device} but {name} is on {t.device}")
    return kernel_size
