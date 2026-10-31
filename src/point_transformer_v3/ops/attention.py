# Vendored from Pointcept2 src/pointcept2/ops/attention.py @ 771a949 (simplified for point_transformer_v3; triton_d16 takes head dims <= 64).
"""Softmax attention inside attention patches, through one ordered table of backends.

Entry points:

- ``serialized_attention(x, qkv, order, inverse, patches, ...)``: multi-head attention inside the
  patches of a serialized level (``PatchLayout`` of its sample sizes, ``ops.patches``) on the
  level's rows. ``qkv`` (the qkv Linear) is called once; the output has the level's rows. PT-v3m1's
  ``SerializedAttention`` runs it.
- ``varlen_attention(q, k, v, cu_q, cu_k, max_q, max_k, scale=...)``: P sequences packed along dim
  0; the queries ``q[cu_q[i]:cu_q[i + 1]]`` attend to ``k[cu_k[i]:cu_k[i + 1]]``. Query and key
  lengths may differ per sequence.

Backends, in the order ``auto`` tries them for each (device, dtype, head_dim, traced or eager); the
first that can run the call takes it:

1. ``triton_d16`` (``serialized_attention`` only; kernels in ``_attention_triton.py``): CUDA,
   bf16/fp16, head_dim <= 64 (other than 16, 32 and 64 in a masked block of the next power of
   two, e.g. Utonia's 18 in 32), compute capability 8.0+ with a working Triton
   (``ops._triton.toolchain_problem``). The kernels read each patch's q/k/v rows through the
   serialization and write the output and gradients back to the same rows, so no permutation, key
   gather or head_dim padding runs outside them; the backward has no atomics (bitwise
   reproducible). Eager calls launch the kernels from a plain ``autograd.Function``; traced graphs
   record ``point_transformer_v3::serialized_attention`` (D15). Configs were tuned for head_dim 16
   (the PTv3 recipes); other head dims are checked for accuracy only.
2. ``varlen``: ``torch.nn.attention.varlen.varlen_attn`` (FlashAttention-2): CUDA, bf16/fp16,
   head_dim <= 256, compute capability 8.0+. A head_dim that is not a multiple of 8 is
   zero-padded (the explicit scale keeps the math). ``varlen_attn`` is imported on its first call
   (its module imports triton and torch._inductor; D22).
3. ``sdpa``: each sequence padded to ``max_q`` / ``max_k`` and run through
   ``F.scaled_dot_product_attention`` with a key-padding mask. Exact in fp32; CPU, CUDA, MPS.

``serialized_attention`` on ``varlen`` / ``sdpa`` permutes the rows into the serialization before
``qkv``, gathers the keys of "overlap" tail patches (``PatchLayout.kv_pos``) and permutes the
output back; both permutations are skipped when the rows already follow the order (``in_order``).

``auto`` is the default (or env ``PTV3_ATTENTION_BACKEND``); ``set_attention_backend`` forces one
backend, and a forced backend that cannot run a call raises. The choice is resolved per device
(the GPUs of a mixed node resolve separately, D22) outside traced graphs (its result is baked in)
and logged once per (entry point, device, dtype, head_dim); on CUDA, a call that ``auto`` does not
give to the entry point's first backend warns once, with the reasons (C18).

Numerics (R1, research/accel/r2/reports/r1-attention-d16.md): ``triton_d16`` against fp64 has the
same relative error as FlashAttention-2 (out 1.6e-3, dq 2.5-5.7e-3, dk/dv 2.3-2.8e-3 in bf16 on
the 1M pack's calls); against ``varlen`` its forward differs by 1-2 bf16 ulp. 22 attention calls
of a 1M step, forward + backward: 82 ms (FA2 with the permutes and gathers) -> 46 ms.

Another backend (for example an adapter for FlashAttention 3 on sm90 or 4 on sm100) would be one
more row of ``_TABLE``, placed where ``auto`` should prefer it: ``layout="packed"`` if it takes
packed q/k/v like ``varlen_attn`` (``serialized_attention`` then does the permutes for it), its
``traceable`` flag saying whether traced graphs may hold it (an op without a fake implementation
cannot be compiled or exported: ``auto`` skips it while tracing), and an ``unsupported(device,
dtype, head_dim)`` check that finds the optional package without importing it
(``importlib.util.find_spec``) and checks the compute capability it was built for. The package
would be imported inside its runner on first use, never at import time and never as a dependency
of point_transformer_v3 (C5 bans ``flash_attn`` imports in the library, so the adapter either lives
in a plugin that adds its row or needs that rule relaxed).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp.autocast_mode import is_autocast_available

from point_transformer_v3.ops._tracing import assume_constant_result, tracing
from point_transformer_v3.ops._triton import (
    MIN_CAPABILITY,
    skip_warning_frames_in,
    toolchain_problem,
    warn_at_caller,
)
from point_transformer_v3.ops.gather import gather_rows, permute_rows
from point_transformer_v3.ops.patches import PatchLayout

__all__ = [
    "AttentionBackend",
    "get_attention_backend",
    "serialized_attention",
    "set_attention_backend",
    "varlen_attention",
]

AttentionBackend = Literal["auto", "triton_d16", "varlen", "sdpa"]

_ENV = "PTV3_ATTENTION_BACKEND"
_LOW_PRECISION = (torch.bfloat16, torch.float16)
_AUTOCAST_INPUTS = (torch.float32, torch.float16, torch.bfloat16)  # what autocast casts
_D16_MAX_HEAD_DIM = 64
_VARLEN_MAX_HEAD_DIM = 256
_VARLEN_MIN_CAPABILITY = (8, 0)
_MAX_ROWS = 2**31 - 1  # int32 positions in the patch table

log = logging.getLogger(__name__)
skip_warning_frames_in(__file__)  # a fallback warning names the caller's line
_backend: str | None = None
_noted: set[tuple[str, ...]] = set()
_capability_problems: dict[int, str] = {}  # CUDA device index -> reason ("" if varlen can run)
_kernels: Any = None  # the _attention_triton module once triton_d16 was chosen


# ---- the backend table


@dataclass(frozen=True)
class _Backend:
    name: str
    label: str  # how messages name it
    layout: Literal["rows", "packed"]  # the level's rows through the order, or packed q/k/v
    traceable: bool  # traced graphs (torch.compile / torch.export) may hold it
    unsupported: Callable[[torch.device, torch.dtype, int], str]  # why not ("" if it can run)


def _triton_d16_problem(device: torch.device, dtype: torch.dtype, head_dim: int) -> str:
    if device.type != "cuda":
        return f"needs CUDA tensors, got {device.type}"
    if dtype not in _LOW_PRECISION:
        return f"needs bf16/fp16 inputs, got {dtype}"
    if head_dim > _D16_MAX_HEAD_DIM:
        return f"supports head_dim <= {_D16_MAX_HEAD_DIM}, got {head_dim}"
    problem = toolchain_problem(device, min_capability=MIN_CAPABILITY)
    return "" if problem is None else f"cannot run here: {problem}"


def _varlen_problem(device: torch.device, dtype: torch.dtype, head_dim: int) -> str:
    if device.type != "cuda":
        return f"needs CUDA tensors, got {device.type}"
    if dtype not in _LOW_PRECISION:
        return f"needs bf16/fp16 inputs, got {dtype}"
    if head_dim > _VARLEN_MAX_HEAD_DIM:
        return f"supports head_dim <= {_VARLEN_MAX_HEAD_DIM}, got {head_dim}"
    return _capability_problem(device)


def _no_problem(device: torch.device, dtype: torch.dtype, head_dim: int) -> str:
    return ""


_TABLE: tuple[_Backend, ...] = (
    _Backend("triton_d16", "the triton_d16 kernels", "rows", True, _triton_d16_problem),
    _Backend("varlen", "varlen_attn", "packed", True, _varlen_problem),
    _Backend("sdpa", "padded SDPA", "packed", True, _no_problem),
)
_BACKENDS: tuple[str, ...] = ("auto", *(b.name for b in _TABLE))


def set_attention_backend(name: AttentionBackend) -> None:
    """Choose the backend of every later attention call (overrides the env var). Graphs already
    traced keep the backend they were traced with."""
    global _backend
    _backend = _check_backend(name, "set_attention_backend")


def get_attention_backend() -> str:
    """The configured backend: ``set_attention_backend``, else env ``PTV3_ATTENTION_BACKEND``,
    else ``"auto"``."""
    global _backend
    if _backend is None:
        _backend = _check_backend(os.environ.get(_ENV, "auto"), _ENV)
    return _backend


# ---- entry points


def serialized_attention(
    x: Tensor,
    qkv: Callable[[Tensor], Tensor],
    order: Tensor,
    inverse: Tensor,
    patches: PatchLayout,
    *,
    num_heads: int,
    head_dim: int,
    scale: float,
    in_order: bool = False,
) -> Tensor:
    """Multi-head softmax attention inside the patches of a level along one serialization.

    ``qkv`` maps rows to ``[q | k | v]`` (each ``num_heads * head_dim`` wide, head-major) row by
    row (a Linear); it is called once, on the rows of the level (``triton_d16``) or on the rows in
    serialized order (the other backends). Patch i of ``patches`` holds serialized positions,
    i.e. the rows ``order[positions]``. ``in_order`` says ``order`` is the identity (the rows
    already follow the serialization), so the packed backends skip both permutations.

    Shapes:
        x: ``(N, C_in)`` float, rows of the level.
        order: ``(N,)`` int64 serialized position -> row (``Serialized.order``, a permutation;
            any stride); inverse: ``(N,)`` int64 its inverse.
        patches: ``build_patches`` of the level's sample sizes (N positions) on x's device.
        returns: ``(N, num_heads * head_dim)`` in qkv's output dtype (the autocast dtype under
        autocast), rows of the level.

    No host sync. Shapes, dtypes and devices (the patch table's too) are checked on every call;
    the checks that need a sync (order is a permutation, the patch table's ranges) run under
    ``PTV3_DEBUG=1`` on eager calls only: a traced graph (torch.compile, torch.export,
    AOTInductor) cannot sync, so debug a model eagerly.
    """
    _check_serialized(x, order, inverse, patches, num_heads, head_dim)
    dtype = _compute_dtype(x)
    traced = tracing(x)
    backend = _choose(x.device, dtype, head_dim, "serialized_attention", traced)
    if backend == "triton_d16":
        return _triton_d16(qkv(x), order, patches, num_heads, head_dim, scale, dtype, traced)
    xs = x if in_order else permute_rows(x, order)
    q, k, v = qkv(xs).unflatten(-1, (3, num_heads, head_dim)).unbind(1)
    if patches.kv_pos is not patches.q_pos:
        k, v = gather_rows(k, patches.kv_pos), gather_rows(v, patches.kv_pos)
    cu_q, cu_k, max_q, max_k = patches.cu_q, patches.cu_k, patches.max_q, patches.max_k
    _check_inputs(q, k, v, cu_q, cu_k, max_q, max_k)
    q, k, v = _autocast(q, k, v)
    out = _packed(q, k, v, cu_q, cu_k, max_q, max_k, scale, backend).flatten(1)
    return out if in_order else permute_rows(out, inverse)


def varlen_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_q: Tensor,
    cu_k: Tensor,
    max_q: int,
    max_k: int,
    *,
    scale: float,
) -> Tensor:
    """Softmax attention of each query sequence over its key/value sequence (backends ``varlen``
    and ``sdpa``; ``triton_d16`` needs ``serialized_attention``'s layout).

    Shapes:
        q: (Tq, H, D); k, v: (Tk, H, D); all the same floating dtype.
        cu_q, cu_k: (P + 1,) int32 on q's device, ``cu[0] == 0``, ``cu_q[-1] == Tq``,
            ``cu_k[-1] == Tk``; sequence i is ``[cu[i], cu[i + 1])``. A sequence with queries
            needs at least one key.
        max_q, max_k: host ints >= every query / key sequence length.
        returns: (Tq, H, D) in q.dtype (in the autocast dtype when autocast is on).

    No host sync; the layout checks that need one run under ``PTV3_DEBUG=1`` on eager calls only.
    """
    _check_inputs(q, k, v, cu_q, cu_k, max_q, max_k)
    q, k, v = _autocast(q, k, v)
    if q.shape[0] == 0:
        return _no_queries(q, k, v)
    backend = _choose(q.device, q.dtype, q.shape[-1], "varlen_attention", tracing(q))
    return _packed(q, k, v, cu_q, cu_k, max_q, max_k, scale, backend)


# ---- backends


def _triton_d16(
    qkv: Tensor,
    order: Tensor,
    patches: PatchLayout,
    num_heads: int,
    head_dim: int,
    scale: float,
    dtype: torch.dtype,
    traced: bool,
) -> Tensor:
    if qkv.dtype != dtype:
        raise TypeError(
            f"qkv returned {qkv.dtype}, but the attention backend was chosen for {dtype} (the "
            "autocast dtype, else x's dtype); make qkv return that dtype"
        )
    if qkv.dim() != 2 or qkv.shape != (order.shape[0], 3 * num_heads * head_dim):
        raise ValueError(
            f"qkv must return (N, 3 * num_heads * head_dim) = ({order.shape[0]}, "
            f"{3 * num_heads * head_dim}), got {tuple(qkv.shape)}"
        )
    # the kernels read qkv's columns, the order and the table with unit strides (no-ops on
    # PointGrid's tensors: argsort / arange orders, build_patches' table)
    qkv, order, table = qkv.contiguous(), order.contiguous(), patches.table.contiguous()
    if not traced and os.environ.get("PTV3_DEBUG") == "1":  # a traced graph cannot sync
        _check_rows_and_table(order, table, patches.max_q)
    args = (qkv, order, table, patches.max_q, num_heads, scale)
    if traced:  # traced graphs keep the op (D15)
        return torch.ops.point_transformer_v3.serialized_attention(*args)[0]
    return _kernels.SerializedAttentionFunction.apply(*args)  # I5: no op dispatch in eager


def _packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_q: Tensor,
    cu_k: Tensor,
    max_q: int,
    max_k: int,
    scale: float,
    backend: str,
) -> Tensor:
    if q.shape[0] == 0:
        return _no_queries(q, k, v)
    if backend == "varlen":
        return _varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale)
    return _sdpa(q, k, v, cu_q, cu_k, max_q, max_k, scale)


def _no_queries(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """The (0, H, D) output of a call without queries, inside the autograd graph of q, k and v
    (zero gradients), so a backward through an empty level runs."""
    return q + (k.sum() + v.sum()) * 0


def _varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_q: Tensor,
    cu_k: Tensor,
    max_q: int,
    max_k: int,
    scale: float,
) -> Tensor:
    # Imported here, not at module level: torch.nn.attention.varlen imports triton and
    # torch._inductor, which CPU inference must not load (D22).
    from torch.nn.attention.varlen import varlen_attn

    head_dim = q.shape[-1]
    pad = -head_dim % 8
    if pad:
        q, k, v = (F.pad(t, (0, pad)) for t in (q, k, v))
    out = cast(Tensor, varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k, scale=scale))
    return out[..., :head_dim] if pad else out


def _sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_q: Tensor,
    cu_k: Tensor,
    max_q: int,
    max_k: int,
    scale: float,
) -> Tensor:
    num = cu_q.shape[0] - 1
    q_slot = _padded_slots(cu_q, max_q, q.shape[0])
    k_slot = _padded_slots(cu_k, max_k, k.shape[0])
    qp, kp, vp = (
        _to_padded(t, slot, num, length).transpose(1, 2)
        for t, slot, length in ((q, q_slot, max_q), (k, k_slot, max_k), (v, k_slot, max_k))
    )
    mask = None
    if k_slot is not None:
        k_len = cu_k.diff()[:, None, None, None]
        mask = torch.arange(max_k, device=q.device) < k_len  # (P, 1, 1, max_k)
    out = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask, scale=scale)
    out = out.transpose(1, 2).flatten(0, 1)  # (P * max_q, H, D)
    return out if q_slot is None else gather_rows(out, q_slot)


def _padded_slots(cu: Tensor, max_len: int, total: int) -> Tensor | None:
    """(total,) int64 row of each token in the (P * max_len) padded layout; None if the layout is
    already dense (every sequence has exactly ``max_len`` tokens)."""
    num = cu.shape[0] - 1
    if total == num * max_len:
        return None
    shift = torch.arange(num, device=cu.device) * max_len - cu[:-1]
    return torch.arange(total, device=cu.device) + shift.repeat_interleave(
        cu.diff(), output_size=total
    )


def _to_padded(x: Tensor, slot: Tensor | None, num: int, length: int) -> Tensor:
    """(T, H, D) tokens -> (P, length, H, D), zeros in the padding."""
    if slot is not None:
        x = x.new_zeros((num * length, *x.shape[1:])).index_copy(0, slot, x)
    return x.unflatten(0, (num, length))


# ---- dispatch


@assume_constant_result
def _choose(
    device: torch.device, dtype: torch.dtype, head_dim: int, entry: str, traced: bool
) -> str:
    """The backend of an ``entry`` call on ``dtype`` inputs with ``head_dim`` on ``device``: the
    forced one, else the first row of ``_TABLE`` the entry can use whose requirements hold. Logs
    each choice once and warns once for a CUDA fallback. Runs eagerly even while torch.compile /
    torch.export trace (its result is baked into the graph), so the probes and the logging never
    enter a graph."""
    global _kernels
    name = get_attention_backend()
    candidates = [b for b in _TABLE if entry == "serialized_attention" or b.layout == "packed"]
    skipped: list[str] = []
    if name != "auto":
        forced = [b for b in candidates if b.name == name]
        if not forced:
            raise ValueError(
                f"attention backend {name!r} reads the rows of a serialized level "
                f"(serialized_attention); {entry} takes packed q, k, v: force 'varlen' or "
                "'sdpa' for it, or use 'auto'"
            )
        chosen = forced[0]
        problem = _problem(chosen, device, dtype, head_dim, traced)
        if problem:
            raise ValueError(
                f"attention backend {name!r} {problem}; call set_attention_backend('auto') to "
                "fall back to the next backend that can run"
            )
    else:
        chosen = candidates[-1]  # sdpa runs everything
        for backend in candidates:
            problem = _problem(backend, device, dtype, head_dim, traced)
            if not problem:
                chosen = backend
                break
            skipped.append(f"{backend.name} {problem}")
    key = (entry, str(device), str(dtype), str(head_dim), name, chosen.name)
    if key not in _noted:
        _noted.add(key)
        reason = "forced" if name != "auto" else "; ".join(skipped) or "first that applies"
        log.info(
            "%s on %s %s, head_dim %d: %s backend (%s: %s)",
            entry,
            device,
            str(dtype).removeprefix("torch."),
            head_dim,
            chosen.name,
            name,
            reason,
        )
        if skipped and device.type == "cuda":
            hint = (
                " Run attention in bf16 (autocast) for the fast path."
                if dtype not in _LOW_PRECISION
                else ""
            )
            warn_at_caller(
                f"{entry} falls back to {chosen.label} on CUDA ({'; '.join(skipped)}); this is "
                f"slower than {candidates[0].label}.{hint}"
            )
    if chosen.name == "triton_d16" and _kernels is None:
        from point_transformer_v3.ops import _attention_triton  # imports triton, registers the ops

        _kernels = _attention_triton
    return chosen.name


def _problem(
    backend: _Backend, device: torch.device, dtype: torch.dtype, head_dim: int, traced: bool
) -> str:
    """Why ``backend`` cannot run this call ("" if it can)."""
    if traced and not backend.traceable:
        return "cannot run inside traced graphs (torch.compile / torch.export)"
    return backend.unsupported(device, dtype, head_dim)


@assume_constant_result
def _capability_problem(device: torch.device) -> str:
    """Why the GPU ``device`` cannot run ``varlen_attn`` (empty if it can), cached per device
    index so the GPUs of a mixed node resolve separately. Runs eagerly even while torch.compile /
    torch.export trace (its result is baked into the graph)."""
    index = torch.cuda.current_device() if device.index is None else device.index
    problem = _capability_problems.get(index)
    if problem is None:
        major, minor = torch.cuda.get_device_capability(index)
        problem = ""
        if (major, minor) < _VARLEN_MIN_CAPABILITY:
            problem = (
                f"needs compute capability 8.0+ (FlashAttention-2, Ampere or newer), but "
                f"cuda:{index} ({torch.cuda.get_device_name(index)}) is sm_{major}{minor}"
            )
        _capability_problems[index] = problem
    return problem


def _check_backend(name: str, source: str) -> str:
    if name not in _BACKENDS:
        raise ValueError(f"{source}: unknown attention backend {name!r}; choose one of {_BACKENDS}")
    return name


def _compute_dtype(x: Tensor) -> torch.dtype:
    """The dtype a Linear returns for ``x``: the autocast dtype under autocast (which casts
    fp32/fp16/bf16 inputs), else x's dtype."""
    device_type = x.device.type
    if (
        x.dtype in _AUTOCAST_INPUTS
        and is_autocast_available(device_type)
        and torch.is_autocast_enabled(device_type)
    ):
        return torch.get_autocast_dtype(device_type)
    return x.dtype


def _autocast(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Cast fp32 inputs to the autocast dtype, as ``F.scaled_dot_product_attention`` would."""
    device_type = q.device.type
    if q.dtype != torch.float32 or not torch.is_autocast_enabled(device_type):
        return q, k, v
    dtype = torch.get_autocast_dtype(device_type)
    return q.to(dtype), k.to(dtype), v.to(dtype)


# ---- validation


def _check_serialized(
    x: Tensor,
    order: Tensor,
    inverse: Tensor,
    patches: PatchLayout,
    num_heads: int,
    head_dim: int,
) -> None:
    """Host checks of a ``serialized_attention`` call (shapes, dtypes, devices; no sync)."""
    if x.dim() != 2 or not x.is_floating_point():
        raise TypeError(f"x must be a float (N, C) tensor, got {x.dtype} {tuple(x.shape)}")
    n = x.shape[0]
    for name, t in (("order", order), ("inverse", inverse)):
        if t.dtype != torch.int64 or t.shape != (n,):
            raise TypeError(
                f"{name} must be an int64 ({n},) row permutation (Serialized.order / .inverse), "
                f"got {t.dtype} {tuple(t.shape)}"
            )
    table, rows = patches.table, patches.cu_q.shape[0]  # P + 1: a row per patch, the sentinel
    if table.dtype != torch.int32 or table.dim() != 2 or table.shape[1] != 4:
        raise TypeError(
            f"patches.table must be (P + 1, 4) int32, got {table.dtype} {tuple(table.shape)}; "
            "build the layout with build_patches"
        )
    if table.shape[0] != rows:
        raise ValueError(
            f"patches.table has {table.shape[0]} rows but cu_q describes {rows - 1} patches (a "
            "row per patch and the sentinel); build the layout with build_patches"
        )
    if not isinstance(patches.max_q, int) or isinstance(patches.max_q, bool):
        raise TypeError(
            f"patches.max_q must be a host int (keep sizes on the host), got "
            f"{type(patches.max_q).__name__}"
        )
    tensors = (order, inverse, table, patches.cu_q)
    if any(t.device != x.device for t in tensors):
        raise ValueError(
            f"order, inverse and patches must be on x's device {x.device}, got "
            f"{sorted({str(t.device) for t in tensors})}; build them on the level's device"
        )
    if patches.q_pos.shape[0] != n:
        raise ValueError(
            f"patches cover {patches.q_pos.shape[0]} positions but x has {n} rows; build them "
            "from this level's sample sizes"
        )
    if not tracing(x) and n > _MAX_ROWS:  # eager only: on a traced size export would add a guard
        raise ValueError(f"{n} rows overflow the int32 patch table; split the batch")
    for name, value in (("num_heads", num_heads), ("head_dim", head_dim)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise TypeError(f"{name} must be a positive host int, got {value!r}")


def _check_rows_and_table(rows: Tensor, table: Tensor, max_q: int) -> None:
    """Debug-only checks (host sync) of what the ``triton_d16`` kernels index with: ``rows`` and
    the table are contiguous (the kernels read both with unit stride); ``rows`` is a permutation
    of ``[0, T)``; the table's query runs tile ``[0, T)`` in order with lengths in ``[0, max_q]``;
    a patch with queries has keys; its keys end inside its own queries' range and start no
    earlier than the previous patch (the single-owner rule of the backward); the last row is the
    sentinel ``(T, 0, T, 0)``."""
    if table.dtype != torch.int32 or table.dim() != 2 or table.shape[1] != 4:
        raise TypeError(f"patches.table must be (P + 1, 4) int32, got {table.dtype}")
    for name, tensor in (("order", rows), ("patches.table", table)):
        if not tensor.is_contiguous():
            raise ValueError(
                f"the triton_d16 kernels read {name} with unit stride, got stride "
                f"{tensor.stride()}; pass .contiguous()"
            )
    t = rows.shape[0]
    if not torch.equal(rows.sort().values.cpu(), torch.arange(t)):
        raise ValueError(f"order must be a permutation of [0, {t}) (rows of the level)")
    tab = table.cpu().long()
    q_start, q_len, k_start, k_len = tab[:-1].unbind(1)
    if tab[-1].tolist() != [t, 0, t, 0]:
        raise ValueError(f"the table's last row must be the sentinel {[t, 0, t, 0]}")
    q_end = q_start + q_len
    starts = torch.cat([q_start, tab[-1:, 0]])
    if int(starts[0]) != 0 or not torch.equal(starts[1:], q_end):
        raise ValueError(f"the table's query runs must tile [0, {t}) in order")
    if bool((q_len < 0).any() | (q_len > max_q).any()):
        raise ValueError(f"the table's query lengths must lie in [0, max_q={max_q}]")
    prev_start = torch.cat([q_start[:1], q_start[:-1]])
    bad = ((q_len > 0) & (k_len < 1)) | (k_start < prev_start) | (k_start + k_len > q_end)
    if bool(bad.any()):
        i = int(bad.nonzero()[0])
        raise ValueError(
            f"patch {i} has keys {tab[i, 2:].tolist()} outside its own and the previous patch's "
            "rows (or no keys for its queries); build the table with build_patches"
        )


def _check_inputs(
    q: Tensor, k: Tensor, v: Tensor, cu_q: Tensor, cu_k: Tensor, max_q: int, max_k: int
) -> None:
    if q.dim() != 3 or k.dim() != 3 or k.shape != v.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError(
            "expected q (Tq, H, D) and k, v (Tk, H, D) with the same H and D, got "
            f"{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if not q.dtype.is_floating_point or not q.dtype == k.dtype == v.dtype:
        raise TypeError(
            f"q, k, v must share one floating dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    for name, cu in (("cu_q", cu_q), ("cu_k", cu_k)):
        if cu.dtype != torch.int32 or cu.dim() != 1 or cu.shape[0] < 1:
            raise TypeError(
                f"{name} must be a 1-D int32 tensor (P + 1,), got {cu.dtype} {tuple(cu.shape)}"
            )
        if cu.device != q.device:
            raise ValueError(f"{name} is on {cu.device} but q is on {q.device}")
    if cu_q.shape != cu_k.shape:
        raise ValueError(
            f"cu_q and cu_k must have the same length, got {cu_q.shape[0]} and {cu_k.shape[0]}"
        )
    for name, value in (("max_q", max_q), ("max_k", max_k)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"{name} must be a host int (keep sizes on the host), got {type(value).__name__}"
            )
    if cu_q.shape[0] == 1 and q.shape[0]:
        raise ValueError(f"no sequences (cu_q has one entry) but q has {q.shape[0]} rows")
    if os.environ.get("PTV3_DEBUG") == "1" and not tracing(q):  # a traced graph cannot sync
        _check_layout(cu_q, q.shape[0], max_q, "q")
        _check_layout(cu_k, k.shape[0], max_k, "k")
        if bool(((cu_q.diff() > 0) & (cu_k.diff() == 0)).any()):
            raise ValueError("a sequence has queries but no keys")


def _check_layout(cu: Tensor, total: int, max_len: int, name: str) -> None:
    """Debug-only checks (host sync)."""
    lens = cu.diff()
    first, last = int(cu[0]), int(cu[-1])
    if first != 0 or last != total:
        raise ValueError(f"cu_{name} must run from 0 to {total}, got {first}..{last}")
    if lens.numel() and (int(lens.min()) < 0 or int(lens.max()) > max_len):
        raise ValueError(
            f"cu_{name} sequence lengths must lie in [0, max_{name}={max_len}], "
            f"got [{int(lens.min())}, {int(lens.max())}]"
        )
