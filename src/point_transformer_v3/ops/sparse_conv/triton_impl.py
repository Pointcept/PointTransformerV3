# Vendored from Pointcept2 src/pointcept2/ops/sparse_conv/triton_impl.py @ 771a949 (simplified for point_transformer_v3).
"""Triton submanifold sparse conv: implicit-GEMM kernels on the ``build_kmap`` kernel map (CUDA).

The kernels (``_triton_kernels``) and their ``torch.library.triton_op`` wrappers (``_triton_ops``,
namespace ``point_transformer_v3::``) are adapted from ocnn-pytorch (commit 7521c22; MIT, Copyright
(c) 2022 Peng-Shuai Wang), whose kernels come from FlexGEMM (MIT, Copyright (c) 2025 Jianfeng
Xiang); licenses in ``THIRD_PARTY_LICENSES.md`` next to this file. This module holds what needs no
triton import: the backend gate (``require``), the launch-config tables and the user-facing
``subm_conv3d``.
Importing it does not import triton; the ops are imported on first use.

Numerics: products accumulate in fp32; fp32 inputs use IEEE fp32 dot products (no TF32). The
input gradient is a gather on the mirrored kernel-map column (no atomics) and the weight gradient a
split-K reduction with a split fixed per GPU and conv shape, so forward and backward are bitwise
reproducible, also across processes. fp32 results are within ~6e-6 of fp64 (relative to the
largest magnitude) up to C=512; they are not bit-identical to spconv's.

Launch configs depend only on the GPU and the conv shape, never on N: the pinned table
``triton_configs.json`` (from an offline sweep) for known GPUs and shapes, a generic heuristic
otherwise (logged once). Nothing is autotuned at runtime, so a traced graph has no guard on N and
Inductor never benchmarks the kernels with random indices.

Toolchain: ``available`` and ``toolchain_problem`` (``ops._triton``, shared by every op with
Triton kernels; re-exported here) check triton, CUDA, the compute capability and Triton's C
driver helper; ``require()`` raises naming the missing piece and registers the ops. GPUs below
compute capability 8.0 (``MIN_CAPABILITY``) are refused: Triton has no tensor-core dot there and
these tiles spill. Kernels without ``tl.dot`` (the depthwise conv) pass
``min_capability=TRITON_MIN_CAPABILITY``, Triton's own floor (7.0, the check torch.compile
makes). The checks take a device (``available(device)``, ``require(device)``), so the default
``auto`` backend takes triton on the sm_80+ GPUs of a mixed node; without one, every visible GPU
must pass (what ``set_backend("triton")`` checks).
"""

from __future__ import annotations

import functools
import importlib.resources
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from point_transformer_v3.ops._tracing import tracing
from point_transformer_v3.ops._triton import (
    MIN_CAPABILITY,
    TRITON_MIN_CAPABILITY,
    available,
    toolchain_problem,
)
from point_transformer_v3.ops.neighbors import check_kmap

__all__ = [
    "MIN_CAPABILITY",
    "TRITON_MIN_CAPABILITY",
    "GemmConfig",
    "WgradConfig",
    "available",
    "gemm_config",
    "require",
    "subm_conv3d",
    "toolchain_problem",
    "wgrad_config",
]

log = logging.getLogger(__name__)

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_CONFIG_FILE = "triton_configs.json"
_ready = False  # require() passed and the ops are registered
_eager_function: Any = None  # _triton_ops.SubmConv3dFunction once the ops are loaded


def subm_conv3d(x: Tensor, weight: Tensor, bias: Tensor | None, kmap: Tensor) -> Tensor:
    """Submanifold conv ``out[p] = bias + sum_k W[:, k] @ x[kmap[p, k]]`` with Triton kernels.

    Shapes:
        x: ``(N, C_in)`` float on CUDA. weight: ``(C_out, k, k, k, C_in)``. bias: ``(C_out,)`` or
        None. kmap: ``(N, k**3)`` int32 from ``build_kmap`` / ``kmap_from_coarse`` (-1 = none).
        returns: ``(N, C_out)`` in the autocast dtype when autocast is on for x's device, else in
        x's dtype (fp32, fp16 or bf16); products accumulate in fp32. Gradients come back in the
        dtype of x, weight and bias; the weight and bias gradients are reduced in fp32.

    The backward reads the kernel map mirrored (``kmap[p, k] == q`` iff ``kmap[q, K-1-k] == p``,
    which ``build_kmap`` guarantees); ``PTV3_DEBUG=1`` checks that and the index range (host
    syncs). The kernels treat entries outside ``[0, N)`` as missing, so a corrupt map gives wrong
    numbers but never reads outside the tensors. Launches run on x's device (device guard).
    Eager calls launch the kernels from ``SubmConv3dFunction``; traced ones (``ops._tracing``)
    record ``point_transformer_v3::subm_conv3d``. Both give the same bits.
    """
    device_type = x.device.type
    dtype = (
        torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else x.dtype
    )
    _check_inputs(x, weight, bias, kmap, dtype)
    if os.environ.get("PTV3_DEBUG") == "1":
        check_kmap(kmap)
    if not _ready:
        _load_ops()
    x = x.to(dtype)
    if tracing(x):  # traced graphs keep the op (D15)
        with torch.autocast(device_type, enabled=False):
            return torch.ops.point_transformer_v3.subm_conv3d(x, weight.flatten(1, 3), bias, kmap)
    return _eager_function.apply(x, weight.flatten(1, 3), bias, kmap)  # I5: same kernels


# ---- toolchain


def require(
    device: torch.device | None = None, *, min_capability: tuple[int, int] = MIN_CAPABILITY
) -> None:
    """Raise ``RuntimeError`` naming the missing piece unless the triton backend can run on
    ``device`` (every visible GPU when None) at ``min_capability`` (see ``toolchain_problem``);
    otherwise register the ``point_transformer_v3::`` conv ops (imports triton) and log the toolchain once."""
    global _ready
    problem = toolchain_problem(device, min_capability=min_capability)
    if problem is not None:
        others = "'auto' or 'torch'"
        raise RuntimeError(
            f"the triton sparse conv backend cannot run here: {problem}. Use backend {others} "
            "instead."
        )
    if _ready:
        return
    _load_ops()
    import triton
    from triton import knobs

    log.info(
        "triton sparse conv backend: Triton %s, kernel cache %s",
        triton.__version__,
        knobs.cache.dir,
    )
    _ready = True


def _load_ops() -> None:
    """Import ``_triton_ops``: imports triton and registers the ops (once per process); keeps
    its eager ``SubmConv3dFunction``."""
    global _eager_function
    from point_transformer_v3.ops.sparse_conv import _triton_ops

    _eager_function = _triton_ops.SubmConv3dFunction


# ---- launch configs


@dataclass(frozen=True)
class GemmConfig:
    """Forward / dgrad launch: ``B1`` rows x ``B2`` output columns per program, ``BK`` reduction
    channels per step."""

    B1: int
    B2: int
    BK: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class WgradConfig:
    """Weight-gradient launch: ``B1`` output channels x ``B2 = BV * BCi`` (offset, input channel)
    columns per program, ``BK`` rows per step, the rows split into ``SPLITK`` fp32 partials."""

    B1: int
    B2: int
    BK: int
    SPLITK: int
    num_warps: int
    num_stages: int
    BV: int = 0  # kernel offsets per program (derived from B2 and C_in)
    BCi: int = 0  # input channels per program


@dataclass(frozen=True)
class _Gpu:
    key: str  # "<name>|sm_<major><minor>|<SM count>", the key of the pinned table
    sm_count: int
    smem: int  # opt-in shared memory per block, bytes


def gemm_config(
    kind: Literal["fwd", "dgrad"],
    c_in: int,
    c_out: int,
    num_offsets: int,
    dtype: torch.dtype,
    device: torch.device,
) -> GemmConfig:
    """The pinned forward / dgrad config for this GPU and shape, else the generic one."""
    gpu = _gpu(device)
    hit = _pinned(gpu, kind, c_in, c_out, num_offsets, dtype)
    if hit is not None:
        return GemmConfig(**hit)
    # fwd reduces over C_in and tiles C_out; dgrad the other way round
    k_ch, n_ch = (c_in, c_out) if kind == "fwd" else (c_out, c_in)
    elem = torch.finfo(dtype).bits // 8
    if dtype == torch.float32:
        b1, b2, bk, stages = 64, _clamp_p2(n_ch, 16, 64), _clamp_p2(k_ch, 16, 32), 2
    else:
        b1, b2, bk, stages = 128, _clamp_p2(n_ch, 16, 64), _clamp_p2(k_ch, 16, 64), 3
    b1, stages = _fit_smem(b1, b2, bk, stages, elem, gpu.smem)
    return GemmConfig(B1=b1, B2=b2, BK=bk, num_warps=4, num_stages=stages)


def wgrad_config(
    c_in: int, c_out: int, num_offsets: int, dtype: torch.dtype, device: torch.device
) -> WgradConfig:
    """The pinned weight-gradient config for this GPU and shape, else the generic one; ``BV`` and
    ``BCi`` filled in (ocnn ``conv_bwd_implicit_gemm_splitk.py`` L86-91)."""
    gpu = _gpu(device)
    hit = _pinned(gpu, "wgrad", c_in, c_out, num_offsets, dtype)
    if hit is not None:
        cfg = WgradConfig(**hit)
    else:
        elem = torch.finfo(dtype).bits // 8
        if dtype == torch.float32:
            b1, b2, bk, stages = _clamp_p2(c_out, 16, 64), 64, 32, 2
        else:
            b1, b2, bk, stages = _clamp_p2(c_out, 16, 64), 128, 64, 3
        b1, stages = _fit_smem(b1, b2, bk, stages, elem, gpu.smem)
        bci = min(_p2(c_in), b2)
        tiles = -(-c_out // b1) * -(-num_offsets // max(1, b2 // bci)) * -(-c_in // bci)
        split = _clamp_p2(max(1, 4 * gpu.sm_count // tiles), 1, 256)  # ~4 programs per SM
        # fp32 (IEEE dot on FMAs) at 4 warps hits 255 registers and spills on sm_75..sm_120 (2x
        # slower); 8 warps use 172 and give the same bits (the split does not depend on warps)
        warps = 8 if dtype == torch.float32 else 4
        cfg = WgradConfig(B1=b1, B2=b2, BK=bk, SPLITK=split, num_warps=warps, num_stages=stages)
    bci = min(_p2(c_in), cfg.B2)
    return WgradConfig(
        B1=cfg.B1,
        B2=cfg.B2,
        BK=cfg.BK,
        SPLITK=cfg.SPLITK,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        BV=max(1, cfg.B2 // bci),
        BCi=bci,
    )


@functools.cache
def _gpu(device: torch.device) -> _Gpu:
    if device.type != "cuda":  # the Triton interpreter (CPU tests): generic configs
        return _Gpu(key=f"{device.type} (Triton interpreter)", sm_count=4, smem=1 << 30)
    props = torch.cuda.get_device_properties(device)
    key = f"{props.name}|sm_{props.major}{props.minor}|{props.multi_processor_count}"
    gpu = _Gpu(key, props.multi_processor_count, props.shared_memory_per_block_optin)
    if key in _pinned_table():
        log.info("triton sparse conv: pinned launch configs for %s", key)
    else:
        log.info("triton sparse conv: no pinned launch configs for %s; using generic ones", key)
    return gpu


@functools.cache
def _pinned_table() -> dict[str, dict[str, dict[str, int]]]:
    """``{gpu key: {"<kind>:<C_in>:<C_out>:<K>:<dtype>": config}}`` from the package data."""
    path = importlib.resources.files("point_transformer_v3.ops.sparse_conv").joinpath(_CONFIG_FILE)
    data = json.loads(path.read_text())
    return {gpu: entry["configs"] for gpu, entry in data["gpus"].items()}


_generic_noted: set[str] = set()


def _pinned(
    gpu: _Gpu, kind: str, c_in: int, c_out: int, num_offsets: int, dtype: torch.dtype
) -> dict[str, int] | None:
    table = _pinned_table().get(gpu.key)
    key = f"{kind}:{c_in}:{c_out}:{num_offsets}:{str(dtype).removeprefix('torch.')}"
    hit = None if table is None else table.get(key)
    if hit is None and table is not None and key not in _generic_noted:
        _generic_noted.add(key)
        log.info(
            "triton sparse conv: no pinned %s config for %s; using the generic one", key, gpu.key
        )
    return hit


def _fit_smem(b1: int, b2: int, bk: int, stages: int, elem: int, smem: int) -> tuple[int, int]:
    """Lower ``num_stages`` (to 2), then the row tile, until the pipelined tiles fit in shared
    memory (estimate: ``(B1 + B2) * BK`` elements per stage)."""
    while stages > 2 and (b1 + b2) * bk * elem * stages > smem:
        stages -= 1
    while b1 > 16 and (b1 + b2) * bk * elem * stages > smem:
        b1 //= 2
    return b1, stages


def _p2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _clamp_p2(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, _p2(n)))


# ---- validation


def _check_inputs(
    x: Tensor, weight: Tensor, bias: Tensor | None, kmap: Tensor, dtype: torch.dtype
) -> None:
    """Shapes, dtypes and devices (no sync)."""
    if weight.dim() != 5 or not weight.shape[1] == weight.shape[2] == weight.shape[3]:
        raise ValueError(f"weight must be (C_out, k, k, k, C_in), got {tuple(weight.shape)}")
    c_out, k, c_in = weight.shape[0], weight.shape[1], weight.shape[4]
    if not x.is_floating_point() or x.dim() != 2 or x.shape[1] != c_in:
        raise ValueError(f"x must be a float (N, {c_in}) tensor, got {x.dtype} {tuple(x.shape)}")
    if dtype not in _DTYPES or not weight.is_floating_point():
        raise TypeError(
            f"the triton backend computes in fp32, fp16 or bf16, got x {x.dtype} (compute dtype "
            f"{dtype}) and weight {weight.dtype}; use the torch backend for other dtypes"
        )
    if kmap.dtype != torch.int32 or kmap.shape != (x.shape[0], k**3):
        raise TypeError(
            f"kmap must be an int32 ({x.shape[0]}, {k**3}) tensor (build_kmap), got "
            f"{kmap.dtype} {tuple(kmap.shape)}"
        )
    if bias is not None and (not bias.is_floating_point() or bias.shape != (c_out,)):
        raise ValueError(
            f"bias must be a float ({c_out},) tensor, got {bias.dtype} {tuple(bias.shape)}"
        )
    for name, t in (("weight", weight), ("bias", bias), ("kmap", kmap)):
        if t is not None and t.device != x.device:
            raise ValueError(f"x is on {x.device} but {name} is on {t.device}")
