# Vendored from Pointcept2 src/pointcept2/ops/serialization.py @ 771a949 (simplified for point_transformer_v3).
"""Space-filling-curve codes of integer grids: z-order and Hilbert (PTv3 serialization).

``encode(grid, depth, order)`` maps (N, 3) grid cells with coordinates in [0, 2**depth) to int64
codes of ``3 * depth`` bits, bit-exact with Pointcept's
``pointcept/models/utils/serialization/default.py:encode(grid, batch=None, depth, order)``.
``serialize`` puts the sample index above the code (Pointcept's layout), so one sort orders a
packed batch by sample and then along the curve. ``decode`` inverts both.

- z-order interleaves the bits as ``... x1 y1 z1 x0 y0 z0`` (x most significant) via a byte table.
- Hilbert is Skilling's transform (Pointcept's port of numpy-hilbert-curve) recast as a 48-state
  automaton that reads the z-order code three levels (9 bits) per step, most significant first.
- ``"-trans"`` orders swap x and y before encoding, as in Pointcept.

Both curves have the prefix property used by m1 pooling: ``encode(grid, d) >> 3 ==
encode(grid >> 1, d - 1)``. Hilbert codes depend on ``depth % 3``: for grids inside
[0, 2**d), ``encode(grid, d) == encode(grid, d + 3)`` but not ``encode(grid, d + 1)``, so never
assume depth 16. Everything is a short chain of elementwise ops and table gathers (a loop over
bit groups, none over points) and issues no host sync on CUDA. The lookup tables (0.4 MiB) are
built once per device; torch.compile bakes them into its graph as constants, and so does
torch.export once an eager call on that device has built them (``_tables``).
"""

from __future__ import annotations

import functools
import os
from typing import Literal, NamedTuple, get_args

import torch
from torch import Tensor

from point_transformer_v3._util import suggest
from point_transformer_v3.ops._tracing import assume_constant_result

__all__ = ["MAX_DEPTH", "ORDERS", "Order", "decode", "encode", "serialize"]

Order = Literal["z", "z-trans", "hilbert", "hilbert-trans"]
ORDERS: tuple[Order, ...] = get_args(Order)
MAX_DEPTH = 16  # bits per axis; 3 * 16 code bits leave 15 bits for the sample index

_INT_DTYPES = frozenset({torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64})


def encode(grid: Tensor, depth: int, order: Order) -> Tensor:
    """Curve code of every grid cell.

    Args:
        grid: (N, 3) integer tensor with coordinates in [0, 2**depth). Higher bits are ignored, as
            in Pointcept; with ``PTV3_DEBUG=1`` out-of-range values raise (one host sync).
        depth: bits per axis, 1..16.
        order: ``"z"``, ``"hilbert"`` or their ``"-trans"`` variants (x and y swapped first).

    Returns:
        (N,) int64 codes in [0, 2**(3 * depth)).
    """
    _check_depth(depth)
    _check_order(order)
    _check_grid(grid, depth)
    return _encode(grid, depth, order)


def serialize(
    grid: Tensor, batch: Tensor, depth: int, order: Order, *, batch_size: int | None = None
) -> Tensor:
    """Sort keys ``batch << 3 * depth | encode(grid, depth, order)``: one sort orders a packed batch
    by sample, then along the curve.

    Args:
        grid: (N, 3) integer tensor, see ``encode``.
        batch: (N,) integer sample index of every row, in [0, batch_size).
        depth: bits per axis, 1..16.
        order: curve, see ``encode``.
        batch_size: number of samples B. When given, ``3 * depth + B.bit_length() <= 63`` is checked
            on the host; otherwise it is checked from ``batch`` only with ``PTV3_DEBUG=1``.

    Returns:
        (N,) int64 keys.
    """
    _check_depth(depth)
    _check_order(order)
    _check_grid(grid, depth)
    if not isinstance(batch, Tensor) or batch.dtype not in _INT_DTYPES:
        raise TypeError(f"serialize: batch must be an integer tensor, got {_describe(batch)}")
    if batch.shape != grid.shape[:1] or batch.device != grid.device:
        raise ValueError(
            f"serialize: batch must have shape ({grid.shape[0]},) on {grid.device} like grid, "
            f"got {tuple(batch.shape)} on {batch.device}"
        )
    if batch_size is not None:
        _check_batch_bits(batch_size, depth)
    if _debug() and batch.numel():
        low, high = int(batch.min()), int(batch.max())
        if low < 0:
            raise ValueError(f"serialize: batch indices must be >= 0, got {low}")
        _check_batch_bits(high + 1, depth)
    return batch.long() << 3 * depth | _encode(grid, depth, order)


def decode(code: Tensor, depth: int, order: Order) -> tuple[Tensor, Tensor]:
    """Invert ``serialize`` (and ``encode``, whose batch index is 0).

    Args:
        code: (N,) int64 keys.
        depth: bits per axis used to encode, 1..16.
        order: curve used to encode.

    Returns:
        grid (N, 3) int32 and batch (N,) int64.
    """
    _check_depth(depth)
    _check_order(order)
    if not isinstance(code, Tensor) or code.dtype != torch.int64 or code.dim() != 1:
        raise TypeError(f"decode: code must be a 1-D int64 tensor, got {_describe(code)}")
    tables = _tables(code)
    bits = code & ((1 << 3 * depth) - 1)
    if order.startswith("hilbert"):
        bits, _ = _walk(tables.hilbert_decode, bits, _hilbert_start(depth), 9, (depth + 2) // 3)
    grid = _deinterleave(bits, depth, order.endswith("-trans"), tables.compact)
    return grid, code >> 3 * depth


# ---- kernels


def _encode(grid: Tensor, depth: int, order: Order) -> Tensor:
    tables = _tables(grid)
    code = _interleave(grid, depth, order.endswith("-trans"), tables.spread)
    if order.startswith("hilbert"):
        code, _ = _walk(tables.hilbert_encode, code, _hilbert_start(depth), 9, (depth + 2) // 3)
    return code


def _interleave(grid: Tensor, depth: int, trans: bool, spread: Tensor) -> Tensor:
    """(N, 3) grid -> (N,) z-order code of its low ``depth`` bits (x and y swapped if ``trans``)."""
    g = grid.long() & ((1 << depth) - 1)
    s = spread[g & 255]
    if depth > 8:
        s |= spread[g >> 8] << 24
    x, y, z = s.unbind(1)
    if trans:
        x, y = y, x
    return x << 2 | y << 1 | z


def _deinterleave(code: Tensor, depth: int, trans: bool, compact: Tensor) -> Tensor:
    """(N,) z-order code of ``depth`` levels -> (N, 3) int32 grid; inverse of ``_interleave``."""
    grid = torch.zeros((code.shape[0], 3), dtype=torch.int64, device=code.device)
    for i in range((depth + 2) // 3):
        grid |= compact[(code >> 9 * i) & 511] << 3 * i
    if trans:
        x, y, z = grid.unbind(1)
        grid = torch.stack((y, x, z), dim=1)
    return grid.int()


def _walk(
    table: Tensor, digits: Tensor, state: Tensor | int, width: int, count: int
) -> tuple[Tensor, Tensor | int]:
    """Run an automaton over the low ``count`` digits of ``width`` bits, most significant first.

    ``table[state << width | digit] == next_state << width | out_digit``. Returns the output digits
    at the same positions and the final state.
    """
    mask = (1 << width) - 1
    out = torch.zeros_like(digits)
    for i in reversed(range(count)):
        entry = table[(state << width) | ((digits >> width * i) & mask)]
        out |= (entry & mask) << width * i
        state = entry >> width
    return out, state


# ---- tables
#
# Skilling's transform ("Programming the Hilbert curve", 2004) visits the levels from the most
# significant one. At each level it reads the current bits (gx, gy, gz), then for axis i = x, y, z
# in turn it inverts the lower bits of x (gi = 1) or exchanges the lower bits of x and axis i
# (gi = 0). Each step acts on all lower levels alike, so their accumulated effect is a signed axis
# permutation T, and the bits read at a level are T applied to the input octant. Pointcept then
# Gray-decodes the read bits (prefix XOR from the top), which only needs the last output bit p.
# (T, p) is the automaton state; 48 states are reachable from (identity, 0).

_Signed = tuple[tuple[int, int], ...]  # ((axis, flip), ...): T(v)[i] = v[axis_i] ^ flip_i
_IDENTITY: _Signed = ((0, 0), (1, 0), (2, 0))


def _level_update(bits: tuple[int, ...]) -> _Signed:
    """Skilling's update of the lower levels after reading one level's bits (x, y, z)."""
    t = list(_IDENTITY)
    for i, bit in enumerate(bits):
        if bit:
            t[0] = (t[0][0], t[0][1] ^ 1)  # invert x
        else:
            t[0], t[i] = t[i], t[0]  # exchange x and axis i
    return tuple(t)


def _then(a: _Signed, b: _Signed) -> _Signed:
    """The signed permutation ``v -> a(b(v))``."""
    return tuple((b[axis][0], b[axis][1] ^ flip) for axis, flip in a)


@functools.cache
def _hilbert_steps() -> tuple[list[int], list[int]]:
    """One-level tables: encode[s << 3 | octant] = next << 3 | digit; decode is its inverse."""
    states = [(_IDENTITY, 0)]
    index = {states[0]: 0}
    encode_step: list[int] = []
    for signed, parity in states:  # breadth-first: the list grows while it is iterated
        for octant in range(8):
            bits = tuple(((octant >> (2 - axis)) & 1) ^ flip for axis, flip in signed)
            digit, last = 0, parity
            for bit in bits:
                last ^= bit
                digit = digit << 1 | last
            nxt = (_then(_level_update(bits), signed), last)
            if nxt not in index:
                index[nxt] = len(states)
                states.append(nxt)
            encode_step.append(index[nxt] << 3 | digit)
    decode_step = [0] * len(encode_step)
    for key, entry in enumerate(encode_step):
        decode_step[(key & ~7) | (entry & 7)] = (entry & ~7) | (key & 7)
    return encode_step, decode_step


@assume_constant_result  # torch.compile reads the result instead of tracing the automaton
def _hilbert_start(depth: int) -> int:
    """Start state for ``depth``. The walk reads ``ceil(depth / 3)`` 9-bit digits, so it first
    meets ``-depth % 3`` zero levels above the grid's top level. A zero octant emits 0 and turns the
    state by a 3-cycle of the axes, so starting ``depth % 3`` turns past the identity puts the walk
    back at the identity on the grid's top level."""
    encode_step, _ = _hilbert_steps()
    state = 0
    for _ in range(depth % 3):
        state = encode_step[state << 3] >> 3
    return state


class _Tables(NamedTuple):
    spread: Tensor  # (256,) bit i of a byte -> bit 3 * i
    compact: Tensor  # (512, 3) 9 interleaved bits -> 3 bits of x, y, z
    hilbert_encode: Tensor  # (48 * 512,) state << 9 | z digits -> next << 9 | Hilbert digits
    hilbert_decode: Tensor  # (48 * 512,) inverse of hilbert_encode for every state


_TABLE_CACHE: dict[torch.device, _Tables] = {}  # real tensors only


def _tables(like: Tensor) -> _Tables:
    """The lookup tables on ``like``'s device.

    Real tensors read the per-device cache (``_cached_tables``), and so does torch.compile: dynamo
    sees plain tensors and runs ``_cached_tables`` eagerly while it traces. Under non-strict
    torch.export or make_fx the inputs are FakeTensors and factory calls make FakeTensors, which
    must never enter the cache, or later eager calls get them (as ``neighbors._octree_luts``). There
    the cached tables of the device are used when an eager call built them (export lifts them into
    the program as constants); otherwise the program builds them itself, uncached.
    """
    if type(like) is Tensor:
        return _cached_tables(like.device)
    cached = _TABLE_CACHE.get(like.device)
    if cached is not None:
        return cached
    return _Tables(*(t.to(like.device) for t in _build_tables()))


@assume_constant_result  # torch.compile bakes the real tables into the graph as constants
def _cached_tables(device: torch.device) -> _Tables:
    """The tables on ``device``, built once per process. On CUDA they are copied from pinned memory
    with ``non_blocking``, so filling the cache does not sync."""
    tables = _TABLE_CACHE.get(device)
    if tables is None:
        if device.type == "cpu":
            tables = _build_tables()
        else:
            cpu = _cached_tables(torch.device("cpu"))
            pinned = device.type == "cuda"
            tables = _Tables(
                *((t.pin_memory() if pinned else t).to(device, non_blocking=pinned) for t in cpu)
            )
        _TABLE_CACHE[device] = tables
    return tables


def _build_tables() -> _Tables:
    """The tables on the CPU (uncached)."""
    byte = torch.arange(256)
    spread = torch.zeros(256, dtype=torch.int64)
    for i in range(8):
        spread |= ((byte >> i) & 1) << 3 * i
    chunk = torch.arange(512)[:, None]
    shift = torch.tensor([2, 1, 0])
    compact = torch.zeros((512, 3), dtype=torch.int64)
    for i in range(3):
        compact |= ((chunk >> (3 * i + shift)) & 1) << i
    encode_step, decode_step = _hilbert_steps()
    return _Tables(spread, compact, _three_levels(encode_step), _three_levels(decode_step))


def _three_levels(step: list[int]) -> Tensor:
    """Compose a one-level table (8 digits per state) into a three-level one (512 per state)."""
    num_states = len(step) // 8
    state = torch.arange(num_states).repeat_interleave(512)
    out, state = _walk(torch.tensor(step), torch.arange(512).repeat(num_states), state, 3, 3)
    return state << 9 | out


# ---- validation


def _debug() -> bool:
    return os.environ.get("PTV3_DEBUG") == "1"


def _describe(x: object) -> str:
    if isinstance(x, Tensor):
        return f"{x.dtype} tensor of shape {tuple(x.shape)}"
    return type(x).__name__


def _check_depth(depth: int) -> None:
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError(f"depth must be an int (bits per axis), got {type(depth).__name__}")
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be in 1..{MAX_DEPTH} (bits per axis), got {depth}")


def _check_order(order: str) -> None:
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}{suggest(str(order), ORDERS)}; one of {ORDERS}")


def _check_grid(grid: Tensor, depth: int) -> None:
    if not isinstance(grid, Tensor) or grid.dtype not in _INT_DTYPES:
        raise TypeError(f"grid must be an integer tensor (int32 grid_coord), got {_describe(grid)}")
    if grid.dim() != 2 or grid.shape[1] != 3:
        raise ValueError(f"grid must have shape (N, 3), got {tuple(grid.shape)}")
    if _debug() and grid.numel():
        low, high = int(grid.min()), int(grid.max())
        if low < 0 or high >= 1 << depth:
            raise ValueError(
                f"grid values must lie in [0, 2**{depth}) for depth={depth}, got [{low}, {high}]; "
                "subtract the per-sample minimum or raise depth"
            )


def _check_batch_bits(batch_size: int, depth: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"serialize: batch_size must be a positive int, got {batch_size!r}")
    if 3 * depth + batch_size.bit_length() > 63:
        raise ValueError(
            f"serialize: batch_size={batch_size} needs {batch_size.bit_length()} bits above the "
            f"{3 * depth} code bits of depth={depth}; int64 keys hold 63. Pack fewer samples or "
            "lower depth."
        )
