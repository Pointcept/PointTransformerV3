# Vendored from Pointcept2 src/pointcept2/ops/patches.py @ 771a949 (simplified for point_transformer_v3).
"""Patch layouts for serialized attention: which positions of a serialized sequence attend together.

Positions index the serialized sequence (samples contiguous, each sample sorted by its code). A
sample of n positions and patch size K is cut into ceil(n / K) patches:

- n <= K: one patch; queries = keys = the n positions.
- n > K, ``"overlap"``: full patches of K; the last partial patch has the r = n % K tail positions
  as queries and the last K positions of the sample as keys. This equals PC's duplicate padding
  (``point_transformer_v3m1_base.py:116-171``), whose duplicated rows only ever act as keys.
- n > K, ``"shrink"``: the last partial patch uses its r positions as both queries and keys.

Queries cover every position exactly once and in order, so ``q_pos`` is ``arange(sum(seq_lens))``.
Every key run is a range of positions (an "overlap" tail's keys are the last K positions of its
sample), which ``PatchLayout.table`` stores as ``(q_start, q_len, k_start, k_len)`` per patch: the
form the ``triton_d16`` attention kernels read (``ops/_attention_triton.py``). Key positions that
are not queries of their own patch lie in the previous patch of the same sample, so every position
has one owning patch.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import Literal

import torch
from torch import Tensor

__all__ = ["PatchLayout", "Tail", "build_patches"]

Tail = Literal["overlap", "shrink"]

_TAILS = ("overlap", "shrink")
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class PatchLayout:
    """P attention patches over a serialized sequence, in two forms: the packed layout
    ``varlen_attention`` takes and the range table of the ``triton_d16`` kernels.

    Patch i has queries ``q_pos[cu_q[i]:cu_q[i + 1]]`` and keys/values
    ``kv_pos[cu_k[i]:cu_k[i + 1]]``; ``table[i]`` is ``(q_start, q_len, k_start, k_len)``: the
    queries ``[q_start, q_start + q_len)`` and the keys ``[k_start, k_start + k_len)`` as position
    ranges (``q_start == cu_q[i]``, ``q_len == cu_q[i + 1] - cu_q[i]``, the same for the keys
    against ``cu_k`` and ``kv_pos``). The last row of ``table`` is the sentinel ``(Tq, 0, Tq, 0)``.
    """

    q_pos: Tensor  # (Tq,) int64 positions; each position exactly once, Tq = sum(seq_lens)
    kv_pos: Tensor  # (Tk,) int64 positions, Tk >= Tq
    cu_q: Tensor  # (P + 1,) int32
    cu_k: Tensor  # (P + 1,) int32
    table: Tensor  # (P + 1, 4) int32 (q_start, q_len, k_start, k_len); sentinel row last
    max_q: int  # host: longest query run
    max_k: int  # host: longest key run


def build_patches(
    seq_lens: Sequence[int],
    patch_size: int,
    tail: Tail = "overlap",
    device: torch.device | str | None = None,
) -> PatchLayout:
    """Cut each sample of a serialized sequence into attention patches of ``patch_size``.

    ``seq_lens`` are the host sample lengths (zeros allowed). Sync-free: the per-patch tables
    (``cu_q``, ``cu_k``, ``table``) are built on the host and copied in one non-blocking transfer
    (pinned on CUDA); per-position tables are expanded on ``device`` (default CPU) with
    ``repeat_interleave(output_size=...)``.
    """
    if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size < 1:
        raise ValueError(f"patch_size must be a positive int, got {patch_size!r}")
    if tail not in _TAILS:
        raise ValueError(f"tail must be one of {_TAILS}, got {tail!r}")
    lens = tuple(seq_lens)
    bad = [n for n in lens if not isinstance(n, int) or isinstance(n, bool) or n < 0]
    if bad:
        raise TypeError(
            f"seq_lens must be non-negative host ints (keep sizes on the host), got {bad[:3]}"
        )
    if sum(lens) > _INT32_MAX:
        raise ValueError(f"{sum(lens)} positions overflow the int32 cu_seqlens; split the batch")
    device = torch.device("cpu" if device is None else device)

    q_lens: list[int] = []
    k_lens: list[int] = []
    shift: list[int] = []  # per patch: kv position - kv row
    table: list[int] = []  # per patch: q_start, q_len, k_start, k_len (positions)
    start = k_start = 0  # first position / first kv row of the current sample
    for n in lens:
        full, r = divmod(n, patch_size)
        runs = [patch_size] * full + ([r] if r else [])
        extra = patch_size - r if tail == "overlap" and full and r else 0  # tail keys added
        q_lens += runs
        k_lens += [patch_size] * len(runs) if extra else runs
        shift += [start - k_start] * full
        for i in range(full):
            s = start + i * patch_size
            table += (s, patch_size, s, patch_size)
        if r:
            shift.append(start - k_start - extra)
            s = start + full * patch_size
            table += (s, r, s - extra, r + extra)
        start += n
        k_start += n + extra

    if k_start > _INT32_MAX:
        raise ValueError(f"{k_start} key positions overflow the int32 cu_seqlens; split the batch")
    num = len(q_lens)
    has_tail = k_start != start  # without overlap tails every shift is 0 and kv_pos == q_pos
    table += (start, 0, start, 0)  # sentinel: the "next patch" of the last one
    # the table first: it starts the (aligned) allocation, as the kernels' own tensors do
    host = [*table, 0, *accumulate(q_lens), 0, *accumulate(k_lens), *(shift if has_tail else ())]
    flat_table, cu_q, cu_k, kv_shift = _host_to_device(host, device).split(
        [4 * (num + 1), num + 1, num + 1, num if has_tail else 0]
    )
    q_pos = torch.arange(start, device=device)
    kv_pos = q_pos
    if has_tail:
        kv_shift = kv_shift.repeat_interleave(cu_k.diff(), output_size=k_start)
        kv_pos = torch.arange(k_start, device=device) + kv_shift
    return PatchLayout(
        q_pos=q_pos,
        kv_pos=kv_pos,
        cu_q=cu_q,
        cu_k=cu_k,
        table=flat_table.view(num + 1, 4),
        max_q=max(q_lens, default=0),
        max_k=max(k_lens, default=0),
    )


def _host_to_device(values: list[int], device: torch.device) -> Tensor:
    """One int32 host table -> ``device``: pinned + non-blocking on CUDA, so no host sync.
    (``array`` + ``frombuffer`` converts a list about 4x faster than ``torch.tensor``.)"""
    host = torch.frombuffer(array("i", values), dtype=torch.int32)
    if device.type != "cuda":
        return host.to(device)
    return host.pin_memory().to(device, non_blocking=True)
