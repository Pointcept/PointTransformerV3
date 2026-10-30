# Adapted from Pointcept2 tests/ops/test_patches.py @ 771a949.
"""build_patches: layout rules, PC padding parity (index sets and fp32 math), sync-free on CUDA."""

from __future__ import annotations

import random
import warnings
from itertools import accumulate, pairwise

import pytest
import torch
from torch import nn

from pointtransformerv3.ops.attention import varlen_attention
from pointtransformerv3.ops.patches import PatchLayout, build_patches
from tests.conftest import DEVICES


def expected_patches(seq_lens, k, tail):
    """Per patch (query positions, key positions), written directly from the spec."""
    patches = []
    start = 0
    for n in seq_lens:
        for a in range(0, n, k):
            b = min(a + k, n)
            queries = list(range(start + a, start + b))
            keys = (
                list(range(start + n - k, start + n))
                if tail == "overlap" and b - a < k < n
                else queries
            )
            patches.append((queries, keys))
        start += n
    return patches


def patches_of(layout: PatchLayout):
    cu_q, cu_k = layout.cu_q.tolist(), layout.cu_k.tolist()
    q_pos, kv_pos = layout.q_pos.tolist(), layout.kv_pos.tolist()
    return [
        (q_pos[cu_q[i] : cu_q[i + 1]], kv_pos[cu_k[i] : cu_k[i + 1]]) for i in range(len(cu_q) - 1)
    ]


def edge_seq_lens(k: int, seed: int) -> list[int]:
    """n < K, n == K, n % K == 0, n % K small, plus random lengths, shuffled."""
    rng = random.Random(seed)
    lens = [max(1, k // 3), k, 3 * k, 2 * k + 1, 2 * k + 2, k + k // 2]
    lens += [rng.randint(1, 5 * k) for _ in range(4)]
    rng.shuffle(lens)
    return lens


def test_overlap_and_shrink_example():
    layout = build_patches((10,), 4, "overlap")
    assert layout.q_pos.tolist() == list(range(10))
    assert layout.kv_pos.tolist() == [*range(8), 6, 7, 8, 9]
    assert layout.cu_q.tolist() == [0, 4, 8, 10]
    assert layout.cu_k.tolist() == [0, 4, 8, 12]
    assert (layout.max_q, layout.max_k) == (4, 4)
    shrink = build_patches((10,), 4, "shrink")
    assert shrink.kv_pos.tolist() == list(range(10))
    assert shrink.cu_k.tolist() == [0, 4, 8, 10]


@pytest.mark.parametrize("tail", ["overlap", "shrink"])
@pytest.mark.parametrize("seed", range(6))
def test_layout_matches_spec(tail, seed):
    rng = random.Random(seed)
    k = rng.choice([1, 2, 7, 16, 48])
    seq_lens = [rng.choice([0, 1, k - 1, k, k + 1, 2 * k, rng.randint(0, 6 * k)]) for _ in range(7)]
    layout = build_patches(tuple(seq_lens), k, tail)
    expected = expected_patches(seq_lens, k, tail)
    assert patches_of(layout) == expected
    assert layout.q_pos.dtype == layout.kv_pos.dtype == torch.int64
    assert layout.cu_q.dtype == layout.cu_k.dtype == torch.int32
    assert layout.max_q == max((len(q) for q, _ in expected), default=0)
    assert layout.max_k == max((len(kv) for _, kv in expected), default=0)
    assert isinstance(layout.max_q, int) and isinstance(layout.max_k, int)


@pytest.mark.parametrize("seq_lens", [(), (0,), (0, 0, 0)])
def test_no_positions(seq_lens):
    layout = build_patches(seq_lens, 8)
    assert layout.cu_q.tolist() == layout.cu_k.tolist() == [0]
    assert layout.q_pos.numel() == layout.kv_pos.numel() == 0
    assert (layout.max_q, layout.max_k) == (0, 0)
    out = varlen_attention(
        torch.zeros(0, 1, 8), torch.zeros(0, 1, 8), torch.zeros(0, 1, 8),
        layout.cu_q, layout.cu_k, layout.max_q, layout.max_k, scale=1.0,
    )  # fmt: skip
    assert out.shape == (0, 1, 8)


def test_argument_checks():
    with pytest.raises(ValueError, match="patch_size"):
        build_patches((4,), 0)
    with pytest.raises(ValueError, match="patch_size"):
        build_patches((4,), True)
    with pytest.raises(ValueError, match="tail"):
        build_patches((4,), 2, "pad")  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="host ints"):
        build_patches((4, -1), 2)
    with pytest.raises(TypeError, match="host ints"):
        build_patches(tuple(torch.tensor([4, 5])), 2)  # pyright: ignore[reportArgumentType]


@pytest.mark.gpu
@pytest.mark.parametrize("tail", ["overlap", "shrink"])
def test_cuda_attention_path_is_sync_free(tail):
    """build_patches -> gather -> varlen_attention -> scatter, forward and backward."""
    seq_lens = tuple(edge_seq_lens(64, seed=1))
    qkv = torch.randn(sum(seq_lens), 3, 2, 16, device="cuda", dtype=torch.bfloat16)
    qkv.requires_grad_()
    order = torch.randperm(sum(seq_lens), device="cuda")
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Synchronization debug mode")
        torch.cuda.set_sync_debug_mode("error")
    try:
        layout = build_patches(seq_lens, 64, tail, device="cuda")
        q_rows, kv_rows = order[layout.q_pos], order[layout.kv_pos]
        q, k, v = qkv.unbind(1)
        out = varlen_attention(
            q[q_rows], k[kv_rows], v[kv_rows],
            layout.cu_q, layout.cu_k, layout.max_q, layout.max_k, scale=0.25,
        )  # fmt: skip
        torch.zeros_like(out).index_copy(0, q_rows, out).sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    host = build_patches(seq_lens, 64, tail)
    for name in ("q_pos", "kv_pos", "cu_q", "cu_k"):
        got = getattr(layout, name)
        assert got.device.type == "cuda"
        assert torch.equal(got.cpu(), getattr(host, name)), name
    assert (layout.max_q, layout.max_k) == (host.max_q, host.max_k)


# ---- parity with PC SerializedAttention (Pointcept point_transformer_v3m1_base.py, inlined)


def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


def pc_get_padding_and_inverse(offset, patch_size):
    """PC ``SerializedAttention.get_padding_and_inverse`` (lines 116-171), verbatim math."""
    bincount = offset2bincount(offset)
    bincount_pad = (
        torch.div(bincount + patch_size - 1, patch_size, rounding_mode="trunc") * patch_size
    )
    # only pad point when num of points larger than patch_size
    mask_pad = bincount > patch_size
    bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
    _offset = nn.functional.pad(offset, (1, 0))
    _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
    pad = torch.arange(_offset_pad[-1], device=offset.device)
    unpad = torch.arange(_offset[-1], device=offset.device)
    cu_seqlens = []
    for i in range(len(offset)):
        unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
        if bincount[i] != bincount_pad[i]:
            pad[
                _offset_pad[i + 1] - patch_size + (bincount[i] % patch_size) : _offset_pad[i + 1]
            ] = pad[
                _offset_pad[i + 1] - 2 * patch_size + (bincount[i] % patch_size) : _offset_pad[
                    i + 1
                ]
                - patch_size
            ]
        pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
        cu_seqlens.append(
            torch.arange(
                _offset_pad[i],
                _offset_pad[i + 1],
                step=patch_size,
                dtype=torch.int32,
                device=offset.device,
            )
        )
    cu_seqlens = nn.functional.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1])
    return pad, unpad, cu_seqlens


def flash_varlen_qkvpacked_fp32(qkv, cu_seqlens, softmax_scale):
    """Exact fp32 stand-in for ``flash_attn_varlen_qkvpacked_func`` (no dropout, no causal)."""
    out = []
    bounds = cu_seqlens.tolist()
    for a, b in pairwise(bounds):
        q, k, v = qkv[a:b].permute(1, 2, 0, 3).unbind(0)  # (H, L, D) each
        attn = ((q * softmax_scale) @ k.transpose(-2, -1)).softmax(-1)
        out.append((attn @ v).transpose(0, 1))
    return torch.cat(out)


def pc_attention_core(qkv, offset, serialized_order, serialized_inverse, patch_size, heads, flash):
    """PC ``SerializedAttention.forward`` from qkv to the feature before ``proj`` (no RPE, eval)."""
    if not flash:
        patch_size = min(offset2bincount(offset).min().tolist(), patch_size)
    H, K, C = heads, patch_size, qkv.shape[1] // 3
    scale = (C // H) ** -0.5
    pad, unpad, cu_seqlens = pc_get_padding_and_inverse(offset, K)
    order = serialized_order[pad]
    inverse = unpad[serialized_inverse]
    qkv = qkv[order]
    if not flash:
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        attn = (q * scale) @ k.transpose(-2, -1)  # (N', H, K, K)
        attn = torch.softmax(attn, dim=-1)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)
    else:
        feat = flash_varlen_qkvpacked_fp32(qkv.reshape(-1, 3, H, C // H), cu_seqlens, scale)
        feat = feat.reshape(-1, C)
    return feat[inverse]


def new_attention_core(qkv, seq_lens, serialized_order, patch_size, heads):
    """Gather by serialized order -> varlen_attention on the overlap layout -> scatter back."""
    layout = build_patches(seq_lens, patch_size, "overlap", device=qkv.device)
    q_rows = serialized_order[layout.q_pos]
    kv_rows = serialized_order[layout.kv_pos]
    q, k, v = qkv.unflatten(1, (3, heads, -1)).unbind(1)
    out = varlen_attention(
        q[q_rows], k[kv_rows], v[kv_rows],
        layout.cu_q, layout.cu_k, layout.max_q, layout.max_k, scale=q.shape[-1] ** -0.5,
    )  # fmt: skip
    return torch.zeros_like(out).index_copy(0, q_rows, out).flatten(1)


def random_serialization(seq_lens, generator):
    """A serialized order that keeps samples contiguous (as batch-prefixed codes do)."""
    starts = [0, *accumulate(seq_lens)]
    order = torch.cat(
        [
            torch.randperm(n, generator=generator) + s
            for n, s in zip(seq_lens, starts[:-1], strict=True)
        ]
    )
    return order, torch.argsort(order)


@pytest.mark.parametrize("patch_size", [3, 16, 128])
@pytest.mark.parametrize("seed", range(3))
def test_layout_equals_pc_padding(patch_size, seed):
    """Patch by patch: PC's kept queries and its (duplicate-padded) key set equal ours."""
    seq_lens = edge_seq_lens(patch_size, seed)
    offset = torch.tensor(list(accumulate(seq_lens)))
    pad, unpad, cu_seqlens = pc_get_padding_and_inverse(offset, patch_size)
    bounds = cu_seqlens.tolist()
    patch_of_padded = torch.repeat_interleave(torch.tensor(bounds).diff())
    query_patch = patch_of_padded[unpad].tolist()  # PC patch that produces each position's output
    pc = [
        (sorted(p for p, j in enumerate(query_patch) if j == i), sorted(pad[a:b].tolist()))
        for i, (a, b) in enumerate(pairwise(bounds))
    ]
    ours = [(q, sorted(kv)) for q, kv in patches_of(build_patches(tuple(seq_lens), patch_size))]
    assert ours == pc


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("flash", [True, False], ids=["flash_path", "nonflash_path"])
@pytest.mark.parametrize(
    ("patch_size", "heads", "channels"), [(16, 3, 48), (48, 4, 64), (1024, 2, 32)]
)
@pytest.mark.parametrize("seed", range(3))
def test_matches_pc_serialized_attention_fp32(device, flash, patch_size, heads, channels, seed):
    g = torch.Generator().manual_seed(seed)
    seq_lens = edge_seq_lens(patch_size, seed)
    if patch_size == 1024:  # keep the CPU reference cheap
        seq_lens = seq_lens[:4]
    order, inverse = random_serialization(seq_lens, g)
    qkv = torch.randn(sum(seq_lens), 3 * channels, generator=g)
    qkv_pc, qkv_new = (qkv.to(device, copy=True).requires_grad_() for _ in range(2))
    offset = torch.tensor(list(accumulate(seq_lens)), device=device)
    order, inverse = order.to(device), inverse.to(device)

    ref = pc_attention_core(qkv_pc, offset, order, inverse, patch_size, heads, flash)
    k = patch_size if flash else min(min(seq_lens), patch_size)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="varlen_attention falls back")
        out = new_attention_core(qkv_new, tuple(seq_lens), order, k, heads)
        grad = torch.randn(ref.shape, generator=g).to(device)
        ref.backward(grad)
        out.backward(grad)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    assert qkv_new.grad is not None and qkv_pc.grad is not None
    torch.testing.assert_close(qkv_new.grad, qkv_pc.grad, atol=1e-5, rtol=1e-5)
