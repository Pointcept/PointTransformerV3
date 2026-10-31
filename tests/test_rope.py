"""Utonia's 3D RoPE (``modules.Rope3d``, ``Config.rope_base``) and PC m3 conversion (CPU)."""

from __future__ import annotations

import pytest
import torch

import point_transformer_v3 as ptv3
from point_transformer_v3.convert import (
    ConvertError,
    convert_state_dict,
    pc_backbone_config,
    pc_version,
)
from point_transformer_v3.model import Config
from point_transformer_v3.modules import Rope3d, SerializedAttention
from point_transformer_v3.presets import UTONIA
from tests.test_model import cloud, small

# head_dim 18 at every stage (Utonia's), tiny widths
ROPE_TINY = dict(
    enc_depths=(1, 1, 1, 1, 1),
    enc_channels=(18, 18, 36, 36, 36),
    enc_num_heads=(1, 1, 2, 2, 2),
    enc_patch_sizes=(32, 32, 32, 32, 32),
)


def utonia_rope(q, k, xyz, base):
    """``Point3DRoPE.forward`` of Utonia's model.py (Pointcept/Utonia-Manifold be53e4a), op by
    op: q, k ``(N, H, D)``, xyz ``(N, 3)``."""
    chunk = q.shape[-1] // 3
    inv_freq = 1.0 / (base ** (torch.arange(0, chunk, 2).float() / chunk))
    freqs = inv_freq.unsqueeze(0)
    emb = [xyz[:, i : i + 1] * freqs for i in range(3)]
    emb = torch.cat([torch.cat((e, e), dim=-1) for e in emb], dim=-1)
    cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    out = []
    for t in (q, k):
        ts, cs, ss = (torch.split(a, chunk, dim=-1) for a in (t, cos, sin))
        out.append(torch.cat([ts[i] * cs[i] + rotate_half(ts[i]) * ss[i] for i in range(3)], -1))
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("heads", "head_dim"), [(3, 18), (2, 12), (1, 36)])
def test_rope_is_utonias_math_bit_for_bit(dtype, heads, head_dim):
    gen = torch.Generator().manual_seed(head_dim)
    n, c = 257, heads * head_dim
    qkv = torch.randn(n, 3 * c, generator=gen).to(dtype)
    coord = (torch.rand(n, 3, generator=gen) - 0.3) * 7
    rope = Rope3d(head_dim, 10)
    out = rope(qkv, coord)
    q, k, v = qkv.unflatten(1, (3, heads, head_dim)).unbind(1)
    ref_q, ref_k = utonia_rope(q, k, coord, 10)
    got_q, got_k, got_v = out.unflatten(1, (3, heads, head_dim)).unbind(1)
    assert out.dtype == dtype and torch.equal(got_v, v)
    # the original rotates in the promoted dtype (fp32) and casts afterwards (to bf16 for flash)
    assert torch.equal(got_q, ref_q.to(dtype)) and torch.equal(got_k, ref_k.to(dtype))


def test_rope_scores_depend_on_relative_positions_only():
    """q.k after RoPE is unchanged by a common translation: the rotations pair up correctly."""
    gen = torch.Generator().manual_seed(0)
    qkv = torch.randn(64, 3 * 2 * 18, generator=gen, dtype=torch.float64)
    coord = torch.rand(64, 3, generator=gen, dtype=torch.float64) * 3
    rope = Rope3d(18, 10).double()

    def scores(c):
        q, k, _ = rope(qkv, c).unflatten(1, (3, 2, 18)).unbind(1)
        return torch.einsum("qhd,khd->hqk", q, k)

    shifted = coord + torch.tensor([1.7, -4.2, 0.9], dtype=torch.float64)
    torch.testing.assert_close(scores(shifted), scores(coord), rtol=1e-9, atol=1e-9)
    assert not torch.allclose(scores(coord * 1.5), scores(coord))


def test_rope_buffer_init_and_config_checks():
    rope = Rope3d(18, 10)
    assert torch.equal(rope.inv_freq, 1.0 / (10 ** (torch.arange(0, 6, 2).float() / 6)))
    with torch.device("meta"):
        meta = ptv3.PointTransformerV3.from_preset("utonia_tiny")
    meta.to_empty(device="cpu")
    meta._init_weights()
    assert torch.equal(meta.enc[3].blocks[0].attn.rope.inv_freq, rope.inv_freq)
    assert "enc.0.blocks.0.attn.rope.inv_freq" in meta.state_dict()
    assert SerializedAttention(32, 2, 16).rope is None
    with pytest.raises(ValueError, match="multiples of 6"):
        ptv3.PointTransformerV3.from_preset("utonia_tiny", enc_num_heads=(1, 2, 4, 8, 8))
    with pytest.raises(ValueError, match="rope_base must be a positive number"):
        Config.from_dict({**UTONIA, "rope_base": 0})


def test_rope_model_invariances():
    """Packing invariance (fixed depth) and row-permutation equivariance with RoPE."""
    model = small("utonia", **ROPE_TINY, serialization_depth=16)
    a, b = cloud(900, seed=1, channels=9), cloud(600, seed=2, channels=9)
    with torch.inference_mode():
        fa, fb = ptv3.upcast(model(a)).feat, ptv3.upcast(model(b)).feat
        fp = ptv3.upcast(model(ptv3.collate([a, b]))).feat
        perm = torch.randperm(len(a.feat), generator=torch.Generator().manual_seed(0))
        fs = ptv3.upcast(model({k: a[k][perm] for k in ("coord", "grid_coord", "feat")})).feat
    torch.testing.assert_close(fp, torch.cat([fa, fb]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fs, fa[perm], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(fa, ptv3.upcast(small("utonia", **ROPE_TINY, rope_base=None)(a)).feat)


def test_pc_m3_config_and_keys_convert():
    kwargs = {
        "in_channels": 9, "enc_channels": (54, 108, 216, 432, 576),
        "enc_num_head": (3, 6, 12, 24, 32), "enc_depths": (3, 3, 3, 12, 3), "enc_mode": True,
        "mask_token": True, "traceable": True, "rope_base": 10,
    }  # fmt: skip
    stem = {"embedding.stem.linear.weight": None}
    assert pc_version(kwargs, stem) == "PT-v3m3" and pc_version({}, stem) == "PT-v3m2"
    values, _ = pc_backbone_config(kwargs, "PT-v3m3")
    assert {k: values[k] for k in UTONIA} == UTONIA
    with pytest.raises(ConvertError, match="RoPE coordinate augmentation"):
        pc_backbone_config({**kwargs, "jitter_coords": 1.2}, "PT-v3m3")
    assert pc_backbone_config({}, "PT-v3m2")[0]["rope_base"] is None
    # PC key names of a RoPE model (inv_freq included) map strictly onto the model
    model = small("utonia", **ROPE_TINY)
    target = model.state_dict()
    state = {_to_pc(k): v.clone() for k, v in target.items()}
    state["embedding.mask_token"] = torch.zeros(1, 18)
    assert pc_version({}, state) == "PT-v3m3"
    out, dropped = convert_state_dict(state, target)
    assert set(dropped) == {"embedding.mask_token"}
    assert all(torch.equal(out[k], v) for k, v in target.items())


def _to_pc(key: str) -> str:
    """Our encoder key -> the PC m2/m3 name."""
    import re

    k = re.sub(r"^enc\.(\d+)\.blocks\.(\d+)\.", r"enc.enc\1.block\2.", key)
    k = re.sub(r"^enc\.(\d+)\.down\.norm\.", r"enc.enc\1.down.norm.0.", k)
    k = re.sub(r"^enc\.(\d+)\.down\.proj\.", r"enc.enc\1.down.proj.", k)
    k = k.replace("embedding.proj.", "embedding.stem.linear.")
    k = k.replace("embedding.norm.", "embedding.stem.norm.")
    k = k.replace(".cpe.conv.", ".cpe.0.").replace(".cpe.proj.", ".cpe.1.")
    k = k.replace(".cpe.norm.", ".cpe.2.").replace(".norm1.", ".norm1.0.")
    return k.replace(".norm2.", ".norm2.0.").replace(".mlp.", ".mlp.0.")
