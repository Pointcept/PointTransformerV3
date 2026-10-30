"""PointTransformerV3: constructors, the compat kwargs, forward contracts, invariances (CPU)."""

from __future__ import annotations

import re

import pytest
import torch

import pointtransformerv3 as ptv3
from pointtransformerv3.convert import RULES, ConvertError
from pointtransformerv3.model import Config, PointTransformerV3
from pointtransformerv3.presets import PC_M1, PRESETS
from pointtransformerv3.structure import auto_depth

TINY = dict(
    enc_depths=(1, 1, 1, 1, 1),
    enc_channels=(16, 16, 32, 32, 32),
    enc_num_heads=(1, 1, 2, 2, 2),
    enc_patch_sizes=(32, 32, 32, 32, 32),
    dec_depths=(1, 1, 1, 1),
    dec_channels=(16, 16, 32, 32),
    dec_num_heads=(1, 1, 2, 2),
    dec_patch_sizes=(32, 32, 32, 32),
)


def cloud(n: int, seed: int = 0, extent: float = 1.5, channels: int = 6) -> dict:
    gen = torch.Generator().manual_seed(seed)
    coord = torch.rand(n, 3, generator=gen) * extent
    point, _ = ptv3.prepare(
        coord,
        torch.rand(n, 3, generator=gen) * 255,
        torch.nn.functional.normalize(torch.randn(n, 3, generator=gen), dim=1),
        feat=("coord", "color", "normal") if channels == 9 else ("color", "normal"),
    )
    return point


def small(preset: str = "ptv3", **over) -> PointTransformerV3:
    torch.manual_seed(0)
    sizes = dict(TINY)
    if not PRESETS[preset].config["dec_depths"]:
        sizes.update(dec_depths=(), dec_channels=(), dec_num_heads=(), dec_patch_sizes=())
    model = PointTransformerV3.from_preset(preset, **{**sizes, **over})
    model.precision = "fp32"
    return model.eval()


# ---- constructors


def test_compat_defaults_are_pc_m1_with_auto_depth():
    model = PointTransformerV3()
    cfg = model.cfg
    for key, value in PC_M1.items():
        assert getattr(cfg, key) == value, key
    assert cfg.serialization_depth == "auto"
    assert cfg.enc_channels == (32, 64, 128, 256, 512) and cfg.dec_channels == (64, 64, 128, 256)
    assert cfg.legacy_cpe_stale_input is False  # model.py has the 74c4fc1 fix
    assert model.precision == "auto"
    assert sum(p.numel() for p in model.parameters()) == 46_166_272


def test_compat_positional_and_options():
    model = PointTransformerV3(6, ("z", "hilbert"), enable_flash=False, cls_mode=True)
    assert model.cfg.orders == ("z", "hilbert")
    assert model.precision == "fp32"  # enable_flash=False: the exact path
    assert model.cfg.dec_depths == () and model.out_channels == 512
    assert PointTransformerV3(order="z").cfg.orders == ("z",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(enable_rpe=True), "relative position encoding"),
        (dict(pdnorm_bn=True), "PDNorm"),
        (dict(pdnorm_ln=True), "PDNorm"),
        (dict(qk_scale=0.5), "qk_scale"),
        (dict(attn_drop=0.1), "attention dropout"),
        (dict(proj_drop=0.1), "projection dropout"),
        (dict(pre_norm=False), "pre-norm"),
    ],
)
def test_compat_unsupported_options_raise_with_fix(kwargs, message):
    with pytest.raises(ConvertError, match=message):
        PointTransformerV3(**kwargs)


def test_compat_default_qk_scale_is_accepted():
    PointTransformerV3(qk_scale=16**-0.5)


def test_compat_unknown_kwarg_suggests():
    with pytest.raises(TypeError, match=re.escape("did you mean 'enc_channels'?")):
        PointTransformerV3(enc_channel=(32, 64, 128, 256, 512))
    with pytest.raises(TypeError, match="either config"):
        PointTransformerV3(config=Config(), in_channels=6)


def test_config_constructors_and_errors():
    model = PointTransformerV3(config={"in_channels": 4, "serialization_depth": 12})
    assert model.cfg.in_channels == 4 and model.cfg.serialization_depth == 12
    assert PointTransformerV3.from_preset("sonata").cfg.stem == "linear"
    with pytest.raises(TypeError, match="did you mean 'in_channels'"):
        Config.from_dict({"in_channel": 6})
    with pytest.raises(ValueError, match="serialization_depth"):
        Config(serialization_depth="adaptive")
    assert PointTransformerV3.from_preset("utonia_tiny").cfg.rope_base == 10.0
    with pytest.raises(ValueError, match="did you mean 'sonata'"):
        PointTransformerV3.from_preset("sonta")


def test_meta_build_then_init_equals_eager():
    torch.manual_seed(3)
    eager = PointTransformerV3.from_preset("sonata_small")
    torch.manual_seed(3)
    with torch.device("meta"):
        meta = PointTransformerV3.from_preset("sonata_small")
    assert all(p.device.type == "meta" for p in meta.parameters())
    meta.to_empty(device="cpu")
    meta._init_weights()
    a, b = eager.state_dict(), meta.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_seed_gives_identical_state_dict():
    torch.manual_seed(5)
    a = PointTransformerV3().state_dict()
    torch.manual_seed(5)
    b = PointTransformerV3().state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


# ---- forward contracts


def test_forward_inputs_offset_batch_or_single():
    model = small()
    point = cloud(800)
    with torch.inference_mode():
        a = model(point)
        b = model({k: v for k, v in point.items() if k != "offset"})
        c = model(
            {
                **{k: v for k, v in point.items() if k != "offset"},
                "batch": torch.zeros(len(point["feat"]), dtype=torch.long),
            }
        )
    assert a.feat.shape == (len(point["feat"]), 16)
    assert torch.equal(a.feat, b.feat) and torch.equal(a.feat, c.feat)
    assert a.offset.tolist() == [len(point["feat"])] and a.batch.max() == 0
    assert isinstance(a, ptv3.Point) and a.grid_size == 0.02


def test_forward_grid_coord_from_coord_and_grid_size():
    """Without grid_coord the original's rule applies: trunc((coord - min) / grid_size)."""
    model = small()
    point = cloud(500)
    gc = torch.div(point.coord - point.coord.min(0)[0], 0.02, rounding_mode="trunc").int()
    with torch.inference_mode():
        a = model({**point, "grid_coord": gc})
        b = model({k: v for k, v in point.items() if k != "grid_coord"})
    assert torch.equal(a.feat, b.feat)


def test_forward_errors():
    model = small()
    point = cloud(200)
    with pytest.raises(KeyError, match="feat"):
        model({k: v for k, v in point.items() if k != "feat"})
    with pytest.raises(ValueError, match="in_channels=6"):
        model({**point, "feat": point.feat[:, :3]})
    with pytest.raises(ValueError, match="cover"):
        model({**point, "offset": torch.tensor([5])})
    with pytest.raises(AttributeError, match="did you mean 'feat'"):
        _ = point.feet


def test_carries_input_keys_and_point_is_a_dict():
    model = small()
    point = cloud(300)
    point["inverse"] = torch.arange(7)
    with torch.inference_mode():
        out = model(point)
    assert torch.equal(out.inverse, torch.arange(7))
    assert isinstance(out, dict) and "feat" in out and out.copy().feat is out.feat


@pytest.mark.parametrize("rule", ["auto", "auto+1"])
@pytest.mark.parametrize("policy", ["shift", "reencode"])
def test_auto_depth_equals_the_explicit_depth(rule, policy):
    """'auto' picks the depth of the batch; a fixed int depth of the same value gives the same
    output (shift); reencode re-derives each level's depth, so compare against auto_depth."""
    point = cloud(1500, extent=3.0)
    top = int(point.grid_coord.max())
    depth = auto_depth(top, rule=rule, minimum=5 if policy == "shift" else 2)
    a = small(serialization_depth=rule, pool_policy=policy)
    torch.manual_seed(0)
    with torch.inference_mode():
        out = a(point)
    assert out.feat.isfinite().all()
    if policy == "shift":
        b = small(serialization_depth=depth, pool_policy=policy)
        with torch.inference_mode():
            assert torch.equal(out.feat, b(point).feat)


def test_auto_depth_rules():
    assert auto_depth(255, rule="auto") == 8 and auto_depth(255, rule="auto+1") == 9
    assert auto_depth(256, rule="auto") == 9 and auto_depth(254, rule="auto+1") == 8
    assert auto_depth(3, rule="auto", minimum=5) == 5  # 2 + 3: Hilbert codes unchanged
    assert auto_depth(0, rule="auto", minimum=5) == 6


def test_tiny_and_single_point_clouds_run():
    model = small(serialization_depth="auto")
    for n in (1, 2, 17):
        gen = torch.Generator().manual_seed(n)
        grid = torch.randint(0, 4, (n, 3), generator=gen).unique(dim=0).int()
        data = {
            "coord": grid.float() * 0.02,
            "grid_coord": grid,
            "feat": torch.randn(len(grid), 6, generator=gen),
        }
        with torch.inference_mode():
            assert model(data).feat.shape == (len(grid), 16)


@pytest.mark.parametrize("preset", ["sonata_small", "ptv3"])
def test_packing_invariance_with_fixed_depth(preset):
    """forward(pack[A, B]) == cat(forward(A), forward(B)) with an int serialization depth and
    LayerNorm (BatchNorm in eval is per-row too)."""
    model = small(preset, serialization_depth=16)
    channels = model.cfg.in_channels
    a, b = cloud(900, seed=1, channels=channels), cloud(600, seed=2, channels=channels)
    pack = ptv3.collate([a, b])
    with torch.inference_mode():
        fa, fb = ptv3.upcast(model(a)).feat, ptv3.upcast(model(b)).feat
        fp = ptv3.upcast(model(pack)).feat
    torch.testing.assert_close(fp, torch.cat([fa, fb]), atol=1e-5, rtol=1e-5)


def test_row_permutation_equivariance():
    model = small("sonata_small", serialization_depth="auto+1")
    point = cloud(1200, channels=9)
    perm = torch.randperm(len(point.feat), generator=torch.Generator().manual_seed(0))
    shuffled = {k: point[k][perm] for k in ("coord", "grid_coord", "feat")}
    with torch.inference_mode():
        a = ptv3.upcast(model(point)).feat
        b = ptv3.upcast(model(shuffled)).feat
    torch.testing.assert_close(b, a[perm], atol=1e-5, rtol=1e-5)


def test_encoder_output_chain_and_upcast():
    model = small("sonata_small")
    point = cloud(1000, channels=9)
    with torch.inference_mode():
        out = model(point)
    depth, p = 0, out
    while "pooling_parent" in p:
        assert p.pooling_inverse.shape == (len(p.pooling_parent.feat),)
        p, depth = p.pooling_parent, depth + 1
    assert depth == 4 and len(p.feat) == len(point.feat)
    full, two = ptv3.upcast(out), ptv3.upcast(out, levels=2)
    assert full.feat.shape[1] == sum(model.cfg.enc_channels) == model.feature_channels
    assert two.feat.shape[1] == sum(model.cfg.enc_channels[2:])
    assert "pooling_parent" in out  # upcast does not modify its input


def test_embed_maps_back_to_input_points():
    model = small("sonata_small")
    gen = torch.Generator().manual_seed(0)
    coord = torch.rand(700, 3, generator=gen) * 2
    color = torch.rand(700, 3, generator=gen) * 255
    normal = torch.nn.functional.normalize(torch.randn(700, 3, generator=gen), dim=1)
    feat = model.embed(coord, color, normal)
    assert feat.shape == (700, sum(model.cfg.enc_channels))
    many = model.embed([coord, coord[:300]], [color, color[:300]], [normal, normal[:300]])
    assert [f.shape[0] for f in many] == [700, 300]
    with pytest.raises(RuntimeError, match="no segmentation head"):
        model.predict(coord, color, normal)
    model.attach_head(5)
    assert model.predict(coord, color, normal).shape == (700, 5)


def test_load_state_dict_converts_pointcept_names():
    """Original model.py / Pointcept names (enc.enc0.block0...) load into the new tree."""
    torch.manual_seed(1)
    src = PointTransformerV3()
    inverse = {
        r"embedding.conv.weight": "embedding.stem.conv.weight",
    }
    state = {}
    for key, value in src.state_dict().items():
        name = inverse.get(key) or _to_pointcept(key)
        state[name] = value
    torch.manual_seed(2)
    dst = PointTransformerV3()
    dst.load_state_dict(state)
    assert all(torch.equal(v, dst.state_dict()[k]) for k, v in src.state_dict().items())
    with pytest.raises(ConvertError, match="not filled"):
        dst.load_state_dict({k: v for k, v in list(state.items())[1:]})


def _to_pointcept(key: str) -> str:
    """Our name -> Pointcept's (the inverse of RULES, for the keys of a PC m1 model)."""
    k = re.sub(r"^(enc|dec)\.(\d+)\.blocks\.(\d+)\.", r"\1.\g<1>\2.block\3.", key)
    k = re.sub(r"^enc\.(\d+)\.down\.norm\.", r"enc.enc\1.down.norm.0.", k)
    k = re.sub(r"^enc\.(\d+)\.down\.proj\.", r"enc.enc\1.down.proj.", k)
    k = re.sub(r"^dec\.(\d+)\.up\.proj\.", r"dec.dec\1.up.proj.0.", k)
    k = re.sub(r"^dec\.(\d+)\.up\.norm\.", r"dec.dec\1.up.proj.1.", k)
    k = re.sub(r"^dec\.(\d+)\.up\.skip_proj\.", r"dec.dec\1.up.proj_skip.0.", k)
    k = re.sub(r"^dec\.(\d+)\.up\.skip_norm\.", r"dec.dec\1.up.proj_skip.1.", k)
    k = k.replace("embedding.norm.", "embedding.stem.norm.")
    k = k.replace(".cpe.conv.", ".cpe.0.").replace(".cpe.proj.", ".cpe.1.")
    k = k.replace(".cpe.norm.", ".cpe.2.").replace(".norm1.", ".norm1.0.")
    k = k.replace(".norm2.", ".norm2.0.").replace(".mlp.", ".mlp.0.")
    assert any(re.fullmatch(p, k) for p, _ in RULES), (key, k)
    return k
