"""Released checkpoints convert strictly and load (marker ``weights``: reads or downloads every
preset's checkpoint and linear-probe head, ~4.6 GB)."""

from __future__ import annotations

import pytest
import torch

import pointtransformerv3 as ptv3
from pointtransformerv3.convert import convert_state_dict, map_name
from pointtransformerv3.modules import Rope3d
from pointtransformerv3.presets import PRESETS
from pointtransformerv3.weights import fetch, read_checkpoint, state_dict_of

pytestmark = pytest.mark.weights
ENCODERS = [name for name, p in PRESETS.items() if p.weights and not p.config["dec_depths"]]


@pytest.mark.parametrize(
    ("preset", "keys", "params"),
    [
        ("sonata", 453, 108_461_328),
        ("sonata_small", 273, 38_648_960),
        ("concerto", 453, 108_461_328),
        ("concerto_large", 453, 207_667_392),
        ("concerto_siglip", 453, 108_461_328),
        ("concerto_origin", 453, 108_461_328),
        ("utonia", 477, 137_253_690),
        ("utonia_small", 287, 37_248_480),
        ("utonia_tiny", 287, 9_326_088),
    ],
)
def test_encoders_convert_strictly(preset, keys, params):
    """Every key but the mask token maps, bit for bit; Utonia's RoPE buffers equal ours."""
    state = state_dict_of(read_checkpoint(fetch(PRESETS[preset].weights)))
    assert len(state) == keys
    with torch.device("meta"):
        model = ptv3.PointTransformerV3.from_preset(preset)
    out, dropped = convert_state_dict(state, model.state_dict())
    assert set(dropped) == {"embedding.mask_token"}
    assert len(out) == keys - 1 == len(model.state_dict())
    assert sum(p.numel() for p in model.parameters()) == params
    for key, value in state.items():
        if key != "embedding.mask_token":
            assert torch.equal(out[map_name(key)], value), key
    rope = [v for k, v in out.items() if k.endswith("rope.inv_freq")]
    assert len(rope) == (sum(model.cfg.enc_depths) if preset.startswith("utonia") else 0)
    assert all(torch.equal(v, Rope3d(18, 10).inv_freq) for v in rope)


@pytest.mark.parametrize(("preset", "channels"), [("sonata", 1232), ("concerto_large", 1728),
                                                  ("utonia", 1386)])  # fmt: skip
def test_linear_probe_heads(preset, channels):
    spec = PRESETS[preset].heads["scannet20"]
    head = state_dict_of(read_checkpoint(fetch(spec.weights)))
    assert set(head) == {"seg_head.weight", "seg_head.bias"}
    assert head["seg_head.weight"].shape == (20, channels) == (20, spec.in_channels)


def test_ptv3_release_segmentor_converts_strictly():
    ckpt = read_checkpoint(fetch(PRESETS["ptv3_scannet"].weights))
    state = state_dict_of(ckpt)
    assert len(state) == 488 and "optimizer" in ckpt  # training state loads as placeholders
    with torch.device("meta"):
        model = ptv3.PointTransformerV3.from_preset("ptv3_scannet")
        model.attach_head(20, 64)
    out, dropped = convert_state_dict(state, model.state_dict())
    assert not dropped and len(out) == 488


@pytest.mark.parametrize("preset", [*ENCODERS, "ptv3_scannet"])
def test_load_on_cpu(preset):
    model = ptv3.load(preset, device="cpu", warmup=False)
    info = model.weights_info["backbone"]
    assert info["sha256"] == PRESETS[preset].weights.sha256
    expected = {"ptv3": "MIT", "sonata": "CC-BY-NC-4.0"}.get(
        preset.split("_")[0], "CC-BY-NC-4.0 + research permission"
    )
    assert info["license"] == expected
    assert (model.seg_head is not None) == (preset == "ptv3_scannet")
    x = torch.rand(500, 3) * 2
    feat = model.embed(x, x * 255, torch.nn.functional.normalize(x - 1, dim=1))
    assert feat.shape == (500, model.feature_channels) and feat.isfinite().all()


@pytest.mark.parametrize("preset", ["sonata", "utonia"])
def test_load_with_head_predicts(preset):
    model = ptv3.load(preset, head="scannet20", device="cpu")
    assert "head" in model.weights_info
    assert model.seg_head.in_features == PRESETS[preset].heads["scannet20"].in_channels
    x = torch.rand(400, 3) * 2
    logits = model.predict(x, x * 255, torch.nn.functional.normalize(x - 1, dim=1))
    assert logits.shape == (400, 20)


def test_load_local_checkpoint_with_config():
    path = fetch(PRESETS["sonata_small"].weights)
    model = ptv3.load(str(path), device="cpu")
    assert model.preset is None and model.cfg.enc_depths == (2, 2, 2, 6, 2)
