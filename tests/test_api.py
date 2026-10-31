"""load / set_backend / the package surface (CPU, no downloads)."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

import point_transformer_v3 as ptv3


def run(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"},
    )
    return out.stdout.strip()


def test_import_is_lazy_and_fast():
    out = run(
        "import sys, time; t = time.perf_counter(); import point_transformer_v3; "
        "dt = time.perf_counter() - t; "
        "print(dt, *[m in sys.modules for m in ('torch', 'triton', 'huggingface_hub')])"
    )
    seconds, *loaded = out.split()
    assert float(seconds) < 1.0 and loaded == ["False", "False", "False"]


def test_cpu_forward_never_imports_triton_or_hub():
    out = run(
        "import sys, torch, point_transformer_v3 as ptv3\n"
        "m = ptv3.load('ptv3', device='cpu')\n"
        "x = torch.rand(400, 3)\n"
        "f = m.embed(x, x * 255, torch.nn.functional.normalize(x, dim=1))\n"
        "print(tuple(f.shape), 'triton' in sys.modules, 'huggingface_hub' in sys.modules)"
    )
    assert out == "(400, 64) False False"


def test_load_random_init_preset_options():
    model = ptv3.load("ptv3", device="cpu")
    assert not model.training and model.preset == "ptv3" and model.weights_info == {}
    assert model.precision == "auto" and model.cfg.serialization_depth == 15
    assert all(p.device.type == "cpu" for p in model.parameters())
    for precision in ("fp32", "fp16", "bf16"):
        assert ptv3.load("ptv3", device="cpu", precision=precision).precision == precision
    with pytest.raises(ValueError, match="precision"):
        ptv3.load("ptv3", device="cpu", precision="fp8")


def test_load_errors():
    with pytest.raises(ValueError, match="did you mean 'utonia'"):
        ptv3.load("utonai")
    with pytest.raises(ValueError, match="did you mean 'sonata_small'"):
        ptv3.load("sonata_smal")
    with pytest.raises(NotImplementedError, match="compile"):
        ptv3.load("ptv3", compile=True)
    with pytest.raises(ValueError, match="no head 'nyu40'"):
        ptv3.load("ptv3", head="nyu40", device="cpu")
    with pytest.raises(FileNotFoundError):
        ptv3.load("/no/such/checkpoint.pth")


def test_set_backend():
    ptv3.set_backend(attention="sdpa", conv="torch", layer_norm="torch")
    from point_transformer_v3.ops.attention import get_attention_backend
    from point_transformer_v3.ops.norm import get_layer_norm_backend
    from point_transformer_v3.ops.sparse_conv import get_backend

    assert (get_attention_backend(), get_backend(), get_layer_norm_backend()) == (
        "sdpa",
        "torch",
        "torch",
    )
    with pytest.raises(ValueError, match="unknown attention backend"):
        ptv3.set_backend(attention="flash3")
    ptv3.set_backend(attention="auto", conv="auto", layer_norm="auto")


def test_presets_are_complete_configs():
    from point_transformer_v3.model import Config

    fields = set(Config.__dataclass_fields__)
    for name, preset in ptv3.PRESETS.items():
        assert set(preset.config) == fields, name
        Config.from_dict(preset.config)
        if preset.weights is not None:
            assert len(preset.weights.sha256) == 64 and preset.weights.revision
    licenses = {n: p.weights.license for n, p in ptv3.PRESETS.items() if p.weights is not None}
    assert licenses.pop("ptv3_scannet") == "MIT"
    assert licenses.pop("sonata") == licenses.pop("sonata_small") == "CC-BY-NC-4.0"  # Meta
    assert set(licenses.values()) == {"CC-BY-NC-4.0 + research permission"}  # Pointcept
    assert (
        ptv3.PRESETS["utonia"].prepare.scale == 0.5 and ptv3.PRESETS["utonia"].config["rope_base"]
    )


def test_warmup_runs_on_cpu(monkeypatch):
    model = ptv3.load("ptv3", device="cpu")
    model.train()
    calls = []
    embed = model.embed
    monkeypatch.setattr(model, "embed", lambda *a, **k: calls.append(len(a[0])) or embed(*a, **k))
    assert ptv3.warmup(model, sizes=(300, 301), points=700) > 0
    assert calls == [700]  # prepare, forward, upcast and map back on a raw cloud
    assert model.training  # restored
    assert torch.is_grad_enabled()
