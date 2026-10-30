"""Model-level CUDA checks (marker ``gpu``; run under the GPU lock): every precision against the
CPU fp32 reference (Utonia's RoPE and head_dim 18 included), seeded init equal on CPU and CUDA,
warm-up. Kernel-level CUDA tests are in tests/ops (Triton conv, attention, LayerNorm against torch
on tiny inputs)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import pointtransformerv3 as ptv3
from tests.test_model import cloud

pytestmark = pytest.mark.gpu


def test_seeded_init_equal_on_cpu_and_cuda():
    torch.manual_seed(4)
    cpu = ptv3.PointTransformerV3.from_preset("sonata_small")
    torch.manual_seed(4)
    with torch.device("cuda"):
        cuda = ptv3.PointTransformerV3.from_preset("sonata_small")
    a, b = cpu.state_dict(), cuda.state_dict()
    assert all(torch.equal(a[k], b[k].cpu()) for k in a)


@pytest.mark.parametrize("preset", ["sonata_small", "ptv3", "utonia_tiny"])
@pytest.mark.parametrize(
    ("precision", "min_cos"), [("fp32", 0.99999), ("fp16", 0.999), ("bf16", 0.99)]
)
def test_precisions_against_cpu_fp32(preset, precision, min_cos):
    torch.manual_seed(0)
    model = ptv3.PointTransformerV3.from_preset(preset).eval()
    model.precision = "fp32"
    channels = model.cfg.in_channels
    point = cloud(3000, extent=2.0, channels=channels)
    with torch.inference_mode():
        ref = ptv3.upcast(model(point)).feat
        model.cuda().precision = precision
        out = ptv3.upcast(
            model({k: v.cuda() if torch.is_tensor(v) else v for k, v in point.items()})
        )
    assert out.feat.dtype == (
        torch.float32
        if precision == "fp32"
        else {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    )
    cos = F.cosine_similarity(out.feat.float().cpu(), ref, dim=1)
    assert cos.min() > min_cos, cos.min()


def test_default_precision_is_fp16_in_eval_and_bf16_in_training():
    model = ptv3.load("ptv3", device="cuda", warmup=False)
    point = {k: v.cuda() if torch.is_tensor(v) else v for k, v in cloud(1000).items()}
    with torch.inference_mode():
        assert model(point).feat.dtype == torch.float16
    model.train()
    out = model(point)
    assert out.feat.dtype == torch.bfloat16
    out.feat.float().square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_warmup_on_cuda():
    model = ptv3.load("ptv3", device="cuda", warmup=False)
    assert ptv3.warmup(model) > 0
