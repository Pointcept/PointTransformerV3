# Adapted from Pointcept2 tests/ops/test_attention.py @ 771a949.
"""varlen_attention: exact SDPA fallback, varlen_attn (bf16) vs fp32 reference, dispatch, checks."""

from __future__ import annotations

import re
import warnings
from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from point_transformer_v3.ops import attention
from point_transformer_v3.ops.attention import (
    get_attention_backend,
    set_attention_backend,
    varlen_attention,
)
from tests.conftest import DEVICES

# (query lengths, key lengths) per sequence: uneven, Tq != Tk (overlap tails), dense, single.
LAYOUTS = {
    "uneven": ([16, 16, 5, 9, 16, 1], [16, 16, 16, 9, 16, 1]),
    "dense": ([8, 8, 8], [8, 8, 8]),
    "single": ([1], [1]),
    "q_dense_k_padded": ([8, 8], [8, 3]),
}


@pytest.fixture
def varlen_calls(monkeypatch):
    """Record the inputs' shapes of every varlen_attn call (``attention._varlen`` imports it from
    torch's module on each call, so the spy goes there)."""
    from torch.nn.attention import varlen

    calls: list[torch.Size] = []
    real = varlen.varlen_attn

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(varlen, "varlen_attn", spy)
    return calls


@pytest.fixture
def backend():
    """Restore the global backend after the test."""
    saved = get_attention_backend()
    yield set_attention_backend
    set_attention_backend(saved)  # pyright: ignore[reportArgumentType]


def cu_of(lens: list[int], device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.tensor([0, *accumulate(lens)], dtype=torch.int32, device=device)


def make_qkv(q_lens, k_lens, heads, dim, *, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(sum(q_lens), heads, dim, generator=g)
    k = torch.randn(sum(k_lens), heads, dim, generator=g)
    v = torch.randn(sum(k_lens), heads, dim, generator=g)
    return [t.to(device, dtype).requires_grad_() for t in (q, k, v)]


def naive_attention(q, k, v, q_lens, k_lens, scale):
    """Per-sequence softmax(q k^T * scale) v in float64."""
    out = []
    for qs, ks, vs in zip(
        q.double().split(q_lens), k.double().split(k_lens), v.double().split(k_lens), strict=True
    ):
        attn = (qs.transpose(0, 1) @ ks.permute(1, 2, 0) * scale).softmax(-1)  # (H, Lq, Lk)
        out.append((attn @ vs.transpose(0, 1)).transpose(0, 1))
    return torch.cat(out)


def run(q, k, v, q_lens, k_lens, scale):
    cu_q, cu_k = cu_of(q_lens, q.device), cu_of(k_lens, q.device)
    return varlen_attention(q, k, v, cu_q, cu_k, max(q_lens), max(k_lens), scale=scale)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("device", DEVICES)
def test_sdpa_fallback_is_exact_in_fp32(layout, device, backend):
    backend("sdpa")
    q_lens, k_lens = LAYOUTS[layout]
    q, k, v = make_qkv(q_lens, k_lens, 3, 16, device=device)
    out = run(q, k, v, q_lens, k_lens, scale=0.3)
    ref_inputs = [t.detach().double().requires_grad_() for t in (q, k, v)]
    ref = naive_attention(*ref_inputs, q_lens, k_lens, scale=0.3)
    assert out.dtype == torch.float32 and out.shape == q.shape
    torch.testing.assert_close(out.double(), ref, atol=1e-5, rtol=1e-5)

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.double())
    for t, r in zip((q, k, v), ref_inputs, strict=True):
        assert t.grad is not None and r.grad is not None
        torch.testing.assert_close(t.grad.double(), r.grad, atol=1e-5, rtol=1e-5)


def test_auto_uses_sdpa_on_cpu_and_cpu_autocast(backend):
    backend("auto")
    q_lens, k_lens = LAYOUTS["uneven"]
    q, k, v = make_qkv(q_lens, k_lens, 2, 12)
    ref = naive_attention(q, k, v, q_lens, k_lens, scale=0.25)
    out = run(q, k, v, q_lens, k_lens, 0.25)
    torch.testing.assert_close(out.double(), ref, atol=1e-5, rtol=1e-5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = run(q, k, v, q_lens, k_lens, 0.25)
    assert out.dtype == torch.bfloat16
    assert cosine(out, ref) > 0.999


def test_empty_input():
    q = torch.zeros(0, 2, 8)
    cu = torch.zeros(1, dtype=torch.int32)
    out = varlen_attention(q, q, q, cu, cu, 0, 0, scale=1.0)
    assert out.shape == (0, 2, 8)


def test_input_checks(monkeypatch):
    q_lens = k_lens = [4, 4]
    q, k, v = make_qkv(q_lens, k_lens, 2, 8)
    cu = cu_of(q_lens)
    with pytest.raises(TypeError, match="int32"):
        varlen_attention(q, k, v, cu.long(), cu, 4, 4, scale=1.0)
    with pytest.raises(TypeError, match="host int"):
        varlen_attention(q, k, v, cu, cu, torch.tensor(4), 4, scale=1.0)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="same H and D"):
        varlen_attention(q, k[:, :1], v[:, :1], cu, cu, 4, 4, scale=1.0)
    with pytest.raises(TypeError, match="dtype"):
        varlen_attention(q, k.double(), v.double(), cu, cu, 4, 4, scale=1.0)
    with pytest.raises(ValueError, match="same length"):
        varlen_attention(q, k, v, cu, cu[:2], 4, 4, scale=1.0)
    with pytest.raises(ValueError, match="no sequences"):
        varlen_attention(q, k, v, cu[:1], cu[:1], 4, 4, scale=1.0)
    monkeypatch.setenv("PTV3_DEBUG", "1")
    with pytest.raises(ValueError, match="max_q=3"):
        varlen_attention(q, k, v, cu, cu, 3, 4, scale=1.0)
    with pytest.raises(ValueError, match="from 0 to 8"):
        varlen_attention(q, k, v, cu, cu_of([4, 3]), 4, 4, scale=1.0)
    with pytest.raises(ValueError, match="no keys"):
        varlen_attention(q, k[:4], v[:4], cu, cu_of([4, 0]), 4, 4, scale=1.0)


def test_backend_setting(monkeypatch, backend):
    with pytest.raises(ValueError, match="unknown attention backend"):
        backend("flash")  # pyright: ignore[reportArgumentType]
    monkeypatch.setattr(attention, "_backend", None)
    monkeypatch.setenv("PTV3_ATTENTION_BACKEND", "sdpa")
    assert get_attention_backend() == "sdpa"
    backend("varlen")
    assert get_attention_backend() == "varlen"
    q, k, v = make_qkv([4], [4], 1, 8)
    with pytest.raises(ValueError, match="needs CUDA"):
        run(q, k, v, [4], [4], 1.0)


def test_capability_check_is_per_device(monkeypatch):
    """D22: a mixed node (cuda:1 is sm_75) resolves each GPU on its own, queried once per index."""
    queried: list[int | None] = []

    def capability(i=None):
        queried.append(i)
        return (7, 5) if i == 1 else (8, 6)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=None: "Some GPU")
    monkeypatch.setattr(attention, "_capability_problems", {})
    old = attention._capability_problem(torch.device("cuda", 1))
    assert "compute capability 8.0+" in old and "cuda:1 (Some GPU) is sm_75" in old
    assert attention._capability_problem(torch.device("cuda", 0)) == ""
    assert attention._capability_problem(torch.device("cuda", 1)) == old
    assert queried == [1, 0]


# ---- CUDA


@pytest.mark.gpu
@pytest.mark.parametrize(("heads", "dim"), [(3, 16), (4, 16), (2, 12), (1, 8), (2, 64), (1, 256)])
@pytest.mark.parametrize("layout", ["uneven", "dense"])
def test_varlen_bf16_matches_fp32_sdpa(heads, dim, layout, backend, varlen_calls):
    q_lens, k_lens = LAYOUTS[layout]
    scale = dim**-0.5
    q, k, v = make_qkv(q_lens, k_lens, heads, dim, device="cuda", dtype=torch.bfloat16, seed=dim)
    backend("auto")
    out = run(q, k, v, q_lens, k_lens, scale)
    assert varlen_calls and varlen_calls[0][-1] % 8 == 0  # varlen ran (head_dim padded to 8k)
    assert out.dtype == torch.bfloat16 and out.shape == q.shape

    backend("sdpa")
    ref_inputs = [t.detach().float().requires_grad_() for t in (q, k, v)]
    ref = run(*ref_inputs, q_lens, k_lens, scale)
    assert cosine(out, ref) >= 0.999
    grad = torch.randn_like(ref)
    out.backward(grad.bfloat16())
    ref.backward(grad)
    for t, r in zip((q, k, v), ref_inputs, strict=True):
        assert t.grad is not None and r.grad is not None
        assert cosine(t.grad, r.grad) >= 0.999


@pytest.mark.gpu
def test_cuda_autocast_takes_varlen(backend, varlen_calls):
    backend("auto")
    q_lens, k_lens = LAYOUTS["uneven"]
    q, k, v = make_qkv(q_lens, k_lens, 2, 16, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = run(q, k, v, q_lens, k_lens, 0.25)
    assert varlen_calls and out.dtype == torch.bfloat16
    assert cosine(out, naive_attention(q, k, v, q_lens, k_lens, 0.25)) >= 0.999


@pytest.mark.gpu
def test_cuda_fallback_warns_once(backend, monkeypatch):
    monkeypatch.setattr(attention, "_noted", set())
    backend("auto")
    q, k, v = make_qkv([4], [4], 1, 8, device="cuda")
    with pytest.warns(UserWarning, match="falls back to padded SDPA"):
        run(q, k, v, [4], [4], 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(q, k, v, [4], [4], 1.0)
    qb, kb, vb = make_qkv([4], [4], 1, 264, device="cuda", dtype=torch.bfloat16)
    with pytest.warns(UserWarning, match="head_dim <= 256"):
        out = run(qb, kb, vb, [4], [4], 1.0)
    assert out.shape == (4, 1, 264)


@pytest.mark.gpu
def test_auto_falls_back_below_sm80(backend, monkeypatch, varlen_calls):
    """An sm_75 GPU (T4 / RTX 20xx, or the old card of a mixed node; simulated as in
    test_sparse_conv): ``auto`` takes SDPA with a warning naming the device instead of calling
    varlen_attn, whose FlashAttention kernel refuses GPUs below sm_80; forcing varlen raises."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: (7, 5))
    monkeypatch.setattr(attention, "_capability_problems", {})
    monkeypatch.setattr(attention, "_noted", set())
    backend("auto")
    q_lens, k_lens = LAYOUTS["uneven"]
    q, k, v = make_qkv(q_lens, k_lens, 2, 16, device="cuda")
    with pytest.warns(UserWarning) as caught, torch.autocast("cuda", dtype=torch.float16):
        out = run(q, k, v, q_lens, k_lens, 0.25)
    (message,) = [str(w.message) for w in caught]
    assert "falls back to padded SDPA" in message and "compute capability 8.0+" in message
    assert re.search(r"cuda:\d \(.*\) is sm_75", message) and "autocast" not in message
    assert not varlen_calls and out.dtype == torch.float16
    assert cosine(out, naive_attention(q, k, v, q_lens, k_lens, 0.25)) >= 0.999
    backend("varlen")
    with pytest.raises(ValueError, match=r"'varlen' needs compute capability 8\.0\+"):
        run(
            q.detach().bfloat16(), k.detach().bfloat16(), v.detach().bfloat16(), q_lens, k_lens, 1.0
        )


@pytest.mark.gpu
@pytest.mark.parametrize("capability", [None, (7, 5)], ids=["native", "sm75"])
def test_fullgraph_capture_of_first_call(capability, backend, monkeypatch):
    """The choice (capability check included) resolves outside the graph and the once-only log is
    not traced: a fullgraph capture whose first attention call is traced compiles once and matches
    eager, on this GPU (varlen) and on a simulated sm_75 (sdpa)."""
    if capability is not None:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: capability)
    monkeypatch.setattr(attention, "_capability_problems", {})
    monkeypatch.setattr(attention, "_noted", set())
    backend("auto")
    q_lens, k_lens = LAYOUTS["uneven"]
    q, k, v = (t.detach() for t in make_qkv(q_lens, k_lens, 2, 16, device="cuda"))
    cu_q, cu_k = cu_of(q_lens, "cuda"), cu_of(k_lens, "cuda")

    def attend(q, k, v):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return varlen_attention(q, k, v, cu_q, cu_k, 16, 16, scale=0.25)

    torch._dynamo.reset()
    compiled = torch.compile(attend, fullgraph=True, backend="eager")
    with torch._dynamo.config.patch(error_on_recompile=True), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = [compiled(q, k, v), compiled(q * 2, k, v)]
        want = [attend(q, k, v), attend(q * 2, k, v)]
    torch._dynamo.reset()
    for g, w in zip(got, want, strict=True):
        torch.testing.assert_close(g, w, rtol=0, atol=0)
    assert attention._capability_problems.keys() == {q.device.index}


@pytest.mark.gpu
@pytest.mark.parametrize(("name", "dtype"), [("varlen", torch.bfloat16), ("sdpa", torch.float32)])
def test_no_host_sync(name, dtype, backend):
    backend(name)
    q_lens, k_lens = LAYOUTS["uneven"]
    q, k, v = make_qkv(q_lens, k_lens, 3, 12, device="cuda", dtype=dtype)
    cu_q, cu_k = cu_of(q_lens, "cuda"), cu_of(k_lens, "cuda")
    grad = torch.randn_like(q)
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Synchronization debug mode")
        torch.cuda.set_sync_debug_mode("error")
    try:
        out = varlen_attention(q, k, v, cu_q, cu_k, 16, 16, scale=0.25)
        out.backward(grad)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert q.grad is not None and torch.isfinite(q.grad).all()
