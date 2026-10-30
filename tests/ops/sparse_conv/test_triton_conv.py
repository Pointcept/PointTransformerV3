# Adapted from Pointcept2 tests/ops/sparse_conv/test_triton_conv.py @ 771a949.
"""Triton sparse conv backend (ops/sparse_conv/triton_impl.py).

CPU: the kernels in the Triton interpreter (subprocess), the toolchain check, backend selection,
launch-config lookup and the kernel-map checks. CUDA (``gpu``, skipped with the missing toolchain
piece as the reason): parity against fp64 (synthetic, and real ScanNet scenes with ``data``),
bf16/fp16 accuracy on every pinned shape, determinism, sync-freedom, torch.compile, packing
invariance (C15) and row-permutation equivariance (C16).

GPU safety: no CUDA launch here gets a kernel-map entry outside its allocation. Corrupt maps on
CUDA use entries that still address sentinel-padded buffers (``_triton_interpret.padded``); far
out-of-range entries go to the interpreter or to ``PTV3_DEBUG=1``, which raises before launching.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from pointtransformerv3.layers.sparse_conv import SubMConv3d
from pointtransformerv3.ops import _triton
from pointtransformerv3.ops.neighbors import build_kmap, check_kmap
from pointtransformerv3.ops.sparse_conv import (
    functional,
    get_backend,
    set_backend,
    subm_conv3d,
    torch_impl,
    triton_impl,
)
from tests.conftest import DEVICES, cell_grid, edge_case_pack, regroup, scannet_pack

REPO = Path(__file__).resolve().parents[3]
PINNED_GPU = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition|sm_120|188"
# (k, C_in, C_out) of every pinned config: what training runs on the pinned GPU
PINNED_SHAPES = sorted(
    {
        ({27: 3, 125: 5}[int(v)], int(c_in), int(c_out))
        for key in triton_impl._pinned_table()[PINNED_GPU]
        for _, c_in, c_out, v, _ in [key.split(":")]
    }
)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max abs error relative to the reference's max magnitude."""
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp(min=1e-30)).item()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.double().flatten(), b.double().flatten(), 0
    ).item()


@pytest.fixture(autouse=True)
def restore_state() -> Iterator[None]:
    saved = functional._backend
    yield
    functional._backend = saved
    functional._noted.clear()
    triton_impl.toolchain_problem.cache_clear()


@pytest.fixture
def triton_cuda() -> None:
    """Skips unless the triton backend can run here; the reason names the missing piece."""
    problem = triton_impl.toolchain_problem()
    if problem is not None:
        pytest.skip(f"triton backend unavailable: {problem}")
    triton_impl.require()


@pytest.fixture
def no_compile_autotune() -> Iterator[None]:
    """CLAUDE.md GPU safety: no compile-time autotuning of data-dependent indexing. Plain
    assignments, undone by ``conftest.restore_compiler_configs``: undoing a monkeypatch of a torch
    config deletes (hides) the entry (see conftest.py)."""
    import torch._inductor.config as inductor_config

    inductor_config.triton.autotune_at_compile_time = False
    inductor_config.max_autotune = False
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def make_grid(synthetic_pack, seq_lens, extent, seed=0, device="cuda", with_edges=False):
    """A ``PointGrid`` of a synthetic pack, optionally after the edge cases of ``edge_case_pack``
    (isolated points, a single point, an empty sample, a corner at 2**16 - 5)."""
    pack = synthetic_pack(seq_lens, extent=extent, seed=seed)
    grid_coord, batch = pack.grid_coord, pack.batch
    if with_edges:
        edge, edge_batch = edge_case_pack()
        grid_coord = torch.cat([edge, grid_coord])
        batch = torch.cat([edge_batch, batch + int(edge_batch.max()) + 1])
    return cell_grid(*regroup(grid_coord.to(device), batch.to(device)))


def fp64_reference(x, weight, bias, kmap, grad):
    """(out, grad_x, grad_weight, grad_bias) in fp64 through the torch backend (checked against a
    dense conv3d at 1e-12 in test_sparse_conv.py)."""
    x = x.detach().double().requires_grad_()
    w = weight.detach().double().requires_grad_()
    b = bias.detach().double().requires_grad_()
    out = torch_impl.subm_conv3d(x, w, b, kmap)
    return (out.detach(), *torch.autograd.grad(out, (x, w, b), grad.double()))


def triton_step(x, weight, bias, kmap, grad, autocast_dtype=None):
    """(out, grad_x, grad_weight, grad_bias) of one triton fwd+bwd."""
    x = x.detach().clone().requires_grad_()
    with torch.autocast(
        "cuda", dtype=autocast_dtype or torch.bfloat16, enabled=bool(autocast_dtype)
    ):
        out = triton_impl.subm_conv3d(x, weight, bias, kmap)
    grads = torch.autograd.grad(out, (x, weight, bias), grad.to(out.dtype))
    return (out.detach(), *grads)


def random_params(c_in, c_out, k, seed=0, device="cuda"):
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(c_out, k, k, k, c_in, generator=gen) / (c_in * k**3) ** 0.5
    bias = torch.randn(c_out, generator=gen)
    return weight.to(device).requires_grad_(), bias.to(device).requires_grad_()


# ---------------------------------------------------------------- CPU: interpreter


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton is not installed")
def test_kernels_in_interpreter():
    """Every kernel on tiny inputs in the Triton interpreter (CPU) against fp64, incl. corrupt
    kernel maps. TRITON_INTERPRET must be set before triton is imported: a subprocess."""
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "tests.ops.sparse_conv._triton_interpret"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL OK" in result.stdout


# ---------------------------------------------------------------- CPU: toolchain and selection


def fake_include(tmp_path: Path, minor: int = sys.version_info.minor) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "Python.h").write_text("")
    (tmp_path / "patchlevel.h").write_text(
        f"#define PY_MAJOR_VERSION {sys.version_info.major}\n#define PY_MINOR_VERSION {minor}\n"
    )
    return tmp_path


def test_toolchain_problem_names_the_missing_piece(monkeypatch, tmp_path):
    def problem() -> str | None:
        triton_impl.toolchain_problem.cache_clear()
        return triton_impl.toolchain_problem()

    real_find_spec = importlib.util.find_spec

    def triton_installed(installed: bool) -> None:
        def find_spec(name, *args):
            if name == "triton":
                return object() if installed else None
            return real_find_spec(name, *args)

        monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    triton_installed(True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=None: "Some GPU")
    monkeypatch.setattr(_triton, "_probe_driver", lambda: None)
    monkeypatch.setenv("CC", "no-such-compiler-pc2")
    assert problem() is None  # the cached driver helper loads: no compiler or headers needed

    triton_installed(False)
    assert "triton is not installed" in (problem() or "")
    triton_installed(True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert "CUDA is not available" in (problem() or "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: (7, 5) if i else (8, 0))
    text = problem() or ""
    assert "cuda:1 (Some GPU) is sm_75" in text and "compute capability 8.0+" in text
    # per device (D22: auto takes triton on the sm_80+ GPUs of a mixed node)
    assert triton_impl.toolchain_problem(torch.device("cuda", 0)) is None
    assert triton_impl.available(torch.device("cuda", 0))
    assert "cuda:1 (Some GPU) is sm_75" in (triton_impl.toolchain_problem(torch.device("cuda", 1)))
    assert not triton_impl.available(torch.device("cuda", 1)) and not triton_impl.available()
    assert "run on CUDA devices, got cpu" in (triton_impl.toolchain_problem(torch.device("cpu")))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=None: (8, 6))
    assert problem() is None

    # the helper does not build: Triton's own compiler and header lookups name the likely cause
    monkeypatch.setattr(_triton, "_probe_driver", lambda: "CalledProcessError: cc (test)")
    good = fake_include(tmp_path / "good")
    monkeypatch.setattr(_triton, "_python_include_dir", lambda: str(tmp_path / "none"))
    monkeypatch.delenv("CPATH", raising=False)
    monkeypatch.setenv("C_INCLUDE_PATH", str(fake_include(tmp_path / "old", minor=0)))
    monkeypatch.setenv("CC", sys.executable)  # any existing executable passes the lookup
    text = problem() or ""
    assert "cuda_utils" in text and "CalledProcessError: cc (test)" in text
    assert "Python.h for Python 3." in text
    monkeypatch.delenv("C_INCLUDE_PATH")
    monkeypatch.setenv("CPATH", f"{tmp_path / 'none'}{os.pathsep}{good}")
    text = problem() or ""  # CPATH counts too: no header hint, a generic one
    assert "Python.h for Python" not in text and "can build a shared library" in text
    monkeypatch.setenv("CC", "no-such-compiler-pc2")
    assert "CC='no-such-compiler-pc2' is set but no such compiler" in (problem() or "")
    monkeypatch.delenv("CC")
    monkeypatch.setattr(_triton.shutil, "which", lambda name: None)
    assert "no C compiler found" in (problem() or "")


def test_set_backend_triton_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(triton_impl, "_ready", False)
    monkeypatch.setattr(
        triton_impl, "toolchain_problem", lambda device=None, **_: "Python.h is missing (test)"
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    functional._backend = "torch"
    with pytest.raises(
        RuntimeError,
        match=r"^set_backend\('triton'\): the triton sparse conv backend cannot run here: "
        r"Python\.h is missing \(test\)\. Use backend 'auto' or 'torch' instead\.$",
    ):
        set_backend("triton")
    assert get_backend() == "torch"  # unchanged
    with pytest.raises(RuntimeError, match=r"^backend='triton': .*Python\.h is missing"):
        functional._check_backend("triton", "backend")

    # an exported env var that cannot work: the message names it and only runnable backends
    functional._backend = None
    monkeypatch.setenv("PTV3_CONV_BACKEND", "triton")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError) as info:
        get_backend()
    message = str(info.value)
    assert message.startswith("PTV3_CONV_BACKEND=triton: ") and "Python.h is missing" in message
    assert "Use backend 'auto' or 'torch' instead." in message and "'spconv'" not in message
    assert message.endswith("Unset PTV3_CONV_BACKEND or set it to one of those.")


@pytest.mark.parametrize("device", DEVICES)
def test_auto_picks_triton_only_on_cuda(synthetic_pack, monkeypatch, device):
    """The default (no env, no set_backend) is auto: triton on CUDA where it can run, never on the
    CPU."""
    calls = []

    def spy(*args):
        calls.append(args[0].device.type)
        return real(*args)

    real = triton_impl.subm_conv3d
    monkeypatch.setattr(triton_impl, "subm_conv3d", spy)
    monkeypatch.delenv("PTV3_CONV_BACKEND", raising=False)
    functional._backend = None
    assert get_backend() == "auto"
    grid = make_grid(synthetic_pack, [200, 100], extent=8, device=device)
    x = torch.randn(300, 16, device=device)
    for k in (3, 5):
        assert SubMConv3d(16, 8, k).to(device)(x, grid).shape == (300, 8)
    expected = ["cuda"] * 2 if device == "cuda" and triton_impl.available(x.device) else []
    assert calls == expected


# ---------------------------------------------------------------- CPU: launch configs


def test_pinned_table_is_well_formed():
    table = triton_impl._pinned_table()
    assert PINNED_GPU in table
    for gpu, configs in table.items():
        assert len(gpu.split("|")) == 3, gpu
        for key, cfg in configs.items():
            kind, c_in, c_out, offsets, dtype = key.split(":")
            assert kind in ("fwd", "dgrad", "wgrad") and dtype in ("bfloat16", "float16")
            assert int(c_in) > 0 and int(c_out) > 0 and int(offsets) in (27, 125)
            cls = triton_impl.WgradConfig if kind == "wgrad" else triton_impl.GemmConfig
            cls(**cfg)  # all fields present, no unknown ones
            for name in ("B1", "B2", "BK"):
                assert cfg[name] & (cfg[name] - 1) == 0, (key, name)  # tl.arange: powers of 2
    data = json.loads((Path(triton_impl.__file__).with_name("triton_configs.json")).read_text())
    assert set(data["gpus"][PINNED_GPU]) == {"measured", "configs"}


def test_launch_config_lookup(monkeypatch, caplog):
    device = torch.device("cuda", 0)
    gpu = triton_impl._Gpu(PINNED_GPU, sm_count=188, smem=101376)
    monkeypatch.setattr(triton_impl, "_gpu", lambda d: gpu)
    triton_impl._generic_noted.clear()
    bf16 = torch.bfloat16
    assert triton_impl.gemm_config("fwd", 32, 32, 27, bf16, device) == triton_impl.GemmConfig(
        B1=128, B2=32, BK=32, num_warps=4, num_stages=3
    )
    wgrad = triton_impl.wgrad_config(32, 32, 27, bf16, device)
    assert (wgrad.SPLITK, wgrad.B2, wgrad.BV, wgrad.BCi) == (107, 128, 4, 32)
    stem = triton_impl.wgrad_config(6, 32, 125, bf16, device)
    assert (stem.BV, stem.BCi) == (16, 8)

    with caplog.at_level(logging.INFO, logger=triton_impl.__name__):
        for _ in range(2):  # unknown shape: generic config, logged once
            cfg = triton_impl.gemm_config("dgrad", 96, 40, 27, bf16, device)
    assert cfg == triton_impl.GemmConfig(B1=128, B2=64, BK=64, num_warps=4, num_stages=3)
    assert len(caplog.records) == 1 and "dgrad:96:40:27:bfloat16" in caplog.records[0].getMessage()

    # an unknown GPU with little shared memory: generic configs that fit it
    small = triton_impl._Gpu("Some GPU|sm_75|40", sm_count=40, smem=48 * 1024)
    monkeypatch.setattr(triton_impl, "_gpu", lambda d: small)
    for dtype, elem in ((bf16, 2), (torch.float32, 4)):
        for kind in ("fwd", "dgrad"):
            g = triton_impl.gemm_config(kind, 64, 64, 27, dtype, device)  # type: ignore[arg-type]
            assert (g.B1 + g.B2) * g.BK * elem * g.num_stages <= small.smem
        w = triton_impl.wgrad_config(64, 64, 27, dtype, device)
        assert (w.B1 + w.B2) * w.BK * elem * w.num_stages <= small.smem
        assert w.BV * w.BCi == w.B2 and 1 <= w.SPLITK <= 256
        assert w.num_warps == (8 if dtype == torch.float32 else 4)  # fp32 at 4 warps spills


# ---------------------------------------------------------------- kernel-map checks


@pytest.mark.parametrize("device", DEVICES)
def test_check_kmap_rejects_corrupt_maps(synthetic_pack, device):
    pack = synthetic_pack([150, 60], extent=6, seed=3, device=device)
    kmap = build_kmap(pack.grid_coord, pack.batch, 3)
    check_kmap(kmap)
    check_kmap(kmap[:0])
    with pytest.raises(TypeError, match="int32"):
        check_kmap(kmap.long())
    p, k = (
        int(i)
        for i in (
            (kmap >= 0) & (kmap != torch.arange(210, device=device)[:, None].int())
        ).nonzero()[0]
    )
    one_sided = kmap.clone()
    one_sided[p, k] = -1  # its mirror entry still points back
    with pytest.raises(ValueError, match="not symmetric"):
        check_kmap(one_sided)
    for bad in (210, -2):
        out_of_range = kmap.clone()
        out_of_range[5, 0] = bad
        with pytest.raises(ValueError, match=r"must lie in \[-1, 210\)"):
            check_kmap(out_of_range)


@pytest.mark.parametrize("device", DEVICES)
def test_debug_mode_rejects_corrupt_kmap(synthetic_pack, monkeypatch, device):
    """Under PTV3_DEBUG=1 an asymmetric or out-of-range kmap raises before any launch. Without it
    the kernels treat an out-of-range entry as missing: on CUDA checked with an entry that still
    addresses the padded features, never one outside the allocation."""
    if device == "cuda" and not triton_impl.available():
        pytest.skip(f"triton backend unavailable: {triton_impl.toolchain_problem()}")
    from tests.ops.sparse_conv._triton_interpret import PAD, padded

    pack = synthetic_pack([120], extent=6, seed=4, device=device)
    kmap = build_kmap(pack.grid_coord, pack.batch, 3)
    weight, bias = random_params(8, 8, 3, device=device)
    x = torch.randn(120, 8, device=device)
    corrupt = kmap.clone()
    corrupt[kmap[:, 0] >= 0, 0] = -1  # drop one direction of every offset-0 pair
    monkeypatch.setenv("PTV3_DEBUG", "1")
    with pytest.raises(ValueError, match="not symmetric"):
        triton_impl.subm_conv3d(x, weight, bias, corrupt)
    for bad in (1 << 30, -(1 << 31), 120, -2):
        corrupt = kmap.clone()
        corrupt[0, 0] = bad
        with pytest.raises(ValueError, match="must lie in"):
            triton_impl.subm_conv3d(x, weight, bias, corrupt)
    if device == "cuda":
        triton_impl.subm_conv3d(x, weight, bias, kmap)  # a valid map passes
        monkeypatch.delenv("PTV3_DEBUG")
        corrupt, missing = kmap.clone(), kmap.clone()
        corrupt[0, 0], missing[0, 0] = 120 + PAD // 2, -1  # inside padded()'s trailing rows
        out = triton_impl.subm_conv3d(padded(x), weight, bias, corrupt)
        assert torch.equal(out, triton_impl.subm_conv3d(x, weight, bias, missing))


def test_input_validation():
    x = torch.randn(10, 4)
    weight, bias = random_params(4, 6, 3, device="cpu")
    kmap = torch.full((10, 27), -1, dtype=torch.int32)
    with pytest.raises(TypeError, match="int32"):
        triton_impl.subm_conv3d(x, weight, bias, kmap.long())
    with pytest.raises(TypeError, match=r"\(10, 27\)"):
        triton_impl.subm_conv3d(x, weight, bias, kmap[:, :8])
    with pytest.raises(TypeError, match="fp32, fp16 or bf16"):
        triton_impl.subm_conv3d(x.double(), weight, bias, kmap)
    with pytest.raises(ValueError, match=r"bias must be a float \(6,\) tensor"):
        triton_impl.subm_conv3d(x, weight, bias[:2], kmap)
    with pytest.raises(ValueError, match=r"bias must be a float \(6,\) tensor"):
        triton_impl.subm_conv3d(x, weight, bias.long(), kmap)
    with pytest.raises(ValueError, match="C_out, k, k, k, C_in"):
        triton_impl.subm_conv3d(x, weight.flatten(1, 3), bias, kmap)


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton is not installed")
def test_op_checks_bias():
    """A direct ``torch.ops.pointtransformerv3.subm_conv3d`` call is checked too (before any launch)."""
    triton_impl._load_ops()
    x, weight = torch.randn(10, 4), torch.randn(6, 27, 4)
    kmap = torch.full((10, 27), -1, dtype=torch.int32)
    for bias in (torch.zeros(7), torch.zeros(6, dtype=torch.int64), torch.zeros(1)):
        with pytest.raises(ValueError, match=r"bias must be a float \(6,\) tensor on cpu"):
            torch.ops.pointtransformerv3.subm_conv3d(x, weight, bias, kmap)


# ---------------------------------------------------------------- CUDA


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_tiny_cuda_against_fp64(triton_cuda, dtype):
    """The interpreter cases on CUDA, fp32 and half features (fp32 weights): strided biases and
    corrupt maps whose bad entries stay inside the padded buffers (``run`` refuses others)."""
    from tests.ops.sparse_conv._triton_interpret import BIAS_CASES, CASES, CORRUPT, run

    results = [run(c, None, "cuda", dtype) for c in CASES]
    results += [
        run(c, None, "cuda", dtype, layout) for c, layout in BIAS_CASES if layout == "strided"
    ]
    results += [run(c, "near", "cuda", dtype) for c in CORRUPT]
    assert all(results)


PARITY = [  # (k, C_in, C_out, sample sizes, extent): the PTv3 shapes, C >= 256 included
    (3, 16, 24, [3000, 1, 0, 2500], 20),
    (3, 32, 32, [4000, 2000], 22),
    (5, 6, 32, [3000, 1500], 20),
    (3, 48, 48, [2500], 18),
    (3, 256, 256, [2000, 1200], 16),
    (3, 512, 512, [1500], 14),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("k", "c_in", "c_out", "seq_lens", "extent"), PARITY)
def test_fp32_parity_against_fp64(triton_cuda, synthetic_pack, k, c_in, c_out, seq_lens, extent):
    grid = make_grid(synthetic_pack, seq_lens, extent, seed=c_in, with_edges=True)
    kmap = grid.kmap(k)
    weight, bias = random_params(c_in, c_out, k)
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(grid.num_rows, c_in, generator=gen).cuda()
    grad = torch.randn(grid.num_rows, c_out, generator=gen).cuda()
    got = triton_step(x, weight, bias, kmap, grad)
    expected = fp64_reference(x, weight, bias, kmap, grad)
    assert [t.dtype for t in got] == [torch.float32] * 4
    for name, a, b in zip(("out", "grad_x", "grad_w", "grad_b"), got, expected, strict=True):
        assert rel(a, b) < 1e-5, (name, rel(a, b))


@pytest.mark.data
@pytest.mark.gpu
def test_fp32_parity_scannet(triton_cuda, scannet_scene):
    """fp32 against fp64 on two real scenes (the pack of test_spconv_parity_fp32_scannet): the L0
    convs and the stem, and C256 / C512 on the stride-8 / stride-16 levels. fp32 uses the generic
    weight-gradient split, which does not grow with N, so real-scene row counts matter."""
    grid_coord, batch = scannet_pack(scannet_scene, [0, 10])
    levels = [cell_grid(*regroup(grid_coord.cuda(), batch.cuda()))]
    for _ in range(4):
        levels.append(levels[-1].downsample(2))
    for level, k, c_in, c_out in [(0, 3, 16, 24), (0, 3, 48, 48), (0, 5, 6, 32), (3, 3, 256, 256),
                                  (4, 3, 512, 512)]:  # fmt: skip
        grid = levels[level]
        kmap = grid.kmap(k)
        weight, bias = random_params(c_in, c_out, k, seed=k * 100 + c_in)
        gen = torch.Generator().manual_seed(c_in)
        x = torch.randn(grid.num_rows, c_in, generator=gen).cuda()
        grad = torch.randn(grid.num_rows, c_out, generator=gen).cuda()
        got = triton_step(x, weight, bias, kmap, grad)
        expected = fp64_reference(x, weight, bias, kmap, grad)
        for name, a, b in zip(("out", "grad_x", "grad_w", "grad_b"), got, expected, strict=True):
            assert rel(a, b) < 1e-5, (level, grid.num_rows, k, c_in, c_out, name, rel(a, b))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("k", "c_in", "c_out"), PINNED_SHAPES)
def test_half_autocast_accuracy(triton_cuda, synthetic_pack, dtype, k, c_in, c_out):
    """Every pinned shape in bf16 and fp16 against fp64 of the same rounded inputs (the kernels
    round x, weight and grad to the compute dtype and accumulate in fp32). Max error relative to
    the largest magnitude, so a wrong tile (e.g. the partial last 128-row tile) cannot hide."""
    grid = make_grid(synthetic_pack, [3000, 1999], 20, seed=k + c_in, with_edges=True)
    n = grid.num_rows
    assert n % 64, n  # a partial last row tile
    kmap = grid.kmap(k)
    weight, bias = random_params(c_in, c_out, k)
    gen = torch.Generator().manual_seed(c_in)
    x = torch.randn(n, c_in, generator=gen).cuda()
    grad = torch.randn(n, c_out, generator=gen).cuda()
    gpu = triton_impl._gpu(x.device)
    if gpu.key == PINNED_GPU:  # the pinned configs are the ones launched
        for kind in ("fwd", "dgrad", "wgrad"):
            assert triton_impl._pinned(gpu, kind, c_in, c_out, k**3, dtype) is not None, kind
    got = triton_step(x, weight, bias, kmap, grad, autocast_dtype=dtype)
    expected = fp64_reference(x.to(dtype), weight.to(dtype), bias, kmap, grad.to(dtype))
    assert [t.dtype for t in got] == [dtype, torch.float32, torch.float32, torch.float32]
    features = {torch.bfloat16: 1e-2, torch.float16: 2e-3}[dtype]  # one rounding of the output
    gates = (features, features, 1e-5, 1e-5)  # weight and bias gradients: fp32 sums
    for name, a, b, gate in zip(
        ("out", "grad_x", "grad_w", "grad_b"), got, expected, gates, strict=True
    ):
        assert rel(a, b) < gate, (name, rel(a, b))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("k", "c_in", "c_out"), [(3, 32, 32), (5, 6, 32)])
def test_deterministic_and_sync_free(triton_cuda, synthetic_pack, dtype, k, c_in, c_out):
    grid = make_grid(synthetic_pack, [5000, 3000], 24, seed=5)
    kmap = grid.kmap(k)  # geometry is built outside the conv
    weight, bias = random_params(c_in, c_out, k)
    x = torch.randn(grid.num_rows, c_in, device="cuda")
    grad = torch.randn(grid.num_rows, c_out, device="cuda")
    half = dtype if dtype != torch.float32 else None
    first = triton_step(x, weight, bias, kmap, grad, autocast_dtype=half)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        second = triton_step(x, weight, bias, kmap, grad, autocast_dtype=half)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


@pytest.mark.gpu
def test_compile_fullgraph_matches_eager(triton_cuda, synthetic_pack, no_compile_autotune):
    weight, bias = random_params(32, 32, 3)

    def conv(x, kmap):
        return triton_impl.subm_conv3d(x, weight, bias, kmap)

    compiled = torch.compile(conv, fullgraph=True, dynamic=True)
    for seed, seq_lens in ((0, [3000, 1000]), (1, [2500])):  # two N, one graph
        grid = make_grid(synthetic_pack, seq_lens, 20, seed=seed)
        kmap = grid.kmap(3)
        x = torch.randn(grid.num_rows, 32, device="cuda")
        with torch.no_grad():
            assert torch.equal(compiled(x, kmap), conv(x, kmap))
        grad = torch.randn(grid.num_rows, 32, device="cuda")
        results = []
        for fn in (conv, compiled):
            x_in = x.clone().requires_grad_()
            out = fn(x_in, kmap)
            results.append((out, *torch.autograd.grad(out, (x_in, weight, bias), grad)))
        for a, b in zip(*results, strict=True):
            assert rel(a, b) < 1e-6


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("k", "c_in", "c_out", "has_bias"), [(3, 32, 32, True), (5, 6, 32, False)])
def test_eager_function_is_the_op_bitwise(
    triton_cuda, synthetic_pack, monkeypatch, dtype, k, c_in, c_out, has_bias
):
    """I5: an eager call launches the kernels from ``SubmConv3dFunction`` (no custom-op dispatch);
    its output and every gradient equal the ``triton_op`` path's (what traced graphs run) bit for
    bit, and it stays sync-free."""
    grid = make_grid(synthetic_pack, [3000, 1200], 18, seed=2)
    kmap = grid.kmap(k)
    weight, bias = random_params(c_in, c_out, k)
    x = torch.randn(grid.num_rows, c_in, device="cuda")
    grad = torch.randn(grid.num_rows, c_out, device="cuda")
    half = dtype if dtype != torch.float32 else None

    def step() -> tuple[torch.Tensor, ...]:
        x_in = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=half or torch.bfloat16, enabled=half is not None):
            out = triton_impl.subm_conv3d(x_in, weight, bias if has_bias else None, kmap)
        node = type(out.grad_fn).__name__
        inputs = (x_in, weight, bias) if has_bias else (x_in, weight)
        return node, out.detach(), *torch.autograd.grad(out, inputs, grad.to(out.dtype))

    step()  # warm-up (first launches compile)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        eager = step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    monkeypatch.setattr(triton_impl, "tracing", lambda x: True)
    op = step()
    assert eager[0] == "SubmConv3dFunctionBackward" and op[0] != eager[0]
    for a, b in zip(eager[1:], op[1:], strict=True):
        assert a.dtype == b.dtype and torch.equal(a, b)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("c", [256, 512])
def test_wgrad_warps_keep_the_bits(triton_cuda, synthetic_pack, monkeypatch, dtype, c):
    """R4(b): the pinned C256 / C512 weight-gradient configs run at 4 warps (8 before); tile, BK
    and split fix the reduction order, so the gradient is the same bit for bit."""
    from pointtransformerv3.ops.sparse_conv import _triton_ops

    grid = make_grid(synthetic_pack, [2500, 900], 16, seed=c)
    kmap = grid.kmap(3)
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(grid.num_rows, c, generator=gen).to("cuda", dtype)
    grad = torch.randn(grid.num_rows, c, generator=gen).to("cuda", dtype)
    pinned = _triton_ops._wgrad(grad, x, kmap)
    config = triton_impl.wgrad_config(c, c, 27, dtype, x.device)
    eight = dataclasses.replace(config, num_warps=8)
    monkeypatch.setattr(_triton_ops, "wgrad_config", lambda *args: eight)
    assert torch.equal(_triton_ops._wgrad(grad, x, kmap), pinned)


@pytest.mark.gpu
def test_export_keeps_the_op(triton_cuda, synthetic_pack):
    """D15 after I5: non-strict torch.export (FakeTensors) and strict export (dynamo) both record
    ``pointtransformerv3.subm_conv3d``, and the exported program computes what eager does."""
    grid = make_grid(synthetic_pack, [1500, 700], 14, seed=4)
    kmap = grid.kmap(3)
    weight, bias = random_params(16, 16, 3)

    class Conv(torch.nn.Module):
        def forward(self, x: torch.Tensor, kmap: torch.Tensor) -> torch.Tensor:
            return triton_impl.subm_conv3d(x, weight.detach(), bias.detach(), kmap)

    x = torch.randn(grid.num_rows, 16, device="cuda")
    expected = Conv()(x, kmap)
    for strict in (False, True):
        exported = torch.export.export(Conv(), (x, kmap), strict=strict)
        targets = {  # the op sits in the autocast-off subgraph (wrap_with_autocast)
            str(n.target)
            for module in exported.graph_module.modules()
            if isinstance(module, torch.fx.GraphModule)
            for n in module.graph.nodes
            if n.op == "call_function"
        }
        assert "pointtransformerv3.subm_conv3d.default" in targets, (strict, targets)
        assert torch.equal(exported.module()(x, kmap), expected)


@pytest.mark.gpu
def test_module_through_backend_switch(triton_cuda, synthetic_pack, caplog):
    grid = make_grid(synthetic_pack, [2000, 700], 16, seed=6)
    conv = SubMConv3d(16, 24, 3).cuda()
    x = torch.randn(grid.num_rows, 16, device="cuda")
    set_backend("triton")
    with caplog.at_level(logging.INFO, logger=functional.__name__):
        out = conv(x, grid)
        conv(x, grid)
    assert [r.getMessage() for r in caplog.records] == [
        "subm_conv3d k=3 C_in=16: triton backend (triton: forced)"
    ]
    assert torch.equal(out, triton_impl.subm_conv3d(x, conv.weight, conv.bias, grid.kmap(3)))
    assert rel(out, subm_conv3d(x, conv.weight, conv.bias, grid, backend="torch")) < 1e-5
    empty = cell_grid(torch.zeros(0, 3, dtype=torch.int32, device="cuda"), [0])
    assert conv(x[:0], empty).shape == (0, 24)  # empty input: torch path, as for spconv
    with pytest.raises(ValueError, match="triton backend needs fp32/fp16/bf16 CUDA"):
        conv.cpu()(x.cpu(), make_grid(synthetic_pack, [2000, 700], 16, seed=6, device="cpu"))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packing_invariance_and_row_permutation(triton_cuda, synthetic_pack, dtype):
    """C15: forward(pack[A, B]) == cat(forward(A), forward(B)); C16: rows permuted within a sample
    permute the outputs. Every row is reduced on its own, so both hold bit for bit."""
    pack = synthetic_pack([1500, 900], extent=14, seed=7, device="cuda")
    a, b = pack.seq_lens
    torch.manual_seed(0)
    conv = SubMConv3d(32, 32, 3).cuda().eval()
    x = torch.randn(a + b, 32, device="cuda")
    set_backend("triton")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        both = conv(x, cell_grid(pack.grid_coord, pack.seq_lens))
        first = conv(x[:a], cell_grid(pack.grid_coord[:a], [a]))
        second = conv(x[a:], cell_grid(pack.grid_coord[a:], [b]))
        perm = torch.randperm(a, generator=torch.Generator().manual_seed(1)).cuda()
        permuted = conv(x[:a][perm], cell_grid(pack.grid_coord[:a][perm], [a]))
    assert torch.equal(both, torch.cat([first, second]))
    assert torch.equal(permuted, first[perm])
