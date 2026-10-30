# Adapted from Pointcept2 tests/ops/test_tracing.py @ 771a949.
"""ops/_tracing.py: ``assume_constant_result`` sets what torch's decorator sets (checked against the
installed torch), dynamo runs a marked function eagerly while it traces, and ``tracing`` tells
eager calls from traced ones."""

from __future__ import annotations

import importlib.util

import torch

from pointtransformerv3.ops._tracing import CONSTANT_RESULT_MARKER, assume_constant_result, tracing


def test_marker_matches_torch():
    def ours() -> int:
        return 1

    def theirs() -> int:
        return 1

    assert assume_constant_result(ours) is ours
    assert torch.compiler.assume_constant_result(theirs) is theirs
    assert vars(ours) == vars(theirs) == {CONSTANT_RESULT_MARKER: True}


def test_marked_function_runs_eagerly_in_a_fullgraph_capture():
    """``importlib.util.find_spec`` cannot be traced (dynamo refuses it); marked, it runs eagerly
    and its result is baked into the graph."""
    calls: list[int] = []

    @assume_constant_result
    def probe() -> bool:
        calls.append(1)
        return importlib.util.find_spec("torch") is not None

    def f(x: torch.Tensor) -> torch.Tensor:
        return x + 1 if probe() else x - 1

    torch._dynamo.reset()
    x = torch.zeros(3)
    out = torch.compile(f, backend="eager", fullgraph=True)(x)
    assert torch.equal(out, torch.ones(3)) and calls == [1]
    torch._dynamo.reset()


def test_tracing_tells_eager_from_traced():
    """Ops with Triton kernels launch them directly when ``tracing`` is False and take their
    ``triton_op`` when it is True: under dynamo, under the fake / proxy dispatch modes of
    non-strict torch.export and make_fx, and for tensor subclasses."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    x = torch.ones(3)
    assert not tracing(x)
    assert tracing(torch.nn.Parameter(x))  # a subclass: the op handles it
    with FakeTensorMode() as mode:
        assert tracing(mode.from_tensor(x))
    seen: list[bool] = []

    def f(t: torch.Tensor) -> torch.Tensor:
        seen.append(tracing(t))
        return t + 1

    make_fx(f)(x)
    torch._dynamo.reset()
    torch.compile(f, backend="eager", fullgraph=True)(x)
    torch._dynamo.reset()
    assert seen == [True, True]
