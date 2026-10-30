"""Golden outputs per preset (random init, seed 0, CPU fp32): any change in the model's math
shows up here (tests/golden/make_golden.py regenerates them on purpose)."""

from __future__ import annotations

import pytest
import torch

from tests.golden.make_golden import PATH, PRESETS, outputs


@pytest.fixture(scope="module")
def current():
    return outputs()


@pytest.mark.parametrize("name", PRESETS)
def test_golden(name, current):
    golden = torch.load(PATH, weights_only=True)[name]
    # relative to the output's scale: CPU thread counts change fp32 reduction orders (~1e-6)
    scale = golden.abs().max().item()
    torch.testing.assert_close(current[name], golden, atol=1e-5 * scale, rtol=0)
