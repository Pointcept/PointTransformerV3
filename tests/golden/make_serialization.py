"""Write tests/golden/serialization.npz: the codes of the original PTv3 serialization
(``serialization/`` of this repository at commit 3229e9b) on the inputs of
tests/ops/test_serialization.py. Generated once; rerunning it needs a git checkout that has that
commit:  .venv/bin/python -m tests.golden.make_serialization

Stored: the random inputs ``grid``, ``batch`` and ``wild``, the codes ``head`` of the first
``GOLDEN_HEAD`` rows of ``grid`` for every order and depth, and the sha256 of every code array the
tests compare (``keys`` and ``sha256``)."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from pointtransformerv3.ops.serialization import MAX_DEPTH, ORDERS
from tests.ops.test_serialization import (
    FULL_DEPTHS,
    GOLDEN,
    GOLDEN_HEAD,
    edge_rows,
    full_grid,
    low_bits,
    sha256,
    wild_head,
)

COMMIT = "3229e9b"
REPO = Path(__file__).resolve().parents[2]


def original(tmp: Path) -> ModuleType:
    """The original ``serialization`` package, extracted from git into ``tmp`` and imported."""
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", COMMIT, "serialization"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tmp, filter="data")
    package = tmp / "serialization"
    spec = importlib.util.spec_from_file_location(
        "_original_serialization",
        package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {package}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    grid = torch.randint(0, 1 << MAX_DEPTH, (512, 3), generator=gen, dtype=torch.int32)
    batch = torch.randint(0, 7, (512,), generator=gen)
    wild = torch.randint(-(2**31), 2**31, (256, 3), generator=gen, dtype=torch.int64).int()
    head = torch.empty(len(ORDERS), MAX_DEPTH, GOLDEN_HEAD, dtype=torch.int64)
    digests: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        orig = original(Path(tmp))
        for o, order in enumerate(ORDERS):
            for depth in range(1, MAX_DEPTH + 1):
                rows = low_bits(grid, depth)
                code = orig.encode(torch.cat([edge_rows(depth), rows]), depth=depth, order=order)
                digests[f"encode/{order}/{depth}"] = sha256(code)
                head[o, depth - 1] = orig.encode(rows[:GOLDEN_HEAD], depth=depth, order=order)
                key = orig.encode(rows, batch=batch, depth=depth, order=order)
                digests[f"serialize/{order}/{depth}"] = sha256(key)
                code = orig.encode(torch.cat([wild_head(), wild]), depth=depth, order=order)
                digests[f"wild/{order}/{depth}"] = sha256(code)
            for depth in FULL_DEPTHS:
                code = orig.encode(full_grid(depth), depth=depth, order=order)
                digests[f"full/{order}/{depth}"] = sha256(code)
    np.savez_compressed(
        GOLDEN,
        grid=grid.numpy(),
        batch=batch.numpy(),
        wild=wild.numpy(),
        head=head.numpy(),
        keys=np.array(list(digests)),
        sha256=np.array(list(digests.values())),
    )
    print("wrote", GOLDEN, f"({GOLDEN.stat().st_size} bytes, {len(digests)} digests)")


if __name__ == "__main__":
    main()
