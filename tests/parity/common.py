"""Paths and cases shared by the parity scripts (no pointtransformerv3 import: dump_original.py
runs in the original environment).

External data and code come from environment variables, checked by the step that needs them
(``env_path``; tests/parity/README.md lists them). Outputs go to ``$PTV3_PARITY_ROOT`` (default
``~/.cache/pointtransformerv3/parity``). The Sonata demo scan and checkpoints are read where the
Sonata package downloads them (``~/.cache/sonata``); the Hugging Face weights from the hub cache.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("PTV3_PARITY_ROOT", "~/.cache/pointtransformerv3/parity")).expanduser()
INPUTS = ROOT / "inputs"
ORIG = ROOT / "orig"
NEW = ROOT / "new"

SAMPLE1 = Path("~/.cache/sonata/data/sample1.npz").expanduser()  # sonata.data.load("sample1")
SCENES = ("scene0342_00", "scene0665_01", "scene0231_01")
BATCH3 = ("scene0665_01", "scene0342_00", "sample1")  # Sonata's batch (as the earlier port)
UTONIA_CASES = ("sample1", "scene0342_00")

# Original code: model.py and serialization/ of this repository at 3229e9b, Sonata at 18c09ff,
# Utonia (github.com/Pointcept/Utonia-Manifold) at be53e4a.
PTV3_COMMIT = "3229e9b"
SONATA_COMMIT = "18c09ff"
UTONIA_COMMIT = "be53e4a29cc7682ccbc4443844cbf3f9f4aa25c8"
UTONIA_REVISION = "9457f025e36c7001ee962a542f0d27a5b600e1e7"  # Pointcept/Utonia (= the preset's)
SONATA_CKPT = Path("~/.cache/sonata/ckpt").expanduser()  # sonata.load()'s download directory
_HUB = Path(
    os.environ.get("HF_HUB_CACHE")
    or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
).expanduser()
PTV3_RELEASE = (
    _HUB / "models--Pointcept--PointTransformerV3/snapshots/"
    "5e25cfaf759289b03db1f90f143b3dc5172929d3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"
)

_VARS = {
    "PTV3_PARITY_RAW": "a directory of raw ScanNet val scenes <name>.npz (float32 coord, color, "
    "normal; segment20) for " + ", ".join(SCENES),
    "PTV3_PARITY_SCANNET": "processed ScanNet (index.json and one .safetensors file per scene) "
    "for the full-val runs",
    "PTV3_PARITY_SONATA_REPO": "a git clone of github.com/facebookresearch/sonata (with "
    + SONATA_COMMIT
    + ")",
    "PTV3_PARITY_UTONIA_REPO": "a git clone of github.com/Pointcept/Utonia-Manifold (with "
    + UTONIA_COMMIT[:7]
    + ")",
}


def env_path(var: str) -> Path:
    """The existing path in environment variable ``var`` (one of ``_VARS``); exits with what to
    set when it is missing."""
    value = os.environ.get(var)
    if not value or not Path(value).expanduser().exists():
        got = f" ({value} does not exist)" if value else ""
        raise SystemExit(f"set {var} to {_VARS[var]}{got}; see tests/parity/README.md")
    return Path(value).expanduser()
