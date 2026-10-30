"""Dump the ORIGINAL implementations' outputs for the parity checks (P1-P4).

Runs in the original environment (torch 2.9.1, spconv, torch_scatter, flash_attn, timm, addict,
huggingface_hub), never imports pointtransformerv3, and reads the original code from git (this
repository's ``model.py`` + ``serialization/`` at 3229e9b, Sonata at 18c09ff from
``$PTV3_PARITY_SONATA_REPO``, Utonia at be53e4a from ``$PTV3_PARITY_UTONIA_REPO``) into
``$ROOT/code``:

    flock /tmp/pointcept2-gpu.lock env PYTHONDONTWRITEBYTECODE=1 \\
        <original env>/bin/python -B tests/parity/dump_original.py [steps]

Outputs under ``$ROOT`` (``$PTV3_PARITY_ROOT``, default ``~/.cache/pointtransformerv3/parity``):

- ``inputs/sonata_<case>.pt``: Sonata's ``transform.default()`` on the raw scenes (numpy seeds
  fixed), collated like Sonata's demo 3 for ``batch3`` (P3 and P4 inputs).
- ``orig/ptv3_random_state.pt``: the seeded original PTv3 weights, BatchNorm statistics and norm
  affines randomized (P1).
- ``orig/ptv3-<mode>-<case>.pt``: P1 levels (encoder + decoder features, grid, batch, codes).
  ``mode``: ``exact`` (enable_flash=False, fp32), ``default`` (enable_flash=True: fp32 with
  fp16 FlashAttention), ``shuffle`` (default mode with the original's order shuffling, seeded).
- ``orig/ptv3-grad-<case>.pt``: P1 parameter gradients of ``sum(out * w)`` (exact mode).
- ``orig/release-<mode>-<case>.pt``: P2, the release weights in the original model.py.
- ``orig/<model>-<mode>-<case>.pt``: P3, Sonata levels, the 1232-d upcast and linear-probe logits.
- ``inputs/utonia_<case>.pt``, ``orig/utonia-<mode>-<case>.pt``: P6, the same for Utonia (its
  ``transform.default(0.5)``, the 1386-d upcast and its linear probe).
"""

from __future__ import annotations

import copy
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    BATCH3,
    INPUTS,
    ORIG,
    PTV3_COMMIT,
    PTV3_RELEASE,
    ROOT,
    SAMPLE1,
    SCENES,
    SONATA_CKPT,
    SONATA_COMMIT,
    UTONIA_CASES,
    UTONIA_COMMIT,
    UTONIA_REVISION,
    env_path,
)

REPO = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def checkout(utonia: bool) -> None:
    """The original sources, extracted from git (never edited); Utonia only for its steps."""
    ptv3 = CODE / "ptv3_orig"
    if not (ptv3 / "model.py").is_file():
        ptv3.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(
            ["git", "-C", str(REPO), "archive", PTV3_COMMIT, "model.py", "serialization"],
            check=True,
            capture_output=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(ptv3)], input=archive, check=True)
        (ptv3 / "__init__.py").write_text("")
    roots = [CODE]
    originals = [("sonata", "PTV3_PARITY_SONATA_REPO", SONATA_COMMIT)]
    if utonia:
        originals.append(("utonia", "PTV3_PARITY_UTONIA_REPO", UTONIA_COMMIT))
    for name, var, commit in originals:
        root = CODE / f"{name}_orig"
        if not (root / name / "model.py").is_file():
            root.mkdir(parents=True, exist_ok=True)
            archive = subprocess.run(
                ["git", "-C", str(env_path(var)), "archive", commit],
                check=True,
                capture_output=True,
            ).stdout
            subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
        roots.append(root)
    sys.path[:0] = [str(r) for r in roots]


def cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().cpu() if t.is_floating_point() else t.detach().cpu()


# ---- PTv3 (P1, P2)


def randomize_norms(model: nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                c = m.num_features
                m.running_mean.copy_(torch.randn(c, generator=gen) * 0.2)
                m.running_var.copy_(torch.rand(c, generator=gen) + 0.5)
            if isinstance(m, nn.BatchNorm1d | nn.LayerNorm):
                c = m.weight.shape[0]
                m.weight.copy_(torch.rand(c, generator=gen) + 0.5)
                m.bias.copy_(torch.randn(c, generator=gen) * 0.1)


def ptv3_hooks(model: nn.Module, record: dict) -> list:
    """Encoder stage outputs (feat, grid, batch, codes, depth) and decoder stage outputs."""
    handles = []

    def enc_hook(s):
        def hook(module, args, out):
            record["enc"][s] = {
                "feat": cpu(out.feat),
                "grid_coord": cpu(out.grid_coord),
                "batch": cpu(out.batch),
                "serialized_code": cpu(out.serialized_code),
                "serialized_depth": int(out.serialized_depth),
            }

        return hook

    def dec_hook(s):
        def hook(module, args, out):
            record["dec"][s] = {
                "feat": cpu(out.feat),
                "grid_coord": cpu(out.grid_coord),
                "batch": cpu(out.batch),
            }

        return hook

    for name, module in model.enc.named_children():
        handles.append(module.register_forward_hook(enc_hook(int(name.removeprefix("enc")))))
    if hasattr(model, "dec"):
        for name, module in model.dec.named_children():
            handles.append(module.register_forward_hook(dec_hook(int(name.removeprefix("dec")))))
    return handles


def ptv3_input(case: str) -> dict:
    data = torch.load(INPUTS / f"ptv3_{case}.pt", weights_only=True)
    return {k: data[k].to(DEV) for k in ("coord", "grid_coord", "feat", "offset")}


def set_shuffle(model: nn.Module, on: bool) -> None:
    from ptv3_orig.model import SerializedPooling

    model.shuffle_orders = on
    for m in model.modules():
        if isinstance(m, SerializedPooling):
            m.shuffle_orders = on


def dump_ptv3() -> None:
    from ptv3_orig.model import PointTransformerV3

    torch.manual_seed(0)
    models = {}
    for flash in (False, True):
        torch.manual_seed(0)
        models[flash] = PointTransformerV3(enable_flash=flash)
    randomize_norms(models[False], seed=1)
    state = models[False].state_dict()
    models[True].load_state_dict(state)
    torch.save(state, ORIG / "ptv3_random_state.pt")
    cases = [*SCENES, "tiny", "batch3"]
    for mode, flash, names in (
        ("exact", False, [*SCENES, "tiny"]),  # enable_flash=False: single samples only
        ("default", True, cases),
        ("shuffle", True, ["scene0342_00", "tiny"]),
    ):
        model = models[flash].to(DEV).eval()
        set_shuffle(model, mode == "shuffle")
        for case in names:
            record = {"enc": {}, "dec": {}}
            handles = ptv3_hooks(model, record)
            torch.manual_seed(123)
            with torch.no_grad():
                out = model(ptv3_input(case))
            for h in handles:
                h.remove()
            record["out"] = cpu(out.feat)
            torch.save(record, ORIG / f"ptv3-{mode}-{case}.pt")
            print("ptv3", mode, case, tuple(out.feat.shape), flush=True)
    models[True].cpu()
    dump_ptv3_grads(models[False])


def dump_ptv3_grads(model: nn.Module | None = None) -> None:
    """P1 gradients in train mode (DropPath off, shuffling off, BatchNorm on batch statistics):
    spconv 2.3.8 runs its eval-mode SubMConv3d on an inference path that records no gradients
    (the CPE convs get none), so eval-mode gradients are no reference."""
    from ptv3_orig.model import PointTransformerV3

    if model is None:
        torch.manual_seed(0)
        model = PointTransformerV3(enable_flash=False)
        model.load_state_dict(torch.load(ORIG / "ptv3_random_state.pt", weights_only=True))
    model = model.to(DEV).train()
    for m in model.modules():
        if hasattr(m, "drop_prob"):
            m.drop_prob = 0.0
    set_shuffle(model, False)
    for case in ("tiny", "scene0342_00"):
        model.zero_grad(set_to_none=True)
        data = ptv3_input(case)
        out = model(data)
        w = torch.randn(out.feat.shape, generator=torch.Generator().manual_seed(2)).to(DEV)
        (out.feat * w).sum().backward()
        grads = {k: cpu(p.grad) for k, p in model.named_parameters() if p.grad is not None}
        torch.save(
            {"w": w.cpu(), "grads": grads, "out": cpu(out.feat)}, ORIG / f"ptv3-grad-{case}.pt"
        )
        print("ptv3 grad", case, len(grads), flush=True)
    model.zero_grad(set_to_none=True)
    model.cpu()


def dump_release() -> None:
    """P2 numerics: the release weights in the original model.py (which has no stale-CPE
    switch); our side runs ptv3_scannet with legacy_cpe_stale_input=False."""
    from ptv3_orig.model import PointTransformerV3

    ckpt = torch.load(PTV3_RELEASE, map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in ckpt["state_dict"].items()}
    head = {k.removeprefix("seg_head."): v for k, v in state.items() if k.startswith("seg_head.")}
    backbone = {
        k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")
    }
    for mode, flash in (("exact", False), ("default", True)):
        torch.manual_seed(0)
        model = PointTransformerV3(enable_flash=flash)
        model.load_state_dict(backbone, strict=True)
        model = model.to(DEV).eval()
        set_shuffle(model, False)
        for case in ("scene0342_00", "scene0665_01"):
            with torch.no_grad():
                out = model(ptv3_input(case))
                logits = out.feat @ head["weight"].to(DEV).T + head["bias"].to(DEV)
            torch.save(
                {"out": cpu(out.feat), "logits": cpu(logits)}, ORIG / f"release-{mode}-{case}.pt"
            )
            print("release", mode, case, flush=True)
        model.cpu()


# ---- Sonata (P3, P4)


def sonata_inputs() -> None:
    sonata = importlib.import_module("sonata")
    raw_dir = env_path("PTV3_PARITY_RAW")
    raw = {"sample1": SAMPLE1, **{s: raw_dir / f"{s}.npz" for s in SCENES}}
    done = {}
    for i, (name, path) in enumerate(raw.items()):
        point = dict(np.load(path))
        for k in ("segment200", "instance"):
            point.pop(k, None)
        segment = point.pop("segment20")
        point["segment"] = segment
        sonata.utils.set_seed(1000 + i)
        point = sonata.transform.default()(point)
        point["segment_full"] = torch.from_numpy(segment.astype(np.int64))
        done[name] = point
        torch.save(point, INPUTS / f"sonata_{name}.pt")
        print("sonata input", name, tuple(point["feat"].shape), flush=True)
    parts = [copy.deepcopy(done[n]) for n in BATCH3]
    seg_full = [p.pop("segment_full") for p in parts]
    batch = sonata.data.collate_fn(parts)
    batch["segment_full"] = torch.cat(seg_full)
    batch["inverse_parts"] = torch.tensor([len(s) for s in seg_full])
    torch.save(batch, INPUTS / "sonata_batch3.pt")


def sonata_val_inputs() -> None:
    """P4 on the full ScanNet val split (CPU): Sonata's transform (seeded) on every val scene."""
    import json

    from safetensors.numpy import load_file

    sonata = importlib.import_module("sonata")
    out = INPUTS / "sonata_val"
    out.mkdir(parents=True, exist_ok=True)
    root = env_path("PTV3_PARITY_SCANNET")
    rows = json.loads((root / "index.json").read_text())["splits"]["val"]
    for i, row in enumerate(rows):
        target = out / f"{row['name']}.npz"
        if target.is_file():
            continue
        scene = load_file(str(root / row["file"]))
        point = {k: scene[k].astype(np.float32) for k in ("coord", "color", "normal")}
        sonata.utils.set_seed(2000 + i)
        point = sonata.transform.default()(point)
        np.savez(
            target,
            grid_coord=point["grid_coord"].int().numpy(),
            feat=point["feat"].numpy(),
            coord=point["coord"].numpy(),
            inverse=point["inverse"].int().numpy(),
        )
    print("sonata val inputs", len(rows), flush=True)


def run_encoder(model: nn.Module, data: dict, head: nn.Module | None) -> dict:
    """Every level of an encoder (Sonata / Utonia) on one input, the full upcast and the probe."""
    keys = ("grid_coord", "serialized_code", "pooling_inverse", "offset", "batch")
    point = {k: data[k].to(DEV) for k in ("coord", "grid_coord", "offset", "feat")}
    torch.manual_seed(0)
    with torch.inference_mode():
        point = model(point)
        levels, p = [], point
        while True:
            lv = {"feat": cpu(p.feat), "serialized_depth": int(p.serialized_depth)}
            lv.update({k: cpu(p[k]) for k in keys if k in p.keys()})
            levels.append(lv)
            if "pooling_parent" not in p.keys():
                break
            p = p["pooling_parent"]
        levels.reverse()
        p, feat = point, point.feat.float()
        while "pooling_parent" in p.keys():
            parent = p["pooling_parent"]
            feat = torch.cat([parent.feat.float(), feat[p["pooling_inverse"]]], -1)
            p = parent
        logits = head(feat) if head is not None and feat.shape[1] == head.in_features else None
    return {
        "levels": levels,
        "upcast": cpu(feat),
        "logits": None if logits is None else cpu(logits),
    }


def linear_head(path: Path | str, channels: int) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    head = nn.Linear(channels, 20)
    head.load_state_dict({k.removeprefix("seg_head."): v for k, v in state.items()})
    return head.to(DEV).eval()


def dump_sonata() -> None:
    sonata = importlib.import_module("sonata")
    head = linear_head(SONATA_CKPT / "sonata_linear_prob_head_sc.pth", 1232)
    for model_name in ("sonata", "sonata_small"):
        for mode, flash in (("exact", False), ("default", True)):
            cfg = None if flash else dict(enable_flash=False)
            model = sonata.load(str(SONATA_CKPT / f"{model_name}.pth"), custom_config=cfg)
            model = model.to(DEV).eval()
            cases = ["sample1", *SCENES] + (["batch3"] if flash else [])
            if model_name == "sonata_small":
                cases = ["sample1", "scene0342_00"]
            for case in cases:
                data = torch.load(INPUTS / f"sonata_{case}.pt", weights_only=False)
                out = run_encoder(model, data, head)
                torch.save(out, ORIG / f"{model_name}-{mode}-{case}.pt")
                shapes = [tuple(lv["feat"].shape) for lv in out["levels"]]
                print(model_name, mode, case, shapes, flush=True)
            model.cpu()
            torch.cuda.empty_cache()


def utonia_inputs() -> None:
    """P6 inputs: Utonia's ``transform.default(0.5)`` (indoor; numpy seeded) on sample1 and a
    ScanNet scene."""
    utonia = importlib.import_module("utonia")
    raw = {"sample1": SAMPLE1, "scene0342_00": env_path("PTV3_PARITY_RAW") / "scene0342_00.npz"}
    for i, case in enumerate(UTONIA_CASES):
        point = dict(np.load(raw[case]))
        segment = point.pop("segment20")
        point = {k: point[k] for k in ("coord", "color", "normal")}
        utonia.utils.set_seed(3000 + i)
        point = utonia.transform.default(0.5)(point)
        point["segment_full"] = torch.from_numpy(segment.astype(np.int64))
        torch.save(point, INPUTS / f"utonia_{case}.pt")
        print("utonia input", case, tuple(point["feat"].shape), flush=True)


def dump_utonia() -> None:
    """P6: Utonia (be53e4a, ``model.py`` with ``Point3DRoPE``) on its own inputs, the weights at
    the pinned Hugging Face revision."""
    from huggingface_hub import hf_hub_download

    utonia = importlib.import_module("utonia")

    def fetch(name: str) -> str:
        return hf_hub_download("Pointcept/Utonia", f"{name}.pth", revision=UTONIA_REVISION)

    head = linear_head(fetch("utonia_linear_prob_head_sc"), 1386)
    for mode, flash in (("exact", False), ("default", True)):
        cfg = None if flash else dict(enable_flash=False)
        model = utonia.model.load(fetch("utonia"), custom_config=cfg).to(DEV).eval()
        for case in UTONIA_CASES:
            data = torch.load(INPUTS / f"utonia_{case}.pt", weights_only=False)
            out = run_encoder(model, data, head)
            torch.save(out, ORIG / f"utonia-{mode}-{case}.pt")
            print(
                "utonia", mode, case, [tuple(lv["feat"].shape) for lv in out["levels"]], flush=True
            )
        model.cpu()
        torch.cuda.empty_cache()


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    ORIG.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1:] or ["sonata_inputs", "ptv3", "release", "sonata", "utonia_inputs", "utonia"]
    checkout(utonia=any(step.startswith("utonia") for step in what))
    print("torch", torch.__version__, "numpy", np.__version__, flush=True)
    if "sonata_inputs" in what:
        sonata_inputs()
    if "ptv3" in what:
        dump_ptv3()
    elif "ptv3_grads" in what:
        dump_ptv3_grads()
    if "release" in what:
        dump_release()
    if "sonata" in what:
        dump_sonata()
    if "sonata_val_inputs" in what:
        sonata_val_inputs()
    if "utonia_inputs" in what:
        utonia_inputs()
    if "utonia" in what:
        dump_utonia()
    os.sync()


if __name__ == "__main__":
    main()
