"""Run this package on the parity inputs and compare with the original implementations' dumps
(``dump_original.py``). Prints one line per comparison and writes ``$ROOT/report.json``.

usage (this package's env, after dump_original.py):

    flock /tmp/pointcept2-gpu.lock .venv/bin/python -m tests.parity.check [p1 p2 p3 p4 p5 p6 val]
    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m tests.parity.check p5cpu   # CPU half of P5

Metrics: ``rel`` = max|new - ref| / max|ref| over the worst level, ``cos`` = per-row cosine
(mean / min), ``ints`` = integer tensors equal (grid-cell sets per level, codes, pooling inverses).
Rows are aligned by (batch, grid_coord): the two implementations order coarse rows differently.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import point_transformer_v3 as ptv3
from point_transformer_v3.convert import map_name
from point_transformer_v3.model import Config
from point_transformer_v3.ops.serialization import serialize
from point_transformer_v3.presets import PC_M1_RELEASE
from tests.parity.common import INPUTS, ORIG, ROOT, SAMPLE1, SCENES, UTONIA_CASES, env_path

REPORT: dict[str, Any] = {}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ORDER_NAMES = ("z", "z-trans", "hilbert", "hilbert-trans")


# ---- helpers


def keys_of(grid_coord: Tensor, batch: Tensor) -> Tensor:
    g = grid_coord.long().cpu()
    return batch.long().cpu() << 48 | g[:, 0] << 32 | g[:, 1] << 16 | g[:, 2]


def align(ref_grid, ref_batch, new_grid, new_batch) -> tuple[Tensor, Tensor, bool]:
    kr, kn = keys_of(ref_grid, ref_batch), keys_of(new_grid, new_batch)
    sr, sn = kr.argsort(), kn.argsort()
    return sr, sn, kr.shape == kn.shape and torch.equal(kr[sr], kn[sn])


def metrics(ref: Tensor, new: Tensor) -> dict[str, float]:
    ref, new = ref.double().cpu(), new.double().cpu()
    diff = (new - ref).abs().max().item() if ref.numel() else 0.0
    scale = ref.abs().max().item() or 1.0
    cos = F.cosine_similarity(ref, new, dim=1, eps=1e-12)
    return {
        "abs": diff,
        "rel": diff / scale,
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
    }


def worst(items: list[dict[str, float]]) -> dict[str, float]:
    return {
        "abs": max(m["abs"] for m in items),
        "rel": max(m["rel"] for m in items),
        "cos_mean": min(m["cos_mean"] for m in items),
        "cos_min": min(m["cos_min"] for m in items),
    }


def show(tag: str, m: dict[str, Any]) -> None:
    REPORT[tag] = m
    parts = []
    for k, v in m.items():
        parts.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    print(f"{tag:48s} " + " ".join(parts), flush=True)


def derive_orders(levels: list[dict[str, Any]]) -> Tensor:
    """(L, K) order indices from the original's per-level serialized codes."""
    out = torch.zeros(len(levels), len(ORDER_NAMES), dtype=torch.long)
    for li, lv in enumerate(levels):
        grid, batch, depth = lv["grid_coord"], lv["batch"], lv["serialized_depth"]
        mine = {
            j: serialize(grid.int(), batch.long(), depth, name)
            for j, name in enumerate(ORDER_NAMES)
        }
        for k, row in enumerate(lv["serialized_code"]):
            hits = [j for j, code in mine.items() if torch.equal(code, row.long())]
            if len(hits) != 1:
                raise RuntimeError(f"level {li} row {k}: cannot identify the order ({hits})")
            out[li, k] = hits[0]
    return out


def miou(pred: Tensor, gt: Tensor, n: int = 20) -> float:
    conf = confusion(pred, gt, n)
    inter = conf.diag().double()
    union = conf.sum(0) + conf.sum(1) - conf.diag()
    seen = union > 0
    return (inter[seen] / union[seen]).mean().item() * 100


def confusion(pred: Tensor, gt: Tensor, n: int = 20) -> Tensor:
    pred, gt = pred.long().cpu(), gt.long().cpu()
    valid = (gt >= 0) & (gt < n)
    return torch.bincount(gt[valid] * n + pred[valid], minlength=n * n).view(n, n)


def ptv3_input(case: str, device=DEV) -> dict[str, Tensor]:
    data = torch.load(INPUTS / f"ptv3_{case}.pt", weights_only=True)
    return {k: data[k].to(device) for k in ("coord", "grid_coord", "feat", "offset")}


def run_ptv3(model, data, orders=None):
    """Our encoder / decoder stage outputs with their grids (forward hooks)."""
    rec: dict[str, dict[int, Any]] = {"enc": {}, "dec": {}}
    hooks = []
    for s, stage in enumerate(model.enc):
        hooks.append(
            stage.register_forward_hook(
                lambda m, a, o, s=s: rec["enc"].__setitem__(s, (o.float(), a[1]))
            )
        )
    for s, stage in enumerate(model.dec):
        hooks.append(
            stage.register_forward_hook(
                lambda m, a, o, s=s: rec["dec"].__setitem__(s, (o.float(), a[2].parent))
            )
        )
    with torch.no_grad():
        out = model(data, orders=orders)
    for h in hooks:
        h.remove()
    return out, rec


def compare_ptv3_levels(ref: dict, rec: dict) -> tuple[dict[str, float], bool]:
    ms, ints = [], True
    for part in ("enc", "dec"):
        for s, lv in ref[part].items():
            feat, grid = rec[part][s]
            sr, sn, ok = align(lv["grid_coord"], lv["batch"], grid.grid_coord, grid.batch_index())
            ints &= ok
            if ok:
                ms.append(metrics(lv["feat"][sr], feat.cpu()[sn]))
    return worst(ms), ints


# ---- P1: PTv3 vs model.py@3229e9b, random weights


def p1(grads_only: bool = False) -> None:
    state = torch.load(ORIG / "ptv3_random_state.pt", weights_only=True)
    model = ptv3.PointTransformerV3().to(DEV)  # the original constructor's defaults
    model.load_state_dict(state)
    model.eval()
    cases = [*SCENES, "tiny", "batch3"]
    for mode, names in (
        ()
        if grads_only
        else (
            ("exact", [*SCENES, "tiny"]),
            ("default", cases),
            ("shuffle", ["scene0342_00", "tiny"]),
        )
    ):
        model.precision = "fp32" if mode == "exact" else "auto"
        ptv3.set_backend(attention="sdpa" if mode == "exact" else "auto")
        for case in names:
            ref = torch.load(ORIG / f"ptv3-{mode}-{case}.pt", weights_only=True)
            orders = None
            if mode == "shuffle":
                levels = [ref["enc"][s] for s in sorted(ref["enc"])]
                orders = derive_orders(levels)
            out, rec = run_ptv3(model, ptv3_input(case), orders)
            m, ints = compare_ptv3_levels(ref, rec)
            final = metrics(ref["out"], out.feat.float().cpu())
            agree = (ref["out"].argmax(1) == out.feat.float().cpu().argmax(1)).float().mean()
            show(
                f"P1 {mode:8s} {case}",
                {**m, "ints": ints, "out_rel": final["rel"], "argmax": agree.item()},
            )
    # gradients in train mode (BatchNorm on batch statistics), DropPath and shuffling off: the
    # original's eval-mode spconv records no gradients for the CPE convs (dump_original.py)
    ptv3.set_backend(attention="sdpa")
    model.precision = "fp32"
    model.train()
    for m in model.modules():
        if hasattr(m, "drop_prob"):
            m.drop_prob = 0.0
    model.load_state_dict(state)  # the original dump starts from the same weights and stats
    identity = torch.arange(4).repeat(model.num_levels, 1)
    for case in ("tiny", "scene0342_00"):
        ref = torch.load(ORIG / f"ptv3-grad-{case}.pt", weights_only=True)
        model.zero_grad(set_to_none=True)
        out = model(ptv3_input(case), orders=identity)
        fwd = metrics(ref["out"], out.feat.detach().float().cpu())
        (out.feat * ref["w"].to(DEV)).sum().backward()
        mine = dict(model.named_parameters())
        rels = {}
        for name, g in ref["grads"].items():
            new = mine[map_name(name)].grad
            rels[name] = (new.cpu().double() - g.double()).abs().max().item() / (
                g.abs().max().item() or 1.0
            )
        # Biases right before a train-mode BatchNorm (pooling / unpooling projections, and the
        # last MLP of each stage, whose output only reaches BatchNorms) have a zero gradient in
        # exact math; both sides hold float noise there (|g| ~1e-5 of the largest gradients).
        scale = max(g.abs().max().item() for g in ref["grads"].values())
        zero = {k for k, g in ref["grads"].items() if g.abs().max().item() < 1e-5 * scale}
        live = {k: v for k, v in rels.items() if k not in zero}
        top = max(live, key=lambda k: live[k])
        noise = max((mine[map_name(k)].grad.abs().max().item() for k in zero), default=0.0) / scale
        show(
            f"P1 grads   {case}",
            {
                "out_rel": fwd["rel"],
                "rel_max": live[top],
                "worst": top,
                "rel_median": float(np.median(list(live.values()))),
                "n": len(live),
                "zero_grad_params": len(zero),
                "zero_grad_noise": noise,
                "missing": len(mine) - len(rels),
            },
        )
    model.eval()
    model.zero_grad(set_to_none=True)
    ptv3.set_backend(attention="auto")


# ---- P2: release weights


def p2() -> None:
    from point_transformer_v3.convert import load_pc_checkpoint
    from tests.parity.common import PTV3_RELEASE

    state = load_pc_checkpoint(PTV3_RELEASE)["state_dict"]
    cfg = Config.from_dict({**PC_M1_RELEASE, "legacy_cpe_stale_input": False})
    model = ptv3.PointTransformerV3(config=cfg)
    model.attach_head(20, 64)
    model.load_state_dict(state)
    model = model.to(DEV).eval()
    for mode in ("exact", "default"):
        model.precision = "fp32" if mode == "exact" else "auto"
        ptv3.set_backend(attention="sdpa" if mode == "exact" else "auto")
        for case in ("scene0342_00", "scene0665_01"):
            ref = torch.load(ORIG / f"release-{mode}-{case}.pt", weights_only=True)
            with torch.no_grad():
                out = model(ptv3_input(case))
                logits = model.seg_head(out.feat.float())
            m = metrics(ref["out"], out.feat.float().cpu())
            agree = (ref["logits"].argmax(1) == logits.cpu().argmax(1)).float().mean().item()
            show(f"P2 release {mode:8s} {case}", {**m, "argmax": agree})
    ptv3.set_backend(attention="auto")


def val_scenes():
    from safetensors.numpy import load_file

    root = env_path("PTV3_PARITY_SCANNET")
    rows = json.loads((root / "index.json").read_text())["splits"]["val"]
    for row in rows:
        yield row["name"], load_file(str(root / row["file"]))


def val_miou(
    model, preset: str, *, precision: str, label: str, head_from_upcast: bool, **prep
) -> None:
    """Single-pass raw-point mIoU over the 312 ScanNet val scenes (``prepare`` with ``prep``,
    default pick "first"; ``pick="random"`` draws from a generator seeded per scene)."""
    model.precision = precision
    conf = torch.zeros(20, 20, dtype=torch.long)
    t0 = time.perf_counter()
    for i, (_, scene) in enumerate(val_scenes()):
        kwargs = dict(prep)
        if kwargs.get("pick") == "random":
            kwargs["generator"] = torch.Generator(device=DEV).manual_seed(i)
        with torch.inference_mode():
            logits = model.predict(scene["coord"], scene["color"], scene["normal"], **kwargs)
        conf += confusion(logits.argmax(1), torch.from_numpy(scene["segment20"].astype(np.int64)))
    inter = conf.diag().double()
    iou = inter / (conf.sum(0) + conf.sum(1) - conf.diag()).double()
    show(
        f"VAL {label}",
        {
            "miou": iou.mean().item() * 100,
            "allacc": (inter.sum() / conf.sum()).item() * 100,
            "seconds": time.perf_counter() - t0,
        },
    )


def val() -> None:
    model = ptv3.load("ptv3_scannet", device=DEV, warmup=False)
    for depth in ("auto", 15):
        cfg = Config.from_dict({**model.cfg.to_dict(), "serialization_depth": depth})
        model.cfg = cfg
        val_miou(
            model,
            "ptv3_scannet",
            precision="fp32",
            label=f"ptv3_scannet fp32 depth={depth}",
            head_from_upcast=False,
        )
    model.cfg = Config.from_dict({**model.cfg.to_dict(), "serialization_depth": "auto"})
    val_miou(
        model,
        "ptv3_scannet",
        precision="auto",
        label="ptv3_scannet fp16 depth=auto",
        head_from_upcast=False,
    )


def valpick() -> None:
    """How the preprocessing moves the ptv3_scannet val mIoU (fp32, depth "auto")."""
    model = ptv3.load("ptv3_scannet", device=DEV, warmup=False)
    for label, prep in (
        ("pick=random", dict(pick="random")),
        ("pick=random float64 lattice", dict(pick="random", lattice="float64")),
        (
            "pick=random, not centered, float64 (Pointcept2's run)",
            dict(pick="random", lattice="float64", center=False),
        ),
        ("pick=first, not centered, float64", dict(lattice="float64", center=False)),
    ):
        val_miou(
            model,
            "ptv3_scannet",
            precision="fp32",
            label=f"ptv3_scannet {label}",
            head_from_upcast=False,
            **prep,
        )


def p4val() -> None:
    """P4 on all 312 ScanNet val scenes: Sonata's own transform (seeded random pick,
    dump_original.py sonata_val_inputs) vs ``prepare`` (pick "first"), both through this
    package's sonata + linear probe in the default precision, orders unshuffled."""
    model = ptv3.load("sonata", head="scannet20", device=DEV, warmup=False)
    conf = {
        "orig_tf": torch.zeros(20, 20, dtype=torch.long),
        "prepare": torch.zeros(20, 20, dtype=torch.long),
    }
    same, t0 = 0, time.perf_counter()
    for name, scene in val_scenes():
        seg = torch.from_numpy(scene["segment20"].astype(np.int64))
        z = np.load(INPUTS / "sonata_val" / f"{name}.npz")
        point = {
            "coord": torch.from_numpy(z["coord"]).to(DEV),
            "grid_coord": torch.from_numpy(z["grid_coord"]).to(DEV),
            "feat": torch.from_numpy(z["feat"]).to(DEV),
        }
        with torch.inference_mode():
            up = ptv3.upcast(model(point)).feat
            ref = model.seg_head(up.to(model.seg_head.weight.dtype)).argmax(1).cpu()
            new, inverse = ptv3.prepare(
                scene["coord"], scene["color"], scene["normal"], preset="sonata", device=DEV
            )
            pred = model.seg_head(ptv3.upcast(model(new)).feat.to(model.seg_head.weight.dtype))
        conf["orig_tf"] += confusion(ref[torch.from_numpy(z["inverse"]).long()], seg)
        conf["prepare"] += confusion(pred.argmax(1)[inverse].cpu(), seg)
        same += int(len(new.feat) == len(z["feat"]))
    out = {}
    for k, c in conf.items():
        iou = c.diag().double() / (c.sum(0) + c.sum(1) - c.diag()).double()
        out[f"miou_{k}"] = iou.mean().item() * 100
    show(
        "P4 val (312 scenes, sonata linear probe)",
        {**out, "same_cell_count": same, "seconds": time.perf_counter() - t0},
    )


# ---- P3: Sonata vs 18c09ff


def sonata_input(case: str, device=DEV, family: str = "sonata") -> dict[str, Any]:
    return torch.load(INPUTS / f"{family}_{case}.pt", weights_only=False)


def run_sonata(model, data, orders):
    point = {k: data[k].to(DEV) for k in ("coord", "grid_coord", "feat", "offset")}
    with torch.inference_mode():
        out = model(point, orders=orders)
        levels, p = [], out
        while True:
            levels.append(p)
            if "pooling_parent" not in p:
                break
            p = p["pooling_parent"]
        levels.reverse()
        up = ptv3.upcast(out)
        logits = (
            model.seg_head(up.feat.to(model.seg_head.weight.dtype))
            if model.seg_head is not None and up.feat.shape[1] == model.seg_head.in_features
            else None
        )
    return levels, up.feat.float().cpu(), None if logits is None else logits.float().cpu()


def compare_sonata(ref, levels, up, logits, data) -> dict[str, Any]:
    ms, ints = [], True
    for lv_ref, lv in zip(ref["levels"], levels, strict=True):
        sr, sn, ok = align(lv_ref["grid_coord"], lv_ref["batch"], lv.grid_coord, lv.batch)
        ints &= ok
        if ok:
            ms.append(metrics(lv_ref["feat"][sr], lv.feat.float().cpu()[sn]))
            if "pooling_inverse" in lv_ref:  # same parent rows -> same coarse keys
                pass
    m = {**worst(ms), "ints": ints, "upcast_rel": metrics(ref["upcast"], up)["rel"]}
    if logits is not None and ref["logits"] is not None:
        m["argmax"] = (ref["logits"].argmax(1) == logits.argmax(1)).float().mean().item()
        if len(data["offset"]) > 1:  # a batch: the collated inverses are per sample
            return m
        inverse, seg = data["inverse"], data["segment_full"]
        m["miou_ref"] = miou(ref["logits"].argmax(1)[inverse], seg)
        m["miou_new"] = miou(logits.argmax(1)[inverse], seg)
    return m


def encoder_parity(tag: str, name: str, cases: list[str], head: str | bool) -> None:
    """An encoder preset against its original's dumps (P3, P6): ours in fp32 (SDPA), the default
    mode, fp16 and bf16; fp32, fp16 and bf16 against the original's exact mode."""
    family = name.split("_")[0]
    model = ptv3.load(name, head=head, device=DEV, warmup=False)
    for mode, precision in (
        ("exact", "fp32"),
        ("default", "auto"),
        ("fp16", "fp16"),
        ("bf16", "bf16"),
    ):
        model.precision = precision
        ptv3.set_backend(attention="sdpa" if mode == "exact" else "auto")
        ref_mode = "exact" if mode in ("exact", "fp16", "bf16") else "default"
        for case in cases:
            path = ORIG / f"{name}-{ref_mode}-{case}.pt"
            if not path.is_file():  # exact mode has no batch3 (enable_flash=False quirk)
                continue
            ref = torch.load(path, weights_only=True)
            data = sonata_input(case, family=family)
            orders = derive_orders(ref["levels"])
            levels, up, logits = run_sonata(model, data, orders)
            show(f"{tag} {name} {mode:7s} {case}", compare_sonata(ref, levels, up, logits, data))
    del model
    torch.cuda.empty_cache()
    ptv3.set_backend(attention="auto")


def p3() -> None:
    encoder_parity("P3", "sonata", ["sample1", *SCENES, "batch3"], "scannet20")
    encoder_parity("P3", "sonata_small", ["sample1", "scene0342_00"], False)


def p6() -> None:
    """Utonia (3D RoPE, head_dim 18) vs the Utonia repository at be53e4a."""
    encoder_parity("P6", "utonia", list(UTONIA_CASES), "scannet20")


# ---- P4: prepare vs the original transform


def p4() -> None:
    model = ptv3.load("sonata", head="scannet20", device=DEV, warmup=False)
    raw_dir = env_path("PTV3_PARITY_RAW")
    raw_files = {"sample1": SAMPLE1, **{s: raw_dir / f"{s}.npz" for s in SCENES}}
    for case, path in raw_files.items():
        raw = np.load(path)
        ref = sonata_input(case)
        point, inverse = ptv3.prepare(
            raw["coord"], raw["color"], raw["normal"], preset="sonata", device=DEV
        )
        kr = keys_of(ref["grid_coord"], torch.zeros(len(ref["grid_coord"]), dtype=torch.long))
        kn = keys_of(point.grid_coord, torch.zeros(len(point.grid_coord), dtype=torch.long))
        same = kr.shape == kn.shape and torch.equal(kr.sort().values, kn.sort().values)
        # every raw point lands in the same grid cell
        vr = keys_of(
            ref["grid_coord"][ref["inverse"]], torch.zeros(len(raw["coord"]), dtype=torch.long)
        )
        vn = keys_of(
            point.grid_coord.cpu()[inverse.cpu()], torch.zeros(len(raw["coord"]), dtype=torch.long)
        )
        model.precision = "auto"
        with torch.inference_mode():
            logits_new = model.predict(raw["coord"], raw["color"], raw["normal"])
            _, _, logits_ref = run_sonata(model, ref, None)
        seg = torch.from_numpy(raw["segment20"].astype(np.int64))
        show(
            f"P4 prepare {case}",
            {
                "cells_ref": len(kr),
                "cells_new": len(kn),
                "same_cells": same,
                "same_point_cell": bool(torch.equal(vr, vn)),
                "miou_orig_tf": miou(logits_ref.argmax(1)[ref["inverse"]], seg),
                "miou_prepare": miou(logits_new.argmax(1).cpu(), seg),
            },
        )


# ---- P5: CPU vs CUDA


def p5(device: str) -> None:
    """Sonata fp32 on scene0342_00 on ``device``; saved for the other half of the pair."""
    model = ptv3.load("sonata", device=device, warmup=False, precision="fp32", head=False)
    ptv3.set_backend(attention="sdpa")
    ref = torch.load(ORIG / "sonata-exact-scene0342_00.pt", weights_only=True)
    data = sonata_input("scene0342_00")
    orders = derive_orders(ref["levels"])
    point = {k: data[k].to(device) for k in ("coord", "grid_coord", "feat", "offset")}
    t0 = time.perf_counter()
    with torch.inference_mode():
        up = ptv3.upcast(model(point, orders=orders)).feat.float().cpu()
    seconds = time.perf_counter() - t0
    torch.save(up, ROOT / f"new_p5_{device}.pt")
    show(f"P5 {device} vs original fp32", {**metrics(ref["upcast"], up), "seconds": seconds})
    other = ROOT / f"new_p5_{'cuda' if device == 'cpu' else 'cpu'}.pt"
    if other.is_file():
        show("P5 cpu vs cuda (new, fp32)", metrics(torch.load(other), up))


def main() -> None:
    what = sys.argv[1:] or ["p1", "p2", "p3", "p4", "p5"]
    print("torch", torch.__version__, "device", DEV, flush=True)
    for step in what:
        if step == "p5cpu":
            p5("cpu")
        elif step == "p1grads":
            p1(grads_only=True)
        elif step == "p5":
            p5("cuda")
        else:
            globals()[step]()
    out = ROOT / f"report-{'-'.join(what)}.json"
    out.write_text(json.dumps(REPORT, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
