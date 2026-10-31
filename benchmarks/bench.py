"""Inference benchmark: this package against the original implementations, same GPU, same inputs.

Two persistent workers, one per environment, take turns on every request (no process ever
times both versions back to back): ``new`` (this package, its ``.venv``) and ``orig`` (the
original ``model.py`` at 3229e9b and Sonata at 18c09ff, in an environment with spconv,
torch_scatter and flash_attn). Inputs are prepared once with ``point_transformer_v3.prepare`` and
fed to both as identical tensors. GPU times are CUDA events around ``model(point)``: 3 warm-up
and 10 timed forwards per cell (two rounds of 5), median; peak memory is
``max_memory_allocated`` minus the allocation before the forward (weights not included; they
are reported separately).

    .venv/bin/python benchmarks/bench.py inputs        # once: prepare the cases (PTV3_BENCH_RAW)
    flock /tmp/pointcept2-gpu.lock .venv/bin/python benchmarks/bench.py gpu
    CUDA_VISIBLE_DEVICES="" .venv/bin/python benchmarks/bench.py cpu
    .venv/bin/python benchmarks/bench.py coldstart     # import / load / warm-up / first forward

Environment (docs/benchmark.md): ``PTV3_BENCH_RAW``, a directory of raw scans ``<case>.npz``
(coord, color, normal: ScanNet ``sn_*``, ScanNet++ ``pp_*``); ``PTV3_BENCH_ORIG_PYTHON``, the
python of the original environment; ``PTV3_BENCH_CACHE`` (default
``~/.cache/point_transformer_v3/bench``) for the prepared inputs. The original code is the one
``tests/parity/dump_original.py`` extracted under ``$PTV3_PARITY_ROOT/code``.

Results go to ``benchmarks/results/<what>.json`` and a table on stdout. Modes: ``default`` (ours:
fp16 autocast; the originals: fp32 with fp16 FlashAttention, i.e. ``enable_flash=True``),
``fp32`` (ours: precision fp32; originals: ``enable_flash=False``) and ``bf16`` (autocast on
both sides). Confounders: the original runs torch 2.9.1, this package 2.12; WSL2.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("PTV3_BENCH_CACHE", "~/.cache/point_transformer_v3/bench")).expanduser()
PARITY_ROOT = os.environ.get("PTV3_PARITY_ROOT", "~/.cache/point_transformer_v3/parity")
CODE = Path(PARITY_ROOT).expanduser() / "code"  # extracted by tests/parity/dump_original.py
SINGLE = ("sn_50k", "sn_150k", "sn_max", "pp_430k", "pp_1100k")
BATCHES = {"sn_batch4": 4, "sn_batch8": 8}
CASES = (*SINGLE, *BATCHES)
MODELS = ("ptv3_scannet", "sonata", "sonata_small")
MODES = ("default", "fp32", "bf16")
SONATA_CKPT = Path("~/.cache/sonata/ckpt").expanduser()
# The originals' fp32 mode (enable_flash=False) builds dense 1024x1024 attention matrices per patch
# and head (~40 GB at 1.1M points, beyond the GPU) and uses the smallest sample's patch size for a
# whole batch: not run there.
ORIG_FP32_SKIP = {
    "pp_1100k": "not run: enable_flash=False needs ~40 GB of attention matrices here",
    "sn_batch4": "not run: enable_flash=False uses the smallest sample's patch size in a batch",
    "sn_batch8": "not run: enable_flash=False uses the smallest sample's patch size in a batch",
}


def env(var: str, what: str) -> str:
    """A required setting (module doc); exits with what to set when it is missing."""
    value = os.environ.get(var)
    if not value:
        raise SystemExit(f"set {var} to {what} (docs/benchmark.md)")
    return value


def raw_dir() -> Path:
    return Path(env("PTV3_BENCH_RAW", "the directory of raw scans <case>.npz")).expanduser()


def orig_python() -> str:
    return env("PTV3_BENCH_ORIG_PYTHON", "the python of the original environment")


# ---- inputs (this package's env)


def make_inputs() -> None:
    import numpy as np
    import torch

    import point_transformer_v3 as ptv3

    CACHE.mkdir(parents=True, exist_ok=True)

    def raw(name: str) -> dict:
        d = dict(np.load(raw_dir() / f"{name}.npz"))
        d["normal"] = np.nan_to_num(d["normal"])  # a few ScanNet++ normals are NaN
        return d

    for prep in ("ptv3_scannet", "sonata"):
        for name in (*SINGLE, *BATCHES):
            out = CACHE / f"{prep}-{name}.npz"
            if out.is_file():
                continue
            if name in BATCHES:
                parts = [
                    ptv3.prepare(**_cols(raw(f"sn_batch_{i}")), preset=prep)[0]
                    for i in range(BATCHES[name])
                ]
                point = ptv3.collate(parts)
            else:
                point = ptv3.prepare(**_cols(raw(name)), preset=prep)[0]
            np.savez(
                out, **{k: point[k].numpy() for k in ("coord", "grid_coord", "feat", "offset")}
            )
            print("input", prep, name, len(point["feat"]), flush=True)
    del torch


def _cols(d: dict) -> dict:
    return {"coord": d["coord"], "color": d["color"], "normal": d["normal"]}


# ---- worker (either env): JSON requests on stdin, answers on the original stdout


def worker(impl: str, device_name: str) -> None:
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    import numpy as np
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    cuda = device.type == "cuda"
    models: dict = {}
    inputs: dict = {}

    def model_for(name: str, mode: str):
        key = (name, mode if impl == "orig" else "-")
        if key not in models:
            models[key] = _build(impl, name, mode, device)
        model = models[key]
        if impl == "new":
            model.precision = {"default": "auto", "fp32": "fp32", "bf16": "bf16"}[mode]
        return model

    def point_for(name: str, case: str) -> dict:
        prep = "ptv3_scannet" if name == "ptv3_scannet" else "sonata"
        key = (prep, case)
        if key not in inputs:
            z = np.load(CACHE / f"{prep}-{case}.npz")
            inputs[key] = {k: torch.from_numpy(z[k]).to(device) for k in z.files}
        return inputs[key]

    def forward(model, point, mode):
        autocast = impl == "orig" and mode == "bf16"
        # no_grad, not inference_mode: spconv's conv tuner fails on inference tensors
        # ("can't find suitable algorithm"); both workers use the same context
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            return model(dict(point))

    def event():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    proto.write(json.dumps({"ready": True, "torch": torch.__version__}) + "\n")
    for line in sys.stdin:
        req = json.loads(line)
        if req["op"] == "quit":
            break
        if req["op"] == "e2e":
            proto.write(json.dumps(_e2e(impl, model_for("sonata", "default"), req, device)) + "\n")
            continue
        model = model_for(req["model"], req["mode"])
        point = point_for(req["model"], req["case"])
        try:  # spconv has no bf16 kernels for some of PTv3's convs: report, do not crash
            forward(model, point, req["mode"])
        except RuntimeError as e:
            proto.write(
                json.dumps(
                    {
                        "error": str(e).strip().splitlines()[-1][-160:],
                        "points": int(point["feat"].shape[0]),
                    }
                )
                + "\n"
            )
            continue
        for _ in range(req.get("warmup", 3) - 1):
            forward(model, point, req["mode"])
        ms = []
        for _ in range(req.get("iters", 5)):
            if cuda:
                b = event()
                forward(model, point, req["mode"])
                e = event()
                torch.cuda.synchronize()
                ms.append(b.elapsed_time(e))
            else:
                t = time.perf_counter()
                forward(model, point, req["mode"])
                ms.append((time.perf_counter() - t) * 1e3)
        out = {"ms": ms, "points": int(point["feat"].shape[0])}
        if req.get("peak") and cuda:
            gc.collect()  # garbage of earlier calls must not be freed inside the measurement
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            res = forward(model, point, req["mode"])
            torch.cuda.synchronize()
            out["peak_mib"] = (torch.cuda.max_memory_allocated(device) - base) / 2**20
            out["weights_mib"] = (
                sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
            )
            del res
        if cuda:
            torch.cuda.empty_cache()  # the two workers share the GPU
        proto.write(json.dumps(out) + "\n")


def _e2e(impl: str, model, req: dict, device) -> dict:
    """Raw points -> per-point Sonata features (1088-d, Sonata's PCA demo upcast) on the input
    points: prepare / transform, forward, upcast, map back; wall time with CUDA syncs."""
    import numpy as np
    import torch

    path = RAW_E2E[req["case"]] or raw_dir() / f"{req['case']}.npz"
    data = dict(np.load(path))
    data = {k: data[k].astype(np.float32) for k in ("coord", "color", "normal")}
    data["normal"] = np.nan_to_num(data["normal"])
    rows = []
    for i in range(req.get("iters", 3) + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if impl == "new":
            import point_transformer_v3 as ptv3

            point, inverse = ptv3.prepare(
                data["coord"], data["color"], data["normal"], preset="sonata", device=device
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.inference_mode():
                feat = ptv3.upcast(model(point), levels=2).feat[inverse]
        else:
            import sonata

            np.random.seed(0)
            point = sonata.transform.default()({k: v.copy() for k, v in data.items()})
            point = {k: v.to(device) if torch.is_tensor(v) else v for k, v in point.items()}
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.inference_mode():
                point = model(point)
                for _ in range(2):
                    parent, inv = point.pop("pooling_parent"), point.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, point.feat[inv]], dim=-1)
                    point = parent
                while "pooling_parent" in point.keys():  # noqa: SIM118 (addict Dict)
                    parent, inv = point.pop("pooling_parent"), point.pop("pooling_inverse")
                    parent.feat = point.feat[inv]
                    point = parent
                feat = point.feat[point.inverse]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i:  # the first run is a warm-up
            rows.append(
                {
                    "prepare_ms": (t1 - t0) * 1e3,
                    "model_ms": (t2 - t1) * 1e3,
                    "total_ms": (t2 - t0) * 1e3,
                }
            )
        del feat
    return {
        "raw_points": len(data["coord"]),
        **{k: statistics.median(r[k] for r in rows) for k in rows[0]},
    }


RAW_E2E: dict[str, Path | None] = {  # None: in PTV3_BENCH_RAW
    "sample1": Path("~/.cache/sonata/data/sample1.npz").expanduser(),  # sonata.data.load's copy
    "pp_1100k": None,
}


def _build(impl: str, name: str, mode: str, device):
    import torch

    if impl == "new":
        import point_transformer_v3 as ptv3

        return ptv3.load(name, device=device, warmup=False, head=False)
    sys.path[:0] = [str(CODE), str(CODE / "sonata_orig")]
    flash = mode != "fp32"
    if name == "ptv3_scannet":
        from ptv3_orig.model import PointTransformerV3

        sys.path.insert(0, str(ROOT / "tests" / "parity"))
        from common import PTV3_RELEASE

        state = torch.load(PTV3_RELEASE, map_location="cpu", weights_only=False)["state_dict"]
        backbone = {
            k.removeprefix("module.backbone."): v
            for k, v in state.items()
            if k.startswith("module.backbone.")
        }
        model = PointTransformerV3(enable_flash=flash)
        model.load_state_dict(backbone)
        return model.to(device).eval()
    import sonata

    cfg = None if flash else {"enable_flash": False}
    return sonata.load(str(SONATA_CKPT / f"{name}.pth"), custom_config=cfg).to(device).eval()


# ---- driver


class Worker:
    def __init__(self, impl: str, device: str, env: dict | None = None) -> None:
        python = sys.executable if impl == "new" else orig_python()
        cmd = [python, "-B", str(Path(__file__).resolve()), "worker", impl, device]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})},
        )
        self.impl = impl
        assert json.loads(self.proc.stdout.readline())["ready"]

    def ask(self, **req) -> dict:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self) -> None:
        self.proc.stdin.write('{"op": "quit"}\n')
        self.proc.stdin.flush()
        self.proc.wait(timeout=120)


def gpu() -> dict:
    make_inputs()
    workers = [Worker("new", "cuda"), Worker("orig", "cuda")]
    results: dict = {}
    cells = [
        (m, c, mode)
        for m in MODELS
        for c in CASES
        for mode in MODES
        if m != "sonata_small" or c in ("sn_150k", "sn_max", "pp_1100k")
    ]
    for rnd in range(2):
        for model, case, mode in cells:
            order = workers if (rnd + len(results)) % 2 == 0 else workers[::-1]
            for w in order:
                key = f"{model}|{case}|{mode}|{w.impl}"
                if w.impl == "orig" and mode == "fp32" and case in ORIG_FP32_SKIP:
                    results[key] = {"ms": [], "error": ORIG_FP32_SKIP[case]}
                    continue
                out = w.ask(
                    op="fwd",
                    model=model,
                    case=case,
                    mode=mode,
                    warmup=3 if rnd == 0 else 1,
                    iters=5,
                    peak=rnd == 1,
                )
                cell = results.setdefault(key, {"ms": [], "points": out["points"]})
                cell.setdefault("points", out["points"])
                if "error" in out:
                    cell["error"] = out["error"]
                    continue
                cell["ms"] += out["ms"]
                for k in ("peak_mib", "weights_mib"):
                    if k in out:
                        cell[k] = out[k]
            print(f"round {rnd} {model} {case} {mode}", file=sys.stderr, flush=True)
    for case in RAW_E2E:
        for w in workers:
            results[f"e2e|{case}|{w.impl}"] = w.ask(op="e2e", case=case, iters=3)
            print("e2e", case, w.impl, results[f"e2e|{case}|{w.impl}"], flush=True)
    for w in workers:
        w.close()
    for cell in results.values():
        if cell.get("ms"):
            cell["median_ms"] = statistics.median(cell["ms"])
    return results


def peaks() -> dict:
    """Peak memory only (one forward per cell after one warm-up, garbage collected first)."""
    make_inputs()
    workers = [Worker("new", "cuda"), Worker("orig", "cuda")]
    results: dict = {}
    for model in MODELS:
        for case in CASES:
            if model == "sonata_small" and case not in ("sn_150k", "sn_max", "pp_1100k"):
                continue
            for mode in MODES:
                for w in workers:
                    if w.impl == "orig" and (
                        (mode == "fp32" and case in ORIG_FP32_SKIP)
                        or (mode == "bf16" and model == "ptv3_scannet")
                    ):
                        continue
                    out = w.ask(
                        op="fwd", model=model, case=case, mode=mode, warmup=1, iters=0, peak=True
                    )
                    results[f"{model}|{case}|{mode}|{w.impl}"] = out
            print("peaks", model, case, file=sys.stderr, flush=True)
    for w in workers:
        w.close()
    return results


def cpu() -> dict:
    """This package only (the original cannot run on CPU): fp32 at 16 and 32 threads."""
    make_inputs()
    results: dict = {}
    for threads in (16, 32):
        w = Worker("new", "cpu", env={"OMP_NUM_THREADS": str(threads), "CUDA_VISIBLE_DEVICES": ""})
        for case in ("sn_50k", "sn_150k", "sn_max"):
            for mode in ("fp32", "bf16") if case == "sn_150k" and threads == 32 else ("fp32",):
                load = os.getloadavg()[0]
                out = w.ask(op="fwd", model="sonata", case=case, mode=mode, warmup=1, iters=5)
                results[f"sonata|{case}|{mode}|{threads}t"] = {
                    **out,
                    "median_ms": statistics.median(out["ms"]),
                    "loadavg": load,
                }
                print(case, mode, threads, statistics.median(out["ms"]), flush=True)
        w.close()
    return results


def coldstart() -> dict:
    """Import, load, warm-up and first-forward costs in fresh processes (GPU)."""
    import tempfile

    make_inputs()
    probe = r"""
import json, os, sys, time
t0 = time.perf_counter()
import point_transformer_v3 as ptv3
import torch
t1 = time.perf_counter()
warm = sys.argv[1] == "1"
model = ptv3.load("sonata", device="cuda", warmup=False, head=False)
torch.cuda.synchronize(); t2 = time.perf_counter()
w = ptv3.warmup(model) if warm else 0.0
import numpy as np
z = np.load(sys.argv[2])
point = {k: torch.from_numpy(z[k]).cuda() for k in z.files}
torch.cuda.synchronize()
times = []
for i in range(4):
    t = time.perf_counter()
    with torch.inference_mode():
        model(point)
    torch.cuda.synchronize(); times.append((time.perf_counter() - t) * 1e3)
print(json.dumps({"import_s": t1 - t0, "load_s": t2 - t1, "warmup_s": w, "first_ms": times[0],
                  "steady_ms": min(times[1:])}))
"""
    results: dict = {}
    data = str(CACHE / "sonata-sn_150k.npz")
    with tempfile.TemporaryDirectory() as empty:
        for label, cache, warm in (
            ("cold triton cache, warmup", empty, "1"),
            ("warm triton cache, warmup", None, "1"),
            ("warm triton cache, no warmup", None, "0"),
        ):
            env = {**os.environ}
            if cache:
                env["TRITON_CACHE_DIR"] = cache
            out = subprocess.run(
                [sys.executable, "-c", probe, warm, data],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            results[label] = json.loads(out.stdout.strip().splitlines()[-1])
            print(label, results[label], flush=True)
    orig_probe = (
        "import json, sys, time\n"
        f"sys.path[:0] = ['{CODE / 'sonata_orig'}']\n"
        "t0 = time.perf_counter()\n"
        "import sonata, torch, numpy as np\n"
        "t1 = time.perf_counter()\n"
        f"model = sonata.load('{SONATA_CKPT / 'sonata.pth'}').cuda().eval()\n"
        "torch.cuda.synchronize(); t2 = time.perf_counter()\n"
        "z = np.load(sys.argv[1])\n"
        "point = {k: torch.from_numpy(z[k]).cuda() for k in z.files}\n"
        "times = []\n"
        "for i in range(4):\n"
        "    t = time.perf_counter()\n"
        "    with torch.no_grad():\n"
        "        model(dict(point))\n"
        "    torch.cuda.synchronize(); times.append((time.perf_counter() - t) * 1e3)\n"
        "print(json.dumps({'import_s': t1 - t0, 'load_s': t2 - t1, 'first_ms': times[0],"
        " 'steady_ms': min(times[1:])}))\n"
    )
    out = subprocess.run(
        [orig_python(), "-B", "-c", orig_probe, data],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    results["original (18c09ff), no warm-up"] = json.loads(out.stdout.strip().splitlines()[-1])
    print("original", results["original (18c09ff), no warm-up"], flush=True)
    imports = {}
    for impl, cmd in (
        ("new", [sys.executable, "-c", "import point_transformer_v3, torch"]),
        (
            "orig",
            [
                orig_python(),
                "-B",
                "-c",
                f"import sys; sys.path[:0] = ['{CODE / 'sonata_orig'}']; import sonata",
            ],
        ),
    ):
        times = []
        for _ in range(3):
            t = time.perf_counter()
            subprocess.run(cmd, check=True, capture_output=True)
            times.append(time.perf_counter() - t)
        imports[impl] = statistics.median(times)
    results["import_process_s"] = imports
    print("imports", imports, flush=True)
    return results


def table(results: dict) -> None:
    rows = {}
    for key, cell in results.items():
        parts = key.split("|")
        if len(parts) != 4 or parts[3] not in ("new", "orig"):
            continue
        rows.setdefault(tuple(parts[:3]), {})[parts[3]] = cell
    print(
        f"{'model':13s} {'case':10s} {'points':>9s} {'mode':8s} {'orig ms':>8s} {'new ms':>8s} "
        f"{'speedup':>7s} {'orig MiB':>9s} {'new MiB':>8s}"
    )
    for (model, case, mode), cell in rows.items():
        o, n = cell.get("orig", {}), cell.get("new", {})
        if "median_ms" not in n:
            continue
        if "median_ms" not in o:
            print(
                f"{model:13s} {case:10s} {n['points']:9d} {mode:8s} {'-':>8s} "
                f"{n['median_ms']:8.1f}  original: {o.get('error', 'n/a')[:60]}"
            )
            continue
        print(
            f"{model:13s} {case:10s} {n['points']:9d} {mode:8s} {o['median_ms']:8.1f} "
            f"{n['median_ms']:8.1f} {o['median_ms'] / n['median_ms']:6.2f}x "
            f"{o.get('peak_mib', 0):9.0f} {n.get('peak_mib', 0):8.0f}"
        )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("gpu", "peaks", "cpu", "coldstart", "inputs"))
    args = parser.parse_args()
    if args.what == "inputs":
        make_inputs()
        return
    results = {"gpu": gpu, "peaks": peaks, "cpu": cpu, "coldstart": coldstart}[args.what]()
    out = ROOT / "benchmarks" / "results" / f"{args.what}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    if args.what == "gpu":
        table(results)
    print("wrote", out)


if __name__ == "__main__":
    main()
