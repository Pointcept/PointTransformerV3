# Parity against the original implementations

Two environments, two GPU sessions. External data and code come from environment variables
(each step checks the ones it needs and says what to set):

| variable | what | used by |
|---|---|---|
| `PTV3_PARITY_RAW` | a directory of raw ScanNet val scenes `<name>.npz` (float32 `coord`, `color`, `normal`; `segment20`) for scene0342_00, scene0665_01, scene0231_01 | inputs, P3/P4/P6 inputs |
| `PTV3_PARITY_SONATA_REPO` | a git clone of github.com/facebookresearch/sonata (with 18c09ff) | dump_original.py (first run) |
| `PTV3_PARITY_UTONIA_REPO` | a git clone of github.com/Pointcept/Utonia-Manifold (with be53e4a) | dump_original.py `utonia*` steps (first run) |
| `PTV3_PARITY_SCANNET` | processed ScanNet: `index.json` and one `.safetensors` file per scene | the full-val steps (`val`, `valpick`, `p4val`, `sonata_val_inputs`) |
| `PTV3_PARITY_ROOT` | outputs (optional; default `~/.cache/pointtransformerv3/parity`) | all |

The Sonata checkpoints and demo scan are read from `~/.cache/sonata` (where `sonata.load` and
`sonata.data.load` put them), the PTv3 release and the Utonia weights from the Hugging Face cache.

1. **Inputs** (this package, CPU): `python -m tests.parity.make_inputs` prepares the PTv3 cases
   (three ScanNet val scenes, a packed batch of the three, a tiny synthetic cloud) with
   `ptv3.prepare(preset="ptv3_scannet", pick="first")`.
2. **Originals** (an environment with torch 2.9.1, spconv, torch_scatter, flash_attn, timm,
   addict; never this package):

   ```bash
   flock /tmp/pointcept2-gpu.lock env PYTHONDONTWRITEBYTECODE=1 \
       <conda env>/bin/python -B tests/parity/dump_original.py
   ```

   It extracts `model.py` + `serialization/` at 3229e9b from this repository, Sonata at 18c09ff
   and Utonia at be53e4a from their clones into `$PTV3_PARITY_ROOT/code`, runs Sonata's and
   Utonia's own transforms on the raw scenes (numpy seeds fixed; Utonia's
   `transform.default(0.5)`), and dumps every level of every case (features, grid, batch,
   serialization codes, pooling inverses), PTv3 parameter gradients, the release weights in
   `model.py`, and the encoders' full upcast and linear-probe logits. Steps (default: all):
   `sonata_inputs ptv3 release sonata utonia_inputs utonia`.
3. **This package**: `flock /tmp/pointcept2-gpu.lock uv run --group parity python -m
   tests.parity.check p1 p2 p3 p4 p5 p6 val` and `CUDA_VISIBLE_DEVICES="" ... check p5cpu`.

The originals shuffle their serialization orders with `randperm` even in eval; `check.py`
derives each level's order permutation from the dumped codes and passes it to
`model(point, orders=...)`, so both sides attend along the same curves. Rows are matched by
(batch, grid_coord) because the two implementations order pooled rows differently. TF32 is off
on both sides; `torch.manual_seed` is set before every forward.

| check | what | gates |
|---|---|---|
| P1 | PTv3 vs model.py@3229e9b, seeded random weights (BatchNorm statistics and norm affines randomized), every encoder and decoder level | integers exact; fp32 (`enable_flash=False`) <= 1e-4 relative per level; grads <= 1e-3; default (fp16) cosine >= 0.999, argmax >= 99.5% |
| P2 | release weights: model.py vs `ptv3_scannet` with the stale-CPE switch off (numerics), and ScanNet val mIoU (312 scenes, single pass) with depth "auto" and 15 | within 0.1 of Pointcept2's measured 76.10 / 76.09 |
| P3 | Sonata and Sonata-small vs 18c09ff on sample1, three ScanNet val scenes and a batch of three | integers exact; fp32 <= 1e-4; cosine >= 0.999 (fp16), >= 0.995 (bf16); linear-probe argmax >= 99.5%, mIoU within 0.1 |
| P4 | `ptv3.prepare` vs Sonata's transform on the raw scenes | same grid cells; every point in the same cell; mIoU within 0.2 |
| P5 | CPU fp32 vs CUDA fp32 (this package) | <= 1e-4 |
| P6 | Utonia (3D RoPE, head_dim 18) vs be53e4a on sample1 and scene0342_00, as P3 | as P3 |

Outputs: `~/.cache/pointtransformerv3/parity/{inputs,orig}` (a few GB) and `report-*.json`.
Results: docs/parity.md.
