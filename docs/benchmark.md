# Benchmark

This package against the original implementations on one RTX PRO 6000 (Blackwell, sm_120, 96 GB)
under WSL2, 2026-10-30. Script: `benchmarks/bench.py`; raw results: `benchmarks/results/*.json`.

**Running it.** The script needs, as environment variables:
- `PTV3_BENCH_RAW`: a directory of raw scans `<case>.npz` with float `coord`, `color` (0-255) and
  `normal`: `sn_50k`, `sn_150k`, `sn_max` (ScanNet scenes of about 50k, 150k and the largest
  val scene), `pp_430k`, `pp_1100k` (ScanNet++), and `sn_batch_0` ... `sn_batch_7` (the
  ScanNet scenes of the batch cases).
- `PTV3_BENCH_ORIG_PYTHON`: the python of an environment with the originals' dependencies (torch,
  spconv, torch_scatter, flash_attn, timm, addict).
- `PTV3_PARITY_ROOT` (optional, default `~/.cache/pointtransformerv3/parity`): where
  `tests/parity/dump_original.py` extracted the original code (`code/`), which the `orig` worker
  imports; `PTV3_BENCH_CACHE` (optional, default `~/.cache/pointtransformerv3/bench`): the prepared
  inputs (`bench.py inputs`).
The Sonata checkpoints and demo scan are read from `~/.cache/sonata`, where the Sonata package
downloads them.

**Setup.**
- Original: this repository's `model.py` at 3229e9b (PTv3, with the released ScanNet weights) and
  Sonata at 18c09ff, in torch 2.9.1 + CUDA 12.8, spconv 2.3.8, flash_attn 2.8.3. New:
  `pointtransformerv3`, torch 2.12.1 + CUDA 13.0, Triton 3.7.1. The torch versions differ, so part
  of every gap is torch itself.
- Two persistent worker processes, one per environment, take turns on every cell; both get the
  identical grid-sampled inputs (`ptv3.prepare`, 2 cm). Times are CUDA events around
  `model(point)`, 3 warm-up and 10 timed forwards in two rounds, median. Peak memory is
  `max_memory_allocated` during one forward minus the allocation before it (weights excluded),
  after a garbage collection. No time scaling.
- The host is shared with other jobs; repeated sessions moved single cells by up to about 10%
  (small inputs, which are bound by host-side launch overhead, move most).
- Modes. **default**: the new package's default (fp16 autocast) against the originals' default
  (fp32 with fp16 FlashAttention, `enable_flash=True`). **fp32**: new `precision="fp32"` against
  `enable_flash=False`. **bf16**: autocast on both sides.
- Not run: the originals' fp32 mode on 1.13M points (its dense per-patch attention matrices need
  about 40 GB) and on batches (it uses the smallest sample's patch size for the whole batch);
  bf16 for the original PTv3 (spconv 2.3.8 has no bf16 kernel for its convs: "can't find
  suitable algorithm").

## Forward time (ms, median) and peak memory

Original → new; speed-up in brackets.

### PTv3 (`ptv3_scannet` vs `model.py` with the release weights)

| input | default ms | fp32 ms | bf16 ms | peak MiB default | peak MiB fp32 |
|---|---|---|---|---|---|
| ScanNet 50k | 45 → 28 (1.6x) | 66 → 36 (1.8x) | – → 29 | 193 → 164 | 1700 → 243 |
| ScanNet 150k | 53 → 31 (1.7x) | 102 → 48 (2.1x) | – → 28 | 577 → 435 | 5100 → 732 |
| ScanNet 298k | 71 → 31 (2.3x) | 172 → 88 (2.0x) | – → 35 | 1142 → 824 | 10095 → 1440 |
| ScanNet++ 432k | 86 → 35 (2.4x) | 198 → 99 (2.0x) | – → 33 | 1634 → 1153 | 14618 → 2046 |
| ScanNet++ 1.13M | 201 → 46 (4.4x) | – → 243 | – → 46 | 4272 → 2969 | – → 5350 |
| ScanNet, batch of 4 (365k) | 90 → 32 (2.8x) | – → 101 | – → 31 | 1399 → 1006 | – → 1775 |
| ScanNet, batch of 8 (883k) | 202 → 45 (4.5x) | – → 230 | – → 43 | 3380 → 2379 | – → 4266 |

### Sonata (`sonata` vs 18c09ff)

| input | default ms | fp32 ms | bf16 ms | peak MiB default | peak MiB fp32 |
|---|---|---|---|---|---|
| ScanNet 50k | 74 → 35 (2.1x) | 98 → 42 (2.3x) | 70 → 34 (2.1x) | 104 → 108 | 1236 → 144 |
| ScanNet 150k | 87 → 35 (2.5x) | 152 → 68 (2.2x) | 87 → 34 (2.5x) | 312 → 226 | 3707 → 439 |
| ScanNet 298k | 112 → 35 (3.2x) | 217 → 119 (1.8x) | 101 → 35 (2.9x) | 617 → 444 | 7333 → 857 |
| ScanNet++ 432k | 117 → 34 (3.4x) | 244 → 132 (1.8x) | 108 → 33 (3.3x) | 897 → 630 | 10636 → 1241 |
| ScanNet++ 1.13M | 271 → 58 (4.7x) | – → 333 | 247 → 55 (4.5x) | 2342 → 1647 | – → 3226 |
| ScanNet, batch of 4 (365k) | 145 → 36 (4.0x) | – → 147 | 128 → 34 (3.8x) | 757 → 544 | – → 1055 |
| ScanNet, batch of 8 (883k) | 285 → 53 (5.4x) | – → 334 | 258 → 53 (4.9x) | 1829 → 1304 | – → 2541 |

### Sonata small (`sonata_small` vs 18c09ff)

| input | default ms | fp32 ms | bf16 ms | peak MiB default | peak MiB fp32 |
|---|---|---|---|---|---|
| ScanNet 150k | 66 → 24 (2.7x) | 85 → 30 (2.8x) | 64 → 22 (2.9x) | 278 → 162 | 2484 → 304 |
| ScanNet 298k | 70 → 21 (3.3x) | 120 → 48 (2.5x) | 73 → 22 (3.4x) | 546 → 316 | 4914 → 596 |
| ScanNet++ 1.13M | 172 → 25 (7.0x) | – → 118 | 158 → 24 (6.6x) | 2074 → 1171 | – → 2232 |

### Concerto large and Utonia (default mode)

A separate session with the same method (two interleaved workers, 3 warm-up and 10 timed forwards
in two rounds, median) and the same grid-sampled inputs; for Utonia the coordinates are scaled by
0.5 on a 1 cm grid, which gives the same grid cells. Originals: Concerto through Sonata's code at
18c09ff (Concerto's inference code is Sonata's; `concerto` has Sonata's architecture, so the
Sonata table applies to it), Utonia at be53e4a (fp32 with bf16 FlashAttention). These two models
are not in `bench.py` yet.

| input | `concerto_large` ms | `utonia` ms | peak MiB `concerto_large` | peak MiB `utonia` |
|---|---|---|---|---|
| ScanNet 150k | 101 → 35 (2.9x) | 108 → 44 (2.5x) | 403 → 286 | 486 → 396 |
| ScanNet 298k | 139 → 38 (3.7x) | 153 → 42 (3.6x) | 799 → 558 | 954 → 782 |
| ScanNet++ 1.13M | 337 → 76 (4.4x) | 419 → 118 (3.5x) | 3031 → 2094 | 3616 → 2940 |

Utonia is slower than `concerto_large` at 1.13M points mostly because of its 3D RoPE, which runs
in plain torch: with `Rope3d` stubbed out (timing only) its forward drops from about 111 to
76-81 ms.

Small inputs are bound by host-side launch overhead (about 30 ms for the PTv3 U-Net at any size
up to 300k points); the gap grows with the input because the Triton conv and attention kernels
scale better than spconv + FlashAttention with their gathers.

## End to end (raw points to per-point Sonata features)

Prepare (center + grid sampling), forward, upcast (Sonata's PCA-demo upcast, 1088-d) and map back
to every raw point; wall time with CUDA syncs, median of 3 after one warm-up. The original runs
Sonata's numpy transform on the CPU; `ptv3.prepare` runs on the GPU.

| scan | raw points | original: transform / model / total | new: prepare / model / total | speed-up |
|---|---|---|---|---|
| sample1 | 273,530 | 66 / 96 / 160 ms | 2.6 / 30 / 33 ms | 4.9x |
| ScanNet++ 578511c8a9 | 4,775,139 | 1124 / 336 / 1460 ms | 15 / 67 / 82 ms | 17.8x |

## Cold start

Fresh processes, Sonata, ScanNet 150k input (`benchmarks/bench.py coldstart`). "First" is the
first real forward after `load`; "steady" the best of the next three.

| | import | load (cached weights) | warm-up | first forward | steady |
|---|---|---|---|---|---|
| original (18c09ff) | 5.6 s | 1.3 s | - | 509 ms | 87 ms |
| new, empty Triton cache, `load()` (warm-up on) | 0.8 s | 0.45 s | 5.0 s | 37 ms | 28 ms |
| new, warm Triton cache, warm-up on | 0.7 s | 0.43 s | 1.6 s | 36 ms | 34 ms |
| new, warm Triton cache, `warmup=False` | 0.7 s | 0.40 s | - | 1295 ms | 36 ms |

`import pointtransformerv3, torch` in a new process takes 0.95 s (`import sonata`: 5.9 s). Loading
memory-maps the checkpoint and converts it on the meta-built model; the sha256 is checked once per
file and remembered. The warm-up compiles every Triton kernel variant the model needs (row counts
are not specialized, see docs/acceleration.md), so the first real input runs at steady-state
speed; without it, the first forward pays about 1.3 s of kernel loading and compilation.

## CPU (new package only; the original cannot run on a CPU)

spconv's default algorithm needs CUDA, and with `ConvAlgo.Native` spconv 2.3.8 asserts that bias
is unsupported on CPU, so the original Sonata does not run there at all. This package runs the
pure-torch paths. Sonata, fp32, median of 5 (load average 8-20 from other jobs on the 32-core
host, so these are upper bounds):

| input | 16 threads | 32 threads |
|---|---|---|
| ScanNet 50k | 0.88 s | 0.91 s |
| ScanNet 150k | 2.37 s | 2.55 s |
| ScanNet 298k | 4.88 s | 5.33 s |

bf16 autocast on the CPU is 3.2x slower (8.1 s at 150k): keep CPU inference in fp32 (the
default). 32 threads are no faster than 16 here.

## Install footprint

`uv sync` from a warm uv cache takes about 0.5 s; the environment holds torch (CUDA 13 wheels),
triton, numpy and huggingface_hub, no compiler toolchain, no spconv / flash-attn / torch-scatter
builds.
