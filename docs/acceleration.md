# Acceleration: backends, precision, numerics

Everything runs in plain PyTorch on any device. On CUDA, three kinds of Triton kernels replace
the slow parts; each op chooses its backend per call and logs the choice once.

| op | backends (`auto` order) | environment variable |
|---|---|---|
| serialized attention | `triton_d16` (fused patch attention, head_dim <= 64: 16/32/64 exact blocks, others such as Utonia's 18 in a masked block of the next power of two; fp16/bf16, sm80+) -> `varlen` (`torch.nn.attention.varlen`, FlashAttention-2, fp16/bf16) -> `sdpa` (padded `scaled_dot_product_attention`, any device and dtype) | `PTV3_ATTENTION_BACKEND` |
| submanifold sparse conv | `triton` (implicit GEMM, fp32/fp16/bf16, sm80+) -> `torch` (gather + GEMM, any device) | `PTV3_CONV_BACKEND` |
| LayerNorm | `triton` (fp32 statistics, output in the activation dtype) -> `torch` | `PTV3_LAYER_NORM_BACKEND` |

```python
import pointtransformerv3 as ptv3
ptv3.set_backend(attention="sdpa", conv="torch", layer_norm="torch")   # e.g. the exact fp32 path
```

`PTV3_DEBUG=1` turns on the value checks that need a host sync (index ranges, kernel
map symmetry, patch tables) before every kernel launch.

**Utonia.** Its head_dim of 18 runs on `triton_d16` in a 32-lane block whose extra lanes are
masked (zeros in q, k and v; addresses clamped inside the head), so scores and outputs are exact;
for head dims 16, 32 and 64 the kernel code is unchanged. Its 3D RoPE (`modules.Rope3d`) rotates
q and k in plain torch (fp32 angles) between the qkv Linear and the attention kernel. That costs
about 30% of Utonia's forward at 1.13M points (111 ms, 76-81 ms with the RoPE stubbed out), which
is why Utonia gains less than Concerto (docs/benchmark.md); fusing it into the kernel is the next
speed-up.

## Precision

`precision` is a model attribute (`load(..., precision=...)`, or `model.precision = ...`):

| value | CUDA | CPU / MPS |
|---|---|---|
| `"auto"` (default) | fp16 autocast in eval, bf16 autocast in training | fp32 |
| `"fp16"` / `"bf16"` | autocast in that dtype | autocast in that dtype (slow on CPU) |
| `"fp32"` | fp32 (attention falls back to SDPA and warns once) | fp32 |

An autocast the caller entered always wins. Under autocast, the Linear stem of Sonata runs in
fp32 (its input holds coordinates), the norms keep fp32 statistics and emit the activation dtype,
and the conv and attention kernels accumulate in fp32. For fine-tuning use bf16 (fp16 needs a
`GradScaler`); `"auto"` does that in training mode.

## Numerics (measured in tests/parity, docs/parity.md)

- fp32 against the original implementations: features within ~1e-5 relative per level; the
  attention kernels differ only by rounding.
- fp16 (the default on CUDA) against the originals' default (fp32 with fp16 attention): up to
  5e-3 relative per level (the original's own fp16 autocast: 3.8e-3), row cosine >= 0.9997,
  the same argmax on >= 99.87% of points, identical linear-probe mIoU. `precision="fp32"` when
  you need the original numerics to 1e-5.
- Concerto and Utonia (4 ScanNet val scenes): fp32 within 1.2e-5 per level; fp16 row cosine
  >= 0.99998 against the original fp32 (Utonia's own default, bf16 attention: 0.99995), probe
  argmax >= 99.976%, mIoU within 0.01.
- bf16: mean row cosine >= 0.9998 against the original fp32, worst single row 0.989 (8 mantissa
  bits); offered for training, where it needs no `GradScaler`.
- Triton conv: fp32 inputs use IEEE fp32 dot products (no TF32); backward without atomics
  (bitwise reproducible).
- No runtime autotuning: launch configurations are pinned per GPU and shape
  (`ops/sparse_conv/triton_configs.json`, measured on sm_120 for widths 32-512) or chosen by a
  generic rule elsewhere (logged once). Sonata's 48/96/192/384 widths use the generic rule.

## Cold start and warm-up

A new process pays four one-off costs before it runs at steady-state speed (Sonata, ScanNet 150k
input, fresh processes, `benchmarks/bench.py coldstart`, docs/benchmark.md):

| step | what happens | time |
|---|---|---|
| `import pointtransformerv3, torch` | torch only; triton and huggingface_hub are imported on first use | 0.7-0.95 s (original: 5.6-5.9 s) |
| `ptv3.load("sonata")`, weights cached | build on the meta device, memory-map the checkpoint, convert its keys, assign, move to the device; the sha256 is checked once per file and remembered | 0.4 s (original: 1.3 s) |
| warm-up (default on CUDA) | forwards on synthetic clouds of 4096, 4100, 40000 and 40004 grid-sampled points: Triton compiles or loads every kernel variant, cuBLAS initializes; then one `embed` on a synthetic raw scan of 600k points (about 260k grid cells): the preprocessing ops run once and the caching allocator grows to a scan's size | 5.0 s with an empty Triton cache, 1.3-2.2 s with a warm one |
| first real forward | | 37 ms after warm-up (steady 28-34 ms); 1.3 s without warm-up; original: 509 ms (steady 87 ms) |
| first real `embed` (demo scan sample1, 274k points, end to end) | | sonata 43 ms (steady 40-45), utonia 47 ms (43-55), ptv3_scannet 37 ms (32-37); with the forwards-only warm-up of the first pre-release: 159 ms |
| first real `embed` (ScanNet scene0011_00, 237k points; median of the next 10) | | concerto 36 ms (34), concerto_large 39 ms (38), utonia 44 ms (41), utonia_small 30-36 ms (30); load with warm-up 2.0-2.5 s |

Triton caches compiled kernels on disk (`~/.triton/cache`), so only the first process after an
install or a Triton upgrade pays the 5 s. The warm-up moves the remaining 1.3 s of kernel loading
and initialization from the first real input into `load`.

- `ptv3.load(...)` warms up on CUDA by default; on CPU and MPS it does not (nothing to compile).
- `ptv3.load(..., warmup=False)` skips it, for example in a script that loads many models;
  `ptv3.warmup(model)` runs it later, for example at server start-up.
- Training needs no warm-up: the first steps pay the compilation once.
- The warm-up covers `prepare`, the forward, `upcast` and the map back. Its scan-sized `embed`
  matters: with an 80k-point raw cloud the first sample1 `embed` still took 69 ms (allocator
  growth), with 600k points it runs at steady speed, also on scan sizes not seen before. The
  pool it grows (about 6.6 GiB reserved for Sonata) is what a scan of that size needs anyway;
  `torch.cuda.empty_cache()` returns it. `ptv3.warmup(model, points=...)` sizes it for other data.
  A model loaded from a local checkpoint (no preset) warms up the forward only.

For the "first real input at steady-state speed" guarantee, the row-count arguments of the
attention, LayerNorm and kernel-map kernels are not specialized (`do_not_specialize`, as
Pointcept2's conv kernels already were): Triton would otherwise compile a separate variant for
every combination of "row count divisible by 16", "equals 1" and pointer alignment, which no
synthetic warm-up covers (with Pointcept2's kernels, 7 more variants compiled on the first real
scan, 1.3 s). The price, measured A/B in one session: steady-state forwards 0-7% slower on large
inputs (Sonata 1.13M points 55.3 -> 57.3 ms, batch of 8 52.4 -> 55.9 ms; PTv3 1.13M
45.0 -> 48.1 ms). `benchmarks/results/gpu_specialized_kernels.json` holds the full benchmark with
the specialized kernels.

## Without a working Triton

Triton needs a C compiler and `Python.h` once per Triton version, to build its launcher (a
uv-managed Python ships the headers). Without them, on Windows and on macOS, every op falls back
to its torch backend and warns once naming the missing piece. MPS works through the torch paths
(not measured here).

## Not supported yet

`torch.compile` of the whole model (`load(compile=True)` raises): the ops carry the
`torch.library` registrations for it, but the data-dependent level hierarchy has not been
validated under compilation in this package.
