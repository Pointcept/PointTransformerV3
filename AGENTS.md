# AGENTS.md: using and working on `pointtransformerv3`

## Using the package

```python
import pointtransformerv3 as ptv3

model = ptv3.load("sonata")                                # eval mode, on cuda/mps/cpu, warmed up
feat = model.embed(coord, color, normal)                   # (N, 1232) for the N input points
logits = ptv3.load("ptv3_scannet").predict(coord, color, normal)   # (N, 20)
```

Input contract (raw points, numpy or torch):
- `coord` `(N, 3)` float, meters (ScanNet-like scale; the grid is 2 cm, Utonia's 1 cm after
  scaling by 0.5).
- `color` `(N, 3)` RGB in **[0, 255]** (`prepare` scales it per preset: `/255` for the encoders,
  `/127.5 - 1` for `ptv3_scannet`).
- `normal` `(N, 3)` unit normals. All three are needed by the released presets; Concerto and
  Utonia accept zeros for a missing color or normal.
- Utonia's `scale` (a `prepare` / `embed` keyword) sets the density it sees: 0.5 for indoor scenes
  (preset default), 0.2 for outdoor LiDAR with `center=False` (ego vehicle at the origin, road on
  the xy-plane); larger scales give finer features.

Lower level, for batching and custom pipelines:

```python
point, inverse = ptv3.prepare(coord, color, normal, preset="sonata", device="cuda")
batch = ptv3.collate([point_a, point_b])         # offsets = cumulative sample ends
out = model(batch)                               # a ptv3.Point (dict with attribute access)
out = ptv3.upcast(out)                           # encoders: back to level 0, levels concatenated
feat_a = out.feat[: out.offset[0]][inverse_a]    # per sample, then to its input points
```

Rules that matter:
- Call `model.eval()` for inference (`load` does). In training mode orders are shuffled and
  DropPath is active.
- `grid_coord` must be unique per sample and non-negative (grid-sample first; `prepare` does).
- Rows of a batch are sample-contiguous; `offset` holds cumulative sample ends.
- With the released presets the serialization depth follows the largest sample of a batch
  (`"auto"`): a sample's features can differ slightly between a batched and a single run. Use an
  int `serialization_depth` when that matters.
- `precision="auto"` is fp16 autocast on CUDA; pass `precision="fp32"` for exact numerics.
- First call: `load` warms up on CUDA (Triton compiles or loads its kernels: 5 s with an empty
  `~/.triton/cache`, 1.3-2.2 s with a warm one; then one scan-sized `embed` runs the
  preprocessing and grows the memory pool), so the first real `embed` runs at steady speed.
  `warmup=False` defers that cost to the first call (about 1.3 s); `ptv3.warmup(model)` runs it
  later. Details: docs/acceleration.md, "Cold start and warm-up".
- Licenses: the code is CC BY-NC 4.0 + a research permission (LICENSE, NOTICE). Weights: PTv3
  ScanNet MIT; Sonata (Meta) CC BY-NC 4.0, non-commercial only;
  Concerto and Utonia (Pointcept) CC BY-NC 4.0 with the code's research permission.

## Demos

`demo/` holds the Sonata, Concerto and Utonia demos ported to this package (README "Demo").
They run on the public API only (`load`, `prepare`, `collate`, `upcast`, `embed`, `predict`);
per-domain settings are `prepare` arguments in the demo code, never package changes. Shared
helpers are in `demo/_common.py` (samples from the Hugging Face dataset `pointcept/demo` at a
pinned revision, `pca_color`, `heat_color`, PLY and PNG writers, the optional open3d viewer).
`pca_color` is an exact PCA with a fixed sign per component, so colors do not depend on the
device or a seed; the demos pass per-model `signs` that reproduce the originals' colors (fitted
against the originals' own output on the same sample; other presets keep the default signs).

| demo | default sample | models | settings |
|---|---|---|---|
| `0_pca`, `1_similarity`, `3_batch` | `sample1` (ScanNet scan) | every encoder preset (default `utonia`) | the preset's |
| `2_sem_seg` | `sample1` | presets with a `scannet20` head | the preset's |
| `4_pca_hk` | `sample5_hk` (40M points) | `utonia*` | `scale=0.0015` |
| `5_pca_manipulation` | `sample4_manipulation` | `utonia*` | `scale=4` |
| `6_pca_object` | `sample3_object` | `utonia*` | `scale=1`, normals estimated, `--normalize` |
| `7_pca_outdoor` | `sample2_outdoor_multiframe` / `sample2_outdoor` | `utonia*`, `concerto_large_outdoor` | `scale=0.2, center=False` |
| `8_pca_video` | a video or images, lifted by VGGT | `utonia*` (scale 1), `concerto*`, `sonata*` | `--tsdf`: open3d `VoxelBlockGrid` surface points |

`concerto_large_outdoor` has no preset: the demo downloads its checkpoint at a pinned revision,
checks its sha256, loads it with `ptv3.load("concerto_large", weights=...)` and centers x and y
after grid sampling, as Concerto's transform does. VGGT is never a dependency or vendored;
`8_pca_video` imports it only when the user installed it. Adapted demo code keeps its Apache-2.0
notices (`demo/THIRD_PARTY_LICENSES.md`, NOTICE). Outputs go to `demo/output/` (ignored by git).
Lint the demos explicitly: `uv run ruff check demo && uv run pyright demo` (pyright's default
scope is `src/`).

## Presets

One model class (`PointTransformerV3`) and one `Config` dataclass; every released model is a
preset in `src/pointtransformerv3/presets.py`: the full `Config`, the weight source (Hugging Face
repo, file, pinned revision, sha256, license), the preparation (`PrepareSpec`: grid size, feature
columns, color range, coordinate scale) and its heads.

| preset | what | weights |
|---|---|---|
| `ptv3` | PTv3 ScanNet architecture, random init | - |
| `ptv3_scannet` | PTv3 + the released ScanNet-20 weights (legacy CPE switches on) | Pointcept/PointTransformerV3, MIT |
| `sonata`, `sonata_small` | Sonata encoders (Linear stem, LayerNorm, encoder only); `sonata` has the linear probe `scannet20` | facebook/sonata, CC BY-NC 4.0 (Meta) |
| `concerto`, `concerto_large`, `concerto_siglip`, `concerto_origin` | Concerto encoders: Sonata's architecture (large: channels 64-768); `concerto_large` has `scannet20` | Pointcept/Concerto, CC BY-NC 4.0 + permission |
| `utonia`, `utonia_small`, `utonia_tiny` | Utonia encoders: Sonata's architecture + 3D RoPE (`rope_base=10`, head_dim 18), 1 cm grid after scaling by 0.5; `utonia` has `scannet20` | Pointcept/Utonia, CC BY-NC 4.0 + permission |

Utonia is `Config.rope_base` (default None): every attention gets a `modules.Rope3d` that
rotates q and k in plain torch (fp32 angles; bit-identical to the original `Point3DRoPE` in fp32)
with the level's coordinates, between the qkv Linear and the unchanged attention backends. Its
head_dim 18 runs on `triton_d16` in a masked 32-lane block (other non-power-of-two head dims
<= 64 likewise); without Triton varlen zero-pads it to 24, and SDPA (fp32, CPU) takes it as
is. The torch RoPE is about 30% of Utonia's forward at 1.13M points: fusing it into the kernel is
the next optimization.

`PointTransformerV3.from_preset(name, **overrides)` builds a randomly initialized model of a
preset with changed options (for example `drop_path=0.0`); `ptv3.load(name)` builds it with the
released weights.

A new release slots in the same way (docs/presets.md has the checklist; Utonia's RoPE is the
worked example): express the backbone as `Config` values; a genuinely new behavior becomes a new
`Config` field whose default keeps every existing preset unchanged (implemented in `modules.py`,
never a second model class); add the checkpoint's key names to `convert.RULES` (strict: every key
mapped once, every parameter filled); add the preset with its `WeightSource`, `PrepareSpec` and
heads; prove it with a parity case against the original code, a `weights` test and a golden
output; document the weights' license in the README model zoo and in NOTICE.

## Working on the package

```bash
uv sync                                   # .venv in the repo; torch from PyPI (CUDA 13 on Linux)
uv run pytest                             # CPU tests (default markers)
uv run pytest -m gpu                      # CUDA tests (tiny inputs; Triton kernels vs torch)
uv run pytest -m weights                  # strict conversion and loading of released weights
uv run ruff check && uv run ruff format --check
uv run --group parity python -m tests.parity.check   # parity: tests/parity/README.md (env vars)
python benchmarks/bench.py gpu                        # benchmark: docs/benchmark.md (env vars)
```

Layout: `src/pointtransformerv3/{model,modules,presets,convert,weights,api,preprocess,heads,
structure}.py`, `layers/` (norms, sparse conv module, init), `ops/` (functional ops and their
Triton kernels). Most files are vendored from Pointcept2 at 771a949; each starts with a
provenance line. Keep them close to the source so fixes can flow both ways. Their docstrings
cite Pointcept2's design notes (C/D/I numbers such as C10 or D23, and `research/...` measurement
reports); those refer to the Pointcept2 repository, not to files here.

Conventions:
- One model class; new releases are presets (docs/presets.md).
- Index dtypes: rows int64, `cu_seqlens` and kernel maps int32, `grid_coord` int32, `coord`
  float32.
- No host sync inside blocks, attention, conv or pooling; `PTV3_DEBUG=1` enables the
  checks that need one.
- Importing the package imports neither triton nor huggingface_hub (tested).
- GPU safety: run new or changed Triton kernels on tiny inputs against a CPU reference first;
  keep single launches short; never autotune at runtime.
