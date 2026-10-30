# Presets: one model class, many released configurations

`pointtransformerv3` has one model class, `PointTransformerV3`, and one `Config` dataclass. Every
released model (PTv3, Sonata, Concerto, Utonia) is a **preset**: a complete `Config`
plus where its weights come from, how raw points are prepared for it and which heads exist.
Later works become Config options of the same class, never new model classes. This keeps one
code path to test, accelerate and maintain.

## What a preset holds (`src/pointtransformerv3/presets.py`)

```python
Preset(
    name="sonata",
    config=SONATA,                         # every Config field, as a plain dict
    weights=WeightSource(repo="facebook/sonata", file="sonata.pth",
                         revision="df99897...", sha256="c5ced5ac...", license="CC-BY-NC-4.0"),
    prepare=PrepareSpec(grid_size=0.02, feat=("coord", "color", "normal")),
    heads={"scannet20": HeadSpec(num_classes=20, in_channels=1232, weights=...)},
)
```

| preset | architecture options | serialization depth | weights | license |
|---|---|---|---|---|
| `ptv3` | conv stem, BatchNorm, shifted codes, torch init | 15 | none (random init) | - |
| `ptv3_scannet` | as `ptv3` + `legacy_dec_cpe_skip_input` + `legacy_cpe_stale_input` | `"auto"` | Pointcept/PointTransformerV3 | MIT |
| `sonata` | Linear stem in fp32, LayerNorm, re-encoded levels, trunc-normal init, encoder only | `"auto+1"` | facebook/sonata `sonata.pth` | CC BY-NC 4.0 (Meta) |
| `sonata_small` | as `sonata`, depths (2,2,2,6,2), channels (32..512) | `"auto+1"` | facebook/sonata `sonata_small.pth` | CC BY-NC 4.0 (Meta) |
| `concerto`, `concerto_siglip`, `concerto_origin` | as `sonata` (Concerto's inference code is Sonata's) | `"auto+1"` | Pointcept/Concerto `concerto_base*.pth` | CC BY-NC 4.0 + permission |
| `concerto_large` | as `sonata`, channels (64..768), heads (4..48) | `"auto+1"` | Pointcept/Concerto `concerto_large.pth` | CC BY-NC 4.0 + permission |
| `utonia` | as `sonata` + `rope_base=10` (3D RoPE), channels (54..576), head_dim 18; prepared at 1 cm after scaling by 0.5 | `"auto+1"` | Pointcept/Utonia `utonia.pth` | CC BY-NC 4.0 + permission |
| `utonia_small`, `utonia_tiny` | as `utonia`, depths (2,2,2,6,2), channels (36..432) / (18..216) | `"auto+1"` | Pointcept/Utonia | CC BY-NC 4.0 + permission |

Concerto: the Hugging Face card names `concerto_base.pth` / `concerto_large.pth` its default models
(a pre-release of Utonia's recipe); `concerto_base_origin.pth` (preset `concerto_origin`) is the
paper's model and `concerto_base_siglip.pth` its SigLIP 2 variant. `concerto_small`,
`concerto_tiny` and `concerto_large_outdoor` (prepared like Utonia outdoors: x0.2, 1 cm, centered
after sampling, x/y only) are further checkpoints there without a preset yet.

## Adding a release

1. **Express the architecture as Config options.** Compare the new backbone with
   `PointTransformerV3` module by module. Anything that is a different value of an existing
   option (depths, channels, heads, patch sizes, `layer_scale`, `stem`, `norm`, `pool_policy`,
   `init`) needs no code. A genuinely new behavior becomes a new Config field whose default
   keeps every existing preset unchanged, implemented in `modules.py` as a new class or branch.
   Never fork `model.py`. Utonia is the worked example: `rope_base` (default None) adds
   `modules.Rope3d` to every attention, which rotates q and k on the level's rows before the
   unchanged attention backends; its checkpoint keys `attn.rope.inv_freq` are one more rule, and
   its constructor kwargs are PC version `PT-v3m3` in `convert.py`.
2. **Map the checkpoint keys.** Add the new names to `convert.RULES` (or `DROPPED` for
   training-only tensors such as mask tokens), so `convert_state_dict` maps every key exactly
   once and fills every parameter. The conversion is strict; a missed key is an error.
3. **Add the preset**: the full config dict, the `WeightSource` (repo, file, pinned revision,
   sha256 from the Hugging Face API, size, license), the `PrepareSpec` (grid size, feature
   columns, color range, coordinate scale) and the heads.
4. **Prove it**: a parity case in `tests/parity` against the original implementation (integers
   exact, fp32 within 1e-4 per level), a `weights` test for strict conversion, and the golden
   output in `tests/golden` (`python -m tests.golden.make_golden`).
5. **Document the license** of the weights in the README model zoo and in NOTICE. Weights are
   downloaded at runtime and never redistributed with this package.

## Serialization depth

The originals choose the serialization depth per batch: `max(grid_coord).bit_length()`
(`model.py`, Pointcept PT-v3m1) or `(max + 1).bit_length()` at every pooled level (Sonata,
Pointcept PT-v3m2). The presets with weights reproduce that (`"auto"` / `"auto+1"`), because the
Hilbert curve depends on the depth (`depth % 3`) and the weights were trained with it. The cost:
one host sync per forward, and a sample's result can depend on the other samples of its batch
(the depth follows the largest one). A fixed int depth (Pointcept2's recipe: 15) removes both,
at the price of different Hilbert orders on about one scene in five (the parity report measures
the effect on mIoU). Pick it with `PointTransformerV3.from_preset(name, serialization_depth=15)`.
