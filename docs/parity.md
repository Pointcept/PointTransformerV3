# Parity with the original implementations

This package was checked against the **original** code, run unmodified in its own environment:
this repository's `model.py` + `serialization/` at commit 3229e9b (PTv3),
facebookresearch/sonata at 18c09ff (Sonata, Concerto) and Pointcept/Utonia-Manifold at be53e4a
(Utonia), in torch 2.9.1 with spconv 2.3.8, torch_scatter, flash_attn 2.8.3 and numpy 1.26.
This package ran torch 2.12.1 (CUDA 13), Triton 3.7.1, on the same RTX PRO 6000 (sm_120, WSL2),
2026-10-30. Method and scripts: `tests/parity/` (README there).

Both originals shuffle their serialization orders with `randperm` even in eval. The harness reads
each level's order permutation off the original's dumped codes and passes it to
`model(point, orders=...)`, so both sides attend along the same curves; rows are matched by
(batch, grid_coord). `rel` = max |new - ref| / max |ref| over the worst level (every encoder and,
for PTv3, decoder level); `cos` = per-row cosine; integers = grid-cell sets, serialization codes and
pooling inverses of every level. TF32 off on both sides.

## Summary

| check | result | gate |
|---|---|---|
| integers (all PTv3 and Sonata cases, every level) | bit-exact | exact |
| PTv3 fp32 vs `model.py` `enable_flash=False` (random weights, 3 scenes + tiny) | rel <= 2.6e-6, output argmax 100% | <= 1e-4 |
| PTv3 gradients (train mode, 447 parameters) | rel <= 2.7e-4 (worst: a pooling projection, behind max pool), median 6.7e-6 | <= 1e-3 |
| PTv3 default (ours fp16 autocast vs `model.py` fp32 + fp16 flash attention), 5 cases incl. a batch of 3 | rel <= 3.0e-3, cos min 0.999996, argmax >= 99.87% | cos >= 0.999, argmax >= 99.5% |
| release weights in `model.py` vs ours (stale-CPE switch off on both sides) | fp32 rel 2.0e-5; default rel 1.1e-2, cos min 0.9997, argmax 99.9% | numerics only |
| `ptv3_scannet` ScanNet val mIoU (312 scenes, single pass, raw points) | **76.10** with Pointcept2's grid sampling (= Pointcept2's 76.10); 75.58 with `ptv3.prepare` defaults (see below) | within 0.1 of 76.10 |
| Sonata fp32 vs 18c09ff `enable_flash=False` (sample1 + 3 scenes) | rel <= 5.9e-6; 1232-d upcast <= 1.2e-6; linear-probe argmax 100% | <= 1e-4 |
| Sonata-small fp32 | rel <= 2.6e-5 | <= 1e-4 |
| Sonata default (ours fp16 vs original fp32 + fp16 flash attention), 5 cases incl. batch3 | rel <= 5.0e-3, upcast <= 1.5e-3, cos min 0.99974, argmax >= 99.98% | <= 1e-3 (see below) |
| Sonata fp16 vs original fp32 | cos min >= 0.99975, argmax >= 99.98% | cos >= 0.999 |
| Sonata bf16 vs original fp32 | cos mean >= 0.99979, cos min 0.9890-0.9975, argmax >= 99.84% | cos min >= 0.995 (missed on 1 of 4 scenes) |
| linear-probe mIoU per scene (sample1, 0342, 0665, 0231) | 49.2 / 51.4 / 66.7 / 76.0, identical to the original in every mode | within 0.1 |
| `ptv3.prepare` vs Sonata's transform | same grid cells on the 3 ScanNet scenes and on all 312 val scenes; every point in the same cell | exact |
| linear probe on ScanNet val, Sonata transform vs `ptv3.prepare` | 70.87 vs 70.92 mIoU | within 0.2 |
| CPU fp32 vs CUDA fp32 (this package, Sonata) | rel 3.3e-6 (CPU vs original: 3.3e-6) | <= 1e-4 |
| Utonia fp32 vs be53e4a `enable_flash=False` (3D RoPE, head_dim 18; sample1 and scene0342_00 through Utonia's `transform.default(0.5)`) | rel <= 3.0e-6, integers exact, 1386-d upcast <= 9.8e-7, linear-probe argmax 100%, mIoU 78.1 / 77.9 on both sides | <= 1e-4 |
| Utonia default (ours fp16 autocast, RoPE in fp32, triton_d16 with head_dim 18 padded to 32, vs original fp32 + bf16 flash attention) | rel <= 1.2e-2, cos min 0.99948, argmax >= 99.94%, mIoU within 0.03 | cos >= 0.999 |
| Utonia fp16 / bf16 vs original fp32 | fp16: cos min 0.99975, argmax >= 99.98%; bf16: cos min 0.991, argmax >= 99.83% | cos >= 0.999 / 0.995 (bf16 missed, as for Sonata) |
| second check, 4 more ScanNet val scenes (see Notes): `concerto` / `concerto_large` fp32 vs 18c09ff, `utonia` / `utonia_small` fp32 vs be53e4a | integers exact; rel <= 1.0e-5 / 1.2e-5 / 3.9e-6 / 9.7e-6; upcast <= 2.6e-6; probe argmax 100%; mIoU concerto_large 69.18, utonia 68.31 on both sides | <= 1e-4 |
| second check, fp16 / bf16 vs original fp32 | fp16: cos min >= 0.99998, argmax >= 99.976%; bf16: cos min >= 0.99929, argmax >= 99.87%; mIoU within 0.01 | cos >= 0.999 / 0.995 |

## Notes

- **Default-mode gate (Sonata, 1e-3).** The originals' default is fp32 with only the attention in
  fp16; this package's default is fp16 autocast everywhere, which is 2x faster. The two differ by
  5e-3 relative per level (1.5e-3 on the upcast feature), the same as the earlier port measured
  for the original's own fp16 autocast against its fp32 (3.8e-3). Against the original fp32,
  our fp16 has cos >= 0.99975 and the same argmax on 99.98% of points; the mIoU is identical.
  With `precision="fp32"` the difference is 6e-6.
- **bf16.** The worst single row of scene0231_01 (298k grid-sampled points) has cosine 0.989;
  means are >= 0.9998 and mIoU is unchanged. bf16 keeps 8 mantissa bits; it is offered for
  training (no GradScaler needed), fp16 is the inference default.
- **Per-scene mIoU is bimodal.** A small class flipping on or off moves one scene's mIoU by about
  8 points (for scene0342_00: 51.4 or 59.9 depending only on which order permutation the blocks
  use, with 99% of argmax unchanged). Comparisons of preprocessing therefore use all 312 val
  scenes.
- **Release weights.** `model.py` never had the stale-CPE behavior the released ScanNet weights
  were trained with, so it cannot reproduce them; the numerics check runs both sides with the
  switch off. With the switch on (`ptv3_scannet`), the val mIoU depends on how the scenes are
  grid-sampled: 76.10 with Pointcept2's grid sampling (lattice of the raw coordinates, float64,
  random pick; Pointcept2 measured 76.10 / 76.09 at depth "auto" / 15), 75.77 with Pointcept's
  own pipeline order (center first, then grid-sample, random pick) and 75.58 with
  `ptv3.prepare`'s default pick "first" (75.56 at depth 15, 75.57 in fp16). The 0.3-0.5 spread
  is lattice alignment of a single pass, not the model: train.log's 77.03 is Pointcept's score
  on the grid-sampled points.
- **Utonia (P6, run 2026-10-30).** The original runs its RoPE in fp32 and casts q, k, v to bf16
  for FlashAttention; this package rotates in fp32 (`modules.Rope3d`, bit-identical to
  `Point3DRoPE` in fp32, tested) and casts to the autocast dtype. Utonia's own module.py also
  forces its spconv convs to fp32 under autocast; this package runs them in the autocast dtype.
  Concerto's inference code is Sonata's (same model, structure and transform files), so P3
  covers its code path, and `tests/test_weights.py` converts every Concerto checkpoint strictly
  against its stored config.
- **Second check (2026-10-30).** An independent run with its own scripts compared `concerto`,
  `concerto_large` (against Sonata 18c09ff), `utonia` and `utonia_small` (against be53e4a) on 4
  ScanNet val scenes not used above (scene0011_00, scene0084_00, scene0300_00, scene0559_00;
  64k-165k grid-sampled points). Each original's transform ran with a fixed seed and fed both
  sides; order shuffling was off on both sides (also inside the original's GridPooling, which
  ignores the config flag). Utonia's own default (bf16 FlashAttention) reaches cos 0.99995
  against its fp32; this package's fp16 default 0.99999. With `ptv3.prepare` instead of Utonia's
  transform, the `utonia` probe gives 68.55 instead of 68.31 mIoU (first point per grid cell, not
  a random one). The bf16 miss of P6 is on sample1; on these scenes bf16 stays >= 0.9993. Not
  compared with an original: `concerto_siglip`, `concerto_origin` (Concerto base's architecture)
  and `utonia_tiny`; on sample1 their fp16 features match this package's fp32
  (cos min >= 0.99999).
- **Serialization depth.** Fixed depth 15 instead of the per-batch depth changes Hilbert orders on
  some scenes and moves the val mIoU by 0.01 (75.58 -> 75.56).
- **sample1** stores float16 coordinates; Sonata's numpy transform centers and grid-samples them
  in float16, `ptv3.prepare` in float32, so 31 of 187,983 grid cells differ there. The ScanNet
  scenes are float32 and match exactly.
- **Gradients.** spconv 2.3.8 runs `SubMConv3d` on an inference path in eval mode that records no
  gradients (the CPE convs get none), so gradients are compared in train mode (BatchNorm on
  batch statistics, DropPath off). 20 biases right before a train-mode BatchNorm have an exactly
  zero gradient; both sides hold noise there (< 2e-7 of the largest gradient).
- **Original quirk.** With `enable_flash=False`, the original uses the smallest sample's patch
  size for every sample of a mixed batch; this package always uses per-sample patches, so batches
  are compared in the default mode only.
