"""Released configurations of ``PointTransformerV3``: one class, many presets.

A preset is a complete ``Config`` (a plain dict: every field, so it does not move when a default
changes), where its weights come from, how a checkpoint maps onto the model, how raw points are
prepared for it (``point_transformer_v3.prepare``) and which segmentation heads exist for it.

Presets (``PRESETS``):

- ``"ptv3"``: Pointcept2's PTv3 recipe for new training (the ScanNet ``semseg-pt-v3m1-0-base``
  architecture with every compatibility switch off, fixed serialization depth 15); random init.
- ``"ptv3_scannet"``: the released ScanNet semantic segmentation weights (Pointcept
  ``semseg-pt-v3m1-0-base``, trained 2023-12 on Pointcept 314afb3) with their 20-class head.
  They need two compatibility switches the original ``model.py`` lost (stale CPE inputs,
  ``legacy_cpe_stale_input``) and signed colors (``color / 127.5 - 1``).
- ``"sonata"`` / ``"sonata_small"``: the Sonata self-supervised encoders (``facebook/sonata``;
  Linear stem in fp32, LayerNorm everywhere, re-encoded pooled levels, no decoder), with the
  ScanNet linear-probe head ``"scannet20"`` (``sonata`` only).
- ``"concerto"``, ``"concerto_large"``, ``"concerto_siglip"``, ``"concerto_origin"``: Concerto
  encoders (``Pointcept/Concerto``), Sonata's architecture and inference code. Per the Hugging
  Face card, ``concerto`` (``concerto_base.pth``) and ``concerto_large`` are the default models
  (a pre-release of Utonia's recipe, robust to missing color and normals: pass zeros);
  ``concerto_origin`` is the paper's model, ``concerto_siglip`` its SigLIP 2 variant.
  ``concerto_large`` has the linear probe ``"scannet20"``.
- ``"utonia"``, ``"utonia_small"``, ``"utonia_tiny"``: Utonia encoders (``Pointcept/Utonia``):
  Sonata's architecture plus 3D RoPE (``rope_base=10``, head_dim 18), prepared at a 1 cm grid
  after scaling coordinates by 0.5 (indoor scenes; ``prepare(scale=...)`` picks the density for
  other domains). ``utonia`` has the linear probe ``"scannet20"``.

Licenses: the PTv3 ScanNet weights are MIT (Hugging Face card of ``Pointcept/PointTransformerV3``).
The Sonata weights (Meta) are CC BY-NC 4.0 (the Sonata README; the ``facebook/sonata`` card states
no license): non-commercial use only. The Concerto and Utonia weights (Pointcept) are CC BY-NC 4.0
with the additional research permission stated in NOTICE. Weights are downloaded from Hugging Face
at runtime and never redistributed with this package.

This module imports neither torch nor model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from point_transformer_v3._util import suggest

__all__ = [
    "ORDERS",
    "PC_M1",
    "PC_M1_RELEASE",
    "PC_M2",
    "PRESETS",
    "PTV3",
    "SCANNET20_CLASSES",
    "SONATA",
    "SONATA_SMALL",
    "UTONIA",
    "HeadSpec",
    "PrepareSpec",
    "Preset",
    "WeightSource",
    "get_preset",
]

ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")

# Pointcept PT-v3m1 (every revision): conv stem, BatchNorm outside the blocks, shifted codes,
# torch's default init, and the first decoder block's CPE fed the skip branch.
PC_M1: dict[str, Any] = dict(
    stem="conv",
    norm="bn",
    pool_policy="shift",
    init="torch",
    legacy_dec_cpe_skip_input=True,
)
# Pointcept PT-v3m2 (Sonata, 10b7490 2025-03-21): Linear stem, LayerNorm everywhere, re-encoded
# levels, trunc-normal init; its unpooling refreshes the features the first decoder CPE sees.
PC_M2: dict[str, Any] = dict(
    stem="linear",
    norm="ln",
    pool_policy="reencode",
    init="trunc_normal",
    legacy_dec_cpe_skip_input=False,
)

_SCANNET_SIZES: dict[str, Any] = dict(
    in_channels=6,
    orders=ORDERS,
    shuffle_orders=True,
    serialization_depth=15,
    strides=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_heads=(2, 4, 8, 16, 32),
    enc_patch_sizes=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_heads=(4, 4, 8, 16),
    dec_patch_sizes=(1024, 1024, 1024, 1024),
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_path=0.3,
    patch_tail="overlap",
    layer_scale=None,
    rope_base=None,
)

# Pointcept2's recipe for new training: Config defaults (fixed depth 15, no compat switches).
PTV3: dict[str, Any] = {
    **_SCANNET_SIZES,
    "stem": "conv",
    "norm": "bn",
    "pool_policy": "shift",
    "init": "torch",
    "legacy_cpe_stale_input": False,
    "legacy_dec_cpe_skip_input": False,
}

# The released ScanNet weights (Pointcept 314afb3, 2023-12): PC m1 with stale CPE inputs, and the
# serialization depth of each batch ("auto", what they were trained and evaluated with).
PC_M1_RELEASE: dict[str, Any] = {
    **_SCANNET_SIZES,
    **PC_M1,
    "serialization_depth": "auto",
    "legacy_cpe_stale_input": True,
}

_ENCODER_ONLY: dict[str, Any] = dict(
    dec_depths=(), dec_channels=(), dec_num_heads=(), dec_patch_sizes=()
)

SONATA: dict[str, Any] = {
    **_SCANNET_SIZES,
    **PC_M2,
    **_ENCODER_ONLY,
    "in_channels": 9,
    "serialization_depth": "auto+1",
    "enc_depths": (3, 3, 3, 12, 3),
    "enc_channels": (48, 96, 192, 384, 512),
    "enc_num_heads": (3, 6, 12, 24, 32),
    "enc_patch_sizes": (1024, 1024, 1024, 1024, 1024),
    "legacy_cpe_stale_input": False,
}

SONATA_SMALL: dict[str, Any] = {
    **SONATA,
    "enc_depths": (2, 2, 2, 6, 2),
    "enc_channels": (32, 64, 128, 256, 512),
    "enc_num_heads": (2, 4, 8, 16, 32),
}

# Concerto: Sonata's architecture (base = SONATA); large is wider (head_dim 16 everywhere).
CONCERTO_LARGE: dict[str, Any] = {
    **SONATA,
    "enc_channels": (64, 128, 256, 512, 768),
    "enc_num_heads": (4, 8, 16, 32, 48),
}

# Utonia (PC m3): Sonata's architecture plus 3D RoPE; head_dim 18 at every stage.
UTONIA: dict[str, Any] = {
    **SONATA,
    "enc_channels": (54, 108, 216, 432, 576),
    "enc_num_heads": (3, 6, 12, 24, 32),
    "rope_base": 10.0,
}
UTONIA_SMALL: dict[str, Any] = {
    **UTONIA,
    "enc_depths": (2, 2, 2, 6, 2),
    "enc_channels": (36, 72, 144, 288, 432),
    "enc_num_heads": (2, 4, 8, 16, 24),
}
UTONIA_TINY: dict[str, Any] = {
    **UTONIA_SMALL,
    "enc_channels": (18, 36, 72, 144, 216),
    "enc_num_heads": (1, 2, 4, 8, 12),
}

SCANNET20_CLASSES = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "otherfurniture",
)  # fmt: skip


@dataclass(frozen=True)
class WeightSource:
    """A checkpoint on Hugging Face, pinned by revision and sha256."""

    repo: str
    file: str
    revision: str
    sha256: str
    license: str
    size: int  # bytes
    repo_type: str = "model"
    legacy_paths: tuple[str, ...] = ()  # earlier download locations that are reused if present

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/blob/{self.revision}/{self.file}"


@dataclass(frozen=True)
class PrepareSpec:
    """How ``point_transformer_v3.prepare`` turns raw points into a model input.

    ``feat``: the feature columns in order (``"coord"``: the centered coordinates of the picked
    point; ``"color"``: RGB scaled by ``color_range``; ``"normal"``: unit normals).
    ``color_range``: ``"unit"`` (``color / 255``) or ``"signed"`` (``color / 127.5 - 1``, Pointcept
    before 2025-03). ``center``: shift x, y to the box center and z to the floor (Pointcept's
    ``CenterShift(apply_z=True)``). ``scale``: multiply the coordinates first (Utonia's fixed
    ``RandomScale``; the coordinates the model sees, features and grid alike, are scaled)."""

    grid_size: float = 0.02
    feat: tuple[str, ...] = ("color", "normal")
    color_range: str = "unit"
    center: bool = True
    scale: float = 1.0


@dataclass(frozen=True)
class HeadSpec:
    """A linear segmentation head over the model's (upcast) features. ``weights`` is None when the
    head is stored in the backbone checkpoint (``seg_head.*`` keys)."""

    num_classes: int
    in_channels: int
    class_names: tuple[str, ...]
    weights: WeightSource | None = None
    description: str = ""


@dataclass(frozen=True)
class Preset:
    name: str
    config: dict[str, Any]
    weights: WeightSource | None
    prepare: PrepareSpec
    heads: dict[str, HeadSpec] = field(default_factory=dict)
    default_head: str | None = None  # attached by load(head=None)
    description: str = ""
    citation: str = ""


_SONATA_REVISION = "df99897472c09f91ba9288da0a034aacffc0b010"
_PTV3_REVISION = "5e25cfaf759289b03db1f90f143b3dc5172929d3"
_CONCERTO_REVISION = "efc130718a564eba0ce4899318f3c20b19d9c752"
_UTONIA_REVISION = "9457f025e36c7001ee962a542f0d27a5b600e1e7"
_CC_BY_NC = "CC-BY-NC-4.0"  # Sonata (Meta): no additional permission from Pointcept
# Concerto and Utonia (Pointcept): CC BY-NC 4.0 with the code's additional research permission
# and commercial contact (NOTICE).
_CC_BY_NC_RESEARCH = "CC-BY-NC-4.0 + research permission"


def _hf(repo: str, revision: str, license: str, file: str, sha256: str, size: int) -> WeightSource:
    """A Hugging Face file; also found where its original repository downloads it
    (``~/.cache/<name>/ckpt``)."""
    legacy = f"~/.cache/{repo.split('/')[1].lower()}/ckpt/{file}"
    return WeightSource(repo, file, revision, sha256, license, size, legacy_paths=(legacy,))


def _concerto(file: str, sha256: str, size: int) -> WeightSource:
    return _hf("Pointcept/Concerto", _CONCERTO_REVISION, _CC_BY_NC_RESEARCH, file, sha256, size)


def _utonia(file: str, sha256: str, size: int) -> WeightSource:
    return _hf("Pointcept/Utonia", _UTONIA_REVISION, _CC_BY_NC_RESEARCH, file, sha256, size)


PTV3_SCANNET_WEIGHTS = WeightSource(
    repo="Pointcept/PointTransformerV3",
    file="scannet-semseg-pt-v3m1-0-base/model/model_best.pth",
    revision=_PTV3_REVISION,
    sha256="40206376ff2f83f48e4d1bc27d5c5d96be7c87c5d11eb45fe7be501959040e7f",
    license="MIT",
    size=554_618_088,
)
SONATA_WEIGHTS = WeightSource(
    repo="facebook/sonata",
    file="sonata.pth",
    revision=_SONATA_REVISION,
    sha256="c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50",
    license=_CC_BY_NC,
    size=434_008_287,
    legacy_paths=("~/.cache/sonata/ckpt/sonata.pth",),
)
SONATA_SMALL_WEIGHTS = WeightSource(
    repo="facebook/sonata",
    file="sonata_small.pth",
    revision=_SONATA_REVISION,
    sha256="ee5dae277dde1d164499579a6b17d0bd38df4e4286b0f1ba21a82f8cc14270d1",
    license=_CC_BY_NC,
    size=154_699_705,
    legacy_paths=("~/.cache/sonata/ckpt/sonata_small.pth",),
)
SONATA_LINEAR_PROBE_SCANNET = WeightSource(
    repo="facebook/sonata",
    file="sonata_linear_prob_head_sc.pth",
    revision=_SONATA_REVISION,
    sha256="9c80789787bef0734a7155f172df62f63b6cd60af5f622ae0339903d089cb707",
    license=_CC_BY_NC,
    size=100_362,
    legacy_paths=("~/.cache/sonata/ckpt/sonata_linear_prob_head_sc.pth",),
)
CONCERTO_WEIGHTS = _concerto(
    "concerto_base.pth", "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
    433_987_358,
)  # fmt: skip
CONCERTO_LARGE_WEIGHTS = _concerto(
    "concerto_large.pth", "04de4954d19b3563523f7a06254a02ed25bdf639e0ed2f12f9106e5fa9b178a7",
    830_812_263,
)  # fmt: skip
CONCERTO_SIGLIP_WEIGHTS = _concerto(
    "concerto_base_siglip.pth", "b9953177dd686136f7b12333b3e4d1c3e8bbb16ed35846a8eef570e87645ba25",
    433_990_557,
)  # fmt: skip
CONCERTO_ORIGIN_WEIGHTS = _concerto(
    "concerto_base_origin.pth", "734ec9cfa7924e9d268c4af697e39dcd22469aa64205b448a9d79634e78c2a89",
    433_990_557,
)  # fmt: skip
CONCERTO_LARGE_LINEAR_PROBE_SCANNET = _concerto(
    "concerto_large_linear_prob_head_sc.pth",
    "af42da21ce11159cbe3dfb927a5322c791e9dd2fbae23f0b395fd196c9c54bc8", 140_090,
)  # fmt: skip
UTONIA_WEIGHTS = _utonia(
    "utonia.pth", "a0a2e5e234ce943082ae9ba26bcff74e285f16ed8d04fed5c36ffe8dc30ea353", 549_168_522
)
UTONIA_SMALL_WEIGHTS = _utonia(
    "utonia_small.pth", "3d7afb731771556ebdf20abd1b6a1f022d9b9f105db8446f5eff02037f7fceb9",
    149_082_467,
)  # fmt: skip
UTONIA_TINY_WEIGHTS = _utonia(
    "utonia_tiny.pth", "e49a1944403d4ec3df459537fb7d5b132326736d0ebdb7c5a74868f3fa50d137",
    37_390_848,
)  # fmt: skip
UTONIA_LINEAR_PROBE_SCANNET = _utonia(
    "utonia_linear_prob_head_sc.pth",
    "8e687f3961b81d374df88024c43f676352f16e4128b9124ad97ab213e111eabd", 112_650,
)  # fmt: skip

_PTV3_CITATION = (
    "Wu et al., Point Transformer V3: Simpler, Faster, Stronger. CVPR 2024. arXiv:2312.10035"
)
_SONATA_CITATION = (
    "Wu et al., Sonata: Self-Supervised Learning of Reliable Point Representations. CVPR 2025. "
    "arXiv:2503.16429"
)
_CONCERTO_CITATION = (
    "Zhang et al., Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial "
    "Representations. NeurIPS 2025. arXiv:2510.23607"
)
_UTONIA_CITATION = (
    "Zhang et al., Utonia: Toward One Encoder for All Point Clouds. ICML 2026. arXiv:2603.03283"
)
_SONATA_PREPARE = PrepareSpec(grid_size=0.02, feat=("coord", "color", "normal"))
# Utonia's transform.default(0.5): coordinates x0.5, then a 1 cm grid (2 cm in the input's meters)
_UTONIA_PREPARE = PrepareSpec(grid_size=0.01, feat=("coord", "color", "normal"), scale=0.5)


def _scannet20(in_channels: int, weights: WeightSource) -> dict[str, HeadSpec]:
    return {
        "scannet20": HeadSpec(
            num_classes=20,
            in_channels=in_channels,
            class_names=SCANNET20_CLASSES,
            weights=weights,
            description=f"ScanNet-20 linear probe on the {in_channels}-d upcast features",
        )
    }


def _encoder(name: str, config: dict[str, Any], weights: WeightSource, description: str,
             citation: str, prepare: PrepareSpec = _SONATA_PREPARE,
             heads: dict[str, HeadSpec] | None = None) -> Preset:  # fmt: skip
    """A released self-supervised encoder (no default head)."""
    return Preset(name, config, weights, prepare, heads or {}, None, description, citation)


PRESETS: dict[str, Preset] = {
    "ptv3": Preset(
        name="ptv3",
        config=PTV3,
        weights=None,
        prepare=PrepareSpec(grid_size=0.02, feat=("color", "normal")),
        description="PTv3 (ScanNet architecture, Pointcept2 recipe), random init",
        citation=_PTV3_CITATION,
    ),
    "ptv3_scannet": Preset(
        name="ptv3_scannet",
        config=PC_M1_RELEASE,
        weights=PTV3_SCANNET_WEIGHTS,
        prepare=PrepareSpec(grid_size=0.02, feat=("color", "normal"), color_range="signed"),
        heads={
            "scannet20": HeadSpec(
                num_classes=20,
                in_channels=64,
                class_names=SCANNET20_CLASSES,
                description="ScanNet-20 semantic segmentation (trained with the backbone)",
            )
        },
        default_head="scannet20",
        description="PTv3 ScanNet semantic segmentation release (val mIoU 77.0 in train.log)",
        citation=_PTV3_CITATION,
    ),
    "sonata": _encoder(
        "sonata", SONATA, SONATA_WEIGHTS,
        "Sonata self-supervised PTv3 encoder (108M parameters, 1232-d upcast features)",
        _SONATA_CITATION, heads=_scannet20(1232, SONATA_LINEAR_PROBE_SCANNET),
    ),
    "sonata_small": _encoder(
        "sonata_small", SONATA_SMALL, SONATA_SMALL_WEIGHTS,
        "Sonata small encoder (39M parameters, 992-d upcast features)", _SONATA_CITATION,
    ),
    "concerto": _encoder(
        "concerto", SONATA, CONCERTO_WEIGHTS,
        "Concerto base encoder (108M parameters, 1232-d upcast features)", _CONCERTO_CITATION,
    ),
    "concerto_large": _encoder(
        "concerto_large", CONCERTO_LARGE, CONCERTO_LARGE_WEIGHTS,
        "Concerto large encoder (208M parameters, 1728-d upcast features)", _CONCERTO_CITATION,
        heads=_scannet20(1728, CONCERTO_LARGE_LINEAR_PROBE_SCANNET),
    ),
    "concerto_siglip": _encoder(
        "concerto_siglip", SONATA, CONCERTO_SIGLIP_WEIGHTS,
        "Concerto base, distilled from SigLIP 2 instead of DINOv2 (108M parameters)",
        _CONCERTO_CITATION,
    ),
    "concerto_origin": _encoder(
        "concerto_origin", SONATA, CONCERTO_ORIGIN_WEIGHTS,
        "Concerto base as in the paper (108M parameters, 1232-d upcast features)",
        _CONCERTO_CITATION,
    ),
    "utonia": _encoder(
        "utonia", UTONIA, UTONIA_WEIGHTS,
        "Utonia encoder, 3D RoPE (137M parameters, 1386-d upcast features)", _UTONIA_CITATION,
        _UTONIA_PREPARE, heads=_scannet20(1386, UTONIA_LINEAR_PROBE_SCANNET),
    ),
    "utonia_small": _encoder(
        "utonia_small", UTONIA_SMALL, UTONIA_SMALL_WEIGHTS,
        "Utonia small encoder (37M parameters, 972-d upcast features)", _UTONIA_CITATION,
        _UTONIA_PREPARE,
    ),
    "utonia_tiny": _encoder(
        "utonia_tiny", UTONIA_TINY, UTONIA_TINY_WEIGHTS,
        "Utonia tiny encoder (9M parameters, 486-d upcast features)", _UTONIA_CITATION,
        _UTONIA_PREPARE,
    ),
}  # fmt: skip


def get_preset(name: str) -> Preset:
    """The preset ``name`` (``PRESETS``); an unknown name raises with a suggestion."""
    if name in PRESETS:
        return PRESETS[name]
    raise ValueError(
        f"unknown preset {name!r}{suggest(name, PRESETS)}; available: {sorted(PRESETS)}"
    )
