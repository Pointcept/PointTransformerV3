# Vendored from Pointcept2 src/pointcept2/models/layers/init.py @ 771a949 (simplified for point_transformer_v3).
"""Weight initialization shared by models: PC m2's trunc-normal rule, and seeded init that gives
the same weights for every build path and device (C14).

A model draws one seed from torch's global CPU generator when it is built (``init_seed``), builds
its modules under ``build_rng()`` (their construction-time draws are replaced) and runs its
``_init_weights`` under ``seeded_rng(seed, devices)`` and ``drawn_on_cpu(model.modules())``. An
eager build (on the CPU or under ``torch.device("cuda")``) and a meta-device build followed by
``to_empty(device=...)`` + ``_init_weights`` then hold the same weights on every device and GPU
model, and none of them moves the global CPU or accelerator generators beyond that one seed draw.

Why the CPU: torch's CUDA ``uniform_`` / ``normal_`` size their launch grid by the SM count, and
both the element-to-thread mapping and the Philox offset each call consumes depend on that grid,
so one seed gives different tensors on different GPU models (D22: mixed GPU types must work).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from typing import Any, cast

import torch
from torch import Tensor, nn

__all__ = ["build_rng", "drawn_on_cpu", "init_seed", "seeded_rng", "trunc_normal_init_"]

_SEED_BOUND = 1 << 62


def init_seed() -> int:
    """A seed for a model's ``_init_weights``: one draw from torch's global CPU generator, made on
    the CPU also inside ``torch.device("meta")`` (so both build paths draw the same seed)."""
    return int(torch.randint(_SEED_BOUND, (), device="cpu"))


@contextlib.contextmanager
def build_rng() -> Iterator[None]:
    """Fork torch's CPU generator and, when the default device (``torch.set_default_device`` or a
    ``torch.device`` context) is an accelerator, that device's generator: for building modules
    whose construction-time init a seeded ``_init_weights`` replaces, so an eager build under
    ``torch.device("cuda")`` leaves the global CUDA generator where a meta build leaves it."""
    device = torch.get_default_device()  # an accelerator comes with its index
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        if device.type not in ("cpu", "meta"):
            index = 0 if device.index is None else device.index
            stack.enter_context(torch.random.fork_rng(devices=[index], device_type=device.type))
        yield


@contextlib.contextmanager
def seeded_rng(seed: int, devices: Iterable[torch.device] = ()) -> Iterator[None]:
    """Run the block with torch's CPU generator, and the generator of every accelerator device in
    ``devices``, seeded with ``seed``; their states are restored afterwards.

    Draws on the CPU give the same numbers for every build path and machine. Draws on a CUDA
    device depend on the GPU model (the RNG kernels' grid follows the SM count), so model init
    also runs under ``drawn_on_cpu``; seeding ``devices`` only keeps a stray device draw
    reproducible and off the global stream.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {seed!r}")
    by_type: dict[str, set[int]] = {}
    for device in devices:
        if device.type in ("cpu", "meta"):
            continue
        index = device.index
        if index is None:
            index = torch.cuda.current_device() if device.type == "cuda" else 0
        by_type.setdefault(device.type, set()).add(index)
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        for kind, indices in by_type.items():
            stack.enter_context(torch.random.fork_rng(devices=sorted(indices), device_type=kind))
        torch.default_generator.manual_seed(seed)
        for kind, indices in by_type.items():
            for index in sorted(indices):
                if kind == "cuda":
                    with torch.cuda.device(index):
                        torch.cuda.manual_seed(seed)
                else:  # one device per process (mps)
                    getattr(torch, kind).manual_seed(seed)
        yield


@contextlib.contextmanager
def drawn_on_cpu(modules: Iterable[nn.Module]) -> Iterator[None]:
    """Run the block with every parameter and buffer held directly by ``modules`` that lives on an
    accelerator swapped for a contiguous CPU copy (same shape, dtype and values); on exit copy the
    CPU values back into the original tensors (in place: the same ``Parameter`` objects and
    storage) and put them back. Pass ``model.modules()`` for a whole model.

    Init drawn inside (``reset_parameters``, ``trunc_normal_init_``) comes from the CPU generator
    for every device, so a seeded init equals the eager CPU build on every rank and GPU model.
    CPU tensors are drawn in place and meta tensors stay (their draws are no-ops). A tensor shared
    by several modules gets one CPU copy. If the block raises, the originals are put back as they
    were.
    """
    swapped: list[tuple[dict[str, Any], str, Tensor]] = []
    copies: dict[int, Tensor] = {}
    for module in modules:
        for store in cast(list[dict[str, Any]], [module._parameters, module._buffers]):
            for name, tensor in list(store.items()):
                if tensor is None or tensor.device.type in ("cpu", "meta"):
                    continue
                copy = copies.get(id(tensor))
                if copy is None:
                    copy = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
                    copy.copy_(tensor.detach())  # values no init rule writes stay as they were
                    if isinstance(tensor, nn.Parameter):
                        copy = nn.Parameter(copy, requires_grad=tensor.requires_grad)
                    copies[id(tensor)] = copy
                store[name] = copy
                swapped.append((store, name, tensor))
    done = False
    try:
        yield
        done = True
    finally:
        written: set[int] = set()
        with torch.no_grad():
            for store, name, original in swapped:
                value = store[name]
                store[name] = original
                if done and value is not None and id(original) not in written:
                    written.add(id(original))
                    original.copy_(value)


def trunc_normal_init_(module: nn.Module, std: float = 0.02) -> None:
    """PC m2's ``_init_weights`` over ``module`` and all its submodules: the weight of every Linear
    and convolution is drawn from ``trunc_normal_(std=std)`` (torch's absolute bounds [-2, 2], as
    PC) and its bias set to 0. Norms and all other parameters keep their own init.

    Convolutions are the modules whose own ``weight`` has 3 or more dims (``SubMConv3d``,
    ``nn.Conv*``), so this layer needs no point-cloud import. Modules are visited in
    ``module.modules()`` order, so a fixed seed gives a fixed state_dict (C14).
    """
    if isinstance(std, bool) or not isinstance(std, int | float) or std <= 0:
        raise ValueError(f"std must be a positive float, got {std!r}")
    for m in module.modules():
        params = dict(m.named_parameters(recurse=False))
        weight = params.get("weight")
        if weight is None or not (isinstance(m, nn.Linear) or weight.ndim >= 3):
            continue
        nn.init.trunc_normal_(weight, std=std)
        if params.get("bias") is not None:
            nn.init.zeros_(params["bias"])
