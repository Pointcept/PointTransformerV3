"""Released weights: find or download a checkpoint, verify it, read it and map it onto a model.

Downloads go through ``huggingface_hub.hf_hub_download`` at the preset's pinned revision, so
``HF_HOME``, ``HF_HUB_CACHE`` and offline mode (``HF_HUB_OFFLINE=1``) work as usual. Before that,
an earlier download in a legacy location (the original repositories' ``~/.cache/<name>/ckpt``) is
reused when its sha256 matches. Every file is checked against the pinned sha256 once; the result
is remembered (path, size, mtime) in ``~/.cache/pointtransformerv3/verified.json``.

Checkpoints are read with ``torch.load(weights_only=True, mmap=True)`` (Pointcept's training
state loads as inert placeholders) and converted at load time (``convert.py``). Nothing is
written next to the weights and no weights ship with this package: they keep their own licenses
(README, "Model zoo") and stay on Hugging Face.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pointtransformerv3.presets import WeightSource

__all__ = ["fetch", "read_checkpoint", "sha256_of", "state_dict_of"]

log = logging.getLogger(__name__)
_LICENSE_NOTED: set[str] = set()


def _cache_dir() -> Path:
    root = os.environ.get("PTV3_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "pointtransformerv3"
    )
    return Path(root)


def sha256_of(path: str | Path) -> str:
    """The sha256 of a file (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, sha256: str) -> bool:
    """Whether ``path`` has the sha256 ``sha256``; hashes once per (path, size, mtime)."""
    stat = path.stat()
    stamp = [stat.st_size, stat.st_mtime_ns, sha256]
    record = _cache_dir() / "verified.json"
    try:
        known = json.loads(record.read_text())
    except (OSError, ValueError):
        known = {}
    if known.get(str(path)) == stamp:
        return True
    ok = sha256_of(path) == sha256
    if ok:
        known[str(path)] = stamp
        try:
            record.parent.mkdir(parents=True, exist_ok=True)
            tmp = record.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(known, indent=1))
            tmp.replace(record)
        except OSError:  # a read-only cache only costs a re-hash next time
            pass
    return ok


def fetch(source: WeightSource, *, revision: str | None = None) -> Path:
    """A local, verified copy of ``source``: a legacy download if one matches, else
    ``hf_hub_download`` at the pinned revision (or ``revision``; then the pinned sha256 is not
    checked, since another revision may hold another file)."""
    pinned = revision is None or revision == source.revision
    if pinned:
        for legacy in source.legacy_paths:
            path = Path(os.path.expanduser(legacy))
            if (
                path.is_file()
                and path.stat().st_size == source.size
                and _verified(path, source.sha256)
            ):
                log.info("weights: using %s (sha256 verified)", path)
                return path
    from huggingface_hub import hf_hub_download  # imported only when weights are fetched

    local = Path(
        hf_hub_download(
            repo_id=source.repo,
            filename=source.file,
            revision=source.revision if pinned else revision,
            repo_type=source.repo_type,
        )
    )
    if pinned and not _verified(local, source.sha256):
        raise RuntimeError(
            f"{local} does not match the pinned sha256 {source.sha256} of {source.url}; delete "
            "it and load again, or pass weights=<path> to use another file"
        )
    if not pinned:
        log.warning(
            "weights: %s at revision %s is not the pinned one; sha256 not checked",
            source.file,
            revision,
        )
    return local


def note_license(source: WeightSource) -> None:
    """Log the weights' license once per process (always for non-commercial ones)."""
    if source.license in _LICENSE_NOTED:
        return
    _LICENSE_NOTED.add(source.license)
    if "research permission" in source.license:  # Pointcept's weights: the code's terms
        log.warning(
            "%s/%s is licensed %s (NOTICE): research, evaluation and education, also in "
            "companies; not in commercial products or services without a commercial license",
            source.repo,
            source.file,
            source.license,
        )
    elif "NC" in source.license:
        log.warning(
            "%s/%s is licensed %s: non-commercial use only (the code license is separate)",
            source.repo,
            source.file,
            source.license,
        )
    else:
        log.info("%s/%s is licensed %s", source.repo, source.file, source.license)


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """A checkpoint file as a dict (``torch.load(weights_only=True, mmap=True)``)."""
    from pointtransformerv3.convert import load_pc_checkpoint

    return load_pc_checkpoint(path)


def state_dict_of(ckpt: Mapping[str, Any]) -> Mapping[str, Any]:
    """The state dict inside a checkpoint (``"state_dict"``, ``"model"``, or the dict itself)."""
    for key in ("state_dict", "model"):
        value = ckpt.get(key)
        if isinstance(value, Mapping) and value and all(isinstance(k, str) for k in value):
            return value
    return ckpt
