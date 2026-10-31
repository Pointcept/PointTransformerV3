# Vendored from Pointcept2 src/pointcept2/ops/_triton.py @ 771a949 (simplified for point_transformer_v3).
"""What the ops with Triton kernels share: the toolchain probe, the slow-fallback warning and the
launch helpers of their kernel files. Importing it does not import triton.

- Toolchain: ``toolchain_problem(device, min_capability=...)`` names what keeps Triton kernels from
  running on a CUDA device (None if nothing); ``available`` is its boolean. Each op passes the
  floor of its kernels: ``MIN_CAPABILITY`` (8.0) for tensor-core ``tl.dot`` tiles (the dense
  sparse conv), ``TRITON_MIN_CAPABILITY`` (7.0, Triton's own floor and the check torch.compile
  makes) for kernels without ``tl.dot`` (depthwise conv, LayerNorm, kernel maps). The checks take
  a device, so ``auto`` backends take Triton on the GPUs of a mixed node that pass; without one,
  every visible GPU must pass.
- Fallback warnings (C18): ``warn_at_caller`` attributes a warning to the first frame outside
  torch and the files registered with ``skip_warning_frames_in`` (the op modules and the module
  wrappers of higher layers), so it names the model line that called the op.
- Launch helpers: an op body (``torch.library.triton_op``) launches ``traceable(kernel)``, so
  torch.compile / torch.export capture the launch; the eager ``autograd.Function`` of the same op
  launches ``raw(kernel)``, the kernel itself (I5). Both run under ``device_guard`` of the inputs'
  device.

Toolchain: Triton compiles the kernels itself (bundled ptxas) and launches them through one C
extension, ``cuda_utils``, which it builds from its ``driver.c`` the first time a Triton version
runs on a platform and caches in ``TRITON_CACHE_DIR`` (default ``~/.triton/cache``). Only that
build needs a host C compiler (``CC``, else gcc or clang on PATH) and the running Python's
``Python.h``; with the helper cached, new kernel signatures compile without either (verified with
Triton 3.7.1: research/integrate/reports/phase1_fix.md). ``toolchain_problem`` loads the helper
and, when that fails, names the missing piece.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import warnings
from typing import Any

import torch
from torch import Tensor
from torch.library import wrap_triton

__all__ = [
    "MIN_CAPABILITY",
    "TRITON_MIN_CAPABILITY",
    "available",
    "device_guard",
    "raw",
    "skip_warning_frames_in",
    "toolchain_problem",
    "traceable",
    "warn_at_caller",
]

MIN_CAPABILITY = (8, 0)  # kernels with tensor-core tl.dot tiles (the dense sparse conv)
TRITON_MIN_CAPABILITY = (7, 0)  # Triton's own floor (torch.compile's check): kernels without tl.dot
_FLOOR_REASONS = {
    MIN_CAPABILITY: (
        "(Ampere or newer): Triton has no tensor-core dot below that and the tiles spill"
    ),
    TRITON_MIN_CAPABILITY: "(Volta or newer), the oldest GPUs Triton compiles for",
}
# Frames a fallback warning skips to reach the user's line: torch (nn.Module.__call__, dynamo) and
# the files registered with ``skip_warning_frames_in``.
_INTERNAL_PREFIXES: tuple[str, ...] = (os.path.dirname(torch.__file__) + os.sep,)


# ---- toolchain


def available(
    device: torch.device | None = None, *, min_capability: tuple[int, int] = MIN_CAPABILITY
) -> bool:
    """Whether Triton kernels can run on ``device`` (a CUDA device), or on every visible GPU when
    ``device`` is None (see ``toolchain_problem``)."""
    return toolchain_problem(device, min_capability=min_capability) is None


@functools.cache
def toolchain_problem(
    device: torch.device | None = None, *, min_capability: tuple[int, int] = MIN_CAPABILITY
) -> str | None:
    """What keeps Triton kernels from running here, with the fix; None if nothing.

    Checks that triton imports, CUDA is available, the GPU has compute capability
    ``min_capability``+ (``device``; every visible GPU when it is None, which is what forcing a
    backend without a tensor at hand needs; 8.0 for the GEMM kernels, ``TRITON_MIN_CAPABILITY``
    for kernels without ``tl.dot``), and Triton's CUDA driver helper loads
    (``triton.runtime.driver.active``: it builds ``cuda_utils`` on first use, then loads it from
    the cache). When the helper fails, the static checks of Triton's build (``CC`` or gcc/clang,
    ``Python.h``) name the likely cause. Cached per device and floor, so ``auto`` can take triton
    on the sm_80+ GPUs of a mixed node.
    """
    if importlib.util.find_spec("triton") is None:
        return (
            "triton is not installed; it ships with torch's CUDA wheels (reinstall torch with "
            "CUDA, e.g. `uv sync` without the cpu extra)"
        )
    if not torch.cuda.is_available():
        return "CUDA is not available (torch.cuda.is_available() is False); run on a CUDA GPU"
    if device is None:
        indices: list[int] = list(range(torch.cuda.device_count()))
    else:
        device = torch.device(device)
        if device.type != "cuda":
            return f"the triton kernels run on CUDA devices, got {device}"
        indices = [torch.cuda.current_device() if device.index is None else device.index]
    floor = tuple(min_capability)
    for i in indices:
        major, minor = torch.cuda.get_device_capability(i)
        if (major, minor) < floor:
            why = _FLOOR_REASONS.get(floor, "")
            return (
                f"cuda:{i} ({torch.cuda.get_device_name(i)}) is sm_{major}{minor}, but the triton "
                f"kernels need compute capability {floor[0]}.{floor[1]}+{' ' + why if why else ''}"
            )
    error = _probe_driver()
    if error is None:
        return None
    return (
        "Triton could not build or load its CUDA driver helper (cuda_utils, built once per "
        "Triton version from driver.c and cached in TRITON_CACHE_DIR; only this build needs a C "
        f"compiler and Python.h): {error}. Fix: {_build_hint()}"
    )


def _probe_driver() -> str | None:
    """Load Triton's CUDA driver helper, building it if it is not cached; the error or None."""
    try:
        from triton.runtime import driver

        driver.active.get_current_target()  # creates the driver, which loads cuda_utils
    except subprocess.CalledProcessError as e:  # the C build; the compiler printed its errors
        command = e.cmd[0] if isinstance(e.cmd, list | tuple) and e.cmd else e.cmd
        return f"the C compiler {command!r} exited with status {e.returncode} (output above)"
    except Exception as e:  # any other build or load failure: no launch can work
        lines = str(e).strip().splitlines()
        text = lines[0].rstrip(".") if lines else ""
        return f"{type(e).__name__}: {text[:300]}"
    return None


def _build_hint() -> str:
    """The likely cause of a failed ``cuda_utils`` build, from Triton's own lookups
    (``triton/runtime/build.py``): the compiler and this Python's headers."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    hints = []
    if _python_header() is None:
        hints.append(
            f"Python.h for Python {version} is not in {_python_include_dir()} or in "
            f"C_INCLUDE_PATH / CPATH: install the Python headers (python{version}-dev, or use a "
            "uv-managed Python, which ships them) or set C_INCLUDE_PATH to a directory holding "
            "this version's Python.h"
        )
    cc = os.environ.get("CC")
    if cc is not None:  # Triton uses CC as is when set, even when empty
        if not cc or shutil.which(cc) is None:
            hints.append(
                f"CC={cc!r} is set but no such compiler was found: set CC to an installed C "
                "compiler (e.g. CC=gcc) or unset it"
            )
    elif shutil.which("gcc") is None and shutil.which("clang") is None:
        hints.append(
            "no C compiler found (Triton uses CC, else gcc or clang on PATH): install gcc or set CC"
        )
    if not hints:
        hints.append(
            f"check that {cc or 'gcc'} can build a shared library against Python.h and libcuda"
        )
    return "; ".join(hints)


def _python_include_dir() -> str:
    """The include dir Triton passes to the compiler (``triton/runtime/build.py``)."""
    scheme = sysconfig.get_default_scheme()
    if scheme == "posix_local":  # Debian's scheme, as Triton handles it
        scheme = "posix_prefix"
    return sysconfig.get_paths(scheme=scheme)["include"]


def _python_header() -> str | None:
    """The first ``Python.h`` of this Python's major.minor on the compiler's include path."""
    dirs = [_python_include_dir()]
    for var in ("C_INCLUDE_PATH", "CPATH"):
        dirs += [d for d in os.environ.get(var, "").split(os.pathsep) if d]
    version = sys.version_info[:2]
    for d in dirs:
        if os.path.isfile(os.path.join(d, "Python.h")) and _header_version(d) == version:
            return d
    return None


def _header_version(include_dir: str) -> tuple[int, int] | None:
    try:
        with open(os.path.join(include_dir, "patchlevel.h")) as f:
            text = f.read()
    except OSError:
        return None
    major = re.search(r"#define\s+PY_MAJOR_VERSION\s+(\d+)", text)
    minor = re.search(r"#define\s+PY_MINOR_VERSION\s+(\d+)", text)
    if major is None or minor is None:
        return None
    return int(major.group(1)), int(minor.group(1))


# ---- fallback warnings


def skip_warning_frames_in(path: str) -> None:
    """Treat frames from ``path`` (a file or a directory ending in ``os.sep``) as internal, so a
    fallback warning skips them and names the caller's line (e.g. a module whose ``forward``
    wraps ``subm_conv3d``)."""
    global _INTERNAL_PREFIXES
    if path not in _INTERNAL_PREFIXES:
        _INTERNAL_PREFIXES = (*_INTERNAL_PREFIXES, path)


def warn_at_caller(message: str) -> None:
    """``warnings.warn(message)`` attributed to the first frame outside torch and the files
    registered with ``skip_warning_frames_in`` (the model line that called the op, whichever entry
    point and module wrappers lie between), as Python 3.12's ``skip_file_prefixes`` does; also on
    3.11."""
    frame = sys._getframe(1)
    level = 2  # stacklevel 2 is the frame that called this function
    while frame.f_back is not None and frame.f_code.co_filename.startswith(_INTERNAL_PREFIXES):
        frame = frame.f_back
        level += 1
    warnings.warn(message, UserWarning, stacklevel=level)


# ---- launch helpers: the op bodies (wrap=traceable) and the eager Functions (wrap=raw)


def traceable(kernel: Any) -> Any:
    """``wrap_triton`` for compiled kernels (traceable by torch.compile / export); kernels made
    under ``TRITON_INTERPRET=1`` (CPU tests) are not ``JITFunction``s and are launched as is."""
    from triton.runtime.jit import JITFunction  # the caller's kernel module imported triton

    return wrap_triton(kernel) if isinstance(kernel, JITFunction) else kernel


def raw(kernel: Any) -> Any:
    """The kernel itself: eager launches (``kernel[grid](...)``, Triton's own launch cache; I5)."""
    return kernel


def device_guard(t: Tensor) -> contextlib.AbstractContextManager[Any]:
    """Make ``t``'s CUDA device current for a launch: Triton launches on the current device and
    stream and accepts pointers of other GPUs (UVA). A no-op when it is current already, and for
    CPU tensors (the Triton interpreter)."""
    return torch.cuda.device(t.device) if t.is_cuda else contextlib.nullcontext()
