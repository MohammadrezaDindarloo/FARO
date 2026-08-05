"""Point the process at the in-repo acados build, before acados is imported.

Section II-G solves the KSO with acados, so it is a hard dependency of Eq. 15.
`acados_template` is a thin wrapper around a C library that must be BUILT (see
scripts/setup_acados.sh); it is not a pip package. Setting this up here rather than
in a shell profile keeps the rule that everything lives inside the repo and the
`faro` environment.

SETTING LD_LIBRARY_PATH IS NOT ENOUGH, which is the whole reason this file exists.
glibc's loader reads it once, at process start; assigning to `os.environ` afterwards
changes a copy it never consults again. So loading `libacados.so` by absolute path
still fails on its DT_NEEDED dependency (`libqpOASES_e.so`). The fix is to load the
dependencies ourselves with RTLD_GLOBAL first -- a library already in the process
satisfies the dependency without any search. BLASFEO underpins HPIPM, so order matters.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

ACADOS_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "acados"


def acados_is_available() -> bool:
    return (ACADOS_ROOT / "lib" / "libacados.so").exists()


def configure_acados() -> bool:
    if not acados_is_available():
        return False
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(ACADOS_ROOT))
    lib = ACADOS_ROOT / "lib"
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if str(lib) not in current.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([str(lib), current]) if current else str(lib)
    for name in ("libblasfeo.so", "libhpipm.so", "libqpOASES_e.so", "libacados.so"):
        path = lib / name
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    return True


def require_acados() -> None:
    if not configure_acados():
        raise RuntimeError(
            "acados is not built, and Eq. 15 is specified to use it:\n"
            '  "The KSO is solved with the SQP solver in acados."  -- Section II-G\n'
            f"Expected {ACADOS_ROOT / 'lib' / 'libacados.so'}.\n"
            "Build it with:  bash scripts/setup_acados.sh"
        )
