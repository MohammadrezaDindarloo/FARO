"""Make Ipopt's verdicts reproducible.

THE BUG THIS EXISTS FOR
-----------------------
Eq. 14's answer is a solver STATUS (Section II-D: a mode is infeasible if the NLP
"does not converge within a number of maximum iterations"), so anything that makes
the solver take a different path can change the answer. On this cluster, it did:

    check(scene, far_reach) x 4  ->  {Solve_Succeeded, Infeasible_Problem_Detected}

The same mode, the same scene, the same process. The NLP itself was verified
byte-identical across builds -- x0, every constraint row, the full Jacobian, and the
cost all hash the same -- so the variation was entirely inside Ipopt's linear
algebra. MUMPS calls a threaded BLAS, thread scheduling changes the order of
floating-point reductions, and on a problem sitting near its feasibility boundary
that is enough to flip which side of it the solve lands on.

Pinning the BLAS to one thread makes it deterministic: 5/5 identical verdicts for
the two modes that had been flipping.

Why this matters more than it looks: Alg. 1 CACHES these verdicts. A
non-deterministic filter does not produce occasional noise -- it produces a
permanent, arbitrary answer, because whichever verdict came back first is the one
the tree search keeps and reuses for the rest of the run.

WHY IT HAS TO RUN THIS EARLY
----------------------------
`OMP_NUM_THREADS` is read when the threading runtime initializes, which happens
when the BLAS library is first loaded -- in practice, on `import numpy`. Setting it
afterwards has no effect, which was measured and not assumed: with the variables set
after `import numpy`, the verdicts went straight back to flipping. So this module is
imported from `faro/__init__.py` before anything else, and it reports honestly when
it was too late rather than pretending to have worked.
"""

from __future__ import annotations

import os
import sys

#: Every threading knob a BLAS shipped by conda-forge might read.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def blas_already_loaded() -> bool:
    """True if a module that pulls in BLAS has been imported already."""
    return any(name in sys.modules for name in ("numpy", "scipy", "casadi", "pinocchio"))


def ensure_single_threaded_blas(*, warn: bool = True) -> bool:
    """Pin the BLAS to one thread. Returns True if it took effect.

    Not silently best-effort: if BLAS is already loaded the setting does nothing, and
    a solver whose answers quietly stopped being reproducible is exactly the failure
    this module exists to prevent. In that case it warns and returns False.

    Set any of the variables yourself to override -- an explicit choice is respected,
    including an explicit choice to run multi-threaded.
    """
    already_set = [v for v in _THREAD_VARS if v in os.environ]
    if already_set:
        return all(os.environ[v] == "1" for v in already_set)

    too_late = blas_already_loaded()
    for var in _THREAD_VARS:
        os.environ[var] = "1"

    if too_late and warn:
        import warnings

        warnings.warn(
            "FARO pinned the BLAS to one thread, but numpy/casadi/pinocchio was "
            "already imported, so the setting will not take effect in this process. "
            "Ipopt's verdicts may not be reproducible: the same contact mode can come "
            "back Solve_Succeeded on one call and Infeasible_Problem_Detected on the "
            "next, and Alg. 1 caches whichever it saw first. Import `faro` before "
            "numpy, or export OMP_NUM_THREADS=1 in the shell.",
            RuntimeWarning,
            stacklevel=2,
        )
    return not too_late
