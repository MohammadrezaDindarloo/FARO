"""Root conftest -- imported by pytest before any test module.

This file exists for exactly one reason, and the ordering is the whole point.

`faro/utils/determinism.py` pins the BLAS to a single thread so Ipopt's verdicts are
reproducible; Eq. 14's answer IS a solver status, and a threaded BLAS's
non-deterministic reductions were observed flipping the same contact mode between
`Solve_Succeeded` and `Infeasible_Problem_Detected` within one process. But
`OMP_NUM_THREADS` is only read when the BLAS library loads, which happens on the
first `import numpy`. Set it afterwards and it does nothing -- measured, not assumed.

Every test module imports numpy at the top. pytest imports the root conftest BEFORE
collecting them, so importing `faro` here is the earliest hook available and the only
place the pin can still take effect. Without it, the suite runs multi-threaded and any
test whose assertion depends on a solver status becomes flaky in a way that looks like
a real regression.
"""

import faro  # noqa: F401  -- imported for its import-time BLAS pin, see above
