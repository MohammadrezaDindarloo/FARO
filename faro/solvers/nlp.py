"""Backend-agnostic NLP interface.

Eqs. 14, 15 and 17 are three different optimization problems solved by three
different solvers in the paper:

    "The KSO is solved with the SQP solver in acados, whereas the TO is solved
     with the SQP solver Hippo. The mode/edge optimizations (14) are solved with
     Ipopt."   -- Section II-G

We use Ipopt for all three to start with (Hippo is not public), so the thing that
matters is that nothing ABOVE this module knows that. A problem is described by its
variables, cost, and a `ConstraintBlock`; the backend is named in a config file.
Swapping in acados later means writing a second `_solve_*` function and changing one
YAML key -- not touching Eq. 14, 15 or 17.

WHY THE RESULT CARRIES `converged` SEPARATELY FROM THE SOLUTION
---------------------------------------------------------------
Section II-D defines feasibility by solver behaviour, not by a residual:

    "A contact mode c is said to be infeasible if the resulting nonlinear program
     does not converge within a number of maximum iterations."

So the verdict IS the solver status, and it has to survive to the caller intact.
`SolveResult` therefore reports `status` and `iterations` verbatim rather than
reducing them to a bool, because the difference between `Infeasible_Problem_Detected`
(the geometry genuinely does not admit this mode) and `Maximum_Iterations_Exceeded`
(we may simply have started somewhere bad) is the difference between a true negative
and a false one -- and the paper's own evaluation hinges on that rate:

    "The few observed false negatives were primarily associated with poor
     initialization or insufficient solver iterations in the KSO."  -- Section IV-C
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from faro.constraints.block import ConstraintBlock
from faro.utils.config import load_config


@dataclass
class NLPProblem:
    """One nonlinear program, in the form every FARO stage produces.

    Attributes
    ----------
    name : for diagnostics, e.g. "mode/left_foot->floor".
    variables : a single CasADi SX column vector of decision variables.
    cost : scalar SX expression.
    block : all constraints. `eq` rows are held at 0, `ineq` rows at <= 0 --
        FARO's convention throughout, and the paper's own (Eq. 14 writes
        `contact(q, c) <= 0`).
    x0 : initial guess. Not optional and not defaulted to zeros: a zero
        configuration is not even a valid quaternion, and the paper is explicit
        that initialization is where false negatives come from.
    lbx, ubx : simple variable bounds. Everything the paper states as a constraint
        lives in `block`; these exist for box bounds a solver handles natively.
    """

    name: str
    variables: ca.SX
    cost: ca.SX
    block: ConstraintBlock
    x0: np.ndarray
    lbx: np.ndarray | None = None
    ubx: np.ndarray | None = None

    @property
    def n_var(self) -> int:
        return int(self.variables.shape[0])


@dataclass
class SolveResult:
    """What came back, including the parts that decide feasibility."""

    converged: bool
    x: np.ndarray
    cost: float
    iterations: int
    status: str
    wall_time: float
    max_eq_violation: float = 0.0
    max_ineq_violation: float = 0.0
    worst_row: str = ""
    backend: str = "ipopt"
    #: Per-row constraint values at the returned point, keyed by label.
    residuals: dict[str, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        """Section II-D's verdict: converged means feasible, anything else does not."""
        return self.converged

    def summary(self) -> str:  # pragma: no cover - display only
        verdict = "FEASIBLE" if self.converged else "INFEASIBLE"
        line = (
            f"{verdict:10s} status={self.status:28s} iters={self.iterations:4d} "
            f"cost={self.cost:9.4f}  t={self.wall_time * 1e3:7.1f} ms"
        )
        if not self.converged and self.worst_row:
            line += f"\n           worst row: {self.worst_row}"
        return line


# =============================================================================
# Ipopt
# =============================================================================
#: Ipopt return strings that mean "this converged". `Solve_Succeeded` is the clean
#: case; `Solved_To_Acceptable_Level` means it hit `acceptable_tol` rather than
#: `tol`, which for a FEASIBILITY question is still a yes -- we are asking whether a
#: configuration exists, not for the last digits of the optimum. Everything else,
#: including `Maximum_Iterations_Exceeded`, is a no, per Section II-D.
_CONVERGED_STATUSES = frozenset({"Solve_Succeeded", "Solved_To_Acceptable_Level"})


def solve(problem: NLPProblem, config: str | dict = "ipopt_default") -> SolveResult:
    """Solve `problem` with the backend named in `config`."""
    cfg = load_config("solvers", config) if isinstance(config, str) else config
    backend = cfg.get("backend", "ipopt")
    if backend != "ipopt":
        raise NotImplementedError(
            f"solver backend {backend!r} is not implemented. Only 'ipopt' exists today; "
            f"the paper additionally uses acados (KSO) and Hippo (TO), and this "
            f"interface is shaped so adding one means writing a _solve_* function here."
        )
    return _solve_ipopt(problem, cfg)


def _solve_ipopt(problem: NLPProblem, cfg: dict) -> SolveResult:
    block = problem.block
    g_parts, lbg_parts, ubg_parts, labels = [], [], [], []

    if block.n_eq:
        g_parts.append(block.eq)
        lbg_parts.append(np.zeros(block.n_eq))
        ubg_parts.append(np.zeros(block.n_eq))
        labels += list(block.eq_labels)
    if block.n_ineq:
        g_parts.append(block.ineq)
        lbg_parts.append(np.full(block.n_ineq, -np.inf))
        ubg_parts.append(np.zeros(block.n_ineq))
        labels += list(block.ineq_labels)

    g = ca.vertcat(*g_parts) if g_parts else ca.SX.zeros(0)
    lbg = np.concatenate(lbg_parts) if lbg_parts else np.zeros(0)
    ubg = np.concatenate(ubg_parts) if ubg_parts else np.zeros(0)

    options = dict(cfg.get("casadi") or {})
    options["ipopt"] = dict(cfg.get("ipopt") or {})

    solver = ca.nlpsol(
        "faro_nlp", "ipopt",
        {"x": problem.variables, "f": problem.cost, "g": g},
        options,
    )

    args = {"x0": problem.x0, "lbg": lbg, "ubg": ubg}
    if problem.lbx is not None:
        args["lbx"] = problem.lbx
    if problem.ubx is not None:
        args["ubx"] = problem.ubx

    start = time.perf_counter()
    solution = solver(**args)
    elapsed = time.perf_counter() - start

    stats = solver.stats()
    status = str(stats.get("return_status", "unknown"))
    x = np.asarray(solution["x"]).ravel()

    # Evaluate every row at the returned point. Cheap, and it is the difference
    # between "infeasible" and "infeasible BECAUSE the left hand cannot reach
    # box_left" -- which is the only form of the answer that is actionable.
    residuals: dict[str, float] = {}
    max_eq = max_ineq = 0.0
    worst_row = ""
    if labels:
        g_fn = ca.Function("g", [problem.variables], [g])
        values = np.asarray(g_fn(solution["x"])).ravel()
        residuals = {label: float(v) for label, v in zip(labels, values)}
        for i, label in enumerate(labels):
            violation = abs(values[i]) if i < block.n_eq else max(0.0, values[i])
            if i < block.n_eq:
                if violation > max_eq:
                    max_eq, worst_row = violation, f"{label} = {values[i]:+.6f}"
            elif violation > max_ineq:
                max_ineq, worst_row = violation, f"{label} = {values[i]:+.6f}"

    return SolveResult(
        converged=status in _CONVERGED_STATUSES,
        x=x,
        cost=float(solution["f"]),
        iterations=int(stats.get("iter_count", -1)),
        status=status,
        wall_time=elapsed,
        max_eq_violation=max_eq,
        max_ineq_violation=max_ineq,
        worst_row=worst_row,
        backend="ipopt",
        residuals=residuals,
    )
