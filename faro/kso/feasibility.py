"""The KSO filter of Section III -- Eq. 15 solved, with a verdict and a cache.

    "3) kinematic sequence feasibility, KSO, using (15)"   -- Section III, filter list

Third and most expensive of the feasibility filters, and the last one before the TO.
It keeps Eq. 14's restart policy, its independent penetration audit, and its rule that
the verdict IS the solver status (Section II-D).

WHAT IT NO LONGER SHARES WITH EQ. 14, AND WHY
---------------------------------------------
The SOLVER. Section II-G assigns Ipopt to Eq. 14 and acados to Eq. 15, and this module
now honours that: the Ipopt path and its outer witness-refresh loop are gone. That loop
existed only because an interior-point method cannot change its constraint functions
mid-solve; an SQP re-linearizes every iteration, so carrying the GJK witnesses as
acados parameters puts Schulman's procedure inside the solve where the paper has it.

Measured on `reach` (2026-08-04): one SQP call, 119 ms, all residuals at machine
precision -- against up to 20 full NLP solves before. Table I reports 0.14-1.93 s.


WHY THE CACHE KEY IS THE SEQUENCE
---------------------------------
Table II attributes the method's advantage on the hard task to caching. A sequence is
frozen and ordered precisely so it can be a key directly; see `ContactSequence`.

WHAT A KSO FAILURE DOES *NOT* TELL YOU
--------------------------------------
Eq. 15 has no forces, so `infeasible` here never means "the robot could not hold
this" -- only "no sequence of configurations satisfies the contacts, the no-slip
condition, collision and the joint limits at once". Balance is the TO's job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from faro.core.modes import ContactSequence
from faro.kso.problem import (
    SequenceWeights, edge_at, mode_at, n_knots, symbolic_knots,
)
from faro.mode_edge.feasibility import _object_poses, _perturb
from faro.scene.collision import SceneCollisionModel
from faro.scene.scene import Scene
from faro.solvers.nlp import SolveResult

#: Eq. 15 is solved by acados, per Section II-G: "The KSO is solved with the SQP solver
#: in acados [26]." Ipopt stays where the paper puts it -- Eq. 14, in
#: `faro/mode_edge/feasibility.py` -- and is no longer an option here.
DEFAULT_SOLVER_CONFIG = "acados_kso"


@dataclass
class KSOReport:
    """One sequence verdict, with everything needed to understand or animate it."""

    sequence: ContactSequence
    feasible: bool
    result: SolveResult
    #: One `{"robot": q_r, "<object>": q_o}` per step, in sequence order. Present even
    #: when infeasible -- the last iterate usually shows WHICH step could not be met.
    configurations: list[dict[str, np.ndarray]] = field(default_factory=list)
    cached: bool = False
    refreshes: int = 1
    wall_time: float = 0.0
    #: Deepest interpenetration over all pairs and ALL STEPS. NaN means not measured
    #: because the solve did not converge -- see the Eq. 14 report for why that is NaN
    #: rather than 0.0.
    max_penetration: float = 0.0
    penetrating_pair: str = ""
    #: Which step produced `max_penetration`; -1 when nothing was measured.
    penetrating_step: int = -1

    def explain(self) -> str:
        head = f"sequence of {len(self.sequence)} modes"
        if self.feasible:
            return (
                f"{head}\n  FEASIBLE  ({self.wall_time:.2f} s, {self.result.iterations} "
                f"SQP iteration(s) via {self.result.backend})"
            )
        if self.result.status == "Maximum_Iterations_Exceeded":
            why = ("hit the iteration cap. A statement about THIS solve, not the "
                   "sequence -- Section IV-C attributes the paper's false negatives "
                   "to exactly this, and a KSO has far more variables to get lost in "
                   "than a single mode does.")
        elif self.result.status == "Infeasible_Problem_Detected":
            why = ("Ipopt converged to a nonzero minimum of constraint violation. "
                   "Eq. 15 is nonconvex, so this is a local certificate, not a proof.")
        else:
            why = f"solver returned {self.result.status}."
        return (f"{head}\n  INFEASIBLE  {why}\n  worst row: {self.result.worst_row or 'n/a'}"
                f"\n  (Eq. 15 has no forces -- this is never a statement about balance.)")

    def step_of(self, label: str) -> int:
        """Which step a constraint label belongs to, or -1. Labels carry `_k<s>`."""
        for s in range(len(self.sequence)):
            if f"_k{s}" in label or f"[{s}->" in label:
                return s
        return -1


class KSOCache:
    """Alg. 1's cache for filter 3, keyed by the sequence itself."""

    def __init__(self) -> None:
        self._entries: dict[ContactSequence, KSOReport] = {}
        self.hits = 0
        self.misses = 0

    def get(self, sequence: ContactSequence) -> KSOReport | None:
        report = self._entries.get(sequence)
        if report is None:
            self.misses += 1
            return None
        self.hits += 1
        return KSOReport(
            sequence=report.sequence, feasible=report.feasible, result=report.result,
            configurations=report.configurations, cached=True, refreshes=report.refreshes,
            wall_time=report.wall_time, max_penetration=report.max_penetration,
            penetrating_pair=report.penetrating_pair, penetrating_step=report.penetrating_step,
        )

    def put(self, report: KSOReport) -> None:
        self._entries[report.sequence] = report

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def feasible_count(self) -> int:
        return sum(1 for r in self._entries.values() if r.feasible)

    def summary(self) -> str:  # pragma: no cover - display only
        total = self.hits + self.misses
        rate = 100.0 * self.hits / total if total else 0.0
        return (f"kso cache: {len(self)} entries ({self.feasible_count} feasible), "
                f"{self.hits} hits / {total} lookups ({rate:.0f}%)")


def _audit(scene, model, states) -> tuple[float, str, int]:
    """Worst interpenetration across every step, audited over ALL pairs."""
    worst, where, step = 0.0, "", -1
    for s, state in enumerate(states):
        penetration, pair = model.worst_penetration(state["robot"], _object_poses(scene, state))
        if penetration < worst:
            worst, where, step = penetration, pair, s
    return worst, where, step


#: Compiled acados OCPs, keyed by what changes their generated C. Code generation for
#: a 689-pair sequence takes several MINUTES while the solve takes ~100 ms, so building
#: per call would make the filter useless; Alg. 1 asks for the same sequence often.
_BUILT: dict[tuple, object] = {}


def _build_key(scene: Scene, sequence: ContactSequence, goal) -> tuple:
    settings = scene.collision
    return (
        scene.name, sequence.label(),
        settings.get("activation_distance", None),
        bool(settings.get("relax_contact_pairs", True)),
        float(settings.get("margin", 0.0)),
        float(settings.get("contact_pair_margin", -1.0e-3)),
        repr(goal),
    )


def _acados_solve(scene, sequence, guess, weights, q_init, goal, collision):
    """Eq. 15 through the acados SQP -- the paper's solver (Section II-G).

    THERE IS NO REFRESH LOOP HERE, AND THAT IS THE POINT
    ----------------------------------------------------
    The Ipopt path wrapped this in up to 20 complete NLP solves, re-freezing the GJK
    witnesses between each, because an interior-point method cannot change its
    constraint functions mid-solve. An SQP re-solves a QP at every iterate already, so
    carrying the witnesses as `model.p` puts Schulman's sequential convex procedure
    INSIDE the iterations, which is where the paper has it. One call, one solve.
    """
    from faro.kso import acados_solver as backend

    key = _build_key(scene, sequence, goal)
    built = _BUILT.get(key)
    if built is None:
        built = backend.build(scene, sequence, weights=weights, q_init=guess[:guess.size //
                              n_knots(sequence)] if q_init is None else q_init,
                              goal=goal, collision=collision)
        _BUILT[key] = built
    result = backend.solve(built, guess, scene)
    return result, built.syms, 1


def check(
    scene: Scene,
    sequence: ContactSequence,
    *,
    cache: KSOCache | None = None,
    weights: SequenceWeights | None = None,
    q0: np.ndarray | None = None,
    collision: bool | SceneCollisionModel = True,
    restarts: int = 2,
    seed: int = 0,
    q_init: np.ndarray | None = None,
    goal: object | None = None,
    solver_config: str | dict = DEFAULT_SOLVER_CONFIG,
) -> KSOReport:
    """Filter 3 of Section III: is this whole contact sequence kinematically feasible?

    `q0` is a STACKED initial guess, one configuration per step. Leaving it None
    starts every step at the grounded nominal -- which is a deliberately poor guess
    for anything but the first step, and is why `faro.kso.warm_start` exists.
    Section IV-C names initialization as the source of the paper's false negatives,
    and a KSO has `len(sequence)` times the variables of Eq. 14 to get lost in.

    A `q0` other than the default is NOT cached against, for the same reason as in
    Eq. 14: a verdict obtained from a lucky warm start must not be served later to a
    query that never had one.
    """
    if cache is not None and q0 is None:
        hit = cache.get(sequence)
        if hit is not None:
            return hit

    started = time.perf_counter()
    sequence.validate(scene)
    weights = weights if weights is not None else SequenceWeights.from_scene(scene)
    model = collision if isinstance(collision, SceneCollisionModel) else (
        SceneCollisionModel.cached(scene) if collision else None
    )
    always_active = model.always_active_pairs() if model is not None else None

    n_steps = n_knots(sequence)
    if q0 is not None:
        start = np.asarray(q0, dtype=float)
    else:
        nominal = symbolic_knots(scene, n_steps)[0].nominal()
        start = np.tile(nominal, n_steps)

    rng = np.random.default_rng(seed)
    n = start.size // n_steps
    best = None
    for attempt in range(max(1, restarts + 1)):
        if attempt == 0:
            point = start
        else:
            # Perturb each step independently: a sequence gets stuck for step-local
            # reasons (one hand on the wrong side of Eq. 7a's branch), and moving
            # every step by the same offset would not break that.
            point = np.concatenate([
                _perturb(scene, start[s * n:(s + 1) * n], rng) for s in range(n_steps)
            ])
        result, syms, refreshes = _acados_solve(
            scene, sequence, point, weights, q_init, goal,
            model if model is not None else False,
        )
        best = (result, syms, refreshes)
        if result.converged:
            break

    result, syms, refreshes = best
    configurations = [syms[s].split(result.x[s * n:(s + 1) * n]) for s in range(n_steps)]

    penetration, pair, step = 0.0, "", -1
    if model is not None:
        if result.converged:
            penetration, pair, step = _audit(scene, model, configurations)
        else:
            penetration, pair = float("nan"), "not audited: the solve did not converge"

    report = KSOReport(
        sequence=sequence, feasible=result.converged, result=result,
        configurations=configurations, refreshes=refreshes,
        wall_time=time.perf_counter() - started,
        max_penetration=penetration, penetrating_pair=pair, penetrating_step=step,
    )
    if cache is not None and q0 is None:
        cache.put(report)
    return report


def warm_start(scene: Scene, sequence: ContactSequence, **kwargs) -> np.ndarray:
    """A stacked initial guess of K+1 knots, built from the EDGE filter E.

    WHY EDGES AND NOT MODES
    -----------------------
    Eq. 15 constrains knot `s` by `contact(q_s, c_{s-1} u c_s) <= 0` -- the EDGE, not
    the mode. Seeding knot `s` with a solution of Eq. 14 on `c_s` alone therefore
    hands the SQP a point that satisfies the WRONG contact set: every knot arrives
    violating its own 7a/7b rows, and the first thing the solver has to do is repair
    them before it can touch the cost. Measured on `reach`, that shows up as
    constraints driven to 1e-16 while stationarity sticks at 2.18 -- the solver
    spending its whole budget on feasibility restoration.

    Seeding from `edge_at(sequence, s)` instead means each knot starts already
    satisfying its own contact equalities, leaving the KSO to reconcile Eq. 8's
    coupling and Eq. 9's collisions -- which is the job Eq. 15 actually adds over
    Eq. 14.

    This is also what Alg. 1 does. Filter E runs on every edge before a sequence is
    ever assembled, so with a `cache` these answers cost nothing by the time the KSO
    sees them; `c_K := c_{K-1}` makes `edge_at(., K)` the last mode alone.
    """
    from faro.mode_edge.feasibility import check as check_eq14

    def stack(report):
        state = report.configurations
        return np.concatenate(
            [state["robot"]] + [state[name] for name in sorted(scene.objects)]
        )

    # Knot 0 is pinned to q_init by Eq. 15 and carries no path constraints, so the
    # mode filter on c_0 is the right seed for it; knots 1..K carry the edge.
    pieces = [stack(check_eq14(scene, mode_at(sequence, 0), **kwargs))]
    for s in range(1, n_knots(sequence)):
        pieces.append(stack(check_eq14(scene, edge_at(sequence, s), **kwargs)))
    return np.concatenate(pieces)
