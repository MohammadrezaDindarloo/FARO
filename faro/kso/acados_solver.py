"""Eq. 15 solved with acados SQP -- the paper's own solver (Section II-G).

    "Both (15) and (17) are implemented using a direct multiple-shooting
     transcription. ... The KSO is solved with the SQP solver in acados [26]."

MULTIPLE SHOOTING, ONE PHASE PER KNOT
-------------------------------------
Eq. 15's knots do not carry the same constraints: s = 0 has none (it is pinned by
`q_0 = q_init`) and each s in 1..K carries `c_{s-1} u c_s`, whose active-pair count
varies along the sequence. A plain `AcadosOcp` applies ONE constraint function to
every intermediate stage, so `AcadosMultiphaseOcp` with `N_list = [1]*K` is what lets
each knot state its own constraints without padding rows that mean nothing where they
were not built.

THE CONTROL IS AN ENCODING, NOT PART OF EQ. 15
----------------------------------------------
Eq. 15 has no controls. An acados path constraint sees only `(x_k, u_k)` and never
`x_{k+1}`, while Eq. 8 couples `q_s` with `q_{s+1}` -- so the increment has to be in
scope:

    x_k = q_s,   u_k = q_{s+1} - q_s,   x_{k+1} = x_k + u_k

exact and linear: a change of variables, not a model. No cost is placed on `u`,
because Eq. 15 places none.

WHY THE WITNESS DATA IS A PARAMETER
-----------------------------------
Eq. 9 freezes a GJK query. Baked in as constants it makes every re-linearization a
new expression and, for a code-generating solver, a new C compilation -- which is
what forced the Ipopt path's outer loop of up to 20 complete NLP solves. As
`model.p` the code is generated once and re-linearizing is a memcpy, which puts
Schulman's sequential convex procedure inside the SQP iterations where the paper has
it. See docs/kso_acados_design.md.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import pathlib
import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from faro.constraints.block import ConstraintBlock, merge
from faro.constraints.collision import witness_parameter_symbols
from faro.core.modes import ContactSequence
from faro.kso.problem import (
    SequenceWeights, contact_blocks, edge_at, limit_block, mode_at, n_knots,
    no_slip_blocks, symbolic_knots,
)
from faro.scene.collision import SceneCollisionModel
from faro.scene.scene import Scene
from faro.utils.acados_env import require_acados
from faro.solvers.nlp import SolveResult

#: Generated C lands beside the acados checkout, outside the package tree.
ACADOS_GEN = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "acados_generated"

_CONVERGED = {0}
_STATUS = {0: "Solve_Succeeded", 1: "Invalid_Number_Detected",
           2: "Maximum_Iterations_Exceeded", 3: "Minimum_Step_Size_Reached",
           4: "QP_Solver_Failed", 5: "Solver_Created_Failed"}


@contextlib.contextmanager
def _build_lock(path: pathlib.Path):
    """Serialize builders of one tag, so two runs cannot clobber each other's export.

    An advisory `flock`, held for the whole generate-and-compile. The second process
    waits rather than racing, and by the time it wakes the first has published a .so
    it can simply load. Advisory locks are per-open-file and released automatically if
    the holder dies, so an interrupted build leaves no stale lock -- it leaves a
    directory with no .so, which the caller already treats as "rebuild".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _bounds(block: ConstraintBlock):
    """FARO's `eq == 0` / `ineq <= 0` as acados' two-sided `lh <= h <= uh`."""
    parts, lh, uh, labels = [], [], [], []
    if block.n_eq:
        parts.append(block.eq)
        lh.append(np.zeros(block.n_eq)); uh.append(np.zeros(block.n_eq))
        labels += list(block.eq_labels)
    if block.n_ineq:
        parts.append(block.ineq)
        lh.append(np.full(block.n_ineq, -1.0e9)); uh.append(np.zeros(block.n_ineq))
        labels += list(block.ineq_labels)
    h = ca.vertcat(*parts) if parts else ca.SX.zeros(0)
    return (h, np.concatenate(lh) if lh else np.zeros(0),
            np.concatenate(uh) if uh else np.zeros(0), labels)


@dataclass
class AcadosKSO:
    """A compiled acados OCP plus everything needed to re-linearize it."""

    solver: object
    syms: list
    pairs: list          # frozen collision pair list, shared by every knot
    model: SceneCollisionModel | None
    n_var: int
    n_knots: int
    labels: list = field(default_factory=list)


def build(scene: Scene, sequence: ContactSequence, *,
          weights: SequenceWeights | None = None, q_init=None, goal=None,
          collision: bool | SceneCollisionModel = True, alignment: str = "column",
          yaw: str = "column", tol: float = 1e-6, max_iter: int = 200,
          nlp_solver_type: str = "SQP", hessian_approx: str = "EXACT",
          globalization: str | None = "MERIT_BACKTRACKING",
          levenberg_marquardt: float = 0.0, qp_iter_max: int = 500,
          regularize_method: str | None = None,
          verbose: bool = False) -> AcadosKSO:
    """Generate and compile the multi-phase OCP for `sequence`.

    `hessian_approx` defaults to EXACT rather than GAUSS_NEWTON. Eq. 15 puts no cost
    on `u`, so a Gauss-Newton Hessian -- built from the cost residual alone -- has an
    exactly zero block in the `u` directions and HPIPM stalls immediately
    (ACADOS_MINSTEP at SQP iteration 1, measured). EXACT takes curvature from the
    constraints too. Putting a cost on `u` would fix it as well and is NOT available:
    it would change Eq. 15.

    `levenberg_marquardt` adds `lambda*I` to the Hessian INSIDE the solver. It is a
    numerical regularization of the QP subproblem, not a term in the objective: the
    cost acados reports and the KKT point it converges to are still Eq. 15's. This is
    the sanctioned way to deal with the zero curvature in `u` -- unlike a ridge on `u`,
    which would change the problem being solved.
    """
    require_acados()
    from acados_template import AcadosModel, AcadosMultiphaseOcp, AcadosOcp, AcadosOcpSolver

    sequence.validate(scene)
    weights = weights if weights is not None else SequenceWeights.from_scene(scene)
    syms = symbolic_knots(scene, n_knots(sequence))
    K, nk = len(sequence), n_knots(sequence)
    n = syms[0].n_var

    model_c = collision if isinstance(collision, SceneCollisionModel) else (
        SceneCollisionModel.cached(scene) if collision else None)

    diagonal = weights.regularization.diagonal(scene)
    q_nom = syms[0].nominal()
    q_init = np.asarray(q_nom if q_init is None else q_init, dtype=float)

    pairs = []
    if model_c is not None:
        settings = scene.collision
        # Eq. 15's own cutoff, which is NOT Eq. 14's -- see `activation_distance_for`.
        activation = model_c.activation_distance_for(settings, "kso")
        # `q_init` is the WHOLE scene configuration (robot + objects); the collision
        # queries want the robot part and the object poses separately.
        from faro.mode_edge.feasibility import _object_poses

        init_state = syms[0].split(q_init)
        pairs = model_c.fixed_pairs(
            init_state["robot"], _object_poses(scene, init_state),
            activation_distance=activation,
            always_active=model_c.always_active_pairs())

    slip = no_slip_blocks(scene, sequence, syms, yaw=yaw)
    slip_by_stage = {}
    for b in slip:
        slip_by_stage.setdefault(int(b.name.split("[")[1].split("->")[0]), []).append(b)

    ocp = AcadosMultiphaseOcp(N_list=[1] * K)
    # EVERYTHING THAT CHANGES THE GENERATED C GOES IN THE TAG. It used to be scene,
    # sequence and pair count only -- which meant flipping `relax_contact_pairs` or
    # `margin` reused a .so compiled for the OTHER problem and reported its answer as
    # this one's. Those settings were measured this session to decide feasibility
    # outright, so a silent stale load is the worst failure mode available here.
    fingerprint = "|".join(str(x) for x in (
        scene.name, sequence.label(), len(pairs), n, nk,
        SceneCollisionModel.activation_distance_for(scene.collision, "kso"),
        scene.collision.get("relax_contact_pairs"),
        scene.collision.get("margin"), scene.collision.get("contact_pair_margin"),
        alignment, yaw, repr(goal), nlp_solver_type, hessian_approx, globalization,
        levenberg_marquardt, qp_iter_max, regularize_method, tol, max_iter,
    ))
    tag = hashlib.sha1(fingerprint.encode()).hexdigest()[:10]
    labels = []

    def knot_blocks(s, sym, params):
        if s == 0:
            return []
        out = [*contact_blocks(sym, edge_at(sequence, s).active_pairs(scene),
                               alignment=alignment),
               limit_block(sym), sym.unit_quaternion_rows()]
        if model_c is not None and pairs:
            out += model_c.parametric_blocks(
                sym, pairs, params,
                margin=float(scene.collision.get("margin", 0.0)),
                relaxed=(model_c.contact_pairs(mode_at(sequence, s))
                         if scene.collision.get("relax_contact_pairs", True) else None),
                relaxed_margin=float(scene.collision.get("contact_pair_margin", -1e-3)))
        if s == nk - 1 and goal is not None:
            out += contact_blocks(sym, goal.active_pairs(scene), alignment=alignment)
        return out

    for phase in range(K):
        x = ca.SX.sym(f"x{phase}", n)
        u = ca.SX.sym(f"u{phase}", n)
        p = witness_parameter_symbols(len(pairs)) if pairs else ca.SX.sym(f"p{phase}", 0)

        m = AcadosModel()
        m.name = f"kso_p{phase}_{tag}"
        m.x, m.u, m.p = x, u, p
        m.disc_dyn_expr = x + u

        sub = AcadosOcp()
        sub.model = m
        sub.cost.cost_type = "LINEAR_LS"
        sub.cost.W = np.block([[np.diag(diagonal), np.zeros((n, n))],
                               [np.zeros((n, n)), np.zeros((n, n))]])
        sub.cost.Vx = np.vstack([np.eye(n), np.zeros((n, n))])
        sub.cost.Vu = np.vstack([np.zeros((n, n)), np.eye(n)])
        sub.cost.yref = np.concatenate([q_nom, np.zeros(n)])
        sub.parameter_values = np.zeros(p.shape[0])

        blocks = knot_blocks(phase, syms[phase], p) + slip_by_stage.get(phase, [])
        if blocks:
            h, lh, uh, lab = _bounds(merge(f"knot{phase}", blocks))
            h = ca.substitute(h, syms[phase].variables, x)
            if phase + 1 < nk:
                h = ca.substitute(h, syms[phase + 1].variables, x + u)
            if phase == 0:
                # THE INITIAL STAGE IS A SEPARATE CONSTRAINT IN ACADOS, and phase 0's
                # only stage IS the initial stage (N_list = [1]*K). `con_h_expr`
                # applies from stage 1 on, so setting it here generated and compiled
                # the rows and then never applied them: Eq. 8 across the FIRST
                # transition silently vanished. Measured on `reach` -- the feet slid
                # 521.656 mm from q_0 to q_1 while acados reported eq = 2.2e-16,
                # because the rows it was reporting on did not include these.
                sub.model.con_h_expr_0 = h
                sub.constraints.lh_0, sub.constraints.uh_0 = lh, uh
            else:
                sub.model.con_h_expr = h
                sub.constraints.lh, sub.constraints.uh = lh, uh
            labels.append(lab)
        else:
            labels.append([])

        if phase == 0:
            sub.constraints.x0 = q_init          # q_0 = q_init, Eq. 15
        else:
            sub.constraints.idxbx_0 = np.array([], dtype=int)
            sub.constraints.lbx_0 = np.array([]); sub.constraints.ubx_0 = np.array([])

        if phase == K - 1:
            sub.cost.cost_type_e = "LINEAR_LS"
            sub.cost.W_e = np.diag(diagonal)
            sub.cost.Vx_e = np.eye(n); sub.cost.yref_e = q_nom.copy()
            term = knot_blocks(nk - 1, syms[nk - 1], p)
            if term:
                h_e, lh_e, uh_e, lab_e = _bounds(merge("terminal", term))
                sub.model.con_h_expr_e = ca.substitute(h_e, syms[nk - 1].variables, x)
                sub.constraints.lh_e, sub.constraints.uh_e = lh_e, uh_e
                labels.append(lab_e)
        ocp.set_phase(sub, phase)

    ocp.solver_options.nlp_solver_type = nlp_solver_type
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = hessian_approx
    # REGULARIZATION, and the two knobs interact. `CONVEXIFY` replaces the Hessian
    # rather than shifting it, so it makes `levenberg_marquardt` inert: measured on
    # `reach`, lm from 0 to 1.0 gave bit-identical results down to the QP iteration
    # count (13 every time). If you need LM to do anything, this must not be CONVEXIFY.
    method = regularize_method if regularize_method is not None else (
        "CONVEXIFY" if hessian_approx == "EXACT" else None)
    if method:
        ocp.solver_options.regularize_method = method
    if levenberg_marquardt:
        ocp.solver_options.levenberg_marquardt = float(levenberg_marquardt)
    # acados defaults to 50, and a KSO knot carries ~230 constraint rows: measured,
    # the third QP hit exactly 50, returned an inaccurate step, and the SQP then died
    # with ACADOS_MINSTEP. This is the QP's iteration budget, not Eq. 15.
    ocp.solver_options.qp_solver_iter_max = int(qp_iter_max)
    if globalization:
        ocp.solver_options.globalization = globalization
    ocp.solver_options.nlp_solver_max_iter = max_iter
    ocp.solver_options.tol = tol
    ocp.solver_options.tf = float(K)
    ocp.mocp_opts.integrator_type = ["DISCRETE"] * K
    ocp.solver_options.print_level = 1 if verbose else 0

    out = ACADOS_GEN / f"kso_{K}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    export = out / "c_generated_code"
    ocp.code_export_directory = str(export)
    json_file = str(out / "ocp.json")
    library = export / "libacados_ocp_solver_multiphase_ocp.so"

    # REUSE THE COMPILED LIBRARY. Code generation plus gcc is ~10-15 minutes for a
    # 689-pair sequence while the solve is ~100 ms, so rebuilding per run makes the
    # filter unusable. The tag covers everything that changes the generated C, so a
    # hit is the same problem by construction.
    #
    # The .so is the COMPLETION MARKER: acados writes the .c files first and only
    # links the library at the very end, so a directory with .c but no .so is a build
    # that died partway (interrupted, killed, out of disk) and must be redone.
    #
    # THREE WAYS THIS RACES, ALL OBSERVED:
    #   * two processes build the same tag at once and clobber each other's export
    #   * one rebuilds while another is mid-solve, deleting the .so under it
    #   * `library.exists()` is true at check time and false by dlopen time
    # The lock serializes builders; the try/except covers the check-to-open window,
    # because on a shared filesystem no lock makes that gap truly zero. Falling back
    # to a full build is always correct -- just slow.
    lock_path = out / ".build.lock"
    with _build_lock(lock_path):
        if library.exists():
            try:
                solver = AcadosOcpSolver(ocp, json_file=json_file, build=False,
                                         generate=False, verbose=False)
            except OSError:
                solver = AcadosOcpSolver(ocp, json_file=json_file, verbose=False)
        else:
            solver = AcadosOcpSolver(ocp, json_file=json_file, verbose=False)

    return AcadosKSO(solver=solver, syms=syms, pairs=pairs, model=model_c,
                     n_var=n, n_knots=nk, labels=labels)


def solve(built: AcadosKSO, x0: np.ndarray, scene: Scene, *,
          refreshes: int = 1) -> SolveResult:
    """Run the SQP, re-linearizing Eq. 9 by writing new PARAMETER VALUES.

    `refreshes` is 1 by default and that is the point. The Ipopt path needs an outer
    loop of up to 20 complete NLP solves because an interior-point method cannot
    change its constraint functions mid-solve. Here each SQP iteration already
    re-solves a QP at the current iterate, so one call is Schulman's sequential convex
    procedure. The parameter is exposed because the witness points are refreshed
    between SQP RUNS here, not between iterations -- acados has no hook for the
    latter -- so a small number may still help on a long sequence.
    """
    n, nk = built.n_var, built.n_knots
    guess = np.asarray(x0, dtype=float).copy()
    started = time.perf_counter()
    status, iters = -1, 0

    for _ in range(max(1, refreshes)):
        stacked = guess.reshape(nk, n)
        if built.pairs and built.model is not None:
            from faro.mode_edge.feasibility import _object_poses
            for k in range(nk):
                state = built.syms[k].split(stacked[k])
                values = built.model.witness_values(
                    built.pairs, state["robot"], _object_poses(scene, state))
                # Stage k, INCLUDING the terminal stage k = nk-1 = N. Clamping to
                # nk-2 here left `con_h_expr_e` linearized around p = 0, which makes
                # every terminal Eq. 9 row `margin - 0 <= 0`: a constant with an
                # exactly zero gradient, i.e. QP degeneracy handed straight to HPIPM.
                built.solver.set(k, "p", values)
        for k in range(nk):
            built.solver.set(k, "x", stacked[k])
        for k in range(nk - 1):
            built.solver.set(k, "u", stacked[k + 1] - stacked[k])

        status = int(built.solver.solve())
        iters += int(built.solver.get_stats("sqp_iter"))
        guess = np.concatenate([np.asarray(built.solver.get(k, "x")).ravel()
                                for k in range(nk)])
        if not np.isfinite(guess).all():
            break

    return SolveResult(
        converged=status in _CONVERGED, x=guess,
        cost=float(built.solver.get_cost()), iterations=iters,
        status=_STATUS.get(status, f"acados_status_{status}"),
        wall_time=time.perf_counter() - started, backend="acados",
    )
