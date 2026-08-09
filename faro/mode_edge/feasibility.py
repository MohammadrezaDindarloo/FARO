"""The mode and edge feasibility filters M and E, plus their cache (Section III).

    "Before admission to the tree, VERIFY applies the selected feasibility filters F.
     We consider the following filters, ordered from cheaper to more expensive:
       1) contact-mode feasibility, M(c+), using (14),
       2) transition-edge feasibility, E(c(v), c+), using (14),
       ...
     Reusable outcomes are stored in the feasible and infeasible caches, K_feas and
     K_infeas, so that previously evaluated checks do not need to be solved again."

This module is the whole of filters 1 and 2. It builds Eq. 14 (`problem.py`), solves
it (`faro/solvers/nlp.py`), and returns a verdict plus the configuration that
produced it -- the configuration matters, because a feasible mode is only useful if
you can look at it and see that it is the pose you meant.

WHY THE CACHE IS NOT AN OPTIMIZATION DETAIL
-------------------------------------------
Table II is unambiguous that caching is where the method's advantage comes from on
the hard task: `M,E,KSO` expands 814.6 nodes against `KSO` alone at 218.2, and the
paper attributes the gap directly to it --

    "Since contact modes and transitions are repeatedly revisited during search,
     cached infeasibility results allow previously checked queries to be rejected
     without resolving the corresponding optimization problems. This substantially
     reduces redundant exploration and explains the large performance gap between
     the KSO and M,E,KSO variants."   -- Section IV-B

So the cache is part of the method, and it lives here rather than inside Alg. 1 so
that the tree search does not own a correctness-critical piece of Eq. 14.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import pinocchio as pin

from faro.core.modes import ContactEdge, ContactMode
from faro.mode_edge.problem import RegularizationWeights, build_problem
from faro.scene.collision import SceneCollisionModel
from faro.scene.scene import Scene
from faro.scene.symbolic import SymbolicScene
from faro.solvers.nlp import SolveResult, solve

#: Eq. 14's own solver settings. See configs/solvers/ipopt_mode_edge.yaml.
DEFAULT_SOLVER_CONFIG = "ipopt_mode_edge"


@dataclass
class FeasibilityReport:
    """One verdict, with everything needed to understand or display it."""

    target: ContactMode | ContactEdge
    feasible: bool
    result: SolveResult
    #: Solved configurations, `{"robot": q_r, "<object>": q_o}`. Present even when
    #: infeasible -- the last iterate is usually the clearest picture of WHY, and
    #: showing it beats reporting a bare False.
    configurations: dict[str, np.ndarray] = field(default_factory=dict)
    #: True when the verdict came from the cache rather than a solve.
    cached: bool = False
    #: How many solve/refresh passes Eq. 9's frozen witness data needed. 1 means the
    #: answer never moved far enough to be worth re-linearizing.
    refreshes: int = 1
    #: Wall time for the WHOLE check -- every restart, every refresh pass, the GJK
    #: queries and the audit. Reported separately from `result.wall_time`, which times
    #: only the final Ipopt call, because the two differ by two orders of magnitude and
    #: the final call is systematically the cheapest: it starts from the previous
    #: pass's answer and only has to confirm it. On `grasp` that is 0.23 s against
    #: 33 s, and quoting the small number makes the filter look ~100x cheaper than it
    #: is -- which is exactly the number Table II has to be compared against.
    wall_time: float = 0.0
    #: Deepest interpenetration in the returned configuration, audited against ALL
    #: collision pairs -- not just the ones inside `activation_distance` that got
    #: rows. 0.0 means clear; NaN means NOT MEASURED, because the solve returned a
    #: non-finite iterate and there was no configuration to audit. NaN rather than
    #: 0.0 for that case on purpose: 0.0 reads as 'clean', and a failed solve is not
    #: clean, it is unknown. (Auditing the garbage directly produced -1.8e308.)
    max_penetration: float = 0.0
    #: Which pair produced `max_penetration`.
    penetrating_pair: str = ""

    @property
    def kind(self) -> str:
        return "edge" if isinstance(self.target, ContactEdge) else "mode"

    def explain(self) -> str:
        """One-paragraph account of the verdict, naming the constraint that failed.

        `Infeasible_Problem_Detected` and `Maximum_Iterations_Exceeded` are reported
        differently on purpose. The first is the stronger claim; the second is a
        statement about this solve, and Section IV-C is explicit that such cases are
        where the paper's false negatives come from. Collapsing them into "infeasible"
        throws away the only signal distinguishing a real pruning from a lost branch.

        Neither is a PROOF, and the wording below is careful about that. Eq. 14 is
        nonconvex, so Ipopt's infeasibility detection means it converged to a local
        minimum of constraint violation that is not zero -- a local certificate, not a
        global one. We have watched a mode go from `Infeasible_Problem_Detected` to
        `Solve_Succeeded` purely by grounding q_nom, which is exactly what a local
        result is entitled to do.
        """
        head = f"{self.kind} {self.target.label()}"
        if self.feasible:
            return (
                f"{head}\n  FEASIBLE  ({self.wall_time:.2f} s over {self.refreshes} "
                f"refresh pass(es); the final one took {self.result.iterations} "
                f"iterations, {self.result.wall_time * 1e3:.0f} ms)"
            )
        if self.result.status == "Contradictory_Contacts":
            return (
                f"{head}\n  INFEASIBLE  decided on geometry, with no solve at all: "
                f"the union of the two modes asks one flat patch to lie against two "
                f"faces of the same rigid body at once. This is the SAME verdict "
                f"Eq. 14 returns -- it just costs microseconds instead of ~40 s.\n"
                f"  {self.result.worst_row}"
            )
        if self.result.status == "Maximum_Iterations_Exceeded":
            why = (
                "hit the iteration cap without converging. This is a verdict about "
                "THIS SOLVE, not about the scene -- a better initial guess or a "
                "higher max_iter may still find a configuration. Section IV-C "
                "attributes the paper's false negatives to exactly this."
            )
        elif self.result.status == "Infeasible_Problem_Detected":
            why = (
                "Ipopt converged to a nonzero minimum of constraint violation -- no "
                "configuration satisfies these contacts NEAR the points it searched. "
                "Eq. 14 is nonconvex, so this is a local certificate, not a proof."
            )
        else:
            why = f"solver returned {self.result.status}."
        return f"{head}\n  INFEASIBLE  {why}\n  worst row: {self.result.worst_row or 'n/a'}"


class FeasibilityCache:
    """Alg. 1's `K_feas` / `K_infeas`, as one dict keyed by the mode or edge itself.

    Kept as a single mapping rather than the paper's two sets because the verdict is
    a value, not a membership: storing `{target: report}` cannot represent the state
    "in both caches", which two independent sets can and which would be a silent
    contradiction. `ContactMode` and `ContactEdge` are frozen and sort-ordered
    precisely so they can serve as keys directly.
    """

    def __init__(self) -> None:
        self._entries: dict[ContactMode | ContactEdge, FeasibilityReport] = {}
        self.hits = 0
        self.misses = 0

    def get(self, target) -> FeasibilityReport | None:
        report = self._entries.get(target)
        if report is None:
            self.misses += 1
            return None
        self.hits += 1
        # Copy, so a caller cannot mark the stored entry as a fresh solve.
        return FeasibilityReport(
            target=report.target, feasible=report.feasible, result=report.result,
            configurations=report.configurations, cached=True, refreshes=report.refreshes,
            wall_time=report.wall_time,
            max_penetration=report.max_penetration, penetrating_pair=report.penetrating_pair,
        )

    def put(self, report: FeasibilityReport) -> None:
        self._entries[report.target] = report

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def feasible_count(self) -> int:
        return sum(1 for r in self._entries.values() if r.feasible)

    def summary(self) -> str:  # pragma: no cover - display only
        total = self.hits + self.misses
        rate = 100.0 * self.hits / total if total else 0.0
        return (
            f"cache: {len(self)} entries ({self.feasible_count} feasible), "
            f"{self.hits} hits / {total} lookups ({rate:.0f}%)"
        )


def _object_poses(scene: Scene, state: dict) -> dict:
    """`{name: SE3}` from a split configuration, for the numeric collision queries."""
    return {
        name: pin.SE3(
            pin.Quaternion(state[name][6], *state[name][3:6]).toRotationMatrix(),
            state[name][:3],
        )
        for name in scene.objects
    }


def _is_configuration(x: np.ndarray, *, bound: float = 1e3) -> bool:
    """Is this iterate a point worth re-linearizing at, or is it wreckage?

    `np.isfinite` alone is not enough, and that is measured rather than argued. A
    failed Eq. 14 solve returns entries around 1e308 -- finite, so the isfinite guard
    sails past them -- and feeding those back as the next linearization point is not
    merely a slow path, it is a broken one:

      * Ipopt rejects the point WITHOUT RUNNING (`Invalid_Number_Detected`, 0
        iterations) and returns it unchanged, so the refresh loop re-linearizes at
        the same rejected point and spins to its cap: 19 no-op passes at ~0.5 s
        each, measured on `edge-regrasp`, twice over.
      * the per-pass penetration audit hands those coordinates to GJK, which raised
        `coincident body origins` and took the whole process down. Inside Alg. 1 that
        is not a slow candidate, it is a lost two-hour run and its entire cache.

    `bound` is deliberately loose. A configuration holds metres (a few), quaternion
    components (at most 1) and joint angles (at most pi), so 1e3 cannot reject a real
    answer while catching the wreckage by twenty orders of magnitude.
    """
    x = np.asarray(x, dtype=float)
    return bool(np.isfinite(x).all() and float(np.max(np.abs(x))) < bound)


def _can_be_flush(patch_a, patch_b) -> bool:
    """Could ONE patch lie flush against both of these at the same instant?

    Only decidable when both are rigidly attached to the same body: then their
    relative pose is fixed, and Eq. 7a's two requirements can be checked directly --
    the outward normals must be parallel, and the two patch planes must coincide.
    Patches on DIFFERENT bodies can move relative to each other, so nothing is
    claimed about them and the solver decides as usual.
    """
    if (patch_a.attachment, patch_a.parent) != (patch_b.attachment, patch_b.parent):
        return True
    normal_a = patch_a.placement.rotation[:, 2]
    normal_b = patch_b.placement.rotation[:, 2]
    if float(np.dot(normal_a, normal_b)) < 1.0 - 1e-9:
        return False
    offset = float(np.dot(normal_a, patch_b.placement.translation - patch_a.placement.translation))
    return abs(offset) <= 1e-9


def contradictory_contacts(scene: Scene, target) -> str:
    """Eq. 7a decided symbolically: is this edge impossible on geometry alone?

    An edge is the union `c1 u c2` (Section II-D), so an interface that changes
    partner appears with BOTH. Eq. 7a then asks one flat patch to lie flush against
    two patches at once. When those two are on the same rigid body that is decidable
    without any optimization -- and on `box_placement` three partner pairs fail it:

        box_bottom  floor + tabletop      parallel normals, planes 0.40 m apart
        left_hand   box_left + box_front  perpendicular normals
        right_hand  box_right + box_rear  perpendicular normals

    This is not a heuristic and not a relaxation: it returns the SAME verdict Eq. 14
    returns, for the same reason, in microseconds instead of tens of seconds.
    Measured over the full edge set, it settles 6176 of 11556 ordered edges (53.4%),
    and on a sample of eight of those Eq. 14 agreed eight times out of eight -- at
    35-51 s each.

    Modes cannot contradict themselves this way: Eq. 1 gives every interface exactly
    one partner, so only an edge is ever tested.

    Returns a human-readable reason, or "" when nothing is decidable here. An empty
    string is NOT a feasibility claim -- it means the solver still has to answer.
    """
    if not isinstance(target, ContactEdge):
        return ""
    for name in sorted(scene.interfaces):
        before = target.before.partner(name)
        after = target.after.partner(name)
        if before is None or after is None or before == after:
            continue
        if not _can_be_flush(scene.patches[before], scene.patches[after]):
            return (f"Eq. 7a: {name} would have to lie flush against {before} and "
                    f"{after} simultaneously, and they are fixed to the same body")
    return ""


def _perturb(scene: Scene, q: np.ndarray, rng, *, joint_scale: float = 0.6) -> np.ndarray:
    """A restart point: same base and object poses, randomized ARM AND LEG angles.

    Only the actuated joints are moved. Perturbing the floating base or an object pose
    would change which side of a contact the solve starts on for reasons unrelated to
    the branch trap, and would make one restart's answer incomparable with another's.
    The quaternions are left exactly unit, so no restart begins off the manifold.

    `joint_scale` is in radians and is deliberately large: escaping the anti-aligned
    branch needs the wrist to swing, not to jitter.
    """
    out = np.asarray(q, dtype=float).copy()
    nq = scene.robot.nq
    lower, upper = scene.robot.joint_limits()
    actuated = slice(7, nq)
    out[actuated] = np.clip(
        out[actuated] + rng.normal(scale=joint_scale, size=nq - 7),
        lower[actuated], upper[actuated],
    )
    return out


def _solve_with_refresh(scene, target, point, weights, model, margin, activation,
                        excluded, relaxed_margin, max_refresh, tolerance, solver_config,
                        always_active=None):
    """One initial guess, run through the Eq. 9 witness-refresh sequence.

    STOPPING CRITERION. Two conditions, and both were measured into place:

      * the solve CONVERGED, and
      * the answer is actually collision-free when re-queried against ALL pairs.

    Stopping on movement alone is not enough, and that is measured rather than
    argued. Eq. 9 is a FIRST-ORDER model at frozen witness points, so a pass can
    satisfy every row it was given -- rows linearized at the previous iterate -- while
    the true signed distance at the new configuration is negative. With a fixed three
    passes the `lift` mode came back converged, feasible, and 9.3 cm inside the box,
    with every one of its 689 collision rows satisfied.

    So the loop runs until the audit is clean, which is what Schulman et al.'s
    sequential convex procedure does: re-linearize until the convex model and the true
    geometry agree, not a fixed number of times. `refresh_iterations` is the cap, and
    hitting it is reported through `max_penetration` rather than hidden.

    WHAT THIS RULE REPLACED, AND WHY IT TOOK TWO TRIES
    --------------------------------------------------
    It used to be `moved < tolerance AND clean`. The movement half never fired:
    re-linearizing at a new point perturbs the rows enough to keep `max|dq|` above
    `refresh_tolerance` = 1e-4 indefinitely, so the iterate circles the constraint
    boundary and the CAP, not convergence, ended every loop. Measured 2026-08-08:
    `lift` used 20 of 20 passes and was already clean at pass 1.

    Dropping it alone made the ten-target tour SLOWER -- 413 s to 548 s -- because the
    loop could then exit on a pass whose Ipopt solve had FAILED but whose geometry was
    clean. `check` discards a non-converged result and restarts, so an early exit
    bought a fresh twenty passes: `grasp` 25 s -> 133 s, `edge-reach` 18 s -> 114 s.

    Requiring convergence as well is what made it a win. Section II-D puts the verdict
    in the solver status, so an iterate that did not converge is not an answer whose
    collision audit means anything -- `check` already refuses to audit one, reporting
    NaN. Measured over the tour, against the old rule:

        old (moved AND clean)     408 s    40.8 s per check
        this (converged AND clean) 337 s   33.7 s per check   identical verdicts

    Not slower on any single entry, and 1.66x on the feasible ones. The total is held
    up by `edge-regrasp`, an INFEASIBLE edge at 223 s -- 66% of the tour -- which no
    early-exit rule can help, since it never converges to exit on.
    """
    guess = np.asarray(point, dtype=float)
    result = sym = None
    refreshes = 0

    for iteration in range(max(1, max_refresh)):
        # One SymbolicScene per pass, shared by the contact rows and the collision
        # rows -- they must be written in the same symbols to mean anything.
        sym = SymbolicScene(scene)

        blocks = None
        if model is not None:
            # Linearize Eq. 9 at the CURRENT point, not at q_nom.
            state = sym.split(guess)
            poses = _object_poses(scene, state)
            blocks = model.blocks(sym, state["robot"], poses, margin=margin,
                                  activation_distance=activation, relaxed=excluded,
                                  relaxed_margin=relaxed_margin,
                                  always_active=always_active)

        problem, _ = build_problem(
            scene, target, weights=weights, q0=guess, collision_blocks=blocks, sym=sym
        )
        result = solve(problem, solver_config)
        refreshes = iteration + 1

        # A FAILED pass is not by itself a reason to stop -- `place` reports
        # Infeasible_Problem_Detected on passes 1, 2 and 3 and converges on pass 4,
        # so the loop must keep going as long as the iterate is still a
        # CONFIGURATION. What it must never do is re-linearize at wreckage; see
        # `_is_configuration` for what that costs.
        if not _is_configuration(result.x):
            break  # the restart loop will retry from a perturbed start
        guess = result.x

        if model is None:
            break
        state = sym.split(guess)
        penetration, _ = model.worst_penetration(state["robot"], _object_poses(scene, state))
        # The allowance has to match what the rows were actually asked for. Contact
        # pairs carry `relaxed_margin` (1 mm of slack), so demanding `>= -tolerance`
        # here asks the answer to be cleaner than any row required and the loop can
        # never terminate -- it ran to its cap on every solve until this matched.
        allowance = min(-tolerance, relaxed_margin if excluded else 0.0) - tolerance
        # CONVERGED, and clean. See the docstring for why both halves are here and
        # what each of them cost when it was not.
        if result.converged and penetration >= allowance:
            break

    return result, sym, refreshes


def check(
    scene: Scene,
    target: ContactMode | ContactEdge,
    *,
    cache: FeasibilityCache | None = None,
    weights: RegularizationWeights | None = None,
    q0: np.ndarray | None = None,
    collision: bool | SceneCollisionModel = True,
    restarts: int = 2,
    seed: int = 0,
    solver_config: str | dict = DEFAULT_SOLVER_CONFIG,
) -> FeasibilityReport:
    """Filter M (a `ContactMode`) or filter E (a `ContactEdge`), via Eq. 14.

    One function for both, because Section II-D defines them as the same
    optimization with `c` replaced by `c1 u c2`. Splitting them would invite the two
    paths to drift apart, and an edge filter that is not literally the mode filter on
    the union is not the paper's edge filter.

    THE COLLISION REFRESH LOOP
    --------------------------
    Eq. 9 is a first-order model around FROZEN GJK witness points, so it is only
    valid near the configuration they were taken at. Solving once from q_nom would
    therefore impose a collision constraint that describes the nominal pose, not the
    answer -- and Milestone 2 measured what that costs: 0.5 rad of box rotation makes
    the frozen model report 0.038 m of clearance where the true signed distance is
    -0.034 m. Translation of a flat face is exact; ROTATION is what invalidates it.

    So Eq. 14 is solved as a short sequence -- solve, re-query the witnesses at the
    answer, solve again -- stopping when the configuration stops moving. This loop is
    Schulman et al.'s own sequential convex procedure and the paper does not describe
    it for Eq. 14; see docs/ambiguities.md #12. `scene.collision["refresh_iterations"]`
    bounds it.

    Only the FINAL solve decides the verdict. An intermediate iterate that failed
    against stale witness data is not evidence about the scene.

    `collision=False` drops Eq. 9 entirely, which is not the paper's Eq. 14 -- it is
    there so the contact-only behaviour stays measurable, since the difference between
    the two is the whole reason to look at collision at all.

    RESTARTS, AND THE BRANCH THEY EXIST TO ESCAPE
    ---------------------------------------------
    A failed solve is retried from a perturbed configuration. That is not generic
    defensiveness; there is one specific trap it addresses.

    Our Eq. 7a uses `R[0,2] = R[1,2] = 0` with `R[2,2] >= 0` (docs/ambiguities.md #23).
    The two EQUALITIES are satisfied by both `R[2,2] = +1` (patches facing each other)
    and `R[2,2] = -1` (facing away); the inequality is what picks the first. But the
    two are disconnected on the surface the equalities define -- there is no path from
    one to the other that keeps them satisfied -- so a solve that starts nearer the
    wrong one gets pulled onto it and cannot climb off. That is not hypothetical: at
    q_nom the left hand patch sits at `R[2,2] = -0.19` relative to `box_left`, i.e.
    already inside the forbidden branch, and the grasp solve duly stalls with
    `7a:R_zz>=0` violated by 1.000071.

    No single nominal posture fixes it, because `box_left` (+y) and `box_front` (+x)
    are perpendicular: an arm pose that starts aligned to one starts anti-aligned to
    the other. So the escape has to come from the initial guess, which is exactly what
    Section IV-C says the paper's own false negatives come down to.

    Restarts are seeded, so verdicts stay reproducible -- see
    `faro/utils/determinism.py` for why that is not negotiable here.

    A `q0` other than q_nom is NOT cached against, so pass a cache only when using
    the default initial guess -- otherwise a verdict obtained from a lucky warm start
    would be served later to a query that never had it.
    """
    if cache is not None and q0 is None:
        hit = cache.get(target)
        if hit is not None:
            return hit

    # Timed from here rather than from the function entry: a cache hit returns the
    # cost of the solve it replaces, which is the number Table II's totals are built
    # from, not the microsecond the dict lookup took.
    started = time.perf_counter()

    # Eq. 7a, decided on geometry before any solver is started. Cached like any other
    # verdict so the tree search sees one uniform kind of answer.
    reason = contradictory_contacts(scene, target)
    if reason:
        nominal = SymbolicScene(scene)
        report = FeasibilityReport(
            target=target, feasible=False,
            result=SolveResult(
                converged=False, x=nominal.nominal(), cost=0.0, iterations=0,
                status="Contradictory_Contacts", wall_time=0.0, worst_row=reason,
            ),
            configurations=nominal.split(nominal.nominal()),
            wall_time=time.perf_counter() - started,
            max_penetration=float("nan"),
            penetrating_pair="not audited: no solve was attempted",
        )
        if cache is not None and q0 is None:
            cache.put(report)
        return report
    model = collision if isinstance(collision, SceneCollisionModel) else (
        SceneCollisionModel.cached(scene) if collision else None
    )
    settings = scene.collision
    margin = float(settings.get("margin", 0.0))
    activation = settings.get("activation_distance", 0.10)
    activation = None if activation is None else float(activation)
    # Mode-dependent: the pairs Eq. 7 already holds in contact. On by default -- not
    # because the rows contradict (at margin 0 they coexist at exactly sd = 0) but
    # because leaving them in makes the refresh loop run to its cap every time,
    # which cost 2.7x. See the scene config for the measurement.
    excluded = (
        model.contact_pairs(target)
        if model is not None and settings.get("relax_contact_pairs", True)
        else None
    )
    relaxed_margin = float(settings.get("contact_pair_margin", -1.0e-3))
    # Pair classes exempt from the cutoff -- see `always_active_pairs` for why the
    # cutoff's assumption does not hold for a movable object.
    always_active = model.always_active_pairs() if model is not None else None
    max_refresh = int(settings.get("refresh_iterations", 3)) if model is not None else 1
    tolerance = float(settings.get("refresh_tolerance", 1e-4))

    rng = np.random.default_rng(seed)
    start = np.asarray(q0, dtype=float) if q0 is not None else SymbolicScene(scene).nominal()

    best = None
    for attempt in range(max(1, restarts + 1)):
        point = start if attempt == 0 else _perturb(scene, start, rng)
        result, sym, refreshes = _solve_with_refresh(
            scene, target, point, weights, model, margin, activation, excluded,
            relaxed_margin, max_refresh, tolerance, solver_config, always_active,
        )
        best = (result, sym, refreshes, attempt)
        if result.converged:
            break

    result, sym, refreshes, attempts_used = best

    configurations = sym.split(result.x)

    # AUDIT. The solve only ever saw the pairs inside `activation_distance`; this
    # re-queries ALL of them on the answer. A cutoff that was too small shows up here
    # as a negative number instead of as a robot standing inside the platform.
    penetration, pair = 0.0, ""
    if model is not None:
        if result.converged:
            penetration, pair = model.worst_penetration(
                configurations["robot"], _object_poses(scene, configurations)
            )
        else:
            # A failed solve has no ANSWER to audit. Its last iterate is not a proposed
            # configuration, and auditing it is not conservative, it is noise: the
            # infeasible `edge-regrasp` returned entries at -1.8e308 -- finite, so a
            # `np.isfinite` guard sails past it -- and the audit dutifully reported
            # 1.8e308 metres of interpenetration.
            penetration, pair = float("nan"), "not audited: the solve did not converge"

    report = FeasibilityReport(
        target=target,
        feasible=result.converged,
        result=result,
        configurations=configurations,
        refreshes=refreshes,
        wall_time=time.perf_counter() - started,
        max_penetration=penetration,
        penetrating_pair=pair,
    )
    if cache is not None and q0 is None:
        cache.put(report)
    return report
