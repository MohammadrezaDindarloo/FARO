"""Milestone 3, part 2: Eq. 14 -- mode and edge feasibility.

Two kinds of check, and the distinction is the same one Milestone 2 settled for the
integrators:

  * that the NLP CONTAINS what Eq. 14 says it contains -- no more (which would make
    the cheap filter reject reachable modes) and no less (which would make it accept
    unreachable ones);
  * that the CONFIGURATION it returns is physically what the mode described --
    measured with numeric forward kinematics, not by re-reading the residuals the
    solver just drove to zero. A residual check would pass on a solver that returned
    its own initial guess.

    pytest tests/test_mode_edge.py -v
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pytest

from faro.constraints.frames import normal_alignment_rows
from faro.core.modes import ContactEdge, ContactMode
from faro.mode_edge.feasibility import FeasibilityCache, check
from faro.mode_edge.problem import RegularizationWeights, build_problem
from faro.scene.scene import Scene
from faro.solvers.nlp import solve


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_config("box_placement")


def mode(**kwargs) -> ContactMode:
    base = {"left_hand": None, "right_hand": None,
            "left_foot": None, "right_foot": None, "box_bottom": None}
    base.update(kwargs)
    return ContactMode.from_dict(base)


STAND = mode(left_foot="floor", right_foot="floor", box_bottom="floor")
GRASP = mode(left_foot="floor", right_foot="floor", box_bottom="floor",
             left_hand="box_left", right_hand="box_right")


@pytest.fixture(scope="module")
def stand_report(scene):
    return check(scene, STAND)


# =============================================================================
# Eq. 7a -- the two alignment forms, and why the default changed
# =============================================================================
def _alignment_values(form: str, R):
    eq, _, ineq, _ = normal_alignment_rows(ca.DM(R), form)
    rows = [float(ca.DM(r)) for r in eq] + [float(ca.DM(r)) for r in ineq]
    return np.array(rows)


def test_the_two_alignment_forms_agree_on_random_rotations(scene):
    """`log3(R)_{x,y} = 0` and `R[0,2] = R[1,2] = 0, R[2,2] >= 0` are the same set.

    Checked as a CLASSIFIER over random rotations rather than by comparing values --
    the two forms produce different numbers, and the claim is only that they are
    satisfied by the same rotations. Aligned rotations (a pure yaw) must satisfy
    both; general ones must fail both.
    """
    rng = np.random.default_rng(0)

    for _ in range(200):
        yaw = rng.uniform(-np.pi, np.pi)
        aligned = pin.exp3(np.array([0.0, 0.0, yaw]))
        assert np.abs(_alignment_values("log3", aligned)[:2]).max() < 1e-9
        column = _alignment_values("column", aligned)
        assert np.abs(column[:2]).max() < 1e-9
        assert column[2] <= 1e-9, "R_zz >= 0 must hold for an aligned pair"

    for _ in range(200):
        tilt = pin.exp3(rng.normal(scale=0.6, size=3))
        log3_violated = np.abs(_alignment_values("log3", tilt)[:2]).max() > 1e-6
        column_violated = np.abs(_alignment_values("column", tilt)[:2]).max() > 1e-6
        assert log3_violated == column_violated


def test_the_column_form_excludes_back_to_back_patches(scene):
    """R[0,2] = R[1,2] = 0 alone admits R[2,2] = -1: patches facing AWAY.

    The log3 form excludes that automatically -- an axis along z gives R[2,2] = +1 --
    so dropping the extra inequality would leave the column form strictly weaker and
    let a foot satisfy Eq. 7a while pointing at the sky.
    """
    flipped = pin.exp3(np.array([np.pi, 0.0, 0.0]))  # R[2,2] = -1
    column = _alignment_values("column", flipped)
    assert np.abs(column[:2]).max() < 1e-9, "equalities alone do not catch this"
    assert column[2] > 0.5, "the R_zz >= 0 row must reject it"


def test_the_log3_failure_survives_every_ordering_of_the_contact_rows(scene):
    """The failure is a property of the FORM, not of how the rows happen to be stacked.

    Row order changes Ipopt's pivoting and therefore the trial points it visits, so a
    single failing run proves very little -- a knife-edge that one ordering happens to
    step on is a different claim from a formulation that does not work. Shuffling the
    five contact blocks and re-solving separates the two.

    It also corrects an earlier reading of ours. Bisecting the mode ONE PAIR AT A TIME
    appeared to blame the pairs already aligned at the nominal pose, but that result
    did not replicate under a different pair ordering -- so no per-pair attribution is
    made here. What replicates is the whole-form result below, and that is all this
    test claims.
    """
    from faro.constraints.block import merge
    from faro.mode_edge.problem import contact_blocks, limit_block
    from faro.scene.symbolic import SymbolicScene
    from faro.solvers.nlp import NLPProblem

    pairs = GRASP.active_pairs(scene)

    def status(order, alignment):
        sym = SymbolicScene(scene)
        nominal = sym.nominal()
        blocks = contact_blocks(sym, [pairs[i] for i in order], alignment=alignment)
        blocks += [limit_block(sym), sym.unit_quaternion_rows()]
        delta = sym.variables - ca.DM(nominal.reshape(-1, 1))
        return solve(
            NLPProblem(name="probe", variables=sym.variables, cost=ca.dot(delta, delta),
                       block=merge("probe", blocks), x0=nominal),
            "ipopt_mode_edge",
        ).status

    rng = np.random.default_rng(0)
    orderings = [tuple(range(len(pairs)))] + [
        tuple(int(i) for i in rng.permutation(len(pairs))) for _ in range(3)
    ]

    for order in orderings:
        assert status(order, "log3") == "Invalid_Number_Detected", (
            f"ordering {order} solved with the literal form; the fragility has moved "
            f"and the default deserves re-measuring rather than assuming"
        )
        assert status(order, "column") == "Solve_Succeeded"


def test_the_grasp_mode_is_only_solvable_with_the_smooth_form(scene):
    """The failure that motivated the change, pinned as a regression test.

    At the nominal pose the hands sit 114 and 142 degrees from the box faces they
    must reach, so the solve rotates THROUGH the singularity. With the literal form
    Ipopt returns `Invalid_Number_Detected` -- an INFEASIBLE verdict produced by
    arithmetic rather than by geometry, which is a false negative of exactly the kind
    Section IV-C measures.
    """
    smooth, _ = build_problem(scene, GRASP, alignment="column")
    literal, _ = build_problem(scene, GRASP, alignment="log3")

    assert solve(smooth, "ipopt_mode_edge").converged
    literal_result = solve(literal, "ipopt_mode_edge")
    assert not literal_result.converged
    assert literal_result.status == "Invalid_Number_Detected"


def test_both_forms_reach_the_same_optimum_where_both_converge(scene):
    """Equivalent feasible sets and an identical objective means an identical optimum.

    The mode is SEARCHED for rather than named. This test used to hardcode `lift`,
    which both forms could solve -- until `base_orientation` went to 10.0 and moved the
    solve path enough that the log3 form started failing on it. That is a fact about
    log3's fragility (ambiguity #23), not about the equivalence claim, and hardcoding
    one mode let an unrelated config change masquerade as a broken equivalence.

    Asserting a candidate was FOUND is what keeps this from degrading into a silent
    skip: if the log3 form ever stops converging on everything, that is a finding, and
    it should show up as a failure here rather than as a green run testing nothing.
    """
    candidates = [
        mode(left_foot="floor", right_foot="floor",
             left_hand="box_left", right_hand="box_right"),
        mode(left_foot="floor", right_foot="floor", box_bottom="floor"),
        mode(left_foot="floor", right_foot="floor"),
        mode(left_foot="floor"),
    ]

    compared = 0
    for candidate in candidates:
        results = {}
        for form in ("column", "log3"):
            problem, _ = build_problem(scene, candidate, alignment=form)
            results[form] = solve(problem, "ipopt_mode_edge")
        if not all(r.converged for r in results.values()):
            continue
        assert results["column"].cost == pytest.approx(results["log3"].cost, rel=1e-6), (
            f"{candidate.label()}: column {results['column'].cost} vs "
            f"log3 {results['log3'].cost}"
        )
        compared += 1

    assert compared > 0, (
        "no mode was solvable by BOTH alignment forms, so the equivalence was never "
        "actually compared end to end. Either log3 has become unusable everywhere -- "
        "worth knowing -- or these candidates need revisiting."
    )


# =============================================================================
# Eq. 14 contains exactly what Section II-D lists
# =============================================================================
def test_eq_14_uses_only_7a_7b_and_13a(scene):
    """Eq. 14 cites (7a), (7b), (9), (13a) -- and deliberately nothing else.

    No 7c/7d because there are no forces in an IK problem, and no 13b/13c because
    there is no time. Adding them would make the cheap filter reject modes a full TO
    could realize, which is the false-negative rate Table III measures. Checked on
    the row LABELS, so it cannot be satisfied by a comment.
    """
    problem, _ = build_problem(scene, GRASP)
    labels = list(problem.block.eq_labels) + list(problem.block.ineq_labels)
    joined = " ".join(labels)

    for present in ("7a:", "7b:", "13a:"):
        assert present in joined, f"Eq. 14 must contain {present} rows"
    for absent in ("7c:", "7d:", "8:", "13b:", "13c:", "10a:", "10b:", "11a:", "11b:"):
        assert absent not in joined, f"Eq. 14 must NOT contain {absent} rows"


def test_the_objective_is_zero_exactly_at_the_nominal_configuration(scene):
    """Eq. 14's cost is `(q - q_nom)^T W (q - q_nom)`, so q_nom is its unique minimum."""
    problem, sym = build_problem(scene, STAND)
    cost = ca.Function("f", [problem.variables], [problem.cost])
    assert float(cost(sym.nominal())) == pytest.approx(0.0, abs=1e-12)
    assert float(cost(sym.nominal() + 0.1)) > 0.0


def test_the_scene_config_supplies_w_and_it_reaches_the_cost(scene):
    """W is a config value, and a config value that never arrives is worse than none.

    `base_orientation: 10` is the "stay upright" prior the paper never writes down.
    Eq. 14 has no gravity and no balance, so under W = I it returns the robot diving
    over the box with its knees at their hyperextension limit -- a pose that satisfies
    every constraint and that Alg. 1 would then warm-start the KSO from.

    Checked end to end: the YAML value must show up in the weights, in the diagonal,
    AND in the assembled cost. Reading it into a dataclass that the NLP never consults
    would pass a shallower test and change nothing about the answer.
    """
    # Whatever the YAML currently says -- these are TUNING values and are meant to be
    # edited. Pinning the numbers here made this test fail on exactly the knob it
    # exists to prove is live, twice: once when `hip` went to 2.0 and once when
    # `base_orientation` came back to 1.0.
    declared = scene.regularization
    weights = RegularizationWeights.from_scene(scene)
    assert weights.base_orientation == pytest.approx(float(declared["base_orientation"]))
    assert weights.joints == pytest.approx(float(declared["joints"]))

    problem, sym = build_problem(scene, STAND)   # no explicit weights -> from config
    cost = ca.Function("f", [problem.variables], [problem.cost])
    nominal = sym.nominal()

    # Compare the base quaternion against a real joint, at that joint's OWN resolved
    # weight -- which is the fallback only if no group matches it.
    model = scene.robot.model
    joint_name = scene.robot.actuated_joint_names[0]
    idx_q = model.joints[model.getJointId(joint_name)].idx_q
    joint_weight = weights.weight_for_joint(joint_name)

    # Tip both by the same amount; the costs must be in the ratio of their weights.
    tilted = nominal.copy()
    tilted[3] += 0.01
    bent = nominal.copy()
    bent[idx_q] += 0.01
    assert float(cost(tilted)) * joint_weight == pytest.approx(
        weights.base_orientation * float(cost(bent)), rel=1e-9
    )

    # The check above goes vacuous whenever the two weights happen to be equal, which
    # is the config's own default. So prove the ratio machinery once with weights that
    # cannot coincide, and that no group can silently override.
    explicit = RegularizationWeights(base_orientation=7.0, joints=1.0)
    problem, sym = build_problem(scene, STAND, weights=explicit)
    cost = ca.Function("f", [problem.variables], [problem.cost])
    nominal = sym.nominal()
    idx_q = model.joints[model.getJointId(joint_name)].idx_q
    tilted = nominal.copy()
    tilted[3] += 0.01
    bent = nominal.copy()
    bent[idx_q] += 0.01
    assert float(cost(tilted)) == pytest.approx(7.0 * float(cost(bent)), rel=1e-9)


def _with_groups(scene, groups):
    """Resolve W with `joint_groups` swapped in, restoring the scene afterwards."""
    original = scene.regularization
    try:
        scene.regularization = {**original, "joint_groups": groups}
        return RegularizationWeights.from_scene(scene)
    finally:
        scene.regularization = original


def test_a_joint_group_covers_the_left_right_pair_and_nothing_else(scene):
    """One number per group is the whole point -- `knee` must not catch an ankle.

    Matching is by SUBSTRING of the joint name, which is what makes a left/right pair
    a single knob without a hand-written index list that a robot swap would silently
    misalign.
    """
    weights = _with_groups(scene, {"knee": 5.0, "ankle": 0.2})
    per_joint = weights.per_joint(scene)

    assert per_joint["left_knee_joint"] == 5.0
    assert per_joint["right_knee_joint"] == 5.0
    assert per_joint["left_ankle_pitch_joint"] == 0.2
    assert per_joint["right_ankle_roll_joint"] == 0.2
    # Everything unnamed keeps the fallback.
    assert per_joint["left_hip_pitch_joint"] == weights.joints
    assert per_joint["waist_yaw_joint"] == weights.joints


def test_the_more_specific_group_wins(scene):
    """`hip: 1` plus `hip_pitch: 9` must give the pitch joints 9 and the rest 1.

    Anything else makes refining a group require rewriting it, and the result would
    depend on which key happened to be iterated first.
    """
    per_joint = _with_groups(scene, {"hip": 1.0, "hip_pitch": 9.0}).per_joint(scene)
    assert per_joint["left_hip_pitch_joint"] == 9.0
    assert per_joint["right_hip_pitch_joint"] == 9.0
    assert per_joint["left_hip_roll_joint"] == 1.0
    assert per_joint["left_hip_yaw_joint"] == 1.0


def test_a_full_joint_name_is_a_group_of_one(scene):
    """Granularity has to reach a single joint, or the feature stops short."""
    per_joint = _with_groups(scene, {"left_knee_joint": 3.0}).per_joint(scene)
    assert per_joint["left_knee_joint"] == 3.0
    assert per_joint["right_knee_joint"] != 3.0


def test_a_group_matching_no_joint_is_rejected(scene):
    """A typo here is silent in the worst way: the joints keep the fallback and the
    config reads as though it took effect."""
    with pytest.raises(KeyError, match="match no joint"):
        _with_groups(scene, {"kneee": 2.0})


def test_two_equally_specific_groups_are_an_error_not_a_coin_flip(scene):
    """`left_knee` and `knee_join` are both 9 characters and both match one joint.

    There is no defensible winner, and picking one would make the weights depend on
    dict insertion order -- a config that behaves differently after an innocuous
    reordering is worse than one that refuses to load.
    """
    weights = _with_groups(scene, {"left_knee": 1.0, "knee_join": 2.0})
    with pytest.raises(ValueError, match="equal specificity"):
        weights.per_joint(scene)


def test_group_weights_land_on_the_right_entries_of_the_diagonal(scene):
    """The group resolves per NAME; the diagonal is indexed by `idx_q`.

    Those are two different orderings, and the bug this guards against -- placing
    weights by counting from index 7 instead of asking each joint where it lives --
    would shift every later joint's weight by one slot the moment a joint was not
    1-DoF, with nothing to report it.
    """
    model = scene.robot.model
    weights = _with_groups(scene, {"knee": 7.0})
    diagonal = weights.diagonal(scene)

    for name in ("left_knee_joint", "right_knee_joint"):
        idx = model.joints[model.getJointId(name)].idx_q
        assert diagonal[idx] == 7.0, f"{name} weight landed somewhere else"
    assert list(diagonal[7:scene.robot.nq]).count(7.0) == 2, "exactly two knees"


def test_an_unknown_regularization_key_is_rejected(scene):
    """A typo in the YAML must not silently leave that group at its default."""
    import dataclasses

    typo = dataclasses.replace(scene) if False else scene
    original = typo.regularization
    try:
        typo.regularization = {**original, "base_orientaton": 10.0}  # missing an 'i'
        with pytest.raises(KeyError, match="unknown regularization"):
            RegularizationWeights.from_scene(typo)
    finally:
        typo.regularization = original


def test_regularization_weights_reach_every_block_of_q(scene):
    """W is diagonal over the STACKED q -- robot base, joints, and each object pose.

    A weight vector built for the robot alone would silently leave the object poses
    unweighted, and the box would then be free to drift anywhere the contacts allow
    at no cost. Checked by giving each group a distinct value and reading the
    diagonal back.
    """
    weights = RegularizationWeights(
        base_position=2.0, base_orientation=3.0, joints=5.0,
        object_position=7.0, object_orientation=11.0,
    )
    diagonal = weights.diagonal(scene)
    assert diagonal.size == scene.robot.nq + 7 * len(scene.objects)
    assert set(diagonal[:3]) == {2.0}
    assert set(diagonal[3:7]) == {3.0}
    assert set(diagonal[7:scene.robot.nq]) == {5.0}
    assert set(diagonal[scene.robot.nq:scene.robot.nq + 3]) == {7.0}
    assert set(diagonal[scene.robot.nq + 3:]) == {11.0}


# =============================================================================
# The ANSWER is physically the mode that was asked for
# =============================================================================
def test_a_feasible_mode_puts_the_feet_on_the_floor(scene, stand_report):
    """Measured with forward kinematics, NOT by re-reading the solver's residuals.

    Re-evaluating the constraint the solver just minimized proves only that Ipopt
    reported its own work honestly. Running FK on the returned configuration and
    looking at where the soles actually are is an independent measurement, and it is
    the one that would catch a patch offset applied in the wrong frame.
    """
    assert stand_report.feasible, stand_report.explain()
    q = stand_report.configurations["robot"]

    for patch_name in ("left_foot_sole", "right_foot_sole"):
        placement = scene.patch_world_placement(scene.patches[patch_name], q)
        assert placement.translation[2] == pytest.approx(0.0, abs=1e-6), (
            f"{patch_name} is {placement.translation[2]:+.4f} m from the floor"
        )
        # Sole flat: its own +z is the outward (downward) normal, so the world z
        # component of that axis must be -1.
        assert placement.rotation[2, 2] == pytest.approx(-1.0, abs=1e-6)


def test_a_feasible_mode_respects_the_joint_limits(scene, stand_report):
    """Eq. 13a, checked on the returned configuration rather than on its rows."""
    lower, upper = scene.robot.joint_limits()
    q = stand_report.configurations["robot"]
    actuated = slice(7, scene.robot.nq)
    assert np.all(q[actuated] >= lower[actuated] - 1e-6)
    assert np.all(q[actuated] <= upper[actuated] + 1e-6)


def test_the_returned_poses_are_on_the_manifold(scene, stand_report):
    """The unit-quaternion rows the paper does not write (docs/ambiguities.md #21).

    Without them the 7-vector is not a rigid transform and every rotation derived
    from it is scaled by |quat|^2 -- so this is not tidiness, it is whether the
    answer means anything.
    """
    q = stand_report.configurations["robot"]
    assert np.linalg.norm(q[3:7]) == pytest.approx(1.0, abs=1e-8)
    for name in scene.objects:
        assert np.linalg.norm(stand_report.configurations[name][3:7]) == pytest.approx(1.0, abs=1e-8)


def test_placing_the_box_on_the_tabletop_actually_moves_the_box(scene):
    """A mode that names a different support must produce a different object pose.

    The box starts on the floor at z = 0.15. If Eq. 14 optimized only over the robot
    -- a plausible reading of "inverse kinematics optimization" -- this mode would
    still report feasible while the box never left the ground, and nothing in a
    residual check would notice.
    """
    on_table = mode(left_foot="floor", right_foot="floor", box_bottom="tabletop")
    report = check(scene, on_table)
    assert report.feasible, report.explain()

    box = report.configurations["box"]
    tabletop_z = scene.patches["tabletop"].placement.translation[2]
    half_height = scene.objects["box"].half_extents[2]
    assert box[2] == pytest.approx(tabletop_z + half_height, abs=1e-6)
    assert box[2] > 0.15 + 1e-3, "the box did not leave the floor"


def test_an_impossible_edge_is_reported_infeasible(scene):
    """Section II-D's edge filter, on a transition that cannot happen at an instant.

    Moving the left hand from `box_left` to `box_front` means, at the transition
    instant, pressing one flat patch against two perpendicular faces at once. Eq. 7a
    cannot align a single normal with both. This is the check that an edge is a real
    test and not a formality -- both endpoint MODES are individually feasible.
    """
    on_front = mode(left_foot="floor", right_foot="floor", box_bottom="floor",
                    left_hand="box_front", right_hand="box_rear")
    assert check(scene, GRASP).feasible
    assert check(scene, on_front).feasible
    assert not check(scene, ContactEdge(GRASP, on_front)).feasible


# =============================================================================
# The cache (Section III / Table II)
# =============================================================================
def test_the_cache_returns_the_same_verdict_without_resolving(scene):
    cache = FeasibilityCache()
    first = check(scene, STAND, cache=cache)
    second = check(scene, STAND, cache=cache)

    assert not first.cached and second.cached
    assert first.feasible == second.feasible
    assert cache.hits == 1 and cache.misses == 1


def test_the_cache_distinguishes_a_mode_from_an_edge_that_contains_it(scene):
    """`ContactEdge(c, c)` has the same contact pairs as `c` but is a different query.

    They happen to pose the same NLP, so a cache keyed on the PAIRS rather than on
    the query would conflate them. That is harmless here and would not stay harmless:
    once collision pairs become mode-dependent, the two stop being the same problem.
    """
    cache = FeasibilityCache()
    check(scene, STAND, cache=cache)
    check(scene, ContactEdge(STAND, STAND), cache=cache)
    assert len(cache) == 2


def test_a_warm_started_solve_is_not_cached(scene):
    """A verdict that depended on a lucky initial guess must not be served to a query
    that did not have one. Section IV-C attributes false negatives to initialization,
    so caching across different starts would make the filter's answer depend on
    solve order.
    """
    cache = FeasibilityCache()
    _, sym = build_problem(scene, STAND)
    check(scene, STAND, cache=cache, q0=sym.nominal())
    assert len(cache) == 0
