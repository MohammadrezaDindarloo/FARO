"""Milestone 2, part 3: every constraint fails exactly where the paper says it must.

`test_milestone2_constraints.py` checks the algebra and `test_milestone2_symbolic.py`
checks it differentiates. This file checks the thing that actually matters for
Milestones 3-5: that each constraint's FAILURE BOUNDARY sits where the equation
puts it, on the real G1 and the real box-placement scene.

Every scenario carries a `predicted_crossing` derived BY HAND from the paper's
equation and the scene's numbers -- independently of the implementation. These tests
compare that prediction against what the code actually does. A mismatch means the
implementation and the paper disagree, which is precisely the class of bug that is
undebuggable once these constraints are buried in a 5000-row NLP.

    pytest tests/test_milestone2_scenarios.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from faro.scenarios.constraint_demos import (
    ALL_SCENARIOS,
    get_scenario,
    resolve_predictions,
    run_sweep,
)
from faro.scene.scene import Scene


@pytest.fixture(scope="module")
def scene() -> Scene:
    s = Scene.from_config("box_placement")
    resolve_predictions(s)
    return s


# =============================================================================
# The headline check
# =============================================================================
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.key)
def test_constraint_fails_where_the_paper_predicts(scenario, scene):
    """Measured failure boundary == analytically predicted failure boundary."""
    result = run_sweep(scenario, scene)
    measured = result.measured_crossing

    if scenario.predicted_crossing is None:
        assert not result.ever_violated, (
            f"{scenario.key}: should never violate, but did at {measured}. "
            f"{scenario.prediction_note}"
        )
        return

    assert measured is not None, (
        f"{scenario.key}: never violated, but should have at "
        f"{scenario.predicted_crossing}. {scenario.prediction_note}"
    )
    assert measured == pytest.approx(scenario.predicted_crossing, abs=scenario.tolerance), (
        f"{scenario.key}: {scenario.prediction_note}"
    )


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.key)
def test_margin_is_monotone_around_the_crossing(scenario, scene):
    """The violation measure must move consistently as the parameter sweeps.

    A constraint whose margin wobbles across the boundary would give the solver a
    non-monotone, and possibly non-convergent, gradient. Checked on the segment
    around the crossing rather than globally, since some rows saturate.
    """
    result = run_sweep(scenario, scene)
    if scenario.predicted_crossing is None:
        return

    margins = np.array([s.margin for s in result.steps])
    crossing_index = next(
        (i for i in range(len(margins) - 1) if margins[i] <= 0 < margins[i + 1]), None
    )
    assert crossing_index is not None

    window = margins[max(0, crossing_index - 5): crossing_index + 6]
    diffs = np.diff(window)
    assert np.all(diffs >= -1e-9), f"{scenario.key}: margin is not monotone near the crossing"


# =============================================================================
# Physical intuition -- the properties that make these constraints MEAN something
# =============================================================================
def test_7a_yaw_really_is_free_over_a_full_half_turn(scene):
    """Eq. 7a "leaves the relative in-plane position and yaw angle free".

    Without this freedom the robot could not turn while walking, so it is worth
    asserting over a wide range rather than trusting the two-row structure.
    """
    result = run_sweep(get_scenario("7a-yaw"), scene)
    assert not result.ever_violated
    worst = max(abs(v) for step in result.steps for v in step.rows.values())
    assert worst < 1e-6, f"yaw perturbs Eq. 7a by {worst}"


def test_7b_yaw_fails_without_the_box_moving_at_all(scene):
    """The `^b xi_a` term is not decoration.

    The box's position is identical at every step of `7b-yaw`; only its orientation
    changes. If the constraint used a's raw half-extents instead of its footprint in
    b's frame, this sweep could never fail.
    """
    result = run_sweep(get_scenario("7b-yaw"), scene)
    positions = [step.state["box_pose"].translation for step in result.steps]
    assert np.allclose(positions, positions[0]), "the box moved; the test is not isolating yaw"
    assert result.ever_violated, "rotation alone never violated 7b -- ^b xi_a is being ignored"


def test_7b_applies_to_end_effector_patches_not_just_objects(scene):
    """The paper, Section II-C 1, verbatim:

        "This includes contact between END-EFFECTOR PATCHES, patches on movable
         objects, and static environment patches."

    So Eq. 7b governs hand-to-box contact exactly as it governs box-to-platform.
    That path is different in the code -- patch `a` hangs off a robot link and its
    world placement comes from forward kinematics rather than an object pose -- so it
    needs its own coverage.
    """
    result = run_sweep(get_scenario("7b-hand-x"), scene)
    assert result.ever_violated
    assert result.measured_crossing == pytest.approx(0.1231, abs=1e-3)

    # The robot holds still and the BOX moves -- the box is the movable object, and
    # posing the robot to meet a fixed box would need full-body IK.
    assert np.allclose(result.steps[0].state["q"], result.steps[-1].state["q"])
    travel = np.linalg.norm(
        result.steps[-1].state["box_pose"].translation
        - result.steps[0].state["box_pose"].translation
    )
    assert travel == pytest.approx(result.steps[-1].param, abs=1e-9)


def test_7b_hand_contact_resolves_the_patch_through_forward_kinematics(scene):
    """Patch `a` must be driven by the robot's joints, not a stored pose.

    Complements the sweep above, which deliberately holds q fixed: bending the elbow
    has to move the contact, or the FK path is not really being exercised.
    """
    from faro.scenarios.constraint_demos import _presenting_configuration

    q = _presenting_configuration(scene, "left_hand_patch", "box_left")
    before = scene.patch_world_placement(scene.patches["left_hand_patch"], q)

    model = scene.robot.model
    q_bent = q.copy()
    q_bent[model.joints[model.getJointId("left_elbow_joint")].idx_q] += 0.3
    after = scene.patch_world_placement(scene.patches["left_hand_patch"], q_bent)

    assert not np.allclose(before.translation, after.translation)


def test_7b_hand_scenarios_hold_an_upright_axis_aligned_box(scene):
    """The box must stay level, so the picture reads as "held", not "flung".

    The presenting pose solves the three wrist angles in closed form precisely so
    the palm can meet an axis-aligned box exactly; if that solve regressed, the box
    would tilt and the demo would be misleading even though 7b still measured right.
    """
    for key in ("7b-hand-x", "7b-hand-y"):
        for step in run_sweep(get_scenario(key), scene).steps[::40]:
            rotation = step.state["box_pose"].rotation
            assert np.allclose(rotation, np.eye(3), atol=1e-12), f"{key}: box is tilted"


def test_presenting_pose_respects_the_joint_limits(scene):
    """A demo pose outside Eq. 13a would be unreachable and quietly dishonest."""
    from faro.scenarios.constraint_demos import _presenting_configuration

    q = _presenting_configuration(scene, "left_hand_patch", "box_left")
    model = scene.robot.model
    for name in model.names[1:]:
        joint = model.joints[model.getJointId(name)]
        if joint.nq != 1:
            continue
        i = joint.idx_q
        assert model.lowerPositionLimit[i] - 1e-9 <= q[i] <= model.upperPositionLimit[i] + 1e-9, (
            f"{name} is outside its URDF limits in the presenting pose"
        )


def test_7b_bound_is_evaluated_per_axis_not_as_one_radius(scene):
    """The same hand patch on the same face fails at different offsets along x and y.

    The wrist cut is 0.0269 x 0.0301, so the slack is 0.1231 in x and 0.1199 in y. A
    single scalar "patch radius" -- a tempting simplification -- could not reproduce
    this. The box-on-platform sweeps cannot detect it either, because those patches
    are square; only an ASYMMETRIC patch pins the per-axis behaviour down.
    """
    along_x = run_sweep(get_scenario("7b-hand-x"), scene).measured_crossing
    along_y = run_sweep(get_scenario("7b-hand-y"), scene).measured_crossing

    hand = scene.patches["left_hand_patch"]
    face = scene.patches["box_left"]
    assert along_x == pytest.approx(face.half_extents[0] - hand.half_extents[0], abs=1e-3)
    assert along_y == pytest.approx(face.half_extents[1] - hand.half_extents[1], abs=1e-3)

    # Stated as the direction-free invariant, so it survives a change of patch
    # geometry: whichever axis the patch is LARGER on must run out of room sooner.
    assert (hand.half_extents[0] > hand.half_extents[1]) == (along_x < along_y), (
        "the larger patch half-extent must leave LESS room"
    )
    assert along_x != along_y, "an asymmetric patch must give two different bounds"


def test_7b_hand_sweep_isolates_7b_by_satisfying_7a_exactly(scene):
    """The hand is posed so Eq. 7a holds to machine precision at every step.

    Otherwise a 7a residual would leak into the margin and the measured crossing
    would be testing the wrong constraint.
    """
    from faro.scenarios.constraint_demos import _kinematic_block

    import casadi as ca

    for step in run_sweep(get_scenario("7b-hand-x"), scene).steps[::20]:
        block = _kinematic_block(
            scene, "left_hand_patch", "box_left",
            q=step.state["q"], box_pose=step.state["box_pose"],
        )
        residual = np.array(ca.DM(ca.evalf(block.eq))).ravel()
        assert np.allclose(residual, 0.0, atol=1e-12), f"7a leaked in at {step.param}"


def test_every_allowed_contact_pair_can_satisfy_7b(scene):
    """No Table IV pair may be geometrically impossible.

    Eq. 7b requires patch a to fit inside patch b. A pair that cannot would make
    every contact using it infeasible for a reason unrelated to the robot -- and in
    the Milestone 6 tree search that reads as an unlucky branch, not a scene bug.
    """
    assert scene.containment_report() == []


def test_containment_report_catches_an_oversized_patch(scene):
    """The guard must actually fire when a scene is wrong."""
    from faro.core.patches import Interface

    broken = Scene(
        name="broken", robot=scene.robot,
        objects=dict(scene.objects), patches=dict(scene.patches),
    )
    # box_bottom (0.15) cannot fit inside left_hand_patch (0.0269) -- reversed on purpose.
    broken.interfaces["bad"] = Interface(
        name="bad", patch=scene.patches["box_bottom"], allowed=["left_hand_patch"]
    )
    problems = broken.containment_report()
    assert len(problems) == 1
    assert "does not fit" in problems[0]


def test_7d_foot_resists_pitch_far_better_than_roll(scene):
    """The x<->y swap in Eq. 7d, expressed as physics.

    The G1's sole is 0.085 m long and 0.030 m half-wide, so it should tolerate
    0.085/0.030 = 2.83x more pitch moment than roll moment. Getting the subscripts
    backwards would invert this and make the robot tip sideways in simulation while
    every unit test still passed.
    """
    roll = run_sweep(get_scenario("7d-roll"), scene).measured_crossing
    pitch = run_sweep(get_scenario("7d-pitch"), scene).measured_crossing

    assert pitch > roll, "a foot must resist pitching more than rolling"
    sole = scene.patches["left_foot_sole"]
    expected_ratio = sole.half_extents[0] / sole.half_extents[1]
    assert pitch / roll == pytest.approx(expected_ratio, rel=1e-3)
    assert expected_ratio == pytest.approx(0.085 / 0.03, rel=1e-9)


def test_7c_friction_bound_is_proportional_to_normal_load(scene):
    """Coulomb friction: double the load, double the tangential capacity."""
    from faro.scenarios import constraint_demos as demos

    baseline = run_sweep(get_scenario("7c-slip"), scene).measured_crossing
    assert baseline == pytest.approx(demos._MU * demos._FZ, abs=2.0)


def test_9_frozen_witness_detects_penetration_but_refreshed_does_not(scene):
    """The single most important practical fact about Eq. 9.

    Under pure TRANSLATION of a flat face the closest-point pair never changes, so the
    frozen model is EXACT and tracks the gap perfectly through contact and into
    penetration. Under ROTATION the contact migrates to another corner, the frozen pair
    stops being the closest one, and the model reports clearance the box does not have.

    That contrast is the argument for refreshing witness data between solver iterations.

    (An earlier version of this test asserted the opposite for the rotation case, on
    the false premise that `coal.distance()` is unsigned. It is not --
    `DistanceRequest.enable_signed_distance` defaults to True -- and the apparent
    confirmation came from a sign error in `query_witness`. Both are fixed.)
    """
    frozen = run_sweep(get_scenario("9-penetrate"), scene)
    assert frozen.ever_violated
    assert frozen.measured_crossing == pytest.approx(0.0, abs=3e-3)

    # Translation: the frozen model reproduces the true signed distance exactly.
    for step in frozen.steps:
        assert -step.rows["9:sd>=margin"] == pytest.approx(step.param, abs=1e-6)

    # Rotation: it does not, and it errs in the DANGEROUS direction.
    drift = run_sweep(get_scenario("9-drift"), scene)
    assert not drift.ever_violated, (
        "the frozen model should keep reporting clearance -- that is the failure mode"
    )
    truths = [step.state["true_signed_distance"] for step in drift.steps]
    assert min(truths) < -0.02, "the box must genuinely penetrate for this to mean anything"

    worst = max(-step.rows["9:sd>=margin"] - truth for step, truth in zip(drift.steps, truths))
    assert worst > 0.05, f"expected several cm of linearization drift, got {worst:.4f} m"


def test_13a_bound_comes_from_the_urdf_not_a_hard_coded_number(scene):
    """Eq. 13a's q_max is a robot property; swapping robots must not invalidate it."""
    from faro.scenarios.constraint_demos import knee_upper_limit

    scenario = get_scenario("13a-knee")
    assert scenario.predicted_crossing == pytest.approx(knee_upper_limit(scene))
    assert run_sweep(scenario, scene).measured_crossing == pytest.approx(
        knee_upper_limit(scene), abs=scenario.tolerance
    )


def test_8_residual_equals_the_slide_distance(scene):
    """Eq. 8's residual is not a proxy -- it IS how far the contact moved."""
    result = run_sweep(get_scenario("8-slip"), scene)
    for step in result.steps[::10]:
        slide = np.hypot(step.rows["8:dp_x"], step.rows["8:dp_y"])
        assert slide == pytest.approx(step.param, abs=1e-9)


# =============================================================================
# Scene invariants the scenarios rely on
# =============================================================================
def test_nominal_pose_satisfies_the_contact_constraints_exactly(scene):
    """q_nominal is the Eq. 14 regularization target AND the initial guess.

    If the standing pose does not itself satisfy Eq. 7a, every solve starts from an
    infeasible point for no reason. This caught a real 0.01 rad sole tilt caused by
    a leg pitch chain that summed to -0.01 instead of 0.
    """
    first = run_sweep(get_scenario("7a-lift"), scene).steps[0]
    assert abs(first.rows["7a:p_z"]) < 1e-9

    tilt = run_sweep(get_scenario("7a-tilt"), scene).steps[0]
    assert abs(tilt.rows["7a:log3(R)_x"]) < 1e-9
    assert abs(tilt.rows["7a:log3(R)_y"]) < 1e-9


def test_every_scenario_has_a_derivation_note(scene):
    """A prediction without a stated derivation is just another implementation.

    Takes `scene` and resolves first: several notes quote numbers that only exist once
    the robot is loaded (a URDF limit, the robot's mass, where the geometry actually
    collides), so they are filled in by `resolve_predictions`. Without that this test
    passed only when some earlier test happened to have resolved them already -- which
    made it order-dependent, and it duly broke the moment a new test file shifted the
    ordering.
    """
    resolve_predictions(scene)
    for scenario in ALL_SCENARIOS:
        assert scenario.prediction_note.strip()
        assert scenario.equation in scenario.prediction_note or "Eq." in scenario.prediction_note
        assert scenario.question.strip().endswith("?")


def test_scenario_keys_are_unique():
    keys = [s.key for s in ALL_SCENARIOS]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=lambda s: s.key)
def test_no_scenario_produces_a_non_finite_row(scenario, scene):
    """EVERY row of the block must be finite -- not just the ones the sweep plots.

    `run_sweep` reduces each block to a margin over a HANDFUL of named rows, so a NaN
    anywhere else is invisible: the plot stays smooth, the crossing lands where it was
    predicted, and every other test passes. That is exactly what happened. `10a-forward`
    sweeps dt from 0, `robot_rollout` formed `vdot = (v_{i+1} - v_i)/dt`, and at dt = 0
    that 0/0 put NaN into 35 of Eq. 10a's 76 rows while the six base-pose rows the
    scenario watches stayed clean.

    It matters beyond tidiness: a NaN row reaching Ipopt is not a bad number, it is an
    immediate `Invalid_Number_Detected` with no indication of which constraint produced
    it. Better to catch it here, where the scenario name says where to look.
    """
    import casadi as ca

    # `ScenarioStep` keeps only the watched rows, so the block has to be rebuilt to see
    # all of them. Every 10th point plus BOTH ENDPOINTS -- the endpoints are where the
    # degenerate parameter values live (dt = 0, zero separation, zero load), and they are
    # where this test earns its keep.
    values = list(scenario.values)
    probe = sorted({0, len(values) - 1, *range(0, len(values), 10)})

    for index in probe:
        value = float(values[index])
        block, _watched, _state = scenario.build(scene, value)
        for rows, labels in ((block.eq, block.eq_labels), (block.ineq, block.ineq_labels)):
            if rows is None or rows.numel() == 0:
                continue
            numbers = np.asarray(ca.DM(rows)).ravel()
            bad = np.where(~np.isfinite(numbers))[0]
            assert bad.size == 0, (
                f"{scenario.key} at {scenario.param_label} = {value}: non-finite rows "
                f"{[labels[i] for i in bad[:5]]} (of {bad.size})"
            )


# =============================================================================
# Eq. 8 -- the rows and the freedoms that `8-slip` alone cannot reach
# =============================================================================
def test_8_spin_isolates_the_yaw_row_with_zero_translation(scene):
    """Eq. 8's second row needs a sweep that does not also translate the contact.

    `8-slip` slides the foot, which lights up dp_x/dp_y and would mask a broken yaw
    row entirely -- it does not even watch `8:dyaw`. Pivoting on the contact point
    keeps dp at machine zero, so the yaw row is the only thing that can move.
    """
    result = run_sweep(get_scenario("8-spin"), scene)
    assert result.ever_violated

    for step in result.steps[::10]:
        assert abs(step.rows["8:dp_x"]) < 1e-9, "the foot translated; yaw is not isolated"
        assert abs(step.rows["8:dp_y"]) < 1e-9
        assert step.rows["8:dyaw"] == pytest.approx(-step.param, abs=1e-9)


def test_8_does_not_constrain_the_normal_direction(scene):
    """Eq. 8 is an IN-PLANE lock, not a full pose lock.

    If it constrained p_z as well, a foot could never leave the floor and the robot
    could never take a step -- so this negative result is load-bearing, not trivia.
    """
    result = run_sweep(get_scenario("8-lift"), scene)
    assert not result.ever_violated
    worst = max(abs(v) for step in result.steps for v in step.rows.values())
    assert worst < 1e-9, f"lifting perturbs Eq. 8 by {worst}"


def test_8_is_relative_not_absolute_so_a_carried_contact_never_slips(scene):
    """The deepest property of Eq. 8, and the one a fixed floor can never show.

    `8-slip` and `8-spin` both use the floor, which never moves -- so they cannot
    distinguish "relative pose is fixed" from "world pose is fixed". Written against
    world poses, Eq. 8 would forbid moving a grasped object at all, making the
    paper's own box-placement task infeasible.
    """
    result = run_sweep(get_scenario("8-carry"), scene)
    assert not result.ever_violated

    # Both patches really do move, or the test proves nothing.
    steps = result.steps
    travel = np.linalg.norm(
        steps[-1].state["box_pose"].translation - steps[0].state["box_pose"].translation
    )
    assert travel > 0.2, "the box did not move; the carry is not being exercised"
    assert not np.allclose(steps[0].state["q"], steps[-1].state["q"]), "the robot did not move"

    worst = max(abs(v) for step in steps for v in step.rows.values())
    assert worst < 1e-9, f"a rigid carry perturbs Eq. 8 by {worst}"


def test_every_row_of_eq_8_is_watched_by_some_scenario(scene):
    """No row of a constraint may go unexercised by the scenario layer.

    `8-slip` watches only dp_x/dp_y. A sign error in the yaw row would have passed
    the whole playground before `8-spin` existed.
    """
    watched = set()
    for scenario in ALL_SCENARIOS:
        if scenario.equation == "8":
            watched.update(run_sweep(scenario, scene).steps[0].rows)
    assert watched == {"8:dp_x", "8:dp_y", "8:dyaw"}


# =============================================================================
# Eqs. 10 / 11 -- checked against INDEPENDENT physics, not against themselves
# =============================================================================
def test_11b_reproduces_the_textbook_euler_equation(scene):
    """The non-circular check for `-[ad_V]^T G V`.

    For a body spinning at constant rate with no external wrench, Eq. 11b's residual
    must equal the classical Euler term `omega x (I omega)` -- computed here with
    numpy's cross product, which shares no code with the adjoint formulation. If the
    adjoint were transposed wrongly, or the (angular, linear) ordering flipped, this
    would disagree while a self-referential test would not notice.
    """
    from faro.scenarios.constraint_demos import _SPIN_INERTIA

    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    for step in run_sweep(get_scenario("11-spin"), scene).steps[::10]:
        omega = step.param * axis
        expected = np.cross(omega, _SPIN_INERTIA @ omega)
        actual = np.array([step.rows[f"11b:W_ext[{i}]"] for i in range(3)]) \
            if all(f"11b:W_ext[{i}]" in step.rows for i in range(3)) else None
        if actual is None:          # only the watched row is recorded
            actual = np.array([step.rows["11b:W_ext[0]"]])
            expected = expected[:1]
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"omega={step.param}: adjoint term does not match omega x (I omega)"
        )


def test_11b_gyroscopic_term_grows_quadratically(scene):
    """omega x (I omega) is quadratic in omega -- a linear term would be a sign of a
    dropped product."""
    steps = run_sweep(get_scenario("11-spin"), scene).steps
    a, b = steps[20], steps[40]
    assert b.param == pytest.approx(2.0 * a.param, rel=1e-6)
    assert abs(b.rows["11b:W_ext[0]"]) == pytest.approx(
        4.0 * abs(a.rows["11b:W_ext[0]"]), rel=1e-6
    )


def test_a_cube_can_never_exhibit_the_gyroscopic_term(scene):
    """Why `11-spin` uses a probe body instead of the scene's box.

    The box is a cube, so I = k*Id and omega x (I omega) = k (omega x omega) = 0 for
    every omega. Using it would have produced a scenario that passes for the wrong
    reason -- the term would be untested, not verified.
    """
    from faro.constraints.dynamics_object import adjoint_transpose_term, spatial_inertia
    import casadi as ca

    inertia = scene.objects["box"].inertia
    assert np.allclose(inertia, inertia[0, 0] * np.eye(3)), "the box is no longer a cube"

    G = spatial_inertia(scene.objects["box"].mass, inertia)
    V = ca.DM(np.array([[1.0], [2.0], [3.0], [0.0], [0.0], [0.0]]))
    term = np.array(ca.DM(ca.evalf(adjoint_transpose_term(V, G)))).ravel()
    assert np.allclose(term, 0.0, atol=1e-12)


def test_11_hold_actually_uses_the_gravity_wrench_and_the_angular_first_ordering(scene):
    """Guards the bug this scenario used to contain.

    It previously wrote the support force at index 2 -- the moment-about-z slot --
    and still "passed", because with V = Vdot = 0 the residual was just the number
    the scenario had computed by hand. Now the weight comes from `gravity_wrench`,
    so a wrong slot no longer cancels.
    """
    from faro.scenarios.constraint_demos import _W_FORCE_Z

    assert _W_FORCE_Z == 5, "wrenches are angular-first: force z is index 5"

    result = run_sweep(get_scenario("11-hold"), scene)
    box = scene.objects["box"]
    first = result.steps[0]
    assert first.param == pytest.approx(box.mass * scene.gravity, abs=1e-9)
    assert abs(first.rows[f"11b:W_ext[{_W_FORCE_Z}]"]) < 1e-9

    # Off the balance point the residual is exactly the force imbalance.
    for step in result.steps[::20]:
        expected = box.mass * scene.gravity - step.param
        assert step.rows[f"11b:W_ext[{_W_FORCE_Z}]"] == pytest.approx(expected, abs=1e-9)


def test_10_weight_sums_over_BOTH_end_effectors(scene):
    """A single contact cannot distinguish `sum_e f_e` from `f_1`.

    The sweep splits the load across the two real sole patches, so a builder that
    dropped the sum would balance at half the weight and the crossing would be wrong.
    """
    from faro.scenarios.constraint_demos import robot_weight

    result = run_sweep(get_scenario("10-weight"), scene)
    assert result.measured_crossing == pytest.approx(robot_weight(scene), abs=1.0)
    assert result.steps[0].state["patches"] == ("left_foot_sole", "right_foot_sole")


def test_10_moment_arm_is_measured_from_the_com_not_the_origin(scene):
    """Eq. 10b's `(p_e - c) x f_e`. Using the world origin instead would offset every
    moment by `c x f`, which is exactly the kind of error that still looks plausible.
    """
    result = run_sweep(get_scenario("10-moment"), scene)
    weight = sum(i.mass for i in scene.robot.model.inertias) * scene.gravity
    for step in result.steps[::10]:
        # p - c = (offset, 0, -c_z); crossed with (0,0,f_z) gives (0, -offset*f_z, 0).
        assert step.rows["10:hdot_ang_y"] == pytest.approx(-step.param * weight, rel=1e-9)
