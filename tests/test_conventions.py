"""Robotics-convention audit: the errors that a passing test suite hides.

Every check here was written after finding the corresponding bug by hand. They all
share a shape: the code was self-consistent and produced plausible numbers, but
disagreed with the physics in a regime the existing scenarios never entered.

  * Eq. 7d was evaluated in the wrong frame. It only matters at nonzero relative
    yaw, and every Eq. 7d scenario happened to place the foot square with the floor.
  * Eq. 9's normal flipped sign under penetration, so interpenetration read as
    clearance. Only reachable once bodies actually overlap, which no demo did.
  * The spatial 6-vector ordering differs between Eqs. 10b and 11b, correctly, and
    nothing yet crossed the boundary where that becomes a swap.

The tests are therefore written against INDEPENDENT ground truth -- brute-force
corner enumeration, textbook Euler equations, analytically known distances -- not
against a second copy of the implementation.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pytest

from faro.constraints.collision import query_witness, signed_distance_expr
from faro.constraints.contact import contact_wrench
from faro.constraints.dynamics_object import (
    adjoint_transpose_term,
    spatial_inertia,
    transform_wrench_to_body,
)
from faro.constraints.dynamics_robot import centroidal_momentum_rate
from faro.constraints.limits import actuated_torque, joint_position_limits
from faro.constraints.frames import (
    RX_PI,
    log3,
    relative_patch_transform,
    rotated_half_extent_rows,
    swap_spatial_ordering,
)
from faro.scene.scene import Scene


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_config("box_placement")

XI_A = np.array([0.085, 0.030])   # the G1 sole: deliberately NOT square
FZ = 100.0


def _yaw(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _evalf(expr) -> np.ndarray:
    return np.array(ca.DM(ca.evalf(expr))).ravel()


# =============================================================================
# Eq. 7d -- centre of pressure, and the frame it is measured in
# =============================================================================
def _cop_rows(cop_a, R_rel, p_rel, f=None):
    """Eq. 7d's four CoP rows for a contact whose CoP is `cop_a` in patch a's frame."""
    f = np.array([0.0, 0.0, FZ]) if f is None else f
    cop_b = R_rel @ np.append(cop_a, 0.0) + p_rel
    kappa_b = np.cross(cop_b, f)
    block = contact_wrench(
        ca.DM(f.reshape(3, 1)), ca.DM(kappa_b.reshape(3, 1)), ca.DM(XI_A.reshape(2, 1)),
        mu=0.7, mu_torsional=1e4,        # huge, so only the CoP rows can bite
        R_rel=ca.DM(R_rel), p_rel=ca.DM(p_rel.reshape(3, 1)),
    )
    rows = dict(zip(block.ineq_labels, _evalf(block.ineq)))
    return max(rows[k] for k in ("7d:+kappa_x", "7d:-kappa_x", "7d:+kappa_y", "7d:-kappa_y"))


@pytest.mark.parametrize("yaw_deg", [0.0, 15.0, 45.0, 90.0, 137.0])
def test_eq_7d_bounds_the_cop_by_patch_a_at_any_relative_yaw(yaw_deg):
    """`^a xi_a` means a's extents in A'S frame -- so the bound must follow a's yaw.

    Eq. 7a leaves the relative yaw FREE, so a foot planted at 45 degrees is perfectly
    admissible. Evaluating Eq. 7d in frame b instead confines the CoP to a rectangle
    of a's DIMENSIONS but b's ORIENTATION, which is a different set: with an 85x30 mm
    sole at 45 degrees it rejects a CoP well inside the foot and accepts one outside.
    """
    R = _yaw(np.deg2rad(yaw_deg))
    origin = np.zeros(3)
    assert _cop_rows(XI_A * 0.99, R, origin) <= 1e-9, "a CoP inside the foot was rejected"
    assert _cop_rows(XI_A * 1.60, R, origin) > 1e-9, "a CoP outside the foot was accepted"


def test_eq_7d_agrees_with_ground_truth_over_random_contacts():
    """Brute force: the CoP is inside a's rectangle iff Eq. 7d is satisfied."""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        R = _yaw(rng.uniform(-np.pi, np.pi))
        p_rel = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.0])
        cop_a = rng.uniform(-1.5, 1.5, 2) * XI_A
        if np.min(np.abs(np.abs(cop_a) - XI_A)) < 1e-6:
            continue                       # exactly on the boundary: no verdict to check
        inside = bool(np.all(np.abs(cop_a) <= XI_A))
        assert (_cop_rows(cop_a, R, p_rel) <= 1e-9) == inside, (
            f"Eq. 7d disagreed with ground truth at cop_a={cop_a}, p_rel={p_rel}"
        )


def test_eq_7d_cop_rows_keep_the_paper_s_x_y_swap():
    """kappa_x is bounded by the half-extent in Y, because the lever arm is along y.

    Guards against "tidying" the subscripts into matching pairs, which reads better
    and is wrong: with an 85x30 mm sole the two bounds differ by 2.8x.
    """
    eye, origin = np.eye(3), np.zeros(3)
    # A CoP displaced along +y only: it loads kappa_x, which must see the Y extent.
    assert _cop_rows(np.array([0.0, XI_A[1] * 0.99]), eye, origin) <= 1e-9
    assert _cop_rows(np.array([0.0, XI_A[0] * 0.99]), eye, origin) > 1e-9, (
        "kappa_x was bounded by the X extent -- the x<->y swap has been lost"
    )


# =============================================================================
# Eq. 9 -- the signed distance must actually be SIGNED
# =============================================================================
@pytest.mark.parametrize("gap", [0.10, 0.02, -0.02, -0.06])
def test_eq_9_reports_true_signed_distance_through_penetration(gap):
    """Two 0.2 m boxes offset by 0.2 + gap: the signed distance IS `gap`, by geometry.

    The bug this guards: the normal was built from `w_A - w_B`, which reverses when
    the bodies overlap. Eq. 9 then reported +0.06 for a 6 cm interpenetration, so
    `0 <= sd` was most comfortably satisfied exactly where the bodies were most
    deeply merged -- the solver would have been rewarded for driving them together.
    """
    coal = pytest.importorskip("coal")

    box = coal.Box(0.2, 0.2, 0.2)
    eye = np.eye(3)
    offset = np.array([0.2 + gap, 0.0, 0.0])
    witness = query_witness(box, coal.Transform3s(eye, np.zeros(3)),
                            box, coal.Transform3s(eye, offset))

    sd = float(ca.DM(ca.evalf(signed_distance_expr(
        ca.DM(eye), ca.DM(np.zeros((3, 1))),
        ca.DM(eye), ca.DM(offset.reshape(3, 1)), witness))))

    assert sd == pytest.approx(gap, abs=1e-6), (
        f"Eq. 9 reported sd={sd:+.4f} for a true signed distance of {gap:+.4f}"
    )
    assert np.sign(sd) == np.sign(gap), "penetration must give a NEGATIVE signed distance"


def test_eq_9_witness_normal_is_unit_and_points_from_b_to_a():
    coal = pytest.importorskip("coal")
    box, eye = coal.Box(0.2, 0.2, 0.2), np.eye(3)
    w = query_witness(box, coal.Transform3s(eye, np.zeros(3)),
                      box, coal.Transform3s(eye, np.array([0.5, 0.0, 0.0])))
    assert np.linalg.norm(w.normal) == pytest.approx(1.0)
    # A is at the origin, B is at +x, so separation grows along -x as seen from A.
    assert w.normal[0] < 0


# =============================================================================
# Eq. 11b -- body-frame Newton-Euler, against the textbook form
# =============================================================================
def test_eq_11b_gyroscopic_term_is_exactly_the_euler_equation():
    """-[ad_V]^T G V must reduce to (omega x I omega, m omega x v), by hand.

    This is the check that the (angular, linear) ordering and the `ad_V` block layout
    agree with each other. Swap either and the term still evaluates, but the moment
    picks up the mass and the force picks up the inertia.
    """
    inertia = np.diag([0.10, 0.50, 0.90])     # anisotropic: an isotropic one hides errors
    mass, G = 2.0, None
    G = spatial_inertia(mass, inertia)
    rng = np.random.default_rng(0)
    for _ in range(50):
        omega, v = rng.normal(size=3), rng.normal(size=3)
        got = _evalf(adjoint_transpose_term(ca.DM(np.concatenate([omega, v]).reshape(6, 1)), G))
        want = np.concatenate([np.cross(omega, inertia @ omega), mass * np.cross(omega, v)])
        np.testing.assert_allclose(got, want, atol=1e-12)


def test_spatial_inertia_is_angular_first_not_pinocchio_s_ordering():
    """G = diag(I, m I3). Pinocchio's `Inertia.matrix()` is the other way round."""
    G = spatial_inertia(3.0, np.diag([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(np.diag(G), [1.0, 2.0, 4.0, 3.0, 3.0, 3.0])


def test_external_wrench_assembles_the_papers_definition_line():
    """W_ext := W_env + W_grav + sum_a W_a -- which had no home in the code at all.

    `object_newton_euler` took W_ext as a parameter, the sum was assembled by hand in
    one scenario, `W_env` appeared nowhere outside a printed string, and
    `transform_wrench_to_body` was called by no production code -- only by a test.
    """
    from faro.constraints.dynamics_object import external_wrench

    mass, g = 2.0, 9.81
    # Gravity alone, body frame aligned with the world: pure downward force, no moment.
    only_gravity = _evalf(external_wrench(mass, ca.DM.eye(3), gravity=g))
    np.testing.assert_allclose(only_gravity, [0, 0, 0, 0, 0, -mass * g], atol=1e-12)

    # A support at the centre of mass exactly cancels it.
    balanced = _evalf(external_wrench(
        mass, ca.DM.eye(3), gravity=g,
        contacts=[(ca.DM([0.0, 0.0, mass * g]), ca.DM.zeros(3), ca.DM.zeros(3))]))
    np.testing.assert_allclose(balanced, 0.0, atol=1e-12)

    # W_env is carried through rather than silently assumed zero.
    with_env = _evalf(external_wrench(
        mass, ca.DM.eye(3), gravity=g, W_env=ca.DM([0.0, 0.0, 0.7, 1.5, 0.0, 0.0])))
    np.testing.assert_allclose(with_env - only_gravity, [0, 0, 0.7, 1.5, 0, 0], atol=1e-12)


def test_external_wrench_transports_off_centre_contacts_into_a_moment():
    """The reason this cannot be a plain sum of force vectors.

    A hand pressing a box FACE acts well away from the centre of mass, so its wrench
    contributes `p x f` to the moment even with no applied torque. Summing raw forces
    gives an object that translates correctly and NEVER ROTATES when pushed off-centre
    -- and Eq. 11b's residual balances perfectly for that wrong physics, so nothing
    downstream would catch it.
    """
    from faro.constraints.dynamics_object import external_wrench

    push = ca.DM([0.0, 0.0, 10.0])
    point = np.array([0.15, 0.0, 0.0])       # 15 cm off-axis: a box face, not the centre

    centred = _evalf(external_wrench(
        0.0, ca.DM.eye(3), gravity=0.0,
        contacts=[(push, ca.DM.zeros(3), ca.DM.zeros(3))]))
    offset = _evalf(external_wrench(
        0.0, ca.DM.eye(3), gravity=0.0,
        contacts=[(push, ca.DM.zeros(3), ca.DM(point.reshape(3, 1)))]))

    # Same force either way...
    np.testing.assert_allclose(centred[3:], offset[3:], atol=1e-12)
    # ...but only the off-centre one produces a moment, and it is exactly p x f.
    np.testing.assert_allclose(centred[:3], 0.0, atol=1e-12)
    np.testing.assert_allclose(offset[:3], np.cross(point, np.array([0.0, 0.0, 10.0])), atol=1e-12)
    assert np.abs(offset[:3]).max() > 1.0, "an off-centre push must generate a real moment"


def test_transform_wrench_to_body_transports_the_moment_arm():
    R = pin.utils.rpyToMatrix(0.2, 0.3, -0.4)
    point = np.array([0.1, -0.2, 0.05])
    force_w = np.array([1.0, 2.0, -3.0])
    got = _evalf(transform_wrench_to_body(
        ca.DM(R), ca.DM(point.reshape(3, 1)),
        ca.DM(force_w.reshape(3, 1)), ca.DM(np.zeros((3, 1)))))
    force_b = R.T @ force_w
    np.testing.assert_allclose(got[3:], force_b, atol=1e-12)
    np.testing.assert_allclose(got[:3], np.cross(point, force_b), atol=1e-12)


# =============================================================================
# Eq. 11a -- object pose integration on SE(3)
# =============================================================================
def _se3_sx(M: pin.SE3):
    import pinocchio.casadi as cpin

    return cpin.SE3(ca.SX(ca.DM(M.rotation)), ca.SX(ca.DM(M.translation.reshape(3, 1))))


_M_I = pin.SE3(pin.utils.rpyToMatrix(0.3, -0.2, 0.7), np.array([0.4, -0.1, 0.6]))
_OMEGA = np.array([0.9, -1.4, 2.1])
_LINVEL = np.array([0.5, 0.2, -0.3])
_V = np.concatenate([_OMEGA, _LINVEL])       # ANGULAR-FIRST, body twist
_DT = 0.05


def _pose_residual(M_next: pin.SE3) -> np.ndarray:
    from faro.constraints.dynamics_object import object_pose_residual

    return _evalf(object_pose_residual(
        _se3_sx(_M_I), ca.DM(_V.reshape(6, 1)), _se3_sx(M_next), _DT).eq)


def test_eq_11a_pose_line_is_satisfied_by_a_consistent_pose_pair():
    """The first line of Eq. 11a existed only as a comment; `q_i`/`q_next` were ignored.

    `object_integration` accepted both poses and returned a six-row block that used
    neither, so a caller could hand it two wildly inconsistent poses and get a zero
    residual. Its docstring pointed at a `pose_residual` function that was never
    written. Ground truth here is built with NUMERIC pinocchio, independently of the
    CasADi path under test.
    """
    truth = _M_I * pin.exp6(np.concatenate([_LINVEL, _OMEGA]) * _DT)
    np.testing.assert_allclose(_pose_residual(truth), 0.0, atol=1e-12)


def test_eq_11a_pose_line_rejects_an_inconsistent_pose_pair():
    """The check the old code could not make: a wrong next pose must show up."""
    truth = _M_I * pin.exp6(np.concatenate([_LINVEL, _OMEGA]) * _DT)
    drifted = pin.SE3(truth.rotation, truth.translation + np.array([0.05, 0.0, 0.0]))
    assert np.abs(_pose_residual(drifted)).max() > 1e-3


def test_eq_11a_uses_a_body_twist_not_a_spatial_one():
    """V is a BODY twist, so it composes on the RIGHT: M_i * exp6(V dt).

    Left-multiplying rotates the object about the WORLD origin instead of its own
    centre. The two agree only when the object sits at the origin with identity
    orientation -- exactly the pose a first test reaches for, which is why this one
    deliberately uses a rotated, translated body.
    """
    spatial = pin.exp6(np.concatenate([_LINVEL, _OMEGA]) * _DT) * _M_I
    assert np.abs(_pose_residual(spatial)).max() > 1e-3


def test_eq_11a_pose_line_respects_the_angular_first_convention():
    """pinocchio's exp6 is LINEAR-first; Eq. 11b and this module are ANGULAR-first.

    This is the one place a twist genuinely crosses that boundary. Forgetting the swap
    integrates the angular rate as a translation and vice versa -- it still runs, and
    the answer is silently wrong.
    """
    swapped = _M_I * pin.exp6(np.concatenate([_OMEGA, _LINVEL]) * _DT)
    assert np.abs(_pose_residual(swapped)).max() > 1e-3


def test_full_object_integration_returns_both_lines_of_eq_11a():
    from faro.constraints.dynamics_object import full_object_integration

    truth = _M_I * pin.exp6(np.concatenate([_LINVEL, _OMEGA]) * _DT)
    V_i = np.array([0.4, -0.6, 1.0, 0.2, 0.1, -0.1])
    Vdot = (_V - V_i) / _DT

    block = full_object_integration(
        _se3_sx(_M_I), ca.DM(V_i.reshape(6, 1)), _se3_sx(truth),
        ca.DM(_V.reshape(6, 1)), ca.DM(Vdot.reshape(6, 1)), _DT)

    assert block.n_eq == 12, "6 pose rows + 6 twist rows"
    np.testing.assert_allclose(_evalf(block.eq), 0.0, atol=1e-12)


def test_eq_11a_pose_line_is_differentiable():
    """It has to survive being handed to Ipopt, not just evaluated."""
    from faro.constraints.dynamics_object import object_pose_residual

    truth = _M_I * pin.exp6(np.concatenate([_LINVEL, _OMEGA]) * _DT)
    V = ca.SX.sym("V", 6)
    block = object_pose_residual(_se3_sx(_M_I), V, _se3_sx(truth), _DT)
    jac = ca.Function("J", [V], [ca.jacobian(block.eq, V)])(_V)
    assert np.all(np.isfinite(np.array(jac)))
    assert np.abs(np.array(jac)).max() > 1e-6, "the residual must actually depend on V"


# =============================================================================
# Eq. 10b -- centroidal ordering, pinned against Pinocchio itself
# =============================================================================
def test_eq_10b_is_linear_first_and_matches_pinocchios_centroidal_map(scene):
    """Eq. 10b is written [linear; angular], and `Ag` must be read the same way.

    Pinned against Pinocchio rather than asserted: a pure vertical base velocity has
    momentum `m * v` in the LINEAR rows and none in the angular ones.
    """
    model = scene.robot.model
    data = model.createData()
    v = np.zeros(model.nv)
    v[2] = 1.0
    h = pin.computeCentroidalMap(model, data, pin.neutral(model)) @ v

    assert h[2] == pytest.approx(pin.computeTotalMass(model), rel=1e-9)
    np.testing.assert_allclose(h[3:], 0.0, atol=1e-9)

    # And `centroidal_momentum_rate` puts gravity in those same rows.
    hdot = _evalf(centroidal_momentum_rate([], [], [], ca.DM.zeros(3), mass=10.0))
    np.testing.assert_allclose(hdot, [0.0, 0.0, -98.1, 0.0, 0.0, 0.0], atol=1e-9)


def test_static_scenarios_cannot_test_the_centroidal_map_at_all(scene):
    """Why `10-swing` had to exist: at v = 0, `h - A(q)v` vanishes for ANY matrix A.

    `10-weight` and `10-moment` both pin the robot in static equilibrium, so they
    exercise only the FIRST line of Eq. 10b. The second line, h = A(q)v, is identically
    satisfied at zero velocity no matter what A contains -- a centroidal map filled
    with garbage would pass both of them. This makes that concrete rather than
    asserting it, by feeding a deliberately wrong A and showing the residual is
    unchanged at v = 0 and different the moment the robot moves.
    """
    model = scene.robot.model
    q = scene.robot.q_nominal
    A = pin.computeCentroidalMap(model, model.createData(), q)
    nonsense = np.zeros_like(A)          # as wrong as a centroidal map can be

    at_rest = np.zeros(model.nv)
    np.testing.assert_allclose(A @ at_rest, nonsense @ at_rest, atol=1e-12)

    moving = np.zeros(model.nv)
    moving[model.joints[model.getJointId("left_shoulder_pitch_joint")].idx_v] = 1.0
    assert np.abs(A @ moving - nonsense @ moving).max() > 1e-3, (
        "a moving robot must distinguish the real centroidal map from a wrong one"
    )


def test_10_swing_generates_angular_momentum_from_joint_motion_alone(scene):
    """The physical claim `10-swing` is built on, checked against Pinocchio directly.

    Swinging the arms produces centroidal ANGULAR momentum with the base perfectly
    still and nothing in contact -- the falling-cat effect. If this were zero the
    scenario would be demonstrating nothing, and `h = A(q)v` could be dropped from
    Eq. 10b without consequence.
    """
    from faro.scenarios.constraint_demos import _SWING_JOINTS, _SWING_RATE_MAX

    model = scene.robot.model
    v = np.zeros(model.nv)
    for joint in _SWING_JOINTS:
        v[model.joints[model.getJointId(joint)].idx_v] = -_SWING_RATE_MAX

    h = pin.computeCentroidalMap(model, model.createData(), scene.robot.q_nominal) @ v
    assert np.abs(h[3:]).max() > 1e-2, (
        f"arm swing must generate angular momentum; got {h[3:]}"
    )


def test_eq_10b_moment_arm_is_measured_from_the_com_not_the_origin():
    """What makes the momentum CENTROIDAL. Using the world origin is a silent bug."""
    f = [ca.DM([0.0, 0.0, 10.0])]
    p = [ca.DM([0.5, 0.0, 0.0])]
    com = ca.DM([0.5, 0.0, 1.0])     # directly above the contact: zero lever arm
    hdot = _evalf(centroidal_momentum_rate(f, [ca.DM.zeros(3)], p, com, mass=0.0, gravity=0.0))
    np.testing.assert_allclose(hdot[3:], 0.0, atol=1e-12)

    com_off = ca.DM([0.0, 0.0, 1.0])   # 0.5 m to the side: a real moment appears
    hdot = _evalf(centroidal_momentum_rate(f, [ca.DM.zeros(3)], p, com_off, mass=0.0, gravity=0.0))
    np.testing.assert_allclose(hdot[3:], np.cross([0.5, 0.0, -1.0], [0.0, 0.0, 10.0]), atol=1e-12)


def test_swap_spatial_ordering_is_an_involution_and_moves_the_blocks():
    w = ca.DM([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(_evalf(swap_spatial_ordering(w)), [4, 5, 6, 1, 2, 3])
    np.testing.assert_allclose(_evalf(swap_spatial_ordering(swap_spatial_ordering(w))),
                               [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError):
        swap_spatial_ordering(ca.DM([1.0, 2.0, 3.0]))


# =============================================================================
# Symbolic type compatibility -- Milestone 4 builds MX graphs
# =============================================================================
@pytest.mark.parametrize("sym", [ca.SX.sym, ca.MX.sym])
def test_dynamics_terms_build_on_both_sx_and_mx_graphs(sym):
    """Hard-coded `ca.SX.zeros` made these unusable from an MX graph.

    Large multiple-shooting transcriptions are normally built with MX, so this would
    have surfaced at Milestone 4 as a type error deep inside an unrelated assembly.
    """
    x = sym("x", 6)
    adjoint_transpose_term(x, spatial_inertia(2.0, np.eye(3)))
    centroidal_momentum_rate([x[:3]], [x[3:6]], [x[:3]], x[:3], mass=1.0)


# =============================================================================
# Eq. 13 -- limits are PER JOINT, from each joint's own URDF specification
# =============================================================================
def test_eq_13_limits_are_per_joint_and_genuinely_heterogeneous(scene):
    """Eq. 13 is componentwise, so each joint must carry its OWN bound.

    Worth pinning because a single shared scalar would look right on any scenario that
    only ever drives one joint. The G1's 29 actuated joints have 13 distinct position
    limits, 5 distinct velocity limits and 5 distinct effort limits -- a knee is rated
    at 139 N.m and 20 rad/s while a wrist is 25 N.m and 37 rad/s.
    """
    from faro.constraints.limits import actuated_slice, default_velocity_limits

    model = scene.robot.model
    q_slice, v_slice = actuated_slice(model), actuated_slice(model, velocity=True)
    upper = scene.robot.joint_limits()[1][q_slice]

    assert len(set(np.round(upper, 6))) > 1, "position limits must differ between joints"
    assert len(set(np.round(default_velocity_limits(model)[v_slice], 6))) > 1
    assert len(set(np.round(np.asarray(model.effortLimit)[v_slice], 6))) > 1

    # And the block really is two rows per joint, not one shared pair.
    block = joint_position_limits(
        ca.DM(scene.robot.q_nominal[q_slice].reshape(-1, 1)),
        scene.robot.joint_limits()[0][q_slice], upper)
    assert block.ineq.shape[0] == 2 * (model.nv - 6)


def test_eq_13c_is_correct_for_more_than_one_joint(scene):
    """13c has only ever been CALLED with a single joint -- the knee.

    So the vectorised path was unexercised: the elementwise `slope * v`, the per-joint
    `tau_max` subtraction, and the label-to-row alignment across four sign blocks. A
    scalar-broadcast bug there would be invisible to every scenario.
    """
    from faro.constraints.limits import torque_speed_limits

    tau_max = np.array([139.0, 25.0, 5.0])       # knee, wrist, and a weak joint
    v_tau_max = np.array([20.0, 37.0, 22.0])
    tau = np.array([70.0, 24.0, 0.5])
    v = np.array([1.0, 1.0, 1.0])

    block = torque_speed_limits(ca.DM(tau.reshape(3, 1)), ca.DM(v.reshape(3, 1)),
                                tau_max=tau_max, v_tau_max=v_tau_max)
    assert block.ineq.shape[0] == 4 * 3 == len(block.ineq_labels)

    values = _evalf(block.ineq)
    for i in range(3):
        by_hand = abs(tau[i]) + (tau_max[i] / v_tau_max[i]) * abs(v[i]) - tau_max[i]
        from_code = max(v_ for lab, v_ in zip(block.ineq_labels, values) if lab.endswith(f"[{i}]"))
        assert from_code == pytest.approx(by_hand, abs=1e-12), (
            f"joint {i} used the wrong tau_max/v_tau_max pair"
        )


def test_missing_velocity_limits_are_sanitised_not_taken_as_zero(scene):
    """Pinocchio reports 0 for a URDF that omits a joint's velocity limit.

    Zero as a BOUND freezes the joint and makes Eq. 13b violated at any non-zero rate --
    which reads as a solver bug, not a model gap. The G1 specifies all 29, so the raw
    array happens to work; the robot is meant to be swappable, so the scenarios go
    through `default_velocity_limits` instead.
    """
    from faro.constraints.limits import default_velocity_limits

    model = scene.robot.model
    assert np.all(default_velocity_limits(model) > 0)

    # Simulate a URDF that forgot one.
    patched = model.copy()
    patched.velocityLimit[10] = 0.0
    sanitised = default_velocity_limits(patched)
    assert sanitised[10] > 1.0, "a missing limit must not become a zero bound"
    assert np.all(sanitised[:10] == default_velocity_limits(model)[:10]), "others untouched"


# =============================================================================
# Eq. 12 -- the torque definition, against Pinocchio's own RNEA
# =============================================================================
def test_eq_12_matches_rnea_with_external_forces(scene):
    """tau = S(M vdot + b - sum J_e^T lambda_e), checked against a second Pinocchio path.

    `pin.rnea(..., fext)` computes exactly that quantity internally, from joint-local
    forces, without ever forming a Jacobian. Agreeing with it pins THREE things at
    once: the minus sign (contact wrenches offload the actuators), the [force; moment]
    ordering that `getFrameJacobian` expects, and the moment transport to the joint.
    """
    model = scene.robot.model
    data = model.createData()
    rng = np.random.default_rng(0)

    q = pin.randomConfiguration(model)
    q[:3] = rng.normal(size=3)
    q[3:7] /= np.linalg.norm(q[3:7])
    v, a = rng.normal(size=model.nv) * 0.3, rng.normal(size=model.nv) * 0.3
    wrench = rng.normal(size=6) * 10.0        # [force; moment], LOCAL_WORLD_ALIGNED

    fid = model.getFrameId("left_ankle_roll_link")
    jid = model.frames[fid].parentJoint
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
    drift = pin.rnea(model, data, q, v, a)    # M(q) vdot + b(q, v)

    ours = _evalf(actuated_torque(
        ca.DM(drift.reshape(-1, 1)), ca.DM(np.zeros((model.nv, 1))),
        [ca.DM(J)], [ca.DM(wrench.reshape(6, 1))], slice(0, model.nv)))

    # The same wrench handed to Pinocchio as a joint-local external force.
    lever = data.oMf[fid].translation - data.oMi[jid].translation
    moment_at_joint = wrench[3:] + np.cross(lever, wrench[:3])
    R_joint = data.oMi[jid].rotation
    fext = pin.StdVec_Force()
    for _ in range(model.njoints):
        fext.append(pin.Force.Zero())
    fext[jid] = pin.Force(R_joint.T @ wrench[:3], R_joint.T @ moment_at_joint)

    np.testing.assert_allclose(ours, pin.rnea(model, data, q, v, a, fext), atol=1e-9)


def test_eq_12_contact_term_sign_is_not_arbitrary(scene):
    """Control for the test above: flipping the sign must change the answer a lot.

    Guards the paper's own warning -- get it backwards and the apparent torque roughly
    doubles instead of cancelling, so Eq. 13c starts rejecting perfectly good motions.
    """
    model = scene.robot.model
    data = model.createData()
    q, v = pin.neutral(model), np.zeros(model.nv)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J = pin.getFrameJacobian(model, data, model.getFrameId("left_ankle_roll_link"),
                             pin.LOCAL_WORLD_ALIGNED)
    drift = ca.DM(pin.rnea(model, data, q, v, v).reshape(-1, 1))
    zero, wrench = ca.DM(np.zeros((model.nv, 1))), ca.DM(np.full((6, 1), 50.0))

    right = _evalf(actuated_torque(drift, zero, [ca.DM(J)], [wrench], slice(0, model.nv)))
    wrong = _evalf(actuated_torque(drift, zero, [ca.DM(-J)], [wrench], slice(0, model.nv)))
    assert np.abs(right - wrong).max() > 1.0


# =============================================================================
# Eq. 10a -- `(+)` is manifold integration, not vector addition
# =============================================================================
def test_eq_10a_integrates_on_the_manifold_not_by_vector_addition(scene):
    """The floating base is quaternion-parameterised; adding omega*dt to it leaves SO(3).

    The error is SECOND order in |omega| dt -- `sqrt(1 + |omega dt|^2) ~= 1 + |omega
    dt|^2 / 2` -- which is exactly what makes it dangerous. At a walking-scale base
    rate it is a few parts in 10^5 and looks like round-off; at the 3 rad/s a dynamic
    task actually reaches it is over 1%, and every rotation built from that quaternion
    is a scaled non-rotation that quietly rescales the forces and moments passing
    through it.
    """
    import pinocchio.casadi as cpin

    from faro.constraints.dynamics_robot import integrate_configuration

    model = scene.robot.model
    dt = 0.05
    v = np.zeros(model.nv)
    v[3:6] = 3.0                       # a rate the paper's dynamic tasks really reach
    q = pin.neutral(model)

    got = _evalf(integrate_configuration(cpin.Model(model), q, v, dt))
    np.testing.assert_allclose(got, pin.integrate(model, q, v * dt), atol=1e-12)
    assert np.linalg.norm(got[3:7]) == pytest.approx(1.0, abs=1e-12)

    naive = q.copy()
    naive[3:6] += v[3:6] * dt          # the usual mistake: omega dt into the vector part
    assert np.linalg.norm(naive[3:7]) == pytest.approx(
        np.sqrt(1.0 + np.linalg.norm(v[3:6] * dt) ** 2), rel=1e-12)
    assert abs(np.linalg.norm(naive[3:7]) - 1.0) > 1e-2, "the naive form must leave the manifold"
    # ...and it is not merely off-scale: renormalising still gives the wrong rotation.
    assert np.linalg.norm(naive[3:7] / np.linalg.norm(naive[3:7]) - got[3:7]) > 1e-4


# =============================================================================
# Eq. 7a / 7b -- the patch-frame convention, against brute force
# =============================================================================
def test_rx_pi_flip_turns_facing_outward_normals_into_the_papers_convention():
    """FARO uses outward normals; the paper implies parallel z. One Rx(pi) reconciles.

    Built the physical way round: place a face-to-face on b at a known yaw, then check
    Eq. 7a's residual is zero in x and y and reports exactly that yaw in z.
    """
    R_b = pin.utils.rpyToMatrix(0.3, -0.2, 0.7)
    p_b = np.array([0.1, 0.2, 0.3])
    yaw = 0.6
    R_a = R_b @ np.array(ca.DM(RX_PI)) @ _yaw(yaw)

    R_rel, p_rel = relative_patch_transform(
        ca.DM(R_a), ca.DM(p_b.reshape(3, 1)), ca.DM(R_b), ca.DM(p_b.reshape(3, 1)))

    np.testing.assert_allclose(_evalf(log3(R_rel)), [0.0, 0.0, yaw], atol=1e-12)
    np.testing.assert_allclose(_evalf(p_rel), 0.0, atol=1e-12)


def test_eq_7b_matches_brute_force_corner_enumeration():
    """`^b xi_a` is the support of a's rotated rectangle -- checkable by enumeration."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        R = _yaw(rng.uniform(-np.pi, np.pi))
        xi_a = rng.uniform(0.01, 0.20, 2)
        xi_b = rng.uniform(0.01, 0.40, 2)
        p = rng.uniform(-0.3, 0.3, 3)
        corners = np.array([[sx * xi_a[0], sy * xi_a[1], 0.0]
                            for sx in (1, -1) for sy in (1, -1)])
        for axis in (0, 1):
            rows, _ = rotated_half_extent_rows(
                ca.DM(R), ca.DM(p.reshape(3, 1)),
                ca.DM(xi_a.reshape(2, 1)), ca.DM(xi_b.reshape(2, 1)), axis)
            want = abs(p[axis]) + np.max(np.abs((R @ corners.T)[axis])) - xi_b[axis]
            assert float(np.max(_evalf(rows))) == pytest.approx(want, abs=1e-12)
