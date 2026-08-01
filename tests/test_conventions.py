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
from faro.constraints.limits import actuated_torque
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
