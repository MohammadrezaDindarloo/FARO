"""Milestone 2, part 2: the constraint blocks used SYMBOLICALLY on the real robot.

`test_milestone2_constraints.py` checks the algebra with numeric constants. That is
necessary but not sufficient: Milestones 3-5 feed these blocks CasADi SX variables
wired to Pinocchio's symbolic kinematics, and hand them to Ipopt. Two things must
hold that constant-folding tests cannot show:

  1. the expressions BUILD against symbolic q (no numeric-only code paths), and
  2. they are DIFFERENTIABLE in q -- otherwise Ipopt has nothing to work with.

This file is the bridge between "the formula is right" and "the NLP will run".

    pytest tests/test_milestone2_symbolic.py -v
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import pytest

from faro.constraints.contact import contact_kinematic
from faro.constraints.dynamics_robot import (
    centroidal_consistency,
    center_of_mass,
    integrate_configuration,
    robot_dynamics,
    total_mass,
)
from faro.constraints.limits import actuated_slice, joint_position_limits
from faro.core.patches import Attachment
from faro.scene.scene import Scene


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_config("box_placement")


@pytest.fixture(scope="module")
def standing(scene: Scene) -> np.ndarray:
    """Nominal pose with the soles dropped onto the floor."""
    q = scene.robot.q_nominal.copy()
    soles = [scene.patches["left_foot_sole"], scene.patches["right_foot_sole"]]
    q[2] += -min(scene.patch_world_placement(p, q).translation[2] for p in soles)
    return q


@pytest.fixture(scope="module")
def sym(scene: Scene):
    """Symbolic model + a symbolic q, with frame placements computed once."""
    cmodel = scene.robot.casadi_model()
    cdata = cmodel.createData()
    q = ca.SX.sym("q", scene.robot.nq)
    cpin.forwardKinematics(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)
    return cmodel, cdata, q


def patch_world_symbolic(scene: Scene, cdata, patch):
    """World placement of a robot-attached patch as a symbolic (R, p) pair.

    The symbolic twin of `Scene.patch_world_placement` -- the numeric one is the
    reference these tests check against.
    """
    frame_id = scene.robot.frame_id(patch.parent)
    oMf_R = cdata.oMf[frame_id].rotation
    oMf_p = cdata.oMf[frame_id].translation
    local_R = ca.DM(patch.placement.rotation)
    local_p = ca.DM(patch.placement.translation.reshape(3, 1))
    return oMf_R @ local_R, oMf_R @ local_p + oMf_p


# =============================================================================
def test_contact_kinematic_builds_and_matches_numeric(scene, sym, standing):
    """Eq. 7a/7b built from symbolic FK must equal the numeric evaluation.

    This is the test that would catch a frame-composition error: the symbolic path
    composes patch offsets onto Pinocchio's oMf, the numeric path uses pinocchio's
    SE3 operator, and they must agree to machine precision.
    """
    cmodel, cdata, q = sym
    sole = scene.patches["left_foot_sole"]
    floor = scene.patches["floor"]

    R_a, p_a = patch_world_symbolic(scene, cdata, sole)
    floor_world = scene.patch_world_placement(floor)

    block = contact_kinematic(
        R_a, p_a,
        ca.DM(floor_world.rotation), ca.DM(floor_world.translation.reshape(3, 1)),
        ca.DM(list(sole.half_extents)), ca.DM(list(floor.half_extents)),
    )

    fn = ca.Function("contact", [q], [block.eq, block.ineq])
    eq_sym, ineq_sym = (np.array(v).ravel() for v in fn(standing))

    # Numeric reference, via the Milestone 1 scene machinery.
    sole_world = scene.patch_world_placement(sole, standing)
    ref = contact_kinematic(
        ca.DM(sole_world.rotation), ca.DM(sole_world.translation.reshape(3, 1)),
        ca.DM(floor_world.rotation), ca.DM(floor_world.translation.reshape(3, 1)),
        ca.DM(list(sole.half_extents)), ca.DM(list(floor.half_extents)),
    )
    eq_ref = np.array(ca.DM(ca.evalf(ref.eq))).ravel()
    assert eq_sym == pytest.approx(eq_ref, abs=1e-12)

    # And it really is a contact: p_z = 0 exactly, normals aligned to ~1e-2
    # (the deliberate 0.01 rad ankle residual in the nominal pose).
    assert abs(eq_sym[2]) < 1e-9
    assert np.linalg.norm(eq_sym[:2]) < 0.02
    assert ineq_sym.max() <= 0


def test_contact_kinematic_is_differentiable_in_q(scene, sym, standing):
    """Ipopt needs dc/dq. A zero or non-finite Jacobian means the NLP cannot move."""
    cmodel, cdata, q = sym
    sole, floor = scene.patches["left_foot_sole"], scene.patches["floor"]
    R_a, p_a = patch_world_symbolic(scene, cdata, sole)
    floor_world = scene.patch_world_placement(floor)

    block = contact_kinematic(
        R_a, p_a,
        ca.DM(floor_world.rotation), ca.DM(floor_world.translation.reshape(3, 1)),
        ca.DM(list(sole.half_extents)), ca.DM(list(floor.half_extents)),
    )

    jac = ca.Function("dcontact", [q], [ca.jacobian(block.eq, q)])
    J = np.array(jac(standing))

    assert J.shape == (3, scene.robot.nq)
    assert np.all(np.isfinite(J))
    assert np.linalg.norm(J) > 1e-6, "constraint does not respond to q at all"

    # The base z-coordinate (index 2) must move the normal-separation row p_z:
    # lifting the robot separates the sole from the floor, one-for-one.
    assert abs(J[2, 2]) == pytest.approx(1.0, abs=1e-6)


def test_finite_difference_agrees_with_analytic_jacobian(scene, sym, standing):
    """Cross-check autodiff against finite differences.

    Guards against an expression that is differentiable but wrong -- e.g. a frozen
    constant where a variable belongs, which autodiff reports happily as zero.
    """
    cmodel, cdata, q = sym
    sole, floor = scene.patches["left_foot_sole"], scene.patches["floor"]
    R_a, p_a = patch_world_symbolic(scene, cdata, sole)
    floor_world = scene.patch_world_placement(floor)
    block = contact_kinematic(
        R_a, p_a,
        ca.DM(floor_world.rotation), ca.DM(floor_world.translation.reshape(3, 1)),
        ca.DM(list(sole.half_extents)), ca.DM(list(floor.half_extents)),
    )

    fn = ca.Function("c", [q], [block.eq])
    jac = ca.Function("dc", [q], [ca.jacobian(block.eq, q)])
    J = np.array(jac(standing))

    eps = 1e-6
    # Perturb a knee, which genuinely moves the foot -- a joint with no effect would
    # make this test pass vacuously.
    idx = scene.robot.model.joints[scene.robot.model.getJointId("left_knee_joint")].idx_q
    q_plus, q_minus = standing.copy(), standing.copy()
    q_plus[idx] += eps
    q_minus[idx] -= eps
    fd = (np.array(fn(q_plus)).ravel() - np.array(fn(q_minus)).ravel()) / (2 * eps)

    assert fd == pytest.approx(J[:, idx], abs=1e-5)
    assert np.linalg.norm(fd) > 1e-4, "the knee should move the foot contact"


def test_centroidal_consistency_builds_symbolically(scene, sym):
    """Eq. 10b second line: h = A(q) v, with A the centroidal momentum matrix."""
    cmodel, cdata, q = sym
    v = ca.SX.sym("v", scene.robot.nv)
    h = ca.SX.sym("h", 6)

    block = centroidal_consistency(cmodel, cdata, q, v, h)
    assert block.n_eq == 6

    fn = ca.Function("cc", [q, v, h], [block.eq])
    # At zero velocity and zero momentum the residual must vanish identically.
    out = np.array(fn(scene.robot.q_nominal, np.zeros(scene.robot.nv), np.zeros(6))).ravel()
    assert out == pytest.approx(np.zeros(6), abs=1e-9)


def test_centroidal_map_reproduces_pinocchio_numerically(scene, sym):
    """The symbolic A(q) must equal Pinocchio's numeric ccrba."""
    cmodel, cdata, q = sym
    v = ca.SX.sym("v", scene.robot.nv)
    h = ca.SX.sym("h", 6)
    block = centroidal_consistency(cmodel, cdata, q, v, h)
    fn = ca.Function("cc", [q, v, h], [block.eq])

    q0 = scene.robot.q_nominal
    v0 = np.random.default_rng(0).normal(size=scene.robot.nv) * 0.1

    model, data = scene.robot.model, scene.robot.model.createData()
    A_numeric = pin.ccrba(model, data, q0, v0)
    h_expected = np.asarray(A_numeric @ v0).ravel()

    # residual = h - A v, so feeding h = A v must give zero.
    residual = np.array(fn(q0, v0, h_expected)).ravel()
    assert residual == pytest.approx(np.zeros(6), abs=1e-8)


def test_total_mass_matches_pinocchio(scene):
    m = total_mass(scene.robot.model)
    data = scene.robot.model.createData()
    expected = pin.computeTotalMass(scene.robot.model, data)
    assert m == pytest.approx(expected, rel=1e-12)
    # Sanity: a G1 is a ~30-45 kg humanoid, not a 3 kg toy or a 300 kg machine.
    assert 20.0 < m < 70.0


def test_center_of_mass_symbolic_matches_numeric(scene, sym):
    cmodel, cdata, q = sym
    com = center_of_mass(cmodel, cdata, q)
    fn = ca.Function("com", [q], [com])

    q0 = scene.robot.q_nominal
    data = scene.robot.model.createData()
    expected = np.asarray(pin.centerOfMass(scene.robot.model, data, q0)).ravel()
    assert np.array(fn(q0)).ravel() == pytest.approx(expected, abs=1e-9)


def test_manifold_integration_is_not_vector_addition(scene, sym):
    """`(+)` in Eqs. 10a/11a is manifold integration.

    Adding a tangent vector to a quaternion configuration would leave the base off
    the unit sphere. This asserts the quaternion stays normalised, which plain
    addition would break.
    """
    cmodel, _, q = sym
    v = ca.SX.sym("v", scene.robot.nv)
    dt = ca.SX.sym("dt")

    q_next = integrate_configuration(cmodel, q, v, dt)
    fn = ca.Function("integ", [q, v, dt], [q_next])

    q0 = scene.robot.q_nominal
    v0 = np.zeros(scene.robot.nv)
    v0[3:6] = [0.5, -0.3, 0.2]          # pure base rotation
    out = np.array(fn(q0, v0, 0.1)).ravel()

    assert np.linalg.norm(out[3:7]) == pytest.approx(1.0, abs=1e-9)
    # Naive addition would NOT stay normalised -- confirm the test is meaningful.
    naive = q0.copy()
    naive[3:6] += v0[3:6] * 0.1
    assert abs(np.linalg.norm(naive[3:7]) - 1.0) > 1e-6


def test_robot_dynamics_defect_vanishes_on_a_consistent_triple(scene, sym):
    """Eq. 10a: a state generated BY the integrator must have zero defect.

    Constructing (q_next, v_next, h_next) from (q_i, v_i, h_i) with backward Euler
    and then asking the constraint for its residual is the cleanest possible check
    that the transcription matches the paper.
    """
    cmodel, cdata, _ = sym
    nq, nv = scene.robot.nq, scene.robot.nv

    q_i = ca.SX.sym("q_i", nq)
    v_i = ca.SX.sym("v_i", nv)
    h_i = ca.SX.sym("h_i", 6)
    v_n = ca.SX.sym("v_n", nv)
    vdot_n = ca.SX.sym("vdot_n", nv)
    hdot_n = ca.SX.sym("hdot_n", 6)
    dt = ca.SX.sym("dt")

    # Build the consistent successor exactly as Eq. 10a prescribes.
    q_n = integrate_configuration(cmodel, q_i, v_n, dt)
    h_n = h_i + hdot_n * dt

    block = robot_dynamics(
        cmodel, cdata, q_i, v_i, h_i, q_n, v_n, h_n, vdot_n, hdot_n, dt
    )
    fn = ca.Function("dyn", [q_i, v_i, h_i, v_n, vdot_n, hdot_n, dt], [block.eq])

    rng = np.random.default_rng(1)
    q0 = scene.robot.q_nominal
    v0 = rng.normal(size=nv) * 0.05
    h0 = rng.normal(size=6) * 0.1
    dt0 = 0.02
    vdot0 = rng.normal(size=nv) * 0.5
    hdot0 = rng.normal(size=6) * 0.5
    vn = v0 + vdot0 * dt0          # satisfies the velocity row by construction

    residual = np.array(fn(q0, v0, h0, vn, vdot0, hdot0, dt0)).ravel()
    assert residual == pytest.approx(np.zeros_like(residual), abs=1e-9)
    assert block.n_eq == nv + nv + 6      # tangent-space q defect + v + h


def test_robot_dynamics_detects_forward_euler(scene, sym):
    """Using v_i instead of v_{i+1} must NOT satisfy the backward-Euler defect.

    The two transcriptions differ only in an index, so this pins that the paper's
    implicit form is what was implemented.
    """
    cmodel, cdata, _ = sym
    nq, nv = scene.robot.nq, scene.robot.nv
    q_i = ca.SX.sym("q_i", nq)
    v_i = ca.SX.sym("v_i", nv)
    h_i = ca.SX.sym("h_i", 6)
    v_n = ca.SX.sym("v_n", nv)
    vdot_n = ca.SX.sym("vdot_n", nv)
    hdot_n = ca.SX.sym("hdot_n", 6)
    dt = ca.SX.sym("dt")

    # FORWARD Euler successor: integrate with v_i, not v_{i+1}.
    q_n_wrong = integrate_configuration(cmodel, q_i, v_i, dt)
    h_n = h_i + hdot_n * dt

    block = robot_dynamics(
        cmodel, cdata, q_i, v_i, h_i, q_n_wrong, v_n, h_n, vdot_n, hdot_n, dt
    )
    fn = ca.Function("dyn", [q_i, v_i, h_i, v_n, vdot_n, hdot_n, dt], [block.eq])

    rng = np.random.default_rng(2)
    v0 = rng.normal(size=nv) * 0.05
    vdot0 = rng.normal(size=nv) * 0.5
    dt0 = 0.02
    residual = np.array(
        fn(scene.robot.q_nominal, v0, np.zeros(6), v0 + vdot0 * dt0, vdot0, np.zeros(6), dt0)
    ).ravel()
    assert np.linalg.norm(residual) > 1e-6


def test_joint_limits_block_on_the_real_robot(scene, sym):
    """Eq. 13a over the ACTUATED slice only, built symbolically."""
    cmodel, _, q = sym
    robot = scene.robot
    lower, upper = robot.joint_limits()
    sl = actuated_slice(robot.model)

    block = joint_position_limits(q[sl], lower[sl], upper[sl])
    assert block.n_ineq == 2 * 29          # 29 actuated joints, two bounds each

    fn = ca.Function("lim", [q], [block.ineq])
    assert np.array(fn(robot.q_nominal)).ravel().max() <= 0

    # Drive one joint past its upper limit and confirm the right row reacts.
    q_bad = robot.q_nominal.copy()
    idx = robot.model.joints[robot.model.getJointId("left_knee_joint")].idx_q
    q_bad[idx] = upper[idx] + 0.25
    values = np.array(fn(q_bad)).ravel()
    assert values.max() == pytest.approx(0.25, abs=1e-9)
