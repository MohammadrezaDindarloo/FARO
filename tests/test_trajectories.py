"""Eqs. 10a and 11a validated on hand-built trajectories, with no solver involved.

These are the only Milestone 2 constraints that span more than one state, so the
tempting place to exercise them is inside the TO (Eq. 17). That is the wrong place to
exercise them FIRST: if an integrator's debut is inside a solver and the solve fails,
`Infeasible_Problem_Detected` cannot distinguish a bad transcription from a bad initial
guess, bad scaling, or a genuinely infeasible problem. There is nothing to bisect.

So the trajectory is written by hand, the defect is evaluated on it directly, and the
transcription is right or wrong on its own terms.

The decisive evidence is the CONVERGENCE ORDER. A trajectory built to satisfy backward
Euler gives machine zero; the same motion transcribed forward gives a defect that
quarters when dt halves -- O(dt^2), the local truncation error of a first-order method.
A defect that merely "looks small" proves nothing; one that scales at exactly the
predicted rate over four step sizes is a real measurement.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pytest

from faro.constraints.dynamics_object import full_object_integration
from faro.constraints.dynamics_robot import robot_dynamics, total_mass
from faro.scenarios.trajectories import object_rollout, robot_rollout
from faro.scene.scene import Scene


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_config("box_placement")


def _worst_defect(scene, dt: float, *, backward: bool) -> float:
    model = scene.robot.model
    cmodel = scene.robot.casadi_model()
    cdata = cmodel.createData()
    roll = robot_rollout(model, dt=dt, steps=8, backward=backward,
                         mass=total_mass(model), gravity=scene.gravity)

    worst = 0.0
    for i in range(len(roll) - 1):
        block = robot_dynamics(
            cmodel, cdata,
            ca.DM(roll.q[i].reshape(-1, 1)), ca.DM(roll.v[i].reshape(-1, 1)),
            ca.DM(roll.h[i].reshape(6, 1)),
            ca.DM(roll.q[i + 1].reshape(-1, 1)), ca.DM(roll.v[i + 1].reshape(-1, 1)),
            ca.DM(roll.h[i + 1].reshape(6, 1)),
            ca.DM(roll.vdot[i + 1].reshape(-1, 1)), ca.DM(roll.hdot.reshape(6, 1)),
            dt,
        )
        worst = max(worst, float(np.abs(np.array(ca.DM(ca.evalf(block.eq)))).max()))
    return worst


def test_eq_10a_is_exact_on_a_backward_euler_trajectory(scene):
    """Built to satisfy the paper's own recurrence, so the defect must be machine zero.

    At any dt -- an integrator that is only accurate for small steps has a bug, not a
    discretization error.
    """
    for dt in (0.04, 0.02, 0.01, 0.005):
        assert _worst_defect(scene, dt, backward=True) < 1e-12, f"failed at dt={dt}"


def test_eq_10a_is_backward_euler_and_not_forward(scene):
    """The distinction the paper draws, and the one most likely to be lost silently.

    Forward Euler still 'works' -- it is a valid, less stable transcription of the same
    ODE -- so nothing downstream would complain. The defect on a forward-built
    trajectory is `(v_{i+1} - v_i) dt = vdot dt^2`, and the RATIO is the evidence:
    halving dt must quarter it. A merely-small defect would prove nothing.
    """
    steps = [0.04, 0.02, 0.01, 0.005]
    defects = [_worst_defect(scene, dt, backward=False) for dt in steps]

    assert defects[0] > 1e-4, "forward Euler must produce a clearly non-zero defect"
    for coarse, fine in zip(defects, defects[1:]):
        assert coarse / fine == pytest.approx(4.0, rel=0.05), (
            f"expected O(dt^2) local error (ratio 4.0), got {coarse / fine:.3f} "
            f"from defects {defects}"
        )


# =============================================================================
# Does the recurrence actually REPRODUCE the true motion?
#
# The defect tests above ask: given (q_i, q_{i+1}, v_{i+1}), is the constraint zero?
# That is what the NLP sees -- but the trajectory was BUILT from that same recurrence,
# so the answer is zero by construction. It proves the implementation matches the
# recurrence; it cannot prove the recurrence reproduces the motion.
#
# This section closes that gap the other way round: march forward with Eq. 10a from
# q_0 alone and compare against a CLOSED-FORM ground truth. Backward Euler is
# first-order, so the accumulated error must fall as O(dt) -- halving dt halves it.
# =============================================================================
_HORIZON = 0.36        # a PARTIAL period on purpose; see below
_YAW_RATE = 2.0
_JOINT_RATE = 1.3
_PERIOD = 1.0


def _reference_velocity(model, t: float) -> np.ndarray:
    """Body twist + one joint rate, chosen so the exact solution is known in closed form.

    The base twist is ANGULAR ONLY and about a FIXED body axis (z). That is what makes
    the ground truth exact rather than a fine-dt approximation: rotations about one
    fixed axis commute, so the time-ordered exponential collapses to
    `R(t) = R_0 exp3(z * integral of omega)`. A twist with a changing axis has no such
    closed form, and comparing against our own integrator at small dt would be circular.
    """
    v = np.zeros(model.nv)
    v[5] = _YAW_RATE * np.cos(2 * np.pi * t / _PERIOD)
    v[model.joints[model.getJointId("left_knee_joint")].idx_v] = (
        _JOINT_RATE * np.cos(2 * np.pi * t / _PERIOD))
    return v


def _reference_pose(model, t: float, q0: np.ndarray) -> np.ndarray:
    """The exact solution at time t -- analytic integrals of the rates above."""
    scale = _PERIOD / (2 * np.pi) * np.sin(2 * np.pi * t / _PERIOD)
    q = q0.copy()
    R = pin.Quaternion(q0[6], q0[3], q0[4], q0[5]).toRotationMatrix() @ pin.exp3(
        np.array([0.0, 0.0, _YAW_RATE * scale]))
    quat = pin.Quaternion(R)
    q[3:7] = [quat.x, quat.y, quat.z, quat.w]
    q[model.joints[model.getJointId("left_knee_joint")].idx_q] += _JOINT_RATE * scale
    return q


def _march(scene, steps: int):
    """Integrate with Eq. 10a's own code path. Returns (base error, joint error)."""
    from faro.constraints.dynamics_robot import integrate_configuration

    model = scene.robot.model
    cmodel = scene.robot.casadi_model()
    dt = _HORIZON / steps
    q0 = pin.neutral(model)
    q = q0.copy()
    for i in range(steps):
        # Backward Euler: the rate at i+1, exactly as `robot_dynamics` consumes it.
        v = _reference_velocity(model, (i + 1) * dt)
        q = np.array(ca.DM(ca.evalf(integrate_configuration(cmodel, q, v, dt)))).ravel()

    exact = _reference_pose(model, _HORIZON, q0)
    R_got = pin.Quaternion(q[6], q[3], q[4], q[5]).toRotationMatrix()
    R_exp = pin.Quaternion(exact[6], exact[3], exact[4], exact[5]).toRotationMatrix()
    knee = model.joints[model.getJointId("left_knee_joint")].idx_q
    return (float(np.linalg.norm(pin.log3(R_exp.T @ R_got))),
            float(abs(q[knee] - exact[knee])))


def test_eq_10a_integrates_toward_the_true_motion(scene):
    """March forward with the equation; compare against the closed-form solution.

    This is the test the defect checks cannot be: those build the trajectory from the
    recurrence, so they are zero by construction. Here the trajectory comes from the
    EQUATION and the answer comes from calculus, and the two are compared.

    Backward Euler is first order, so the accumulated error must scale as O(dt) --
    halving dt halves it. Measured 2.02x, 2.01x, 2.00x over four step sizes.

    Two details that are easy to get wrong and would make this test meaningless:

      * the horizon is a PARTIAL period. Over a FULL period the rectangle sum of a
        cosine cancels exactly, the error collapses to ~1e-16 at every dt, and the test
        passes while measuring nothing;
      * dt DIVIDES the horizon exactly. Using round(horizon/dt) ends each run at a
        different time, and the ratios come out as noise (1.12, 1.76, 3.22).
    """
    counts = [9, 18, 36, 72]
    base = [_march(scene, n)[0] for n in counts]
    joint = [_march(scene, n)[1] for n in counts]

    assert base[0] > 1e-3, "the coarsest step must have visible error, or nothing converges"
    for series, name in ((base, "base rotation"), (joint, "joint angle")):
        for coarse, fine in zip(series, series[1:]):
            assert coarse / fine == pytest.approx(2.0, rel=0.05), (
                f"{name}: expected first-order convergence (ratio 2.0), got "
                f"{coarse / fine:.3f} from {series}"
            )


def test_naive_quaternion_integration_does_not_converge(scene):
    """The control: a wrong integrator must FAIL to converge, not merely be less accurate.

    Adding `omega dt` into the quaternion's vector part is the usual mistake. It leaves
    SO(3), so refining dt does not drive the error to zero at first order -- which is
    what distinguishes a wrong method from a coarse one. Without this control, the
    convergence test above could be satisfied by anything roughly right.
    """
    model = scene.robot.model
    knee_v = model.joints[model.getJointId("left_knee_joint")].idx_v
    knee_q = model.joints[model.getJointId("left_knee_joint")].idx_q
    q0 = pin.neutral(model)

    errors = []
    for steps in (9, 18, 36, 72):
        dt = _HORIZON / steps
        q = q0.copy()
        for i in range(steps):
            v = _reference_velocity(model, (i + 1) * dt)
            q[3:6] += v[3:6] * dt          # the mistake: omega dt into the vector part
            q[knee_q] += v[knee_v] * dt
        errors.append(abs(float(np.linalg.norm(q[3:7])) - 1.0))

    assert min(errors) > 1e-4, "the naive form should never land back on the manifold"
    ratios = [c / f for c, f in zip(errors, errors[1:])]
    assert not all(abs(r - 2.0) < 0.05 for r in ratios), (
        f"the naive integrator converged at first order, so this control proves nothing: {ratios}"
    )


def test_the_gait_is_a_walk_and_not_an_integrator_exercise(scene):
    """The trajectory has to be legible, or it only tests the algebra.

    An earlier version used random accelerations. It satisfied Eq. 10a just as exactly
    and was useless: the robot tumbled through the air, so no one could look at it and
    judge whether the formulation described something a robot could do. This checks the
    properties that make it a WALK.
    """
    from faro.scenarios.constraint_demos import _ground_the_walk, standing_configuration

    model = scene.robot.model
    data = model.createData()
    roll = robot_rollout(model, dt=0.02, steps=50, mass=total_mass(model),
                         q0=standing_configuration(scene))
    _ground_the_walk(scene, roll)

    idx = {n: model.joints[model.getJointId(n)].idx_q for n in
           ("left_knee_joint", "right_knee_joint", "left_hip_pitch_joint")}
    feet = [model.getFrameId(f"{s_}_ankle_roll_link") for s_ in ("left", "right")]

    heights = []
    for q in roll.q:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        heights.append([float(data.oMf[f].translation[2]) for f in feet])
    heights = np.array(heights)

    # Each foot lifts in turn, and they are NOT in lockstep.
    assert np.ptp(heights[:, 0]) > 0.05, "the left foot never lifts"
    assert np.ptp(heights[:, 1]) > 0.05, "the right foot never lifts"
    assert np.abs(heights[:, 0] - heights[:, 1]).max() > 0.05, (
        "both feet move identically -- the legs are not in antiphase. A planar leg's "
        "foot height is even in (hip, knee), so negating both leaves it unchanged."
    )
    assert heights.min() > -1e-6, "a foot goes through the floor"

    # Knees only ever flex the way a knee bends.
    lower, upper = scene.robot.joint_limits()
    for name, i in idx.items():
        angles = np.array([q[i] for q in roll.q])
        assert angles.min() >= lower[i] and angles.max() <= upper[i], (
            f"{name} leaves its Eq. 13a range: [{angles.min():.3f}, {angles.max():.3f}] "
            f"vs [{lower[i]:.3f}, {upper[i]:.3f}]"
        )

    # And it goes somewhere.
    assert roll.q[-1][0] - roll.q[0][0] > 0.3, "the robot does not advance"


def test_grounding_the_walk_leaves_eq_10a_untouched(scene):
    """The floor shift is a constant world-z translation, so the defect must not move.

    Worth checking rather than assuming: `q_{i+1}.p = q_i.p + R_i v_lin dt`, and a
    constant added to both sides cancels -- but only because the shift is applied to
    EVERY configuration and does not touch the rotation.
    """
    from faro.scenarios.constraint_demos import _ground_the_walk, standing_configuration

    model = scene.robot.model
    cmodel = scene.robot.casadi_model()
    cdata = cmodel.createData()

    def worst(roll):
        out = 0.0
        for i in range(len(roll) - 1):
            block = robot_dynamics(
                cmodel, cdata,
                ca.DM(roll.q[i].reshape(-1, 1)), ca.DM(roll.v[i].reshape(-1, 1)),
                ca.DM(roll.h[i].reshape(6, 1)),
                ca.DM(roll.q[i + 1].reshape(-1, 1)), ca.DM(roll.v[i + 1].reshape(-1, 1)),
                ca.DM(roll.h[i + 1].reshape(6, 1)),
                ca.DM(roll.vdot[i + 1].reshape(-1, 1)), ca.DM(roll.hdot.reshape(6, 1)),
                roll.dt)
            out = max(out, float(np.abs(np.array(ca.DM(ca.evalf(block.eq)))).max()))
        return out

    roll = robot_rollout(model, dt=0.02, steps=8, mass=total_mass(model),
                         q0=standing_configuration(scene))
    before = worst(roll)
    _ground_the_walk(scene, roll)
    assert worst(roll) < 1e-12 and before < 1e-12


def test_eq_10a_base_rotation_is_what_makes_the_test_meaningful(scene):
    """The rollout must actually rotate the floating base.

    Revolute joints integrate correctly under plain addition, so a transcription that
    breaks only the exponential map would pass a trajectory whose base merely
    translates. The rollout injects a real angular velocity; this checks it took.
    """
    model = scene.robot.model
    roll = robot_rollout(model, dt=0.02, steps=8, mass=total_mass(model))

    quats = np.array([q[3:7] for q in roll.q])
    assert np.abs(quats - quats[0]).max() > 1e-3, "the base never rotates"
    # ...and every configuration stays a valid one.
    for q in roll.q:
        assert np.linalg.norm(q[3:7]) == pytest.approx(1.0, abs=1e-12)


# =============================================================================
# Eq. 11a -- the same treatment for the object
# =============================================================================
def _object_defect(dt: float, *, backward: bool) -> float:
    import pinocchio.casadi as cpin

    def as_sx(M):
        return cpin.SE3(ca.SX(ca.DM(M.rotation)), ca.SX(ca.DM(M.translation.reshape(3, 1))))

    twist0 = np.array([0.9, -1.4, 2.1, 0.5, 0.2, -0.3])     # angular-first
    rate = np.array([0.2, -0.1, 0.4, 0.0, 0.0, -9.81])      # gravity in the linear part
    traj = object_rollout(pin.SE3(pin.utils.rpyToMatrix(0.3, -0.2, 0.7),
                                  np.array([0.4, -0.1, 0.6])),
                          twist0, dt=dt, steps=8, backward=backward, twist_rate=rate)

    worst = 0.0
    for (M_i, V_i), (M_next, V_next) in zip(traj, traj[1:]):
        block = full_object_integration(
            as_sx(M_i), ca.DM(V_i.reshape(6, 1)), as_sx(M_next),
            ca.DM(V_next.reshape(6, 1)), ca.DM(rate.reshape(6, 1)), dt)
        worst = max(worst, float(np.abs(np.array(ca.DM(ca.evalf(block.eq)))).max()))
    return worst


def test_eq_11a_is_exact_on_a_backward_euler_trajectory():
    """Both lines of Eq. 11a, on a tumbling ballistic object."""
    for dt in (0.04, 0.02, 0.01, 0.005):
        assert _object_defect(dt, backward=True) < 1e-12, f"failed at dt={dt}"


def test_eq_11a_is_backward_euler_and_not_forward():
    """Same convergence-order evidence, on SE(3) rather than the robot's manifold."""
    steps = [0.04, 0.02, 0.01, 0.005]
    defects = [_object_defect(dt, backward=False) for dt in steps]

    assert defects[0] > 1e-4
    for coarse, fine in zip(defects, defects[1:]):
        assert coarse / fine == pytest.approx(4.0, rel=0.05), (
            f"expected O(dt^2), got {coarse / fine:.3f} from {defects}"
        )


def test_object_rollout_stays_on_se3():
    """Rotations must remain rotations -- det = +1 and R^T R = I at every knot."""
    traj = object_rollout(pin.SE3.Identity(), np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
                          dt=0.02, steps=16)
    for M, _ in traj:
        R = M.rotation
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
