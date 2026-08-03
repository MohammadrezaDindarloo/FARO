"""Reference trajectories, so Eqs. 10a and 11a can be validated BEFORE the TO exists.

WHY THIS MODULE EXISTS
----------------------
Eqs. 10a and 11a are integration constraints. Every other constraint in Milestone 2
can be checked at a single configuration, but an integrator only says anything across
consecutive steps -- so the obvious place to exercise it is inside the trajectory
optimization of Eq. 17.

That is the wrong place to exercise it FIRST. If the integrator's first outing is
inside a solver and the solve fails, there is no way to tell a bad transcription from
a bad initial guess, bad scaling, or a genuinely infeasible problem. The failure mode
is `Infeasible_Problem_Detected` with nothing to bisect.

So the trajectory is written by hand here, the defect is evaluated on it directly, and
the integrator is either right or wrong on its own terms. No solver involved.

THE CONSTRUCTION
----------------
`robot_rollout` builds a WALKING trajectory that satisfies Eq. 10a EXACTLY. The motion
is prescribed as a velocity profile -- a real gait, chosen to be recognisable on screen
-- and the configurations follow from the paper's own recurrence:

    vdot_{i+1} = (v_{i+1} - v_i) / dt      <- the gait's own accelerations
    q_{i+1} = q_i (+) v_{i+1} dt           <- BACKWARD Euler: v at i+1, not i

An earlier version used RANDOM accelerations. Those satisfied Eq. 10a just as exactly
and were useless: the robot tumbled through the air in a way no one could look at and
judge. A formulation check has to be legible, or it only tests the algebra.

Evaluated on that trajectory the defect must be zero to machine precision at any dt.
Pass `backward=False` and the same motion is transcribed with `v_i` instead -- forward
Euler -- and the defect becomes non-zero and scales as O(dt^2), the local truncation
error of a first-order method. Halving dt divides it by exactly 4.

That pair is the whole point. The first says "this is an integrator"; the second says
"and it is the BACKWARD one", which is the distinction the paper makes and the one a
transcription is most likely to get wrong silently.

SCOPE -- what this does NOT claim
---------------------------------
Eq. 10a is a KINEMATIC identity: it relates q, v and h across a step and says nothing
about whether the motion is dynamically achievable. Making 10a and 10b hold together
-- so that `h = A(q)v` at every knot AND `hdot` is the net external wrench -- is a
coupled problem with no closed-form rollout. That coupling IS the trajectory
optimization, and it belongs to Milestone 5. These trajectories are deliberately only
10a-consistent, and the scenarios say so rather than implying more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass
class Rollout:
    """A hand-built reference trajectory: states, and the inputs that produced them."""

    dt: float
    q: list
    v: list
    h: list
    vdot: list
    hdot: np.ndarray

    def __len__(self) -> int:
        return len(self.q)


# A walking gait, in the paper's own state variables. Amplitudes are modest and the
# phasing is the textbook one -- legs in antiphase, arms counter-swinging against them
# -- so the motion on screen is recognisable as walking rather than as an integrator
# exercise. Nothing here is tuned to make Eq. 10a pass; the trajectory is built from a
# VELOCITY profile and the constraint is evaluated on whatever configurations follow.
_GAIT_PERIOD = 1.0          # seconds per full stride
_GAIT_BOB = 0.02            # vertical travel of the torso, metres
_GAIT_YAW = 0.06            # torso yaw sway, radians
_GAIT_LEG = 0.35            # hip/knee swing amplitude, radians
_GAIT_ARM = 0.30            # shoulder swing amplitude, radians

# THE GAIT. Defined as an analytic position profile plus its exact derivative, so the
# rollout can start from the gait's own t=0 pose and integrate the gait's own velocity.
#
# The knee term is the one that matters and the one that is easy to get wrong. Writing
# it as a plain antiphase sinusoid, `+A sin` on the left and `-A sin` on the right,
# produces two failures at once:
#
#   * the right knee bends BACKWARDS through its -0.087 rad limit, because a knee is a
#     one-way hinge and a sign flip does not respect that;
#   * both feet stay at exactly the same height. A planar two-link leg has foot height
#     `L1 cos(theta) + L2 cos(theta + phi)`, which is EVEN in (theta, phi) -- so negating
#     hip and knee together leaves the foot exactly where it was. The robot skates.
#
# `(1 - cos)` fixes both: it is non-negative for every t, so the knee only ever flexes
# in the direction it can, and the two legs reach their flexion peaks half a cycle
# apart. That is what lifts each foot in turn.
_GAIT_JOINTS = {
    # joint name                  amplitude, phase, profile
    "left_hip_pitch_joint":       (_GAIT_LEG, 0.0, "sin"),
    "right_hip_pitch_joint":      (_GAIT_LEG, np.pi, "sin"),
    "left_knee_joint":            (1.1 * _GAIT_LEG, 0.0, "flex"),
    "right_knee_joint":           (1.1 * _GAIT_LEG, np.pi, "flex"),
    "left_ankle_pitch_joint":     (-0.4 * _GAIT_LEG, 0.0, "sin"),
    "right_ankle_pitch_joint":    (-0.4 * _GAIT_LEG, np.pi, "sin"),
    # Arms counter-swing: left arm forward when the RIGHT leg is forward.
    "left_shoulder_pitch_joint":  (_GAIT_ARM, np.pi, "sin"),
    "right_shoulder_pitch_joint": (_GAIT_ARM, 0.0, "sin"),
}


def _profile(kind: str, amplitude: float, omega: float, t: float, phase: float):
    """(position offset, velocity) for one joint's gait profile."""
    angle = omega * t + phase
    if kind == "sin":
        return amplitude * np.sin(angle), amplitude * omega * np.cos(angle)
    # "flex": one-way, non-negative, peaking once per stride.
    return amplitude * (1.0 - np.cos(angle)), amplitude * omega * np.sin(angle)


def gait_configuration(model, t: float, q_stand) -> np.ndarray:
    """The gait's configuration at time t -- used for the rollout's STARTING pose.

    The rollout integrates `gait_velocity`, so its configurations drift from this by
    the integrator's own O(dt^2) per step. That drift is real and is not hidden: this
    function only fixes where the trajectory starts, so that a joint whose profile is
    non-zero at t = 0 does not begin from the wrong place and walk out of its limits.
    """
    omega = 2.0 * np.pi / _GAIT_PERIOD
    q = np.asarray(q_stand, dtype=float).copy()
    for name, (amplitude, phase, kind) in _GAIT_JOINTS.items():
        offset, _ = _profile(kind, amplitude, omega, t, phase)
        q[model.joints[model.getJointId(name)].idx_q] += offset
    return q


def gait_velocity(model, t: float, *, speed: float = 0.4) -> np.ndarray:
    """The robot's velocity `v` at time t, in Pinocchio's ordering.

    A TANGENT vector, which is what Eq. 10a integrates. The free-flyer's linear and
    angular components are expressed in the BASE frame per Pinocchio's convention, so
    `v[0]` is forward speed along the torso's own x-axis, not world x.

    Defining the trajectory by its VELOCITY is what lets Eq. 10a hold exactly: the
    paper's recurrence consumes v, so a velocity profile integrates into configurations
    that satisfy it by construction, while still describing a motion chosen for
    physical plausibility rather than for algebraic convenience.
    """
    omega = 2.0 * np.pi / _GAIT_PERIOD
    v = np.zeros(model.nv)

    v[0] = speed                                             # forward, in the base frame
    v[2] = _GAIT_BOB * omega * np.cos(2.0 * omega * t)       # torso bobs twice per stride
    v[5] = _GAIT_YAW * omega * np.cos(omega * t)             # gentle yaw sway

    for name, (amplitude, phase, kind) in _GAIT_JOINTS.items():
        _, rate = _profile(kind, amplitude, omega, t, phase)
        v[model.joints[model.getJointId(name)].idx_v] = rate
    return v


def robot_rollout(model, *, dt: float = 0.02, steps: int = 24, backward: bool = True,
                  mass: float = 1.0, gravity: float = 9.81, speed: float = 0.4,
                  q0=None, seed: int = 0) -> Rollout:
    """A WALKING trajectory for Eq. 10a. Exactly consistent when `backward=True`.

    The motion is a real gait -- torso translating forward with a slight bob and yaw
    sway, legs swinging in antiphase, arms counter-swinging -- so it can be judged by
    eye. An earlier version used random accelerations, which produced a robot tumbling
    in mid-air: exact against Eq. 10a and useless for telling whether the formulation
    describes something a robot could do.

    Two things fall out of using a real gait rather than noise:

      * the torso YAWS, so the quaternion genuinely rotates. That matters because
        revolute joints integrate correctly under plain addition -- a transcription bug
        that only breaks the exponential map would hide in a motion whose base merely
        translates;
      * the accelerations are the gait's own, `vdot = (v_{i+1} - v_i) / dt`, rather
        than numbers chosen to make the constraint pass.
    """
    # Start standing, not at `pin.neutral` -- the neutral pose puts the pelvis at z = 0,
    # which buries the feet 0.76 m under the floor and makes the whole gait unreadable.
    stand = pin.neutral(model) if q0 is None else np.asarray(q0, dtype=float)
    q = [gait_configuration(model, 0.0, stand)]
    v = [gait_velocity(model, 0.0, speed=speed)]
    h = [np.zeros(6)]
    vdot = [np.zeros(model.nv)]

    # `hdot` is Eq. 10b's first line with gravity only. Eq. 10a does not care where it
    # came from -- it just integrates it -- so this gives `h` something real to carry
    # without claiming the gait is dynamically consistent (see the module docstring).
    hdot = np.array([0.0, 0.0, -mass * gravity, 0.0, 0.0, 0.0])

    for i in range(steps):
        v_next = gait_velocity(model, (i + 1) * dt, speed=speed)
        # vdot follows from the gait, so the velocity line of Eq. 10a holds exactly.
        # dt = 0 is a real sweep value (the `10a-forward` scenario starts there), and it
        # makes this 0/0. NaN, not a large number -- and NaN in `vdot` poisons 35 of the
        # 76 rows of Eq. 10a while the base-pose rows the scenario plots stay finite, so
        # nothing on screen would show it. Zero is the correct limit: v_next == v[i] when
        # no time passes, so eq_v = v_next - (v_i + 0) = 0 either way.
        vdot.append((v_next - v[i]) / dt if dt > 0.0 else np.zeros(model.nv))
        # THE line that distinguishes the two transcriptions.
        rate = v_next if backward else v[i]
        q.append(pin.integrate(model, q[i], rate * dt))
        v.append(v_next)
        h.append(h[i] + hdot * dt)

    return Rollout(dt=dt, q=q, v=v, h=h, vdot=vdot, hdot=hdot)


def object_rollout(pose0: pin.SE3, twist0, *, dt: float = 0.02, steps: int = 24,
                   backward: bool = True, twist_rate=None) -> list:
    """A trajectory for Eq. 11a, built the same way. Returns [(pose, twist), ...].

    `twist0` and `twist_rate` are ANGULAR-FIRST body quantities, matching
    `faro.constraints.dynamics_object`. The pose steps with the SE(3) exponential:

        q_{i+1} = q_i * exp6(V_{i+1} dt)

    -- a right multiplication, because V is a body twist. `object_pose_residual`
    documents why that side matters.
    """
    from faro.constraints.frames import swap_spatial_ordering

    import casadi as ca

    rate = np.zeros(6) if twist_rate is None else np.asarray(twist_rate, dtype=float)
    poses, twists = [pose0], [np.asarray(twist0, dtype=float)]

    for i in range(steps):
        V_next = twists[i] + rate * dt
        stepping = V_next if backward else twists[i]
        # angular-first -> pinocchio's linear-first, via the one sanctioned converter
        linear_first = np.asarray(
            ca.DM(swap_spatial_ordering(ca.DM(stepping.reshape(6, 1)))), dtype=float
        ).ravel()
        poses.append(poses[i] * pin.exp6(linear_first * dt))
        twists.append(V_next)

    return list(zip(poses, twists))
