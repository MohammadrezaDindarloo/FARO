"""Constraint demonstrations -- one physical story per paper equation.

Each `Scenario` sweeps a single physical parameter and evaluates one constraint
block along the way. Crucially, each declares `predicted_crossing`: the value at
which the paper's equation says the constraint MUST start failing, derived by hand
from the equation and the scene's actual numbers. `tests/test_milestone2_scenarios.py`
then checks the measured crossing against that prediction.

That is the difference between "the code runs" and "the code implements Eq. 7b":
a prediction computed independently of the implementation, and confirmed by it.

Headless by design -- no Meshcat import anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import casadi as ca
import numpy as np
import pinocchio as pin

from faro.constraints.block import ConstraintBlock
from faro.constraints.collision import WitnessData, collision_avoidance, query_witness
from faro.constraints.contact import contact_kinematic, contact_wrench
from faro.constraints.limits import actuated_slice, joint_position_limits
from faro.constraints.dynamics_object import (
    gravity_wrench,
    object_newton_euler,
    spatial_inertia,
)
from faro.constraints.dynamics_robot import centroidal_momentum_rate, total_mass
from faro.constraints.limits import joint_velocity_limits, torque_speed_limits
from faro.constraints.no_slip import no_slip
from faro.core.patches import Attachment
from faro.scene.scene import Scene

# Equality rows are never exactly zero in floating point, so "satisfied" needs a
# tolerance. 1e-9 is far below any physically meaningful violation (a nanometre of
# separation) and far above numerical noise.
EQ_TOL = 1e-9

# Wall-clock seconds between animation frames, matching the playground's default
# --fps. Scenarios that sweep a RATE integrate against it so the motion on screen is
# the velocity under test, rather than a static pose with a caption.
_FRAME_DT = 1.0 / 25.0


# =============================================================================
# The equations, as the paper writes them.
#
# Printed above each scenario so the algebra and the animation are on screen at the
# same time. Transcribed from the PDF (arXiv:2607.18362v1, Section II-C); note the
# paper's own symbols: xi = half-extents, kappa = contact moment, lambda = wrench,
# mu_r = torsional friction coefficient.
# =============================================================================
EQUATION_TEXT: dict[str, str] = {
    "7a": (
        "(7a)   log3(R)_{x,y} = 0 ,    p_z = 0\n"
        "\n"
        "       p, R = position and rotation of patch a RELATIVE TO patch b,\n"
        "              expressed in frame b.  Patch normals are the local z-axes.\n"
        "       -> aligns the patch normals and forces zero normal separation,\n"
        "          WHILE LEAVING the relative in-plane position and yaw FREE."
    ),
    "7b": (
        "(7b)   |p_{x,y}|  <=  (^b xi_b)_{x,y}  -  (^b xi_a)_{x,y}\n"
        "\n"
        "       xi      = half-extents of a rectangular patch\n"
        "       ^b xi_a = half-extents of patch a EXPRESSED IN FRAME b,\n"
        "                 i.e. a's rectangle rotated into b's frame:\n"
        "                   ^b xi_a,x = |R_00| xi_ax + |R_01| xi_ay\n"
        "                   ^b xi_a,y = |R_10| xi_ax + |R_11| xi_ay\n"
        "       -> keeps patch a INSIDE patch b (containment, not overlap)."
    ),
    "7c": (
        "(7c)   |f_{x,y}|  <=  mu f_z ,      f_z >= 0\n"
        "\n"
        "       f  = contact force of a w.r.t. b, expressed in frame b\n"
        "       mu = Coulomb friction coefficient\n"
        "       -> unilateral contact (push only, never pull) plus a PYRAMIDAL\n"
        "          approximation of Coulomb friction: each component is bounded\n"
        "          separately, so the corner f_x = f_y = mu f_z is admissible."
    ),
    "7d": (
        "(7d)   |kappa_z|  <=  mu_r f_z\n"
        "\n"
        "       | [kappa_x] |            [ (^a xi_a)_y ]\n"
        "       | [kappa_y] |    <=  f_z [ (^a xi_a)_x ]\n"
        "\n"
        "       kappa = contact moment,  mu_r = torsional friction coefficient\n"
        "       -> torsional friction, plus centre-of-pressure bounds from the\n"
        "          FINITE PATCH SIZE.  NOTE THE x<->y SWAP: kappa_x is bounded by\n"
        "          the half-extent in Y, because a moment about x is produced by\n"
        "          the normal force acting at a lever arm along y."
    ),
    "8": (
        "(8)    (p^{s+1} - p^s)_{x,y} = 0 ,    log3( (R^s)^T R^{s+1} )_z = 0\n"
        "\n"
        "       For a sticking contact active across adjacent timesteps s, s+1.\n"
        "       -> constrains BOTH the in-plane position and the in-plane rotation\n"
        "          across timesteps.  Exactly the three DoF that (7a) left free."
    ),
    "9": (
        "(9)    0  <=  sd_AB(x)  ~=  n . ( T^w_A(x) p_A  -  T^w_B(x) p_B )\n"
        "\n"
        "       p_A, p_B = GJK witness points, in the LOCAL frames of each body\n"
        "       n        = separating normal;  T^w = world transform of a body\n"
        "       -> the signed-distance approximation of Schulman et al.  The\n"
        "          witness points and normal are FROZEN at the linearization\n"
        "          point; only the body transforms depend on the variables."
    ),
    "10": (
        "(10b)  hdot^r = [ mg + sum_e f^r_e                                  ]\n"
        "                [ sum_e (p^r_e - c) x f^r_e  +  kappa^r_e           ]\n"
        "\n"
        "       h^r = A(q^r) v^r\n"
        "\n"
        "       m = robot mass, c = centre of mass, p^r_e = contact point,\n"
        "       A(q) = centroidal momentum matrix.\n"
        "       -> the CENTROIDAL dynamics: the rate of change of momentum is\n"
        "          just gravity plus the contact wrenches.  The moment arm is\n"
        "          measured FROM THE COM, which is what makes it centroidal."
    ),
    "11": (
        "(11b)  W^o_ext = G^o Vdot^o  -  [ad_V^o]^T G^o V^o\n"
        "\n"
        "       G^o = spatial inertia of the object, V^o = body twist,\n"
        "       [ad_V]^T = adjoint Lie bracket (the Coriolis/gyroscopic term).\n"
        "       W^o_ext = W_env + W_grav + sum_a W_a\n"
        "       -> Newton-Euler for a rigid object in body coordinates.  The\n"
        "          object has its OWN dynamics; it is not part of the robot model."
    ),
    "12": (
        "(12)   tau_j := S ( M(q) vdot + b(q, v) - sum_e J_e(q)^T lambda_e )\n"
        "\n"
        "       S = selects the ACTUATED rows (drops the 6 floating-base DoF)\n"
        "       -> the torque DEFINITION, not a limit.  Note the MINUS sign:\n"
        "          contact wrenches OFFLOAD the actuators.  Flip it and the\n"
        "          apparent torque doubles, so Eq. 13c rejects good motions."
    ),
    "13b": (
        "(13b)  v_min  <=  v_j  <=  v_max                (componentwise)\n"
        "\n"
        "       -> actuated joint VELOCITY limits.  Needs velocities, so only\n"
        "          the TO (17) enforces it; mode/edge (14) and KSO (15) cannot."
    ),
    "13c": (
        "(13c)  |tau_j|  +  (tau_max / v_tau,max) |v_j|   <=  tau_max\n"
        "\n"
        "       -> a LINEAR torque-speed envelope: the full tau_max is available\n"
        "          at rest and falls affinely to zero at v_tau,max.  Implemented\n"
        "          as 4 linear rows per joint (the sign combinations of tau and\n"
        "          v) so it stays SMOOTH -- fabs would stall Ipopt at v = 0."
    ),
    "13a": (
        "(13a)  q_min  <=  q_j  <=  q_max                (componentwise)\n"
        "\n"
        "       q_j = ACTUATED joint positions (the 6 floating-base DoF are\n"
        "             unactuated and have no limits).\n"
        "       -> the only limit that mode/edge (14) and KSO (15) enforce;\n"
        "          (13b) and (13c) need velocities and torques, which those\n"
        "          kinematic problems do not have."
    ),
}


def paper_equation(equation: str) -> str:
    """The paper's own statement of an equation, for display."""
    return EQUATION_TEXT.get(equation, f"(no transcription for Eq. {equation})")


# =============================================================================
# Result containers
# =============================================================================
@dataclass
class ScenarioStep:
    """One point of a parameter sweep."""

    param: float
    rows: dict[str, float]
    margin: float          # <= 0 satisfied, > 0 violated
    state: dict            # whatever the visualizer needs (q, box pose, wrench, ...)

    @property
    def violated(self) -> bool:
        return self.margin > 0.0


@dataclass
class SweepResult:
    """A whole sweep, plus where the constraint actually started failing."""

    scenario: "Scenario"
    steps: list[ScenarioStep]

    @property
    def measured_crossing(self) -> float | None:
        """Parameter value where `margin` first crosses zero, linearly interpolated.

        Returns None if the constraint never fails over the sweep -- which is the
        CORRECT outcome for scenarios like "yaw is free" that exist to show a
        constraint deliberately does not care about something.
        """
        for prev, cur in zip(self.steps, self.steps[1:]):
            if prev.margin <= 0.0 < cur.margin:
                span = cur.margin - prev.margin
                if abs(span) < 1e-15:
                    return cur.param
                frac = -prev.margin / span
                return prev.param + frac * (cur.param - prev.param)
        return None

    @property
    def ever_violated(self) -> bool:
        return any(step.violated for step in self.steps)


# =============================================================================
# Scenario definition
# =============================================================================
@dataclass
class Scenario:
    """One physical story that exercises one constraint.

    Attributes
    ----------
    key : short id used on the command line, e.g. "7a-lift".
    equation : which paper equation this demonstrates.
    title : one-line summary.
    question : the physical question the sweep answers -- the learning point.
    expect : what should be observed, in words.
    param_label : name and unit of the swept parameter.
    values : the sweep.
    predicted_crossing : value at which the paper's equation says it must fail,
        or None if it should never fail.
    prediction_note : how that number was derived from the equation.
    build : callable(scene, value) -> (ConstraintBlock, watched_labels, state).
    """

    key: str
    equation: str
    title: str
    question: str
    expect: str
    param_label: str
    values: np.ndarray
    predicted_crossing: float | None
    prediction_note: str
    build: Callable
    tolerance: float = 1e-3   # how close measured must be to predicted


def _margin(block: ConstraintBlock, watched: list[str]) -> tuple[dict[str, float], float]:
    """Evaluate watched rows and reduce them to one signed violation measure.

    Equality rows contribute |value| - EQ_TOL (any deviation is a violation);
    inequality rows contribute value directly (the paper's `<= 0` convention).
    """
    rows: dict[str, float] = {}
    margin = -np.inf

    for expr, labels, is_eq in ((block.eq, block.eq_labels, True), (block.ineq, block.ineq_labels, False)):
        if expr is None:
            continue
        values = np.array(ca.DM(ca.evalf(expr))).ravel()
        for label, value in zip(labels, values):
            # PREFIX match, not equality: Eq. 7b expands each `|.|` into sign cases,
            # so "7b:+p_x" names the eight rows "7b:+p_x[++]" ... "7b:+p_x[--]".
            # A scenario names the quantity it is watching, not every row of it.
            if watched and not any(label.startswith(w) for w in watched):
                continue
            rows[label] = float(value)
            margin = max(margin, abs(value) - EQ_TOL if is_eq else value)

    return rows, (0.0 if margin == -np.inf else margin)


def run_sweep(scenario: Scenario, scene: Scene) -> SweepResult:
    """Evaluate a scenario across its parameter range."""
    steps = []
    for value in scenario.values:
        block, watched, state = scenario.build(scene, float(value))
        rows, margin = _margin(block, watched)
        steps.append(ScenarioStep(param=float(value), rows=rows, margin=margin, state=state))
    return SweepResult(scenario=scenario, steps=steps)


# =============================================================================
# Shared helpers
# =============================================================================
def standing_configuration(scene: Scene) -> np.ndarray:
    """Nominal posture with the soles resting exactly on the floor."""
    q = scene.robot.q_nominal.copy()
    soles = [scene.patches["left_foot_sole"], scene.patches["right_foot_sole"]]
    q[2] += -min(scene.patch_world_placement(p, q).translation[2] for p in soles)
    return q


def _patch_world(scene: Scene, patch, q=None) -> pin.SE3:
    return scene.patch_world_placement(patch, q if patch.attachment is Attachment.ROBOT else None)


def _kinematic_block(scene: Scene, a_name: str, b_name: str, q=None, box_pose=None) -> ConstraintBlock:
    """Eq. 7a/7b between two named patches at the given state."""
    a, b = scene.patches[a_name], scene.patches[b_name]
    if box_pose is not None and a.attachment is Attachment.OBJECT:
        T_a = box_pose * a.placement
    else:
        T_a = _patch_world(scene, a, q)
    if box_pose is not None and b.attachment is Attachment.OBJECT:
        T_b = box_pose * b.placement
    else:
        T_b = _patch_world(scene, b, q)

    return contact_kinematic(
        ca.DM(T_a.rotation), ca.DM(T_a.translation.reshape(3, 1)),
        ca.DM(T_b.rotation), ca.DM(T_b.translation.reshape(3, 1)),
        ca.DM(list(a.half_extents)), ca.DM(list(b.half_extents)),
        name=f"{a_name}->{b_name}",
    )


# =============================================================================
# Eq. 7a -- normal alignment and zero separation
# =============================================================================
def _build_7a_lift(scene: Scene, height: float):
    q = standing_configuration(scene)
    q[2] += height                                    # raise the whole robot
    block = _kinematic_block(scene, "left_foot_sole", "floor", q=q)
    return block, ["7a:p_z"], {"q": q, "patches": ("left_foot_sole", "floor")}


def _build_7a_tilt(scene: Scene, angle: float):
    q = standing_configuration(scene)
    idx = scene.robot.model.joints[scene.robot.model.getJointId("left_ankle_pitch_joint")].idx_q
    q[idx] += angle
    block = _kinematic_block(scene, "left_foot_sole", "floor", q=q)
    return block, ["7a:log3(R)_x", "7a:log3(R)_y"], {"q": q, "patches": ("left_foot_sole", "floor")}


def _build_7a_yaw(scene: Scene, yaw: float):
    """Yaw the whole robot about the vertical. Eq. 7a must NOT care."""
    q = standing_configuration(scene)
    q[3:7] = pin.Quaternion(pin.rpy.rpyToMatrix(0.0, 0.0, yaw)).coeffs()

    # Re-drop onto the floor using the SOLE PATCH, exactly as standing_configuration
    # does. Placing by the ankle frame instead leaves the robot hovering 35 mm up,
    # because the G1's sole sits that far below its ankle.
    soles = [scene.patches["left_foot_sole"], scene.patches["right_foot_sole"]]
    q[2] += -min(scene.patch_world_placement(p, q).translation[2] for p in soles)

    block = _kinematic_block(scene, "left_foot_sole", "floor", q=q)
    return block, ["7a:log3(R)_x", "7a:log3(R)_y", "7a:p_z"], {"q": q, "patches": ("left_foot_sole", "floor")}


# =============================================================================
# Eq. 7b -- containment
# =============================================================================
def _build_7b_slide(scene: Scene, offset: float):
    """Slide the box across the platform until it overhangs the edge."""
    platform = _patch_world(scene, scene.patches["tabletop"])
    box = scene.objects["box"]
    pose = pin.SE3(np.eye(3), platform.translation + np.array([offset, 0.0, box.half_extents[2]]))
    block = _kinematic_block(scene, "box_bottom", "tabletop", box_pose=pose)
    return block, ["7b:+p_x", "7b:-p_x"], {
        "q": standing_configuration(scene), "box_pose": pose,
        "patches": ("box_bottom", "tabletop"),
    }


# The left arm's "presenting" posture: forearm held out in front of the chest with
# the wrist cut face pointing at the box, ready for it to be placed against.
# Hand-picked constants,
# not an IK solution -- only the three WRIST angles are computed, and those in
# closed form (see `_presenting_configuration`). Chosen so the box ends up in front
# of the torso and clear of it, with every joint inside its Eq. 13a limits.
_PRESENT_ARM = {
    "left_shoulder_pitch_joint": -0.9,
    "left_shoulder_roll_joint": 0.6,
    "left_shoulder_yaw_joint": -1.1,
    "left_elbow_joint": 0.7,
}

# Rx(pi): the patch-frame flip that makes two outward normals oppose (see frames.py).
_FLIP_R = np.diag([1.0, -1.0, -1.0])


def _presenting_configuration(scene: Scene, hand_name: str, face_name: str) -> np.ndarray:
    """Standing pose whose hand patch is oriented to meet an UPRIGHT, axis-aligned box.

    The robot stands normally; the box is the movable object and comes to the hand,
    which is the physically sensible framing (and the one Table IV describes: "left
    hand -> box left"). Posing the robot to meet a fixed box instead would demand
    full-body IK, and reads as the robot flying onto the box.

    Only the patch's ORIENTATION has to be exact, and that is solved in closed form.
    The G1's three wrist joints are roll(X), pitch(Y), yaw(Z) and -- checked against
    the URDF -- carry no fixed rotation between them, so the patch's world rotation is

        R_hand(a,b,c) = R0 . Rx(a) Ry(b) Rz(c) . R_patch

    with R0 the chain up to the wrist and R_patch the hand patch's fixed offset.
    Requiring R_hand = R_face . Rx(pi) (the orientation that seats an upright box's
    face flat on the cut) gives Rx(a)Ry(b)Rz(c) = R0^T . R_des . R_patch^T, whose
    XYZ Euler angles come straight out of `matrixToRpy` on the transpose -- Pinocchio
    returns the ZYX convention, and Rx(a)Ry(b)Rz(c) = (Rz(-c)Ry(-b)Rx(-a))^T.

    Exact to machine precision, so Eq. 7a holds identically and Eq. 7b is the only
    thing the sweep can break.
    """
    model = scene.robot.model
    idx = lambda name: model.joints[model.getJointId(name)].idx_q  # noqa: E731

    q = standing_configuration(scene)
    for joint, angle in _PRESENT_ARM.items():
        q[idx(joint)] = angle

    wrist = [idx(f"left_wrist_{axis}_joint") for axis in ("roll", "pitch", "yaw")]
    q[wrist] = 0.0

    # R0: world rotation of the wrist-roll joint frame with the wrist zeroed.
    data = scene.robot.model.createData()
    pin.forwardKinematics(model, data, q)
    R0 = data.oMi[model.getJointId("left_wrist_roll_joint")].rotation

    R_des = scene.patches[face_name].placement.rotation @ _FLIP_R   # box_pose = identity
    M = R0.T @ R_des @ scene.patches[hand_name].placement.rotation.T
    q[wrist] = -pin.rpy.matrixToRpy(M.T)
    return q


def _box_face_on_hand(scene: Scene, q, hand_name: str, face_name: str, d_local) -> pin.SE3:
    """Place the box so `face` lies on the hand patch, displaced in-plane by `d_local`.

    Inverts the Eq. 7a relation rather than satisfying it approximately. The contact
    frames satisfy `T_a = (T_b . flip) . Trans(d)`, so with the patch placement `T_a`
    fixed by forward kinematics the face placement is

        T_b = T_a . Trans(-d) . flip

    and the box pose follows by peeling off the patch's fixed offset. The relative
    rotation is then exactly identity and `p = (d_x, d_y, 0)`, i.e. 7a holds to
    machine precision at every step while `d` sweeps 7b straight through its bound.
    """
    T_a = _patch_world(scene, scene.patches[hand_name], q)
    flip = pin.SE3(_FLIP_R, np.zeros(3))
    shift = pin.SE3(np.eye(3), np.array([-d_local[0], -d_local[1], 0.0]))
    T_b = T_a * shift * flip
    return T_b * scene.patches[face_name].placement.inverse()


def _build_7b_hand_x(scene: Scene, offset: float):
    """Slide the box across the stationary hand patch along the face's local x.

    Negated so the box travels AWAY from the torso: face-local +x points back toward
    the robot here, and a box sliding through the robot's own chest would be a
    confusing picture for a constraint that has nothing to do with self-collision.
    Only |p_x| enters Eq. 7b, so the crossing is unaffected.
    """
    q = _presenting_configuration(scene, "left_hand_patch", "box_left")
    pose = _box_face_on_hand(scene, q, "left_hand_patch", "box_left", (-offset, 0.0))
    block = _kinematic_block(scene, "left_hand_patch", "box_left", q=q, box_pose=pose)
    return block, ["7b:+p_x", "7b:-p_x"], {
        "q": q, "box_pose": pose, "patches": ("left_hand_patch", "box_left"),
    }


def _build_7b_hand_y(scene: Scene, offset: float):
    """Same, but along the face's local y -- where the bound is DIFFERENT."""
    q = _presenting_configuration(scene, "left_hand_patch", "box_left")
    pose = _box_face_on_hand(scene, q, "left_hand_patch", "box_left", (0.0, offset))
    block = _kinematic_block(scene, "left_hand_patch", "box_left", q=q, box_pose=pose)
    return block, ["7b:+p_y", "7b:-p_y"], {
        "q": q, "box_pose": pose, "patches": ("left_hand_patch", "box_left"),
    }


def _build_7b_yaw(scene: Scene, yaw: float):
    """Yaw a box that is already near the edge.

    Rotating enlarges its footprint in the platform's frame (`^b xi_a`), so a box
    that fits when aligned can overhang once turned -- without translating at all.
    """
    platform = _patch_world(scene, scene.patches["tabletop"])
    box = scene.objects["box"]
    offset = 0.12                                    # inside when aligned (bound 0.15)
    pose = pin.SE3(
        pin.rpy.rpyToMatrix(0.0, 0.0, yaw),
        platform.translation + np.array([offset, 0.0, box.half_extents[2]]),
    )
    block = _kinematic_block(scene, "box_bottom", "tabletop", box_pose=pose)
    return block, ["7b:+p_x", "7b:-p_x", "7b:+p_y", "7b:-p_y"], {
        "q": standing_configuration(scene), "box_pose": pose,
        "patches": ("box_bottom", "tabletop"),
    }


# =============================================================================
# Eq. 7c / 7d -- friction and centre of pressure
# =============================================================================
_FZ = 200.0        # normal load [N], roughly half a G1's weight on one foot
_MU = 0.7
_MU_TORSION = 0.05


def _wrench_block(scene: Scene, f, kappa):
    sole = scene.patches["left_foot_sole"]
    return contact_wrench(
        ca.DM(np.asarray(f, float).reshape(3, 1)),
        ca.DM(np.asarray(kappa, float).reshape(3, 1)),
        ca.DM(list(sole.half_extents)),
        mu=_MU, mu_torsional=_MU_TORSION,
        name="left_foot_sole",
    )


def _build_7c_slip(scene: Scene, fx: float):
    q = standing_configuration(scene)
    f, kappa = [fx, 0.0, _FZ], [0.0, 0.0, 0.0]
    block = _wrench_block(scene, f, kappa)
    return block, ["7c:+f_x", "7c:-f_x"], {
        "q": q, "patches": ("left_foot_sole", "floor"),
        "force": f, "moment": kappa, "mu": _MU, "anchor": "left_foot_sole",
    }


def _build_7c_pull(scene: Scene, fz: float):
    q = standing_configuration(scene)
    f, kappa = [0.0, 0.0, fz], [0.0, 0.0, 0.0]
    block = _wrench_block(scene, f, kappa)
    return block, ["7c:f_z>=0"], {
        "q": q, "patches": ("left_foot_sole", "floor"),
        "force": f, "moment": kappa, "mu": _MU, "anchor": "left_foot_sole",
    }


def _build_7d_roll(scene: Scene, kx: float):
    """Moment about x -- ROLL. Bounded by the foot's half-extent in Y (the narrow one)."""
    q = standing_configuration(scene)
    f, kappa = [0.0, 0.0, _FZ], [kx, 0.0, 0.0]
    block = _wrench_block(scene, f, kappa)
    return block, ["7d:+kappa_x", "7d:-kappa_x"], {
        "q": q, "patches": ("left_foot_sole", "floor"),
        "force": f, "moment": kappa, "mu": _MU, "anchor": "left_foot_sole",
    }


def _build_7d_pitch(scene: Scene, ky: float):
    """Moment about y -- PITCH. Bounded by the half-extent in X (the long one)."""
    q = standing_configuration(scene)
    f, kappa = [0.0, 0.0, _FZ], [0.0, ky, 0.0]
    block = _wrench_block(scene, f, kappa)
    return block, ["7d:+kappa_y", "7d:-kappa_y"], {
        "q": q, "patches": ("left_foot_sole", "floor"),
        "force": f, "moment": kappa, "mu": _MU, "anchor": "left_foot_sole",
    }


def _build_7d_torsion(scene: Scene, kz: float):
    q = standing_configuration(scene)
    f, kappa = [0.0, 0.0, _FZ], [0.0, 0.0, kz]
    block = _wrench_block(scene, f, kappa)
    # `mu_r` is what tells the viewer to draw the torsion rings. Only this scenario
    # sweeps kappa_z, and it is the ONLY row with no other visual consequence: the
    # force, the pyramid and the CoP are all unchanged by a moment about the normal.
    return block, ["7d:+kappa_z", "7d:-kappa_z"], {
        "q": q, "patches": ("left_foot_sole", "floor"),
        "force": f, "moment": kappa, "mu": _MU, "mu_r": _MU_TORSION,
        "anchor": "left_foot_sole",
    }


# =============================================================================
# Eq. 8 -- no-slip
# =============================================================================
def _build_8_slide(scene: Scene, shift: float):
    """Translate the robot sideways while the foot is supposed to be sticking."""
    q0 = standing_configuration(scene)
    q1 = q0.copy()
    q1[0] += shift

    sole, floor = scene.patches["left_foot_sole"], scene.patches["floor"]
    T_a0, T_a1 = _patch_world(scene, sole, q0), _patch_world(scene, sole, q1)
    T_b = _patch_world(scene, floor)

    block = no_slip(
        ca.DM(T_a0.rotation), ca.DM(T_a0.translation.reshape(3, 1)),
        ca.DM(T_b.rotation), ca.DM(T_b.translation.reshape(3, 1)),
        ca.DM(T_a1.rotation), ca.DM(T_a1.translation.reshape(3, 1)),
        ca.DM(T_b.rotation), ca.DM(T_b.translation.reshape(3, 1)),
        name="left_foot_sole",
    )
    return block, ["8:dp_x", "8:dp_y"], {"q": q1, "patches": ("left_foot_sole", "floor")}


def _no_slip_block(scene: Scene, a_name: str, b_name: str, q0, q1, pose0=None, pose1=None):
    """Eq. 8 between two named patches at two timesteps, either of which may move."""
    a, b = scene.patches[a_name], scene.patches[b_name]
    A0 = pose0 * a.placement if pose0 is not None and a.attachment is Attachment.OBJECT \
        else _patch_world(scene, a, q0)
    A1 = pose1 * a.placement if pose1 is not None and a.attachment is Attachment.OBJECT \
        else _patch_world(scene, a, q1)
    B0 = pose0 * b.placement if pose0 is not None and b.attachment is Attachment.OBJECT \
        else _patch_world(scene, b, q0)
    B1 = pose1 * b.placement if pose1 is not None and b.attachment is Attachment.OBJECT \
        else _patch_world(scene, b, q1)
    return no_slip(
        ca.DM(A0.rotation), ca.DM(A0.translation.reshape(3, 1)),
        ca.DM(B0.rotation), ca.DM(B0.translation.reshape(3, 1)),
        ca.DM(A1.rotation), ca.DM(A1.translation.reshape(3, 1)),
        ca.DM(B1.rotation), ca.DM(B1.translation.reshape(3, 1)),
        name=f"{a_name}->{b_name}",
    )


def _build_8_spin(scene: Scene, yaw: float):
    """Spin the foot about the contact normal, pivoting on the contact point itself.

    Yawing the base about the world origin would drag the foot sideways too and light
    up `dp_x`/`dp_y`, hiding whether the yaw row works at all. Pivoting about the sole
    keeps dp EXACTLY zero, so `8:dyaw` is the only row that can move.
    """
    q0 = standing_configuration(scene)
    pivot = _patch_world(scene, scene.patches["left_foot_sole"], q0).translation

    T_base = pin.SE3(pin.Quaternion(q0[6], q0[3], q0[4], q0[5]).matrix(), q0[:3].copy())
    spin = pin.SE3(pin.rpy.rpyToMatrix(0.0, 0.0, yaw), np.zeros(3))
    to_pivot = pin.SE3(np.eye(3), np.array([pivot[0], pivot[1], 0.0]))
    T_new = to_pivot * spin * to_pivot.inverse() * T_base

    q1 = q0.copy()
    q1[:3] = T_new.translation
    q1[3:7] = pin.Quaternion(T_new.rotation).coeffs()

    block = _no_slip_block(scene, "left_foot_sole", "floor", q0, q1)
    return block, ["8:dyaw", "8:dp_x", "8:dp_y"], {
        "q": q1, "patches": ("left_foot_sole", "floor"),
    }


def _build_8_lift(scene: Scene, height: float):
    """Raise the robot straight up. Eq. 8 constrains only the IN-PLANE pose."""
    q0 = standing_configuration(scene)
    q1 = q0.copy()
    q1[2] += height
    block = _no_slip_block(scene, "left_foot_sole", "floor", q0, q1)
    return block, ["8:dp_x", "8:dp_y", "8:dyaw"], {
        "q": q1, "patches": ("left_foot_sole", "floor"),
    }


def _build_8_carry(scene: Scene, shift: float):
    """Carry the box: hand and box translate together, so nothing slips.

    Eq. 8 holds the RELATIVE pose fixed, not either patch's world pose. Both patches
    sweep a quarter metre through space and the residual stays at zero.
    """
    q0 = _presenting_configuration(scene, "left_hand_patch", "box_left")
    pose0 = _box_face_on_hand(scene, q0, "left_hand_patch", "box_left", (0.0, 0.0))

    q1 = q0.copy()
    q1[0] += shift
    pose1 = pin.SE3(pose0.rotation, pose0.translation + np.array([shift, 0.0, 0.0]))

    block = _no_slip_block(scene, "left_hand_patch", "box_left", q0, q1, pose0, pose1)
    return block, ["8:dp_x", "8:dp_y", "8:dyaw"], {
        "q": q1, "box_pose": pose1, "patches": ("left_hand_patch", "box_left"),
    }


# =============================================================================
# Eq. 9 -- collision
# =============================================================================
_SLAB = 0.2                 # visual thickness of the platform, for the collision box
_LINEARIZE_AT = 0.10        # gap at which the frozen witness data is queried


def _collision_setup(scene: Scene, gap: float, tilt: float = 0.0):
    """Geometry for the box-vs-platform collision scenarios.

    `tilt` rotates the box about its own y-axis, which is what separates a frozen
    first-order model from the true signed distance (see `_build_9_drift`).
    """
    try:
        import coal
    except ImportError:  # pragma: no cover
        import hppfcl as coal

    box = scene.objects["box"]
    platform = scene.patches["tabletop"]
    T_plat = _patch_world(scene, platform)

    # Box approaches the platform's +x side face; `gap` is the face-to-face clearance.
    pos = T_plat.translation + np.array(
        [platform.half_extents[0] + box.half_extents[0] + gap, 0.0, -0.1]
    )
    plat_center = T_plat.translation - np.array([0.0, 0.0, _SLAB / 2.0])

    R_box = pin.utils.rpyToMatrix(0.0, tilt, 0.0)
    geom_box = coal.Box(*box.size)
    geom_plat = coal.Box(2 * platform.half_extents[0], 2 * platform.half_extents[1], _SLAB)
    tf_box, tf_plat = coal.Transform3s(), coal.Transform3s()
    tf_box.setTranslation(pos)
    tf_box.setRotation(R_box)
    tf_plat.setTranslation(plat_center)
    return pos, plat_center, geom_box, geom_plat, tf_box, tf_plat, R_box


def _build_9_penetrate(scene: Scene, gap: float):
    """Drive the box into the platform, with the witness data FROZEN.

    This is how Eq. 9 actually behaves inside an NLP: the GJK query runs once at the
    linearization point (here a 0.10 m gap), its witness points and normal become
    constants, and the resulting LINEAR expression is what the solver sees. Because
    it is linear, it happily reports a negative signed distance -- which is exactly
    what lets the solver feel and push back against penetration.

    Contrast with `9-drift`, which shows where this model stops being exact.
    """
    pos, plat_center, geom_box, geom_plat, tf_box, tf_plat, _ = _collision_setup(scene, gap)
    *_, ref_tf_box, ref_tf_plat, _ = _collision_setup(scene, _LINEARIZE_AT)

    witness = query_witness(geom_box, ref_tf_box, geom_plat, ref_tf_plat)
    block = collision_avoidance(
        ca.DM(np.eye(3)), ca.DM(pos.reshape(3, 1)),
        ca.DM(np.eye(3)), ca.DM(plat_center.reshape(3, 1)),
        witness, margin=0.0, name="box/platform",
    )
    return block, ["9:sd>=margin"], {
        "q": standing_configuration(scene),
        "box_pose": pin.SE3(np.eye(3), pos), "patches": (),
        "highlight_objects": ("box",), "status_at": pos,
    }


_DRIFT_GAP = 0.02           # clearance held while the box is tilted


def _build_9_drift(scene: Scene, tilt: float):
    """Hold the clearance fixed and TILT the box, with the witness data frozen at 0 deg.

    This is the scenario that shows what "approximation" in Eq. 9 actually costs, and
    it replaces an earlier one built on a false premise. That one claimed coal's
    `distance()` is unsigned and therefore cannot report penetration, so a refreshed
    query could never fail. It is not unsigned -- `DistanceRequest.enable_signed_distance`
    defaults to True and coal returns the negative distance correctly. The old scenario
    only appeared to confirm the claim because `query_witness` built its normal from
    `w_A - w_B`, which REVERSES under overlap; the sign error and the false explanation
    propped each other up. Both are fixed.

    The real property is this one. Eq. 9 is exact for the material points it froze and
    drifts once the true closest-point pair changes:

      * under pure TRANSLATION of a flat face, the closest pair never changes, so the
        frozen model is exact everywhere -- which is why `9-penetrate` sweeps a gap and
        tracks the truth perfectly;
      * under ROTATION the contact migrates to a different corner, the frozen pair is
        no longer the closest one, and the model reports clearance the box does not have.

    That gap IS the linearization error, and it is the whole argument for refreshing
    witness data between solver iterations (ambiguity #12).
    """
    pos, plat_center, geom_box, geom_plat, tf_box, tf_plat, R_box = _collision_setup(
        scene, _DRIFT_GAP, tilt=tilt
    )
    *_, ref_tf_box, ref_tf_plat, _ = _collision_setup(scene, _DRIFT_GAP, tilt=0.0)

    witness = query_witness(geom_box, ref_tf_box, geom_plat, ref_tf_plat)
    block = collision_avoidance(
        ca.DM(R_box), ca.DM(pos.reshape(3, 1)),
        ca.DM(np.eye(3)), ca.DM(plat_center.reshape(3, 1)),
        witness, margin=0.0, name="box/platform",
    )
    truth = query_witness(geom_box, tf_box, geom_plat, tf_plat).distance
    return block, ["9:sd>=margin"], {
        "q": standing_configuration(scene),
        "box_pose": pin.SE3(R_box, pos), "patches": (),
        "true_signed_distance": truth,
        # Deliberately tinted by what EQ. 9 believes, not by the truth -- the whole
        # point of this scenario is that the box stays green while it is buried.
        "highlight_objects": ("box",), "status_at": pos,
    }


# =============================================================================
# Eq. 10 / 11 -- dynamics.
#
# These need a VELOCITY and an ACCELERATION, which a static configuration does not
# have. Rather than skip them, each scenario pins the robot or object in static
# equilibrium (V = 0, Vdot = 0) and sweeps the one quantity that must then balance:
# the contact wrench. That is the whole content of Eq. 10b's first line and Eq. 11b
# at rest, and it is checkable by hand against m*g.
# =============================================================================
def robot_weight(scene: Scene) -> float:
    """m*g for the loaded robot -- a URDF number, not a paper number."""
    return total_mass(scene.robot.model) * scene.gravity


def _build_10_weight(scene: Scene, f_z: float):
    """Stand on BOTH feet and share the load. hdot must stay zero.

    Two end-effectors, not one, so the sum over e in Eq. 10b is actually exercised --
    a single contact cannot tell `sum_e f_e` from `f_1`. The contact points are the
    real sole patches from the scene, so the geometry is the robot's, not invented.
    """
    q = standing_configuration(scene)
    model = scene.robot.model
    mass = total_mass(model)
    com = np.asarray(pin.centerOfMass(model, scene.robot.data, q)).ravel()

    soles = ["left_foot_sole", "right_foot_sole"]
    points = [_patch_world(scene, scene.patches[n], q).translation for n in soles]
    half = ca.DM([0.0, 0.0, f_z / 2.0])

    hdot = centroidal_momentum_rate(
        [half, half], [ca.DM.zeros(3, 1)] * 2,
        [ca.DM(p.reshape(3, 1)) for p in points],
        ca.DM(com.reshape(3, 1)), mass, scene.gravity,
    )
    block = ConstraintBlock(
        name="10:hdot", eq=ca.vertcat(hdot[2]), eq_labels=["10:hdot_lin_z"],
    )
    # Draw both sides of Eq. 10b's linear row: the two contact forces pushing up and
    # the weight pulling down. They balance exactly at the crossing.
    vectors = [
        {"origin": p, "vector": [0.0, 0.0, f_z / 2.0], "path": f"overlay/f{i}"}
        for i, p in enumerate(points)
    ]
    vectors.append({"origin": com, "vector": [0.0, 0.0, -mass * scene.gravity],
                    "color": "weight", "path": "overlay/mg"})
    return block, ["10:hdot_lin_z"], {
        "q": q, "patches": tuple(soles), "vectors": vectors, "com": com,
    }


def _build_10_moment(scene: Scene, offset: float):
    """Slide the support sideways under a robot that is already carrying its weight.

    The force still balances gravity exactly, so the LINEAR row stays at zero -- but
    the moment arm (p - c) grows and the ANGULAR row does not. This is why the centre
    of pressure has to sit under the centre of mass.
    """
    q = standing_configuration(scene)
    mass = total_mass(scene.robot.model)
    com = np.asarray(pin.centerOfMass(scene.robot.model, scene.robot.data, q)).ravel()
    contact = np.array([com[0] + offset, com[1], 0.0])

    hdot = centroidal_momentum_rate(
        [ca.DM([0.0, 0.0, mass * scene.gravity])], [ca.DM.zeros(3, 1)],
        [ca.DM(contact.reshape(3, 1))], ca.DM(com.reshape(3, 1)), mass, scene.gravity,
    )
    block = ConstraintBlock(
        name="10:hdot", eq=ca.vertcat(hdot[2], hdot[4]),
        eq_labels=["10:hdot_lin_z", "10:hdot_ang_y"],
    )
    # The support still cancels the weight, so the two vertical arrows stay equal --
    # what moves is the arrow's BASE, sliding out from under the CoM marker. The
    # purple arrow is the moment that opens up as it does.
    moment = np.cross(contact - com, np.array([0.0, 0.0, mass * scene.gravity]))
    return block, ["10:hdot_ang_y"], {
        "q": q, "patches": (), "com": com, "status_at": contact,
        "vectors": [
            {"origin": contact, "vector": [0.0, 0.0, mass * scene.gravity],
             "path": "overlay/support"},
            {"origin": com, "vector": [0.0, 0.0, -mass * scene.gravity],
             "color": "weight", "path": "overlay/mg"},
            {"origin": com, "vector": moment, "color": "moment", "scale": 0.02,
             "path": "overlay/moment"},
        ],
    }


# Wrenches in Eq. 11 are ANGULAR-FIRST: indices 0..2 are the moment, 3..5 the force.
# `spatial_inertia` puts the inertia tensor in G[0:3,0:3] and the mass in G[3:6,3:6],
# and `gravity_wrench` returns (0, 0, 0, f_x, f_y, f_z). Writing a vertical force at
# index 2 puts it in the MOMENT-about-z slot -- a mistake that survives any test in
# which the equation does no work.
_W_FORCE_Z = 5
_W_MOMENT_X = 0


def _build_11_hold(scene: Scene, f_z: float):
    """Hold the box against gravity. At rest Eq. 11b reduces to W_ext = 0.

    The weight comes from `gravity_wrench` rather than being written by hand, so the
    scenario exercises that function AND the angular-first ordering: put the support
    in the wrong slot and the residual no longer cancels.
    """
    box = scene.objects["box"]
    G = spatial_inertia(box.mass, box.inertia)

    support = ca.SX.zeros(6, 1)
    support[_W_FORCE_Z] = f_z
    W_ext = gravity_wrench(box.mass, ca.DM.eye(3), scene.gravity) + support

    block = object_newton_euler(ca.SX.zeros(6, 1), ca.SX.zeros(6, 1), W_ext, ca.DM(G))
    pose = pin.SE3(np.eye(3), np.array([0.45, 0.0, 0.45]))
    centre = pose.translation

    # THREE arrows, not two, and side by side rather than stacked on one line.
    #
    # The two input arrows are drawn offset in y so their LENGTHS can be compared --
    # superimposed on the same axis they just overlap, which is why this scenario read
    # as "two arrows growing from the box". The third, at the centre, is the residual
    # `W_ext` itself: the one quantity Eq. 11b actually constrains, and the only one
    # whose value the verdict depends on. It shrinks to nothing at f_z = mg and grows
    # again past it -- a box pushed too hard does not stay still, it accelerates up.
    weight = box.mass * scene.gravity
    residual = f_z - weight
    scale = 0.02          # a tenth of the robot scale: 19.6 N here vs the robot's 324 N
    return block, [f"11b:W_ext[{_W_FORCE_Z}]"], {
        "q": standing_configuration(scene), "box_pose": pose, "patches": (),
        "highlight_objects": ("box",), "status_at": centre + np.array([0.0, 0.0, 0.28]),
        "vectors": [
            {"origin": centre + np.array([0.0, -0.22, 0.0]), "vector": [0.0, 0.0, f_z],
             "scale": scale, "path": "overlay/support"},
            {"origin": centre + np.array([0.0, 0.22, 0.0]),
             "vector": [0.0, 0.0, -weight],
             "color": "weight", "scale": scale, "path": "overlay/mg"},
            {"origin": centre, "vector": [0.0, 0.0, residual], "color": "moment",
             "scale": scale, "path": "overlay/residual"},
        ],
    }


# An elongated probe body. The scene's box is a CUBE, so its inertia is isotropic and
# `omega x (I omega) = I (omega x omega) = 0` for every omega -- a cube can never
# exhibit the [ad_V]^T G V term at all. Demonstrating it needs an anisotropic body.
#
# The inertia is DERIVED from the body's dimensions rather than typed in, because a
# hand-picked diagonal is easy to make non-physical. The previous value here,
# diag(0.10, 0.50, 0.90), violated the triangle inequality I_x + I_y >= I_z
# (0.60 < 0.90) and so is not the inertia of any rigid body -- the demo was computing
# a real formula on an impossible object. Deriving it from a box makes that
# unrepresentable, and lets the same dimensions be DRAWN.
_SPIN_SIZE = np.array([0.60, 0.20, 0.15])   # long in x: clearly not a cube on screen
_SPIN_MASS = 2.0
_SPIN_INERTIA = np.diag(
    _SPIN_MASS / 12.0 * np.array([
        _SPIN_SIZE[1] ** 2 + _SPIN_SIZE[2] ** 2,
        _SPIN_SIZE[0] ** 2 + _SPIN_SIZE[2] ** 2,
        _SPIN_SIZE[0] ** 2 + _SPIN_SIZE[1] ** 2,
    ])
)
_SPIN_AXIS = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)   # non-principal on purpose
_SPIN_POSE = np.array([0.45, 0.0, 0.60])
_SPIN_OMEGA_MAX = 4.0       # end of the sweep, rad/s
_SPIN_STEPS = 81
_SPIN_DURATION = (_SPIN_STEPS - 1) * _FRAME_DT


def _build_11_spin(scene: Scene, omega: float):
    """Tumble an ELONGATED object at constant twist with no external wrench.

    Eq. 11b then reads `0 = -[ad_V]^T G V`, and the gyroscopic term says that is
    impossible: an anisotropic body spinning about a non-principal axis needs a
    moment just to keep spinning at a constant rate. The residual is the textbook
    Euler term `omega x (I omega)` and grows as omega^2.
    """
    G = spatial_inertia(_SPIN_MASS, _SPIN_INERTIA)
    V = ca.DM(np.concatenate([omega * _SPIN_AXIS, np.zeros(3)]).reshape(6, 1))

    block = object_newton_euler(V, ca.SX.zeros(6, 1), ca.SX.zeros(6, 1), ca.DM(G))

    # The body ACTUALLY TURNS, at the rate under test. As with the knee in 13b-speed,
    # a scenario about a velocity has to contain one: the rate ramps linearly across
    # the sweep, so integrating it gives the angle reached by this frame and the
    # frame-to-frame rotation is the swept omega. Without this the body sat still and
    # only the arrows moved, which is what made this scenario unreadable.
    angle = 0.5 * omega ** 2 * _SPIN_DURATION / _SPIN_OMEGA_MAX
    pose = pin.SE3(pin.exp3(_SPIN_AXIS * angle), _SPIN_POSE)
    centre = pose.translation

    # Yellow = the spin itself. Purple = the moment Eq. 11b says you must supply just
    # to hold that spin steady. At omega = 0 the purple arrow vanishes; it then grows
    # as omega^2 while the yellow one grows linearly -- so the purple one overtakes it.
    gyroscopic = np.cross(omega * _SPIN_AXIS, _SPIN_INERTIA @ (omega * _SPIN_AXIS))
    return block, [f"11b:W_ext[{_W_MOMENT_X}]"], {
        "q": standing_configuration(scene), "patches": (),
        "probe": {"pose": pose, "size": _SPIN_SIZE},
        "status_at": centre + np.array([0.0, 0.0, 0.32]),
        "vectors": [
            {"origin": centre, "vector": omega * _SPIN_AXIS, "scale": 0.12,
             "path": "overlay/omega"},
            {"origin": centre, "vector": gyroscopic, "color": "moment", "scale": 0.85,
             "path": "overlay/gyro"},
        ],
    }


# =============================================================================
# Eq. 13b / 13c -- the limits that need velocities and torques
# =============================================================================
def _joint_axis_world(scene: Scene, joint: str, q) -> tuple[np.ndarray, np.ndarray]:
    """World origin and rotation axis of a revolute joint, for drawing.

    The rate and the torque of Eqs. 13b/13c have no contact patch to hang off, so
    they are drawn as arrows along the joint's own axis. They ANNOTATE the motion
    built by `_knee_at_speed`; they are not a substitute for it.
    """
    model = scene.robot.model
    data = model.createData()
    pin.forwardKinematics(model, data, np.asarray(q))
    jid = model.getJointId(joint)
    placement = data.oMi[jid]
    axes = {"JointModelRX": [1.0, 0.0, 0.0],
            "JointModelRY": [0.0, 1.0, 0.0],
            "JointModelRZ": [0.0, 0.0, 1.0]}
    shortname = model.joints[jid].shortname()
    if shortname not in axes:
        raise ValueError(f"{joint!r} is a {shortname}; only RX/RY/RZ have a single axis")
    return placement.translation.copy(), placement.rotation @ np.asarray(axes[shortname])


def knee_velocity_limit(scene: Scene) -> float:
    model = scene.robot.model
    return float(model.velocityLimit[model.joints[model.getJointId("left_knee_joint")].idx_v])


def knee_torque_limit(scene: Scene) -> float:
    model = scene.robot.model
    return float(model.effortLimit[model.joints[model.getJointId("left_knee_joint")].idx_v])


# The sweep is a RAMP IN TIME, not a list of unrelated numbers: the knee starts at
# rest and its rate climbs linearly to `speed_max` over `_SPEED_STEPS` frames played
# `_FRAME_DT` apart. Integrating that ramp gives a knee angle for every swept speed,
# so the joint on screen is genuinely turning at the rate under test.
#
# Why this matters: an earlier version held the knee at a fixed 0.42 rad and merely
# drew an arrow whose length was `speed`. Nothing had a velocity, so a sign error or
# a wrong joint index in Eq. 13b would have looked identical. The check that keeps
# that from coming back is `test_velocity_scenarios_actually_move_the_joint`, which
# finite-differences the DISPLAYED angles and demands they reproduce the swept rate.
_SPEED_STEPS = 121          # sweep length; resolve_predictions builds `values` to match
_SPEED_MARGIN = 1.2         # 13b sweeps 20% past v_max so the crossing is interior

# SLOW MOTION. The G1's knee is rated at 20 rad/s and its range is only 2.97 rad, so
# at full speed it crosses the whole range in four frames -- on screen that is not
# "fast", it is a flicker with no readable direction. Playing the motion at 1/6 speed
# keeps roughly twenty frames per traverse, which reads as a fast swing.
#
# This changes the PLAYBACK RATE, never the physics: the constraint is still evaluated
# at the true swept velocity, and the crossing still lands on the URDF's v_max. The
# frame-to-frame angle is `v * dt / _SLOWDOWN`, so it stays exactly proportional to
# the rate under test -- which is what `test_velocity_scenarios_actually_move_the_joint`
# checks. The banner in `expect` tells the viewer it is slowed, so the speed on screen
# is never mistaken for the number in the terminal.
_SLOWDOWN = 6.0


def _fold_into(angle: float, lo: float, hi: float) -> float:
    """Reflect `angle` back and forth between `lo` and `hi` (a triangle wave).

    The knee cannot ramp forever -- at 20 rad/s it would leave its 2.97 rad range in
    four frames -- so it swings. Reflection is exact: the map is piecewise linear with
    slope +-1, so |d(folded)/d(angle)| = 1 and the joint's SPEED is untouched. Only the
    sign flips at each turn, and Eq. 13b's limits are symmetric, so the margin still
    depends on |v| alone and the sweep stays monotone.

    A side effect worth watching in the viewer: the knee never leaves [lo, hi], so
    Eq. 13a holds for the entire sweep while 13b is being violated. The two limits are
    genuinely independent.
    """
    span = hi - lo
    phase = (angle - lo) % (2.0 * span)
    return lo + (phase if phase <= span else 2.0 * span - phase)


def _knee_at_speed(scene: Scene, speed: float, speed_max: float) -> tuple[np.ndarray, int]:
    """Standing pose with the left knee moved to where a rate ramp puts it.

    With v(t) = speed_max * t / T, the angle is the integral theta(t) = v^2 T / (2 v_max),
    rewritten here in terms of the swept `speed` so the builder stays a pure function of
    its parameter. Consecutive sweep steps then differ by exactly `dt * (v_k + v_k+1)/2`
    -- the trapezoid rule, which is EXACT for a linear ramp -- so finite-differencing
    the displayed frames returns the swept velocity itself.
    """
    q = standing_configuration(scene)
    model = scene.robot.model
    idx_q = model.joints[model.getJointId("left_knee_joint")].idx_q
    lo, hi = (limit[idx_q] for limit in scene.robot.joint_limits())

    duration = (_SPEED_STEPS - 1) * _FRAME_DT / _SLOWDOWN
    travelled = 0.5 * speed**2 * duration / speed_max
    q[idx_q] = _fold_into(q[idx_q] + travelled, float(lo), float(hi))
    return q, idx_q


def _build_13b_speed(scene: Scene, speed: float):
    """Swing the knee faster and faster until it outruns its URDF velocity limit."""
    model = scene.robot.model
    q, _ = _knee_at_speed(scene, speed, knee_velocity_limit(scene) * _SPEED_MARGIN)
    v = np.zeros(model.nv)
    idx = model.joints[model.getJointId("left_knee_joint")].idx_v
    v[idx] = speed

    actuated = actuated_slice(model, velocity=True)
    v_max = model.velocityLimit[actuated]
    block = joint_velocity_limits(
        ca.DM(v[actuated].reshape(-1, 1)),
        ca.DM(-v_max.reshape(-1, 1)),
        ca.DM(v_max.reshape(-1, 1)),
    )
    # Watch only the knee's own two rows; every other joint sits at zero velocity.
    knee = idx - actuated.start

    # The knee is now genuinely turning at `speed`; these two arrows just put a number
    # on it, since the eye cannot read rad/s off a swinging link. Yellow past blue is
    # the violation, and it happens on the frame where the swing visibly outruns the
    # joint's rated speed.
    origin, axis = _joint_axis_world(scene, "left_knee_joint", q)
    limit = float(model.velocityLimit[idx])
    return block, [f"13b:v<=max[{knee}]", f"13b:v>=min[{knee}]"], {
        "q": q, "patches": (), "status_at": origin,
        "vectors": [
            {"origin": origin, "vector": axis * limit, "color": "capacity",
             "scale": 0.02, "path": "overlay/v_max"},
            {"origin": origin, "vector": axis * speed, "scale": 0.02,
             "path": "overlay/v"},
        ],
    }


_TORQUE_FRACTION = 0.5      # the demand held while the joint speeds up


def _build_13c_envelope(scene: Scene, speed: float):
    """Hold a constant torque demand and speed the joint up until the motor runs out.

    Eq. 13c is an ENVELOPE, not a box: torque and speed trade off linearly. At half
    the peak torque the joint may only reach half its no-load speed.
    """
    tau_max, v_max = knee_torque_limit(scene), knee_velocity_limit(scene)
    q, _ = _knee_at_speed(scene, speed, v_max)
    demand = _TORQUE_FRACTION * tau_max
    block = torque_speed_limits(
        ca.DM([[demand]]), ca.DM([[speed]]), tau_max=[tau_max], v_tau_max=[v_max],
    )

    # The whole point of an envelope: the DEMAND is constant while the AVAILABLE
    # torque shrinks affinely with speed. The blue arrow shortens past the yellow one
    # exactly at the crossing -- a box constraint would leave blue at full length.
    origin, axis = _joint_axis_world(scene, "left_knee_joint", q)
    available = tau_max * (1.0 - abs(speed) / v_max)
    return block, list(block.ineq_labels), {
        "q": q, "patches": (), "status_at": origin,
        "vectors": [
            {"origin": origin, "vector": axis * available, "color": "capacity",
             "scale": 0.004, "path": "overlay/tau_available"},
            {"origin": origin, "vector": axis * demand, "scale": 0.004,
             "path": "overlay/tau_demand"},
        ],
    }


# =============================================================================
# Eq. 13a -- joint limits
# =============================================================================
def _build_13a_knee(scene: Scene, angle: float):
    robot = scene.robot
    q = standing_configuration(scene)
    idx = robot.model.joints[robot.model.getJointId("left_knee_joint")].idx_q
    q[idx] = angle

    lower, upper = robot.joint_limits()
    sl = actuated_slice(robot.model)
    block = joint_position_limits(ca.DM(q[sl].reshape(-1, 1)), lower[sl], upper[sl], name="joints")

    watched = [lab for lab in block.ineq_labels if lab.endswith(f"[{idx - 7}]")]
    origin, _ = _joint_axis_world(scene, "left_knee_joint", q)
    return block, watched, {"q": q, "patches": (), "status_at": origin}


def knee_upper_limit(scene: Scene) -> float:
    """The G1's left knee upper limit, read from the URDF -- Eq. 13a's q_max."""
    robot = scene.robot
    idx = robot.model.joints[robot.model.getJointId("left_knee_joint")].idx_q
    return float(robot.joint_limits()[1][idx])


# =============================================================================
# The catalogue
# =============================================================================
ALL_SCENARIOS: list[Scenario] = [
    Scenario(
        key="7a-lift",
        equation="7a",
        title="Lift the foot off the floor",
        question="What does 'p_z = 0' (zero normal separation) actually forbid?",
        expect="The foot leaves the floor and 7a:p_z grows exactly as fast as the lift. "
               "ANY separation is a violation -- contact is an equality, not a tolerance.",
        param_label="lift height [m]",
        values=np.linspace(0.0, 0.06, 61),
        predicted_crossing=0.0,
        prediction_note="Eq. 7a sets p_z = 0 exactly, so any lift > 0 violates it.",
        build=_build_7a_lift,
        tolerance=2e-3,
    ),
    Scenario(
        key="7a-tilt",
        equation="7a",
        title="Tilt the foot with the ankle pitch",
        question="What does 'log3(R)_{x,y} = 0' (normal alignment) forbid?",
        expect="The sole rotates out of the floor plane and log3(R)_y grows with the "
               "ankle angle. Heel-strike and toe-off are NOT flat contacts.",
        param_label="extra ankle pitch [rad]",
        values=np.linspace(0.0, 0.30, 61),
        predicted_crossing=0.0,
        prediction_note="Eq. 7a sets log3(R)_{x,y} = 0 exactly; any tilt violates it.",
        build=_build_7a_tilt,
        tolerance=1.5e-2,
    ),
    Scenario(
        key="7a-yaw",
        equation="7a",
        title="Yaw the whole robot on the spot  [NEVER VIOLATES]",
        question="The paper says 7a leaves 'the relative in-plane position and yaw "
                 "angle free'. Is that true?",
        expect="NOTHING happens to any 7a row, all the way to a full half-turn. "
               "A foot flat on the floor may point in any direction -- which is exactly "
               "the freedom the robot needs in order to turn while walking.",
        param_label="base yaw [rad]",
        values=np.linspace(0.0, np.pi, 46),
        predicted_crossing=None,
        prediction_note="Eq. 7a constrains only log3(R)_x, log3(R)_y and p_z; yaw is free.",
        build=_build_7a_yaw,
    ),
    Scenario(
        key="7b-slide",
        equation="7b",
        title="Slide the box off the edge of the platform",
        question="Is Eq. 7b an overlap test or a containment test?",
        expect="The constraint fails once the box's FOOTPRINT leaves the platform, not "
               "when their centres separate. Crossing = platform half-extent minus box "
               "half-extent, i.e. containment.",
        param_label="box offset along +x [m]",
        values=np.linspace(0.0, 0.30, 121),
        predicted_crossing=0.15,
        prediction_note="Eq. 7b: |p_x| <= (^b xi_b)_x - (^b xi_a)_x = 0.30 - 0.15 = 0.15 m.",
        build=_build_7b_slide,
        tolerance=3e-3,
    ),
    Scenario(
        key="7b-hand-x",
        equation="7b",
        title="Slide the box across the robot's wrist cut face (face-local x)",
        question="Does Eq. 7b apply to end-effector patches too, or only object/environment ones?",
        expect="It applies to all of them. The paper: 'This includes contact between "
               "END-EFFECTOR PATCHES, patches on movable objects, and static environment "
               "patches.' The robot stands still with its forearm out; the BOX slides "
               "across the wrist cut face and loses support at 0.15 - 0.0269 = 0.1231 m. "
               "Unlike the box-on-platform sweeps, patch a here is attached to a robot "
               "link, so its placement comes from FORWARD KINEMATICS, not an object pose.",
        param_label="box offset along face x [m]",
        # 121 steps, like the other 7b sweeps: ~5 s of animation. Denser sampling buys
        # no accuracy here -- the 7b margin is LINEAR in the offset, so the crossing is
        # found by interpolation and comes out exact (2.8e-17) even at 61 steps.
        values=np.linspace(0.0, 0.20, 121),
        predicted_crossing=0.1231,
        prediction_note="Eq. 7b: |p_x| <= (^b xi_b)_x - (^b xi_a)_x = 0.15 - 0.0269 = 0.1231 m.",
        build=_build_7b_hand_x,
        tolerance=1e-3,
    ),
    Scenario(
        key="7b-hand-y",
        equation="7b",
        title="Slide the box across the wrist cut face the OTHER way  [DIFFERENT bound]",
        question="Is Eq. 7b's bound really evaluated per axis?",
        expect="Same hand, same face, same contact -- but sliding the box VERTICALLY it "
               "fails EARLIER, at 0.1199 m versus 0.1231 m sliding sideways, because the "
               "wrist cut is 0.0269 x 0.0301: it is TALLER than it is wide, so the taller "
               "axis runs out of room first. A single scalar 'patch radius' could not "
               "reproduce this, and the box-on-platform sweeps cannot show it because "
               "those patches are square.",
        param_label="box offset along face y [m]",
        values=np.linspace(0.0, 0.20, 121),
        predicted_crossing=0.1199,
        prediction_note="Eq. 7b: |p_y| <= (^b xi_b)_y - (^b xi_a)_y = 0.15 - 0.0301 = 0.1199 m.",
        build=_build_7b_hand_y,
        tolerance=1e-3,
    ),
    Scenario(
        key="7b-yaw",
        equation="7b",
        title="Yaw a box that already sits near the edge",
        question="Why does Eq. 7b need '^b xi_a' -- a's half-extents expressed in b's frame?",
        expect="The box NEVER MOVES, yet the constraint fails. Rotating enlarges its "
               "axis-aligned footprint on the platform, so the admissible region shrinks "
               "underneath it. Using a's raw half-extents would miss this entirely.",
        param_label="box yaw [rad]",
        values=np.linspace(0.0, 0.6, 121),
        predicted_crossing=0.2278,
        prediction_note=(
            "Eq. 7b at offset 0.12: need 0.30 - 0.15(|cos t| + |sin t|) >= 0.12, "
            "i.e. |cos t| + |sin t| <= 1.2, i.e. sqrt(2) sin(t + pi/4) <= 1.2. "
            "First failure at t = asin(1.2/sqrt2) - pi/4 = 0.2278 rad (13.05 deg)."
        ),
        build=_build_7b_yaw,
        tolerance=1e-2,
    ),
    Scenario(
        key="7c-slip",
        equation="7c",
        title="Push the foot sideways until it slips",
        question="Where is the edge of the friction pyramid?",
        expect=f"Slip begins at f_x = mu*f_z = {_MU} * {_FZ:.0f} = {_MU * _FZ:.0f} N. "
               "Note this is the PYRAMID: each component is bounded separately, so the "
               "diagonal corner (f_x = f_y = mu f_z) is admissible even though it lies "
               "outside the quadratic cone.",
        param_label="tangential force f_x [N]",
        values=np.linspace(0.0, 200.0, 201),
        predicted_crossing=_MU * _FZ,
        prediction_note=f"Eq. 7c: |f_x| <= mu f_z = {_MU} * {_FZ:.0f} = {_MU * _FZ:.0f} N.",
        build=_build_7c_slip,
        tolerance=2.0,
    ),
    Scenario(
        key="7c-pull",
        equation="7c",
        title="Try to pull the foot off the floor with the contact force",
        question="What does unilaterality (f_z >= 0) mean physically?",
        expect="As soon as the normal force turns negative the constraint fails: a "
               "contact can PUSH but never PULL. The floor is not glue. This single "
               "row is what forces the robot to lift a foot rather than hang from it.",
        param_label="normal force f_z [N]",
        values=np.linspace(100.0, -50.0, 151),
        predicted_crossing=0.0,
        prediction_note="Eq. 7c: f_z >= 0.",
        build=_build_7c_pull,
        tolerance=1.5,
    ),
    Scenario(
        key="7d-roll",
        equation="7d",
        title="Roll moment on the foot (about x, the NARROW direction)",
        question="Why does kappa_x use the half-extent in Y?",
        expect=f"The foot tips sideways at kappa_x = f_z * xi_y = {_FZ:.0f} * 0.03 = "
               f"{_FZ * 0.03:.1f} Nm. A roll moment is resisted by the foot's WIDTH, "
               "which is the small dimension -- compare with 7d-pitch.",
        param_label="roll moment kappa_x [Nm]",
        values=np.linspace(0.0, 12.0, 121),
        predicted_crossing=_FZ * 0.03,
        prediction_note="Eq. 7d: |kappa_x| <= f_z (^a xi_a)_y = 200 * 0.03 = 6.0 Nm.",
        build=_build_7d_roll,
        tolerance=0.2,
    ),
    Scenario(
        key="7d-pitch",
        equation="7d",
        title="Pitch moment on the foot (about y, the LONG direction)",
        question="Is the centre-of-pressure bound really asymmetric?",
        expect=f"The foot tips forward only at kappa_y = f_z * xi_x = {_FZ:.0f} * 0.085 = "
               f"{_FZ * 0.085:.1f} Nm -- about 2.8x the roll bound. THIS is why the x<->y "
               "swap in Eq. 7d matters: a foot is long front-to-back, so it resists "
               "pitching far better than rolling. Swapping the subscripts would silently "
               "make the robot tip over sideways in simulation.",
        param_label="pitch moment kappa_y [Nm]",
        values=np.linspace(0.0, 30.0, 121),
        predicted_crossing=_FZ * 0.085,
        prediction_note="Eq. 7d: |kappa_y| <= f_z (^a xi_a)_x = 200 * 0.085 = 17.0 Nm.",
        build=_build_7d_pitch,
        tolerance=0.4,
    ),
    Scenario(
        key="7d-torsion",
        equation="7d",
        title="Twist the foot about the contact normal",
        question="What limits spinning in place?",
        expect=f"Torsional friction caps kappa_z at mu_r * f_z = {_MU_TORSION} * {_FZ:.0f} "
               f"= {_MU_TORSION * _FZ:.0f} Nm -- much weaker than the linear friction "
               "bound, which is why pivoting on one foot is easy. WATCH THE CIRCLES, "
               "not the foot: a moment about the normal leaves the force, the friction "
               "pyramid and the centre of pressure ALL unchanged, so nothing else in "
               "the scene can move. The yellow arc is the applied twist and the blue "
               "ring is the capacity; the arc grows through the ring at the crossing.",
        param_label="torsional moment kappa_z [Nm]",
        values=np.linspace(0.0, 25.0, 126),
        predicted_crossing=_MU_TORSION * _FZ,
        prediction_note=f"Eq. 7d: |kappa_z| <= mu_r f_z = {_MU_TORSION} * {_FZ:.0f} = "
                        f"{_MU_TORSION * _FZ:.0f} Nm.",
        build=_build_7d_torsion,
        tolerance=0.4,
    ),
    Scenario(
        key="8-slip",
        equation="8",
        title="Translate the robot while a foot is supposed to be sticking",
        question="What does the no-slip condition add on top of Eq. 7a?",
        expect="Eq. 7a was perfectly happy for the foot to sit ANYWHERE on the floor. "
               "Eq. 8 says it may not MOVE between timesteps once established. The "
               "residual equals the slide distance exactly -- 1 cm of drift is a 1 cm "
               "violation.",
        param_label="body translation [m]",
        values=np.linspace(0.0, 0.10, 101),
        predicted_crossing=0.0,
        prediction_note="Eq. 8: (p^{s+1} - p^s)_{x,y} = 0 exactly.",
        build=_build_8_slide,
        tolerance=2e-3,
    ),
    Scenario(
        key="8-spin",
        equation="8",
        title="Spin the foot in place about the contact normal",
        question="Eq. 8 has TWO rows. Does the second one -- the yaw row -- work?",
        expect="The foot pivots on its own contact point, so it never TRANSLATES: "
               "dp_x and dp_y stay at zero to machine precision and only `8:dyaw` "
               "moves. Sliding alone can never test this row, because a sliding foot "
               "lights up dp first. Eq. 8 pins all three in-plane DoF -- exactly the "
               "three that Eq. 7a deliberately left free.",
        param_label="foot yaw about the contact point [rad]",
        values=np.linspace(0.0, 0.40, 81),
        predicted_crossing=0.0,
        prediction_note="Eq. 8: log3((R^s)^T R^{s+1})_z = 0 exactly, so any spin violates.",
        build=_build_8_spin,
        tolerance=1e-2,
    ),
    Scenario(
        key="8-lift",
        equation="8",
        title="Raise the robot straight up  [NEVER VIOLATES]",
        question="Is Eq. 8 a full pose lock, or only an IN-PLANE one?",
        expect="Nothing happens, through 20 cm of lift. Eq. 8 constrains only the "
               "in-plane position and the yaw -- the normal direction belongs to "
               "Eq. 7a's p_z = 0. Getting this wrong the other way would weld the foot "
               "to the floor and make stepping impossible, so the fact that Eq. 8 stays "
               "silent here is what lets the robot ever lift a foot.",
        param_label="lift height [m]",
        values=np.linspace(0.0, 0.20, 81),
        predicted_crossing=None,
        prediction_note="Eq. 8 constrains (p)_{x,y} and yaw only; p_z is Eq. 7a's business.",
        build=_build_8_lift,
    ),
    Scenario(
        key="8-carry",
        equation="8",
        title="Carry the box: hand and box move together  [NEVER VIOLATES]",
        question="Is Eq. 8 about the world pose of a contact, or the RELATIVE pose?",
        expect="Both patches sweep a quarter metre through space and the residual "
               "stays at exactly zero. Eq. 8 holds the pose of patch a RELATIVE TO "
               "patch b, so a hand stuck to a box it is carrying never slips. Written "
               "against world poses instead, this would forbid moving a grasped object "
               "at all -- which would make the paper's whole box-placement task "
               "infeasible.",
        param_label="carry distance [m]",
        values=np.linspace(0.0, 0.25, 76),
        predicted_crossing=None,
        prediction_note="Eq. 8 uses the relative transform, so a rigid carry leaves it at 0.",
        build=_build_8_carry,
    ),
    Scenario(
        key="9-penetrate",
        equation="9",
        title="Drive the box into the platform  [witness FROZEN, as in an NLP]",
        question="What does the GJK signed-distance approximation actually measure?",
        expect="The signed distance falls linearly with the gap and keeps going "
               "NEGATIVE past contact. That negative value is what gives a solver a "
               "gradient to push back on. The witness points and normal were frozen at "
               "a 0.10 m gap -- freezing is what makes the expression linear and "
               "differentiable.",
        param_label="gap [m]  (negative = interpenetrating)",
        values=np.linspace(0.10, -0.05, 121),
        predicted_crossing=0.0,
        prediction_note="Eq. 9: 0 <= sd. With the witness frozen, sd = gap exactly for "
                        "this pure translation, so it crosses at gap = 0.",
        build=_build_9_penetrate,
        tolerance=3e-3,
    ),
    Scenario(
        key="9-drift",
        equation="9",
        title="Hold the clearance and tilt the box -- how far can the frozen model be trusted?",
        question="What does the '~=' in Eq. 9 actually cost, and when must GJK be re-run?",
        expect="The box keeps a 2 cm clearance at its original contact, but as it tilts, "
               "a DIFFERENT corner swings down toward the platform. The frozen witness "
               "pair does not know that, so Eq. 9 keeps reporting the clearance of a "
               "corner that is no longer the closest one, and it drifts away from the "
               "true signed distance. It crosses zero only once the frozen pair itself "
               "separates -- long after the real geometry has touched. Compare with "
               "9-penetrate, where a flat face slides straight in: there the closest "
               "pair never changes and the frozen model is EXACT. That contrast is the "
               "whole argument for refreshing witness data between solver iterations "
               "and never inside the constraint (GJK is combinatorial, so it cannot go "
               "in the graph anyway).",
        param_label="box tilt [rad]",
        values=np.linspace(0.0, 0.5, 101),
        predicted_crossing=None,   # the frozen model NEVER crosses -- that is the point
        prediction_note="",        # completed with the true crossing in resolve_predictions
        build=_build_9_drift,
        tolerance=2e-2,
    ),
    Scenario(
        key="10-weight",
        equation="10",
        title="How hard must the floor push to hold the robot up?",
        question="What does Eq. 10 actually equate?",
        expect="hdot is the NET wrench, so a robot standing still (hdot = 0) needs the "
               "contact force to cancel gravity exactly. Push harder than m*g and the "
               "equality breaks -- the robot would have to be accelerating upward. The "
               "crossing is the robot's own weight, read from the URDF, not a "
               "paper number.",
        param_label="vertical contact force f_z [N]",
        values=None,          # filled from the robot's mass in resolve_predictions
        predicted_crossing=None,
        prediction_note="Eq. 10b: hdot_z = -m g + f_z = 0 at f_z = m g.",
        build=_build_10_weight,
        tolerance=1.0,
    ),
    Scenario(
        key="10-moment",
        equation="10",
        title="Slide the support sideways from under the centre of mass",
        question="Why is the moment arm measured from the COM and not the origin?",
        expect="The force still cancels gravity exactly, so the LINEAR row stays at "
               "zero -- but (p - c) x f grows the moment the support leaves the "
               "vertical through the CoM, and the ANGULAR row does not forgive it. "
               "This is 'why you fall over', and it is the reason Eq. 7d's centre of "
               "pressure has to stay under the body.",
        param_label="support offset from the CoM [m]",
        values=np.linspace(0.0, 0.06, 61),
        predicted_crossing=0.0,
        prediction_note="Eq. 10b: hdot_ang = (p - c) x f, zero only when the support is "
                        "directly under the CoM.",
        build=_build_10_moment,
        tolerance=2e-3,
    ),
    Scenario(
        key="11-hold",
        equation="11",
        title="Hold the box still against gravity",
        question="Does the object have its own dynamics, separate from the robot?",
        expect="WATCH THE MIDDLE ARROW. Three are drawn: the support (left), the weight "
               "(right) -- offset so their lengths can be compared -- and at the centre "
               "the RESIDUAL W_ext, which is the only one Eq. 11b actually constrains. "
               "It starts at zero length, and every newton of extra support makes it "
               "grow; the box turns red as it does. A box pushed harder than its weight "
               "does not sit still, it accelerates upward. "
               "Eq. 11 is Newton-Euler for the box alone: at rest (V = 0, "
               "Vdot = 0) the total external wrench must vanish, so the support has to "
               "equal the box's weight, 2.0 * 9.81 = 19.62 N. Note this is a SEPARATE "
               "equation from the robot's Eq. 10: FARO optimizes over robot AND object "
               "states, which is what lets it plan a throw or a placement.",
        param_label="supporting force f_z [N]",
        values=np.linspace(19.62, 29.62, 101),
        predicted_crossing=19.62,
        prediction_note="Eq. 11b at rest: W_ext = 0, so the support must equal the box's "
                        "weight, f_z = m_box g = 2.0 * 9.81 = 19.62 N.",
        build=_build_11_hold,
        tolerance=0.15,
    ),
    Scenario(
        key="11-spin",
        equation="11",
        title="Tumble an elongated object at constant rate, with nothing pushing it",
        question="What is the `-[ad_V]^T G V` term for, and when does it matter?",
        expect="WATCH: a grey elongated slab tumbles about the diagonal axis, spinning "
               "faster every frame. Yellow arrow = the spin rate you asked for. Purple "
               "arrow = the moment Eq. 11b says you MUST supply just to keep that spin "
               "steady -- and nothing is supplying it, so the constraint fails the "
               "instant the body turns at all. The bead above turns red. "
               "With no external wrench and a constant twist, Eq. 11b becomes "
               "0 = -[ad_V]^T G V -- and the gyroscopic term says that is impossible. "
               "An anisotropic body spinning about a non-principal axis needs a moment "
               "just to keep spinning steadily. The residual IS the textbook Euler term "
               "omega x (I omega) and grows as omega^2. NOTE: the scene's box is a CUBE, "
               "whose inertia is isotropic, so omega x (I omega) = 0 for every omega -- "
               "no scenario using that box could ever exercise this term, which is why "
               "an elongated probe body is used here.",
        param_label="angular rate about (1,1,1)/sqrt(3) [rad/s]",
        values=np.linspace(0.0, _SPIN_OMEGA_MAX, _SPIN_STEPS),
        predicted_crossing=0.0,
        prediction_note="Eq. 11b: with W_ext = 0 and Vdot = 0 the residual is "
                        "omega x (I omega), non-zero for any spin about a non-principal "
                        "axis, so it violates immediately.",
        build=_build_11_spin,
        tolerance=1e-1,
    ),
    Scenario(
        key="13b-speed",
        equation="13b",
        title="Drive a joint past its velocity limit",
        question="Which stages can enforce Eq. 13b at all?",
        expect="WATCH: the knee swings back and forth, faster every frame, and the bead "
               "on the knee turns RED the moment the rate passes the limit. Blue arrow = "
               "the limit, yellow = the actual rate. PLAYED AT 1/6 SPEED -- at 20 rad/s "
               "the knee crosses its whole 2.97 rad range in four frames, which reads as "
               "a flicker rather than a motion; the terminal shows the true rates. "
               "The joint really is turning at the "
               "swept rate. It binds exactly at the URDF's "
               "velocity limit. Note what does NOT happen: the knee never leaves its "
               "mechanical range, so Eq. 13a holds throughout while 13b is being "
               "violated -- position and speed limits are independent. Unlike 13a, this "
               "one needs a VELOCITY -- so mode/edge (Eq. 14) and KSO (Eq. 15) cannot "
               "enforce it at all, and only the trajectory optimization (Eq. 17) sees "
               "it. That is precisely the constraint coverage gap Table I reports "
               "between KSO and TO.",
        param_label="knee velocity [rad/s]",
        values=None,          # filled from the URDF in resolve_predictions
        predicted_crossing=None,
        prediction_note="Eq. 13b: v_j <= v_max, read from the G1 URDF.",
        build=_build_13b_speed,
        tolerance=0.1,
    ),
    Scenario(
        key="13c-envelope",
        equation="13c",
        title="Speed a joint up while it is already delivering half its peak torque",
        question="Is Eq. 13c a box on (tau, v), or something else?",
        expect="Something else: an ENVELOPE. The knee speeds up exactly as in 13b-speed, "
               "but now watch the two arrows rather than the leg: the yellow DEMAND "
               "stays fixed while the blue AVAILABLE torque shrinks as the joint turns "
               "faster. Torque and speed trade off linearly, so a "
               "motor holding half its peak torque can only reach HALF its no-load "
               "speed. A box constraint would have allowed full torque at full speed, "
               "which no real actuator can do. Watch which of the four rows bites -- "
               "the sign combination matters, and using fabs instead would be "
               "non-differentiable exactly at v = 0 where joints rest.",
        param_label="knee velocity [rad/s]",
        values=None,          # filled from the URDF in resolve_predictions
        predicted_crossing=None,
        prediction_note="Eq. 13c: |tau| + (tau_max/v_max)|v| <= tau_max with "
                        "|tau| = 0.5 tau_max gives v = 0.5 v_max.",
        build=_build_13c_envelope,
        tolerance=0.1,
    ),
    Scenario(
        key="13a-knee",
        equation="13a",
        title="Drive the knee past its mechanical limit",
        question="What does Eq. 13a protect, and where does it bind?",
        expect="The knee sails through its range and the constraint bites exactly at the "
               "URDF's q_max. This is the ONLY limit that mode/edge (Eq. 14) and KSO "
               "(Eq. 15) enforce -- 13b and 13c need velocities and torques, which "
               "kinematic problems do not have.",
        param_label="left knee angle [rad]",
        values=np.linspace(0.0, 2.9, 146),
        predicted_crossing=None,   # filled in from the URDF at runtime
        prediction_note="Eq. 13a: q_j <= q_max, read from the G1 URDF.",
        build=_build_13a_knee,
        tolerance=3e-2,
    ),
]


def get_scenario(key: str) -> Scenario:
    for scenario in ALL_SCENARIOS:
        if scenario.key == key:
            return scenario
    raise KeyError(f"unknown scenario {key!r}; available: {[s.key for s in ALL_SCENARIOS]}")


def resolve_predictions(scene: Scene) -> None:
    """Fill in predictions that can only be read from the loaded robot.

    The knee limit is a URDF number, not a paper number, so it is looked up rather
    than hard-coded -- swapping robots must not silently invalidate the prediction.
    """
    for scenario in ALL_SCENARIOS:
        if scenario.key == "10-weight" and scenario.values is None:
            weight = robot_weight(scene)
            scenario.values = np.linspace(weight, weight + 60.0, 121)
            scenario.predicted_crossing = weight
            scenario.prediction_note = (
                f"Eq. 10b: hdot_z = -m g + f_z = 0 at f_z = m g = {weight:.2f} N "
                f"(mass from the G1 URDF)."
            )
        if scenario.key == "9-drift" and not scenario.prediction_note:
            # Where the geometry ACTUALLY collides, from a fresh query at every tilt --
            # so the printed verdict contrasts the model against the truth rather than
            # against another copy of itself.
            truth = [(float(t), _build_9_drift(scene, float(t))[2]["true_signed_distance"])
                     for t in scenario.values]
            touch = next((t for t, sd in truth if sd < 0.0), None)
            scenario.prediction_note = (
                "Eq. 9 with FROZEN witness data never crosses zero here -- it reports "
                f"growing clearance. The real geometry touches at tilt = {touch:.3f} rad "
                "and is interpenetrating from there on. The gap between the two IS the "
                "linearization error, and it is why witness data must be refreshed."
            )
        if scenario.key == "13b-speed" and scenario.values is None:
            limit = knee_velocity_limit(scene)
            # `_knee_at_speed` integrates against exactly this ramp, so the endpoint
            # and the step count must come from the same constants it uses.
            scenario.values = np.linspace(0.0, limit * _SPEED_MARGIN, _SPEED_STEPS)
            scenario.predicted_crossing = limit
            scenario.prediction_note = (
                f"Eq. 13b: v_j <= v_max = {limit:.2f} rad/s (from the G1 URDF)."
            )
        if scenario.key == "13c-envelope" and scenario.values is None:
            v_max = knee_velocity_limit(scene)
            crossing = v_max * (1.0 - _TORQUE_FRACTION)
            scenario.values = np.linspace(0.0, v_max, _SPEED_STEPS)
            scenario.predicted_crossing = crossing
            scenario.prediction_note = (
                f"Eq. 13c: |tau| + (tau_max/v_max)|v| <= tau_max with "
                f"|tau| = {_TORQUE_FRACTION} tau_max gives v = "
                f"{_TORQUE_FRACTION} v_max = {crossing:.2f} rad/s."
            )
        if scenario.key == "13a-knee" and scenario.predicted_crossing is None:
            limit = knee_upper_limit(scene)
            scenario.predicted_crossing = limit
            scenario.prediction_note = f"Eq. 13a: q_j <= q_max = {limit:.4f} rad (from the G1 URDF)."
