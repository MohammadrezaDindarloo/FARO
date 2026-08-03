"""Rigid-body object dynamics -- paper Eq. 11 (Section II-C 3).

    q^o_{i+1} = q^o_i (+) V^o_{i+1} (T_bar dt),
    V^o_{i+1} = V^o_i + Vdot^o_{i+1} (T_bar dt)                            (11a)

    W^o_ext = G^o Vdot^o  -  [ad_{V^o}]^T G^o V^o                          (11b)

    (11b) defines the object twist-wrench dynamics driven by the total external
    wrench, following [23] (Lynch & Park, Modern Robotics). Here W^o denotes
    wrenches expressed in the OBJECT BODY FRAME, [ad_{V^o}] is the adjoint Lie
    bracket operator associated with the object body twist V^o, and G^o is the
    spatial inertia matrix of object o.

    W^o_ext := W^o_env + W^o_grav + sum_{a in I} W^o_a

Eq. 11a splits across two functions because its two lines live in different spaces:
`object_pose_residual` for the SE(3) line (needs the exponential map) and
`object_integration` for the twist line (a plain vector-space defect).
`full_object_integration` returns both.

Eq. 11b is the Newton-Euler equation in body coordinates. It is why objects need
their own state (Eq. 5b) rather than being welded into the robot model: a carried
box has its own momentum and its own contact wrenches.

CONVENTIONS THAT MUST MATCH (Modern Robotics ch. 8, the paper's [23]):
  * Twist V = (omega; v) -- ANGULAR FIRST, then linear.
  * Spatial inertia G = diag(I, m*Id3) in that same ordering.
  * Wrench W = (moment; force), matching the twist ordering by duality.

Pinocchio's own spatial types use the opposite (linear-first) ordering, so mixing
the two silently swaps the blocks. Everything in this module is angular-first, and
`spatial_inertia` builds G to match.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from faro.constraints.block import ConstraintBlock
from faro.constraints.frames import skew


def spatial_inertia(mass: float, inertia_tensor) -> np.ndarray:
    """The object's spatial inertia G^o, in (angular, linear) ordering.

        G = [ I   0     ]
            [ 0   m*Id3 ]
    """
    G = np.zeros((6, 6))
    G[:3, :3] = np.asarray(inertia_tensor, dtype=float)
    G[3:, 3:] = float(mass) * np.eye(3)
    return G


def adjoint_transpose_term(V, G):
    """The `-[ad_V]^T G V` term of Eq. 11b, the velocity-product (Coriolis) force.

    With V = (omega; v),

        [ad_V] = [ skew(omega)      0      ]
                 [ skew(v)     skew(omega) ]

    (Modern Robotics eq. 8.40). This term is what makes a spinning object precess;
    dropping it looks harmless at low speed and is wrong for the paper's dynamic
    tasks (Toss to Table, Double Catch, Juggle).
    """
    omega, v = V[:3], V[3:]
    # `ca.DM.zeros`, not `ca.SX.zeros`: a DM block promotes to whichever symbolic type
    # the caller brought, so this stays usable from both SX and MX graphs. Hard-coding
    # SX here made the whole term unusable from an MX graph, which is what a large
    # multiple-shooting transcription is normally built with.
    ad = ca.vertcat(
        ca.horzcat(skew(omega), ca.DM.zeros(3, 3)),
        ca.horzcat(skew(v), skew(omega)),
    )
    return -(ad.T @ (ca.DM(G) @ V))


def object_newton_euler(V, Vdot, W_ext, G, *, name: str = "object_ne") -> ConstraintBlock:
    """Eq. 11b: W_ext = G Vdot - [ad_V]^T G V, as an equality residual.

    All quantities in the object BODY frame, angular-first.
    """
    residual = ca.DM(G) @ Vdot + adjoint_transpose_term(V, G) - W_ext
    return ConstraintBlock(
        name=name,
        eq=residual,
        eq_labels=[f"11b:W_ext[{i}]" for i in range(6)],
    )


def object_pose_residual(M_i, V_next, M_next, dt_scaled, *, name: str = "object_pose"):
    """Eq. 11a, FIRST line: q^o_{i+1} = q^o_i (+) V^o_{i+1} (T_bar dt), on SE(3).

    Parameters
    ----------
    M_i, M_next : object poses at steps i and i+1, as `pinocchio.SE3` (numeric) or
        `pinocchio.casadi.SE3` (symbolic).
    V_next : the object BODY twist at i+1, ANGULAR-FIRST like everything else here.

    Returns 6 equalities, the pose defect expressed in the tangent space at `M_next`.

    THREE things here are easy to get wrong, and all three are silent:

    1. `(+)` is the SE(3) exponential, not addition. A pose is not a vector; adding
       `V dt` to a rotation matrix leaves SO(3) immediately.

    2. V is a BODY twist, so it composes on the RIGHT: `M_i * exp6(V dt)`. Writing
       `exp6(V dt) * M_i` treats it as a spatial twist and rotates the object about
       the world origin instead of its own centre -- the two agree only when the
       object sits at the origin with identity orientation, which is exactly the
       configuration a first test tends to use.

    3. `pinocchio`'s `exp6`/`log6` are LINEAR-FIRST, while Eq. 11b and this whole
       module are ANGULAR-FIRST (Lynch & Park). This is the one place in the codebase
       where a twist genuinely crosses that boundary, so it goes through
       `swap_spatial_ordering` rather than relying on the blocks happening to line up.

    The residual is `log6(predicted^-1 * M_next)`, a difference taken ON the manifold
    for the same reason `robot_dynamics` uses `cpin.difference` rather than `q_next -
    q_pred`: subtracting two poses entrywise is meaningless.
    """
    import pinocchio.casadi as cpin

    from faro.constraints.frames import swap_spatial_ordering

    # angular-first (ours) -> linear-first (pinocchio's exp6)
    twist = swap_spatial_ordering(ca.SX(V_next)) * dt_scaled
    predicted = M_i * cpin.exp6(twist)
    residual = cpin.log6(predicted.actInv(M_next)).vector

    return ConstraintBlock(
        name=name,
        eq=residual,
        eq_labels=[f"11a:q[{i}]" for i in range(6)],
    )


def object_integration(q_i, V_i, q_next, V_next, Vdot_next, dt_scaled, *, name: str = "object_int"):
    """Eq. 11a, SECOND line: V^o_{i+1} = V^o_i + Vdot^o_{i+1} (T_bar dt).

    The twist lives in a vector space, so this one really is plain addition. The POSE
    line is `object_pose_residual`, which needs the SE(3) exponential and is kept
    separate because it needs the poses as SE3 objects rather than as coordinates.

    Note `Vdot_next` on the right-hand side -- backward Euler, as the paper writes it.

    `q_i` and `q_next` are accepted only so the signature mirrors the paper's line;
    they are not used here. An earlier version took them and quietly returned a
    six-row block that ignored them entirely, so a caller could hand it two wildly
    inconsistent poses and get a zero residual back. Pass them to
    `object_pose_residual` -- `full_object_integration` does both.
    """
    eq = V_next - (V_i + Vdot_next * dt_scaled)
    return ConstraintBlock(name=name, eq=eq, eq_labels=[f"11a:V[{i}]" for i in range(6)])


def full_object_integration(M_i, V_i, M_next, V_next, Vdot_next, dt_scaled,
                            *, name: str = "object_11a") -> ConstraintBlock:
    """Both lines of Eq. 11a: 12 equalities, pose defect then twist defect."""
    from faro.constraints.block import merge

    pose = object_pose_residual(M_i, V_next, M_next, dt_scaled, name="pose")
    twist = object_integration(None, V_i, None, V_next, Vdot_next, dt_scaled, name="twist")
    return merge(name, [pose, twist])


def gravity_wrench(mass: float, R_world_body, gravity: float = 9.81):
    """W^o_grav in the object BODY frame, angular-first.

    Gravity acts at the centre of mass, so it produces no moment about the body
    frame when that frame is at the CoM (which is how objects are defined here).
    The force must still be ROTATED into the body frame -- Eq. 11b is written in
    body coordinates throughout.
    """
    f_world = ca.vertcat(0.0, 0.0, -float(mass) * gravity)
    f_body = R_world_body.T @ f_world
    return ca.vertcat(ca.SX.zeros(3), f_body)


def external_wrench(mass: float, R_world_body, *, contacts=(), W_env=None,
                    gravity: float = 9.81):
    """The paper's definition line: W^o_ext := W^o_env + W^o_grav + sum_{a in I} W^o_a.

    Everything returned is in the object BODY frame, angular-first, ready for
    `object_newton_euler`. Until this existed the definition had no home in the code:
    `object_newton_euler` took `W_ext` as a parameter and the sum was assembled inline,
    by hand, in one scenario. `transform_wrench_to_body` -- the ingredient every
    `W^o_a` needs -- was called by no production code at all, only by a test.

    Parameters
    ----------
    contacts : iterable of `(force_world, moment_world, application_point_body)`, one
        per contact interface `a` acting on the object. The application point is in
        BODY coordinates, and it is the whole reason this cannot be a plain sum.
    W_env : any other environment wrench (drag, a spring, a conveyor), body frame,
        angular-first. The paper lists it separately and never says what is in it, so
        it defaults to nothing rather than being silently assumed zero forever.

    WHY THE MOMENT ARM IS THE POINT. A hand pressing a box FACE acts well away from
    the box's centre of mass, so its wrench contributes `p x f` to the moment even
    when the hand applies no torque of its own. Summing the raw force vectors instead
    -- the obvious-looking implementation -- gives an object that translates correctly
    and NEVER ROTATES when pushed off-centre. Nothing in Eq. 11b would flag that; the
    residual balances perfectly for the wrong physics.
    """
    total = gravity_wrench(mass, R_world_body, gravity)
    if W_env is not None:
        total = total + W_env
    for force_world, moment_world, point_body in contacts:
        total = total + transform_wrench_to_body(
            R_world_body, point_body, force_world, moment_world)
    return total


def transform_wrench_to_body(R_world_body, application_point_body, force_world, moment_world):
    """Express a world-frame wrench in the object body frame, angular-first.

    `application_point_body` is where the wrench acts, in body coordinates -- the
    moment arm the paper's `sum_{a in I} W^o_a` needs when a hand pushes a box face
    away from the object's centre of mass.
    """
    f_body = R_world_body.T @ force_world
    m_body = R_world_body.T @ moment_world + ca.cross(application_point_body, f_body)
    return ca.vertcat(m_body, f_body)
