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


def object_integration(q_i, V_i, q_next, V_next, Vdot_next, dt_scaled, *, name: str = "object_int"):
    """Eq. 11a: backward-Euler integration of object pose and twist.

    The pose residual is left to the caller's SE(3) parameterisation via
    `pose_residual`, because objects are stored as SE(3) rather than as a Pinocchio
    model. Only the twist defect is universal, so that is what this returns.

    Note `V_next` and `Vdot_next` on the right-hand sides -- backward Euler again.
    """
    eq = V_next - (V_i + Vdot_next * dt_scaled)
    return ConstraintBlock(name=name, eq=eq, eq_labels=[f"11a:V[{i}]" for i in range(6)])


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


def transform_wrench_to_body(R_world_body, application_point_body, force_world, moment_world):
    """Express a world-frame wrench in the object body frame, angular-first.

    `application_point_body` is where the wrench acts, in body coordinates -- the
    moment arm the paper's `sum_{a in I} W^o_a` needs when a hand pushes a box face
    away from the object's centre of mass.
    """
    f_body = R_world_body.T @ force_world
    m_body = R_world_body.T @ moment_world + ca.cross(application_point_body, f_body)
    return ca.vertcat(m_body, f_body)
