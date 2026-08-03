"""Robot dynamics -- paper Eq. 10 (Section II-C 3).

    q^r_{i+1} = q^r_i (+) v^r_{i+1} (T_bar dt),
    v^r_{i+1} = v^r_i + vdot^r_{i+1} (T_bar dt),
    h^r_{i+1} = h^r_i + hdot^r_{i+1} (T_bar dt),                          (10a)

    hdot^r = [ m g + sum_e f^r_e
               sum_e (p^r_e - c) x f^r_e + kappa^r_e ],
    h^r = A(q^r) v^r                                                      (10b)

    where (10a) integrates the robot configuration, velocity, and centroidal
    momentum using a BACKWARD EULER discretization over the scaled timestep
    T_bar dt, and (+) denotes integration on the configuration manifold.

Two things are easy to get wrong and are called out in the code below:

  * The integration is BACKWARD Euler: the right-hand side uses the state at
    i+1, not i. Writing `v_i` where the paper writes `v_{i+1}` gives forward
    Euler, which still "works" but is a different (and less stable) transcription.

  * `(+)` is manifold integration, not vector addition. q^r lives in SE(3) x R^n
    (Eq. 6a), so the floating base must be integrated with `pin.integrate`;
    adding a 6-vector to a 7-entry quaternion configuration is meaningless.

CENTROIDAL DYNAMICS. Eq. 10b is a *centroidal* model: the momentum rate is just
the net external wrench (gravity plus the end-effector wrenches), and whole-body
consistency is imposed separately by h = A(q) v with A the centroidal momentum
matrix. This is why the robot control in Eq. 5a is (vdot, lambda_1..lambda_nee) --
accelerations and contact wrenches -- rather than joint torques. Torques appear
only in the limits, Eq. 12.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio.casadi as cpin

from faro.constraints.block import ConstraintBlock
from faro.constraints.frames import as_sx


def centroidal_momentum_rate(f_list, kappa_list, p_list, com, mass: float, gravity: float = 9.81):
    """Eq. 10b (first line): hdot^r from gravity and end-effector wrenches.

    Parameters
    ----------
    f_list, kappa_list : per-end-effector force and moment (world frame).
    p_list : per-end-effector contact point positions (world frame), the paper's p^r_e.
    com : robot centre of mass c (world frame).
    mass : total robot mass m.

    Returns a 6-vector [linear; angular].

    The angular part uses the moment arm (p_e - c) about the CENTRE OF MASS, not
    the origin -- that is what makes this centroidal momentum rather than spatial
    momentum about the world origin.
    """
    linear = ca.vertcat(0.0, 0.0, -mass * gravity)
    angular = ca.DM.zeros(3)   # DM, not SX: promotes to the caller's symbolic type
    for f, kappa, p in zip(f_list, kappa_list, p_list):
        linear = linear + f
        angular = angular + ca.cross(p - com, f) + kappa
    return ca.vertcat(linear, angular)


def centroidal_consistency(cmodel, cdata, q, v, h, *, name: str = "centroidal") -> ConstraintBlock:
    """Eq. 10b (second line): h^r = A(q^r) v^r.

    A(q) is the centroidal momentum matrix. This is the constraint that couples the
    reduced centroidal state to the whole-body configuration -- without it, the
    momentum variables would be free to take any value and the model would permit
    motions the actual robot cannot produce.
    """
    cpin.computeCentroidalMap(cmodel, cdata, as_sx(q))
    A = cdata.Ag
    return ConstraintBlock(
        name=name,
        eq=as_sx(h) - A @ as_sx(v),
        eq_labels=[f"10b:h-Av[{i}]" for i in range(6)],
    )


def integrate_configuration(cmodel, q, v, dt_scaled):
    """The paper's `q (+) v (T_bar dt)` -- manifold integration of Eq. 10a.

    Delegates to pinocchio's `integrate`, which applies the exponential map on the
    floating base and plain addition on the revolute joints. Never replace this
    with `q + v * dt`: the base is quaternion-parameterised, so vector addition would
    leave the configuration off the manifold. The error is SECOND order in |omega| dt,
    which is what makes it dangerous -- a few parts in 1e5 at walking rates, where it
    passes for round-off, but over 1% at the 3 rad/s a dynamic task reaches. And
    renormalising does not rescue it: the resulting rotation is still wrong, because
    omega dt belongs in the exponential map, not in the quaternion's vector part.

    Numeric inputs are promoted like `log3`'s, so this can be checked against
    `pin.integrate` directly instead of only inside a symbolic graph.
    """
    return cpin.integrate(cmodel, as_sx(q), as_sx(v * dt_scaled))


def robot_dynamics(
    cmodel, cdata,
    q_i, v_i, h_i,
    q_next, v_next, h_next,
    vdot_next, hdot_next,
    dt_scaled,
    *, name: str = "robot_dyn",
) -> ConstraintBlock:
    """Eq. 10a: backward-Euler integration of configuration, velocity and momentum.

    `dt_scaled` is the paper's `T_bar_s dt` -- the nominal timestep multiplied by
    the per-contact-stage time-scaling factor of Eq. 17. Passing it in (rather than
    a raw dt) is what lets TO stretch or compress each contact stage.

    All three defects are equalities, in the paper's own order.
    """
    # q_{i+1} = q_i (+) v_{i+1} (T_bar dt)   -- note v at i+1: backward Euler.
    q_pred = integrate_configuration(cmodel, q_i, v_next, dt_scaled)
    # Difference on the manifold, not q_next - q_pred: the residual must live in
    # the tangent space or the floating base contributes a meaningless quaternion
    # difference.
    eq_q = cpin.difference(cmodel, as_sx(q_pred), as_sx(q_next))

    eq_v = v_next - (v_i + vdot_next * dt_scaled)
    eq_h = h_next - (h_i + hdot_next * dt_scaled)

    nv = int(eq_v.shape[0])
    return ConstraintBlock(
        name=name,
        eq=ca.vertcat(eq_q, eq_v, eq_h),
        eq_labels=(
            [f"10a:q[{i}]" for i in range(int(eq_q.shape[0]))]
            + [f"10a:v[{i}]" for i in range(nv)]
            + [f"10a:h[{i}]" for i in range(6)]
        ),
    )


def total_mass(model) -> float:
    """Total robot mass m, for Eq. 10b."""
    return float(sum(inertia.mass for inertia in model.inertias))


def center_of_mass(cmodel, cdata, q):
    """Symbolic centre of mass c(q), for the moment arms in Eq. 10b."""
    cpin.centerOfMass(cmodel, cdata, as_sx(q))
    return cdata.com[0]
