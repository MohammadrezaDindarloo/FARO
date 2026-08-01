"""Actuation and limit constraints -- paper Eqs. 12 and 13 (Section II-C 4).

Eq. 12 is a DEFINITION, not a constraint -- the actuated joint torque:

    tau_j := S ( M(q) vdot + b(q, v) - sum_{e=1}^{nee} J_e(q)^T lambda^r_e )   (12)

    where S selects the actuated components of the dynamics.

Eq. 13 holds the actual limits, applied COMPONENTWISE:

    q_min <= q_j <= q_max                                                  (13a)
    v_min <= v_j <= v_max                                                  (13b)
    |tau_j| + (tau_max / v_tau_max) |v_j| <= tau_max                       (13c)

    The last inequality is a linear torque-speed envelope: the available torque
    decreases affinely with joint speed, reaching zero at v_tau_max.

WHICH STAGE USES WHICH (this is the part that is easy to get wrong):
  * Eq. 14  (mode/edge) -> `limits(q) <= 0 (13a)`      position limits ONLY
  * Eq. 15  (KSO)       -> `limits(q_s) <= 0 (13a)`    position limits ONLY
  * Eq. 17  (TO)        -> `limits(x_i, u_i) <= 0 (13)` ALL of 13a, 13b, 13c

Mode/edge and KSO are kinematic problems with no velocity or torque variables, so
13b and 13c are not merely omitted for speed -- they are undefined there.

FLOATING BASE: `q_j` and `v_j` are the ACTUATED joints only. The 6 floating-base
DoF are unactuated and have no limits; including them would impose meaningless
bounds on the base pose. `actuated_slice` centralises that indexing.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from faro.constraints.block import ConstraintBlock


def actuated_slice(model, *, velocity: bool = False) -> slice:
    """Index range of the actuated entries of q (or v) for a floating-base model.

    A quaternion free-flyer occupies the first 7 entries of q and the first 6 of v,
    so the actuated joints start at 7 and 6 respectively.
    """
    if model.nq - model.nv != 1:
        raise ValueError(
            f"expected a quaternion floating base (nq - nv == 1), got nq={model.nq}, nv={model.nv}"
        )
    return slice(6, model.nv) if velocity else slice(7, model.nq)


def joint_position_limits(q, q_min, q_max, *, name: str = "limits_q") -> ConstraintBlock:
    """Eq. 13a, componentwise, in `<= 0` form.

    `q` here is already the ACTUATED slice; see `actuated_slice`.
    """
    q_min = ca.DM(np.asarray(q_min).reshape(-1, 1))
    q_max = ca.DM(np.asarray(q_max).reshape(-1, 1))
    n = int(q.shape[0])
    return ConstraintBlock(
        name=name,
        ineq=ca.vertcat(q - q_max, q_min - q),
        ineq_labels=[f"13a:q<=max[{i}]" for i in range(n)] + [f"13a:q>=min[{i}]" for i in range(n)],
    )


def joint_velocity_limits(v, v_min, v_max, *, name: str = "limits_v") -> ConstraintBlock:
    """Eq. 13b, componentwise. TO only -- KSO and mode/edge have no velocities."""
    v_min = ca.DM(np.asarray(v_min).reshape(-1, 1))
    v_max = ca.DM(np.asarray(v_max).reshape(-1, 1))
    n = int(v.shape[0])
    return ConstraintBlock(
        name=name,
        ineq=ca.vertcat(v - v_max, v_min - v),
        ineq_labels=[f"13b:v<=max[{i}]" for i in range(n)] + [f"13b:v>=min[{i}]" for i in range(n)],
    )


def actuated_torque(M, b, jacobians, wrenches, selection, *, name: str = "tau"):
    """Eq. 12: tau_j := S ( M(q) vdot + b(q, v) - sum_e J_e(q)^T lambda_e ).

    Parameters
    ----------
    M : the term M(q) @ vdot, already multiplied (kept as one argument so callers
        may substitute Pinocchio's RNEA, which returns M vdot + b directly).
    b : nonlinear effects b(q, v). Pass zeros if `M` already includes them.
    jacobians : list of 6 x nv end-effector Jacobians J_e(q).
    wrenches : list of 6-vectors lambda_e, in the SAME frame as their Jacobian.
    selection : S, either a matrix or a slice picking the actuated rows.

    The minus sign matters: contact wrenches OFFLOAD the actuators. Flipping it
    doubles the apparent torque instead of cancelling it, which then makes Eq. 13c
    reject perfectly good motions.
    """
    total = M + b
    for J, wrench in zip(jacobians, wrenches):
        total = total - J.T @ wrench
    return total[selection] if isinstance(selection, slice) else selection @ total


def torque_speed_limits(
    tau, v, *, tau_max, v_tau_max, name: str = "limits_tau_speed"
) -> ConstraintBlock:
    """Eq. 13c: |tau_j| + (tau_max / v_tau_max) |v_j| <= tau_max.

    A linear envelope: at zero speed the full tau_max is available, and it falls
    affinely to zero at v_tau_max. Modelled with the absolute values expanded into
    four linear inequalities per joint (the four sign combinations of tau and v),
    which keeps the constraint SMOOTH -- `fabs` is not differentiable at zero and
    would stall Ipopt exactly where joints are at rest.
    """
    tau_max = ca.DM(np.asarray(tau_max, dtype=float).reshape(-1, 1))
    slope = ca.DM((np.asarray(tau_max).reshape(-1) / np.asarray(v_tau_max, dtype=float).reshape(-1)).reshape(-1, 1))

    n = int(tau.shape[0])
    rows, labels = [], []
    for s_tau in (+1.0, -1.0):
        for s_v in (+1.0, -1.0):
            rows.append(s_tau * tau + slope * (s_v * v) - tau_max)
            labels += [f"13c:tau{'+' if s_tau > 0 else '-'}v{'+' if s_v > 0 else '-'}[{i}]"
                       for i in range(n)]
    return ConstraintBlock(name=name, ineq=ca.vertcat(*rows), ineq_labels=labels)


def default_velocity_limits(model) -> np.ndarray:
    """Velocity limits from the URDF, for Eq. 13b.

    Pinocchio reports 0 for joints whose URDF omits a velocity limit, which as a
    bound would freeze the joint solid. Those are replaced with a large finite
    value and reported, rather than silently producing an over-constrained NLP.
    """
    limits = np.asarray(model.velocityLimit, dtype=float).copy()
    missing = (limits <= 0) | ~np.isfinite(limits)
    limits[missing] = 1e3
    return limits
