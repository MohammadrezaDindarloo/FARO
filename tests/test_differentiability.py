"""Every solver-facing constraint must be symbolic and twice differentiable.

Milestones 3-5 hand these expressions to Ipopt, which needs continuous second
derivatives. A kink does not raise an error -- it degrades convergence, or makes the
solver linearize a constraint as insensitive to a variable it actually depends on,
and the symptom appears as `Infeasible_Problem_Detected` many modules away.

So it is checked mechanically here, three ways:

  1. every builder accepts CasADi SYMBOLS, not just numbers;
  2. the expression graph contains no non-smooth operator (fabs, sign, fmin/fmax,
     floor, copysign) -- scanned via CasADi's instruction op-codes;
  3. first and second derivatives are finite, and the analytic Jacobian agrees with
     a finite difference.

    pytest tests/test_differentiability.py -v
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from faro.constraints.collision import WitnessData, collision_avoidance
from faro.constraints.contact import contact_full, contact_kinematic, contact_wrench
from faro.constraints.limits import (
    joint_position_limits,
    joint_velocity_limits,
    torque_speed_limits,
)
from faro.constraints.no_slip import no_slip

# CasADi op-codes that are not twice differentiable. `copysign` and `sign` are
# piecewise constant; `fabs`, `fmin`, `fmax`, `floor` all have kinks or jumps.
NONSMOOTH_OPS = {
    ca.OP_FABS: "fabs",
    ca.OP_SIGN: "sign",
    ca.OP_FMIN: "fmin",
    ca.OP_FMAX: "fmax",
    ca.OP_FLOOR: "floor",
    ca.OP_COPYSIGN: "copysign",
}


def nonsmooth_operators(expr, args) -> list[str]:
    """Names of any non-smooth operators appearing in an expression graph."""
    f = ca.Function("probe", args, [expr])
    found = []
    for i in range(f.n_instructions()):
        op = f.instruction_id(i)
        if op in NONSMOOTH_OPS:
            found.append(NONSMOOTH_OPS[op])
    return found


def all_rows(block):
    parts = [p for p in (block.eq, block.ineq) if p is not None and p.numel()]
    return ca.vertcat(*parts)


# =============================================================================
# Symbolic builders, exercised with real symbols
# =============================================================================
def _symbolic_contact():
    R_a, p_a = ca.SX.sym("R_a", 3, 3), ca.SX.sym("p_a", 3)
    R_b, p_b = ca.SX.sym("R_b", 3, 3), ca.SX.sym("p_b", 3)
    block = contact_kinematic(R_a, p_a, R_b, p_b, ca.DM([0.085, 0.03]), ca.DM([0.15, 0.15]))
    return block, [R_a, p_a, R_b, p_b]


def _symbolic_wrench():
    f, kappa = ca.SX.sym("f", 3), ca.SX.sym("kappa", 3)
    block = contact_wrench(f, kappa, ca.DM([0.085, 0.03]), mu=0.7, mu_torsional=0.05)
    return block, [f, kappa]


def _symbolic_no_slip():
    syms = [ca.SX.sym(n, 3, 3) if n.startswith("R") else ca.SX.sym(n, 3)
            for n in ("R_a0", "p_a0", "R_b0", "p_b0", "R_a1", "p_a1", "R_b1", "p_b1")]
    return no_slip(*syms), syms


def _symbolic_collision():
    R_A, p_A = ca.SX.sym("R_A", 3, 3), ca.SX.sym("p_A", 3)
    R_B, p_B = ca.SX.sym("R_B", 3, 3), ca.SX.sym("p_B", 3)
    witness = WitnessData(
        p_A=np.array([0.1, 0.0, 0.0]), p_B=np.array([-0.1, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]), distance=0.2,
    )
    return collision_avoidance(R_A, p_A, R_B, p_B, witness), [R_A, p_A, R_B, p_B]


def _symbolic_limits_q():
    q = ca.SX.sym("q", 5)
    return joint_position_limits(q, ca.DM([-1.0] * 5), ca.DM([1.0] * 5)), [q]


def _symbolic_limits_v():
    v = ca.SX.sym("v", 5)
    return joint_velocity_limits(v, ca.DM([-2.0] * 5), ca.DM([2.0] * 5)), [v]


def _symbolic_torque_speed():
    tau, v = ca.SX.sym("tau", 3), ca.SX.sym("v", 3)
    return torque_speed_limits(tau, v, tau_max=[100.0] * 3, v_tau_max=[10.0] * 3), [tau, v]


def _symbolic_contact_full():
    R_a, p_a = ca.SX.sym("R_a", 3, 3), ca.SX.sym("p_a", 3)
    R_b, p_b = ca.SX.sym("R_b", 3, 3), ca.SX.sym("p_b", 3)
    f, kappa = ca.SX.sym("f", 3), ca.SX.sym("kappa", 3)
    block = contact_full(
        R_a, p_a, R_b, p_b, ca.DM([0.085, 0.03]), ca.DM([0.15, 0.15]),
        f, kappa, mu=0.7, mu_torsional=0.05,
    )
    return block, [R_a, p_a, R_b, p_b, f, kappa]


BUILDERS = {
    "7a+7b contact_kinematic": _symbolic_contact,
    "7c+7d contact_wrench": _symbolic_wrench,
    "7 contact_full": _symbolic_contact_full,
    "8 no_slip": _symbolic_no_slip,
    "9 collision_avoidance": _symbolic_collision,
    "13a joint_position_limits": _symbolic_limits_q,
    "13b joint_velocity_limits": _symbolic_limits_v,
    "13c torque_speed_limits": _symbolic_torque_speed,
}


# =============================================================================
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_builder_accepts_symbols(name):
    """A builder that silently required numbers could never enter an NLP."""
    block, args = BUILDERS[name]()
    rows = all_rows(block)
    assert isinstance(rows, ca.SX), f"{name} did not stay symbolic"
    assert rows.numel() > 0
    assert ca.depends_on(rows, ca.vertcat(*[ca.vec(a) for a in args])), (
        f"{name} produced an expression independent of its inputs"
    )


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_no_nonsmooth_operators_anywhere(name):
    """The mechanical version of "is it differentiable".

    This is the check that caught Eq. 7b: `^b xi_a` used `fabs` on rotation entries,
    and its kink sat exactly at R_01 = R_10 = 0 -- axis-aligned patches, which is the
    nominal pose and hence the initial guess of every solve. It now expands into
    smooth sign-case rows, exactly as Eq. 7c and Eq. 13c already did.
    """
    block, args = BUILDERS[name]()
    found = nonsmooth_operators(all_rows(block), [ca.vec(a) for a in args])
    assert not found, f"{name} contains non-smooth operators: {sorted(set(found))}"


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_first_and_second_derivatives_are_finite(name):
    """Ipopt needs a Hessian, so second derivatives must exist and be finite."""
    rng = np.random.default_rng(0)
    block, args = BUILDERS[name]()
    rows = all_rows(block)
    flat = ca.vertcat(*[ca.vec(a) for a in args])

    jac = ca.Function("jac_probe", [flat], [ca.jacobian(rows, flat)])
    hess = ca.Function("hess_probe", [flat], [ca.jacobian(ca.jacobian(ca.sum1(rows), flat), flat)])

    for _ in range(5):
        x = rng.normal(size=flat.numel())
        assert np.all(np.isfinite(np.array(jac(x)))), f"{name}: non-finite Jacobian"
        assert np.all(np.isfinite(np.array(hess(x)))), f"{name}: non-finite Hessian"


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_analytic_jacobian_matches_finite_difference(name):
    """Autodiff is only correct if it agrees with the function it differentiates."""
    rng = np.random.default_rng(1)
    block, args = BUILDERS[name]()
    rows = all_rows(block)
    flat = ca.vertcat(*[ca.vec(a) for a in args])

    fun = ca.Function("f_probe", [flat], [rows])
    jac = ca.Function("j_probe", [flat], [ca.jacobian(rows, flat)])

    x = rng.normal(size=flat.numel())
    analytic = np.array(jac(x))
    step = 1e-6
    numeric = np.zeros_like(analytic)
    for i in range(flat.numel()):
        dx = np.zeros_like(x)
        dx[i] = step
        numeric[:, i] = (np.array(fun(x + dx)) - np.array(fun(x - dx)).ravel().reshape(-1, 1)).ravel() / (2 * step)

    assert np.allclose(analytic, numeric, atol=1e-5), f"{name}: autodiff disagrees with finite differences"


# =============================================================================
# The specific regression
# =============================================================================
def test_eq_7b_is_smooth_through_the_axis_aligned_pose():
    """The exact configuration where `fabs` used to kink.

    At zero relative yaw the patches are axis-aligned, R_01 = R_10 = 0, and BOTH
    absolute values in `^b xi_a` sat on their kinks. CasADi reported a slope of 0
    there while the true one-sided slopes were -/+ xi_ay, so a solver starting from
    q_nominal would have believed Eq. 7b was insensitive to relative yaw.
    """
    theta = ca.SX.sym("theta")
    R = ca.vertcat(
        ca.horzcat(ca.cos(theta), -ca.sin(theta), 0),
        ca.horzcat(ca.sin(theta), ca.cos(theta), 0),
        ca.horzcat(0, 0, 1),
    )
    block = contact_kinematic(
        R, ca.DM([[0.05], [0.0], [0.0]]), ca.DM.eye(3), ca.DM.zeros(3, 1),
        ca.DM([0.085, 0.03]), ca.DM([0.15, 0.15]),
    )
    jac = ca.Function("j", [theta], [ca.jacobian(block.ineq, theta)])

    # Every row's slope must be continuous across theta = 0.
    left = np.array(jac(-1e-6)).ravel()
    at_zero = np.array(jac(0.0)).ravel()
    right = np.array(jac(+1e-6)).ravel()
    assert np.allclose(left, at_zero, atol=1e-6)
    assert np.allclose(right, at_zero, atol=1e-6)

    # ... and non-zero, with the RIGHT magnitude per axis. Differentiating
    # s_0 R_i0 xi_ax + s_1 R_i1 xi_ay at theta = 0, for R = Rz(theta):
    #   axis x: dR_00 = -sin = 0, dR_01 = -cos = -1  ->  slope = -/+ xi_ay = 0.03
    #   axis y: dR_10 =  cos = 1, dR_11 = -sin =  0  ->  slope = +/- xi_ax = 0.085
    # A single "patch radius" would give the same number on both axes.
    assert np.abs(at_zero[:8]).max() == pytest.approx(0.03, abs=1e-6)
    assert np.abs(at_zero[8:]).max() == pytest.approx(0.085, abs=1e-6)


def test_smooth_expansion_is_exact_not_an_approximation():
    """The sign-case rows must reproduce `|.|` exactly -- no epsilon, same feasible set.

    max over the 8 rows == |p_x| + |R_00| xi_ax + |R_01| xi_ay - xi_bx, which is what
    the fabs form computed. If this drifted, every Eq. 7b prediction would shift.
    """
    from faro.constraints.frames import rotated_half_extents

    rng = np.random.default_rng(7)
    xi_a, xi_b = ca.DM([0.085, 0.03]), ca.DM([0.15, 0.15])

    for _ in range(20):
        theta = rng.uniform(-np.pi, np.pi)
        p = rng.normal(scale=0.05, size=3)
        R = ca.DM([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        block = contact_kinematic(R, ca.DM(p.reshape(3, 1)), ca.DM.eye(3),
                                  ca.DM.zeros(3, 1), xi_a, xi_b)
        rows = np.array(ca.DM(ca.evalf(block.ineq))).ravel()

        ext = np.array(ca.DM(ca.evalf(rotated_half_extents(R, xi_a)))).ravel()
        for axis in (0, 1):
            expected = abs(p[axis]) - (float(xi_b[axis]) - ext[axis])
            assert rows[axis * 8:(axis + 1) * 8].max() == pytest.approx(expected, abs=1e-12)
