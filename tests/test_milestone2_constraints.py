"""Milestone 2 tests: the shared constraint blocks, paper Eqs. 7-13.

Strategy throughout: build a configuration that is *physically correct* by
construction, assert the constraint is satisfied, then perturb it in a specific
physical way and assert the constraint is violated -- and that the RIGHT row is the
one that goes positive. A constraint that is merely "satisfied at the right answer"
can still be wrong; it also has to reject the wrong answers, for the right reason.

    pytest tests/test_milestone2_constraints.py -v
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin
import pytest

from faro.constraints.block import ConstraintBlock, merge
from faro.constraints.collision import WitnessData, collision_avoidance, query_witness
from faro.constraints.contact import (
    contact_full,
    contact_kinematic,
    contact_wrench,
    patch_containment_is_possible,
)
from faro.constraints.dynamics_object import (
    adjoint_transpose_term,
    object_newton_euler,
    spatial_inertia,
)
from faro.constraints.dynamics_robot import centroidal_momentum_rate
from faro.constraints.frames import RX_PI, log3, relative_patch_transform, rotated_half_extents, skew
from faro.constraints.limits import (
    actuated_slice,
    joint_position_limits,
    joint_velocity_limits,
    torque_speed_limits,
)
from faro.constraints.no_slip import applies_between, no_slip


def val(expr) -> np.ndarray:
    """Numerically evaluate a CasADi expression built from constants."""
    return np.array(ca.DM(ca.evalf(expr))).ravel()


def rot_z(angle: float) -> np.ndarray:
    return pin.rpy.rpyToMatrix(0.0, 0.0, angle)


# =============================================================================
# Eq. 7a / 7b -- contact kinematics
# =============================================================================
def flat_contact(dx=0.0, dy=0.0, dz=0.0, yaw=0.0, tilt=0.0, xi_a=(0.08, 0.03), xi_b=(1.0, 1.0)):
    """Patch a resting on a horizontal patch b, with optional perturbations.

    b is a floor-like patch at the origin with outward normal +z. a is a sole-like
    patch with outward normal -z (FARO's uniform outward convention), so the two are
    anti-parallel and the Rx(pi) flip inside `relative_patch_transform` applies.
    """
    R_b, p_b = np.eye(3), np.zeros(3)
    R_a = pin.rpy.rpyToMatrix(np.pi + tilt, 0.0, 0.0) @ rot_z(yaw)
    p_a = np.array([dx, dy, dz])
    return contact_kinematic(
        ca.DM(R_a), ca.DM(p_a.reshape(3, 1)),
        ca.DM(R_b), ca.DM(p_b.reshape(3, 1)),
        ca.DM(list(xi_a)), ca.DM(list(xi_b)),
    )


def test_7a_satisfied_by_perfect_flat_contact():
    block = flat_contact()
    assert np.allclose(val(block.eq), 0.0, atol=1e-12)
    assert np.all(val(block.ineq) <= 1e-12)


def test_7a_leaves_yaw_free():
    """The paper: (7a) aligns the normals "while leaving the relative in-plane
    position and yaw angle free". A foot flat on the floor may point any direction."""
    for yaw in (0.0, 0.3, 1.2, -2.0, np.pi / 2):
        assert np.allclose(val(flat_contact(yaw=yaw).eq), 0.0, atol=1e-9), yaw


def test_7a_leaves_in_plane_position_free():
    for dx, dy in [(0.0, 0.0), (0.2, -0.4), (-0.5, 0.5)]:
        assert np.allclose(val(flat_contact(dx=dx, dy=dy).eq), 0.0, atol=1e-12)


def test_7a_detects_normal_separation():
    """p_z = 0 is zero separation: a hovering foot must be rejected."""
    eq = val(flat_contact(dz=0.05).eq)
    assert abs(eq[2]) == pytest.approx(0.05, abs=1e-12)


def test_7a_detects_tilt():
    """log3(R)_{x,y} = 0 is normal alignment: a tilted sole must be rejected."""
    eq = val(flat_contact(tilt=0.07).eq)
    assert np.linalg.norm(eq[:2]) == pytest.approx(0.07, abs=1e-6)


def test_7a_flip_convention_is_what_makes_antiparallel_normals_work():
    """Regression guard for ambiguity #8.

    FARO stores OUTWARD normals, so mating patches are anti-parallel. Eq. 7a as
    written in the paper assumes PARALLEL z-axes. Without the Rx(pi) flip, a
    perfectly flat contact would appear violated by pi radians.
    """
    R_a = pin.rpy.rpyToMatrix(np.pi, 0.0, 0.0)   # outward normal -z
    R_b = np.eye(3)                              # outward normal +z
    p = ca.DM(np.zeros((3, 1)))

    with_flip, _ = relative_patch_transform(ca.DM(R_a), p, ca.DM(R_b), p, flip_b=True)
    without, _ = relative_patch_transform(ca.DM(R_a), p, ca.DM(R_b), p, flip_b=False)

    assert np.allclose(val(log3(with_flip)), 0.0, atol=1e-12)
    assert np.linalg.norm(val(log3(without))) == pytest.approx(np.pi, abs=1e-9)


def test_7b_is_containment_not_overlap():
    """|p_xy| <= ^b xi_b - ^b xi_a forces a INSIDE b.

    At the centre the slack equals the difference of half-extents, so a patch larger
    than its support is infeasible even perfectly centred -- that is containment,
    not mere overlap.
    """
    block = flat_contact(xi_a=(0.08, 0.03), xi_b=(1.0, 1.0))
    assert val(block.ineq).max() == pytest.approx(-(1.0 - 0.08), abs=1e-9)

    too_big = flat_contact(xi_a=(1.5, 0.03), xi_b=(1.0, 1.0))
    assert val(too_big.ineq).max() > 0
    assert not patch_containment_is_possible((1.5, 0.03), (1.0, 1.0))
    assert patch_containment_is_possible((0.08, 0.03), (1.0, 1.0))


def test_7b_rejects_sliding_off_the_edge():
    """Move a until it overhangs b: the corresponding row must go positive."""
    ok = flat_contact(dx=0.5, xi_a=(0.08, 0.03), xi_b=(1.0, 1.0))
    assert val(ok.ineq).max() <= 0

    off = flat_contact(dx=0.95, xi_a=(0.08, 0.03), xi_b=(1.0, 1.0))
    values = val(off.ineq)
    assert values.max() > 0
    # It must be the +p_x row that fails, not some unrelated one.
    assert off.ineq_labels[int(np.argmax(values))].startswith("7b:+p_x")


def test_7b_rotated_half_extents_grow_under_yaw():
    """^b xi_a is a's rectangle expressed in b's frame, so yaw enlarges its footprint.

    A 45-degree-rotated rectangle needs more axis-aligned room; ignoring this (i.e.
    using a's raw half-extents) would let a rotated patch hang over the edge.
    """
    xi_a = ca.DM([0.10, 0.02])
    straight = val(rotated_half_extents(ca.DM(np.eye(3)), xi_a))
    assert straight == pytest.approx([0.10, 0.02])

    turned = val(rotated_half_extents(ca.DM(rot_z(np.pi / 2)), xi_a))
    assert turned == pytest.approx([0.02, 0.10])  # axes swap under a quarter turn

    diagonal = val(rotated_half_extents(ca.DM(rot_z(np.pi / 4)), xi_a))
    expected = (0.10 + 0.02) / np.sqrt(2.0)
    assert diagonal == pytest.approx([expected, expected], abs=1e-12)
    # For a LONG THIN rectangle a 45-degree turn shrinks the x-extent while growing
    # the y-extent -- the bounding box is not uniformly larger.
    assert diagonal[0] < straight[0]
    assert diagonal[1] > straight[1]

    # For a SQUARE patch, though, a 45-degree turn grows BOTH extents by sqrt(2).
    # This is the case that actually matters for Eq. 7b: a rotated foot needs more
    # axis-aligned room, so using a's raw half-extents would let it overhang.
    square = ca.DM([0.05, 0.05])
    square_straight = val(rotated_half_extents(ca.DM(np.eye(3)), square))
    square_turned = val(rotated_half_extents(ca.DM(rot_z(np.pi / 4)), square))
    assert square_turned == pytest.approx([0.05 * np.sqrt(2.0)] * 2, abs=1e-12)
    assert np.all(square_turned > square_straight)


def test_7a_row_count_is_three_not_two():
    """log3(R)_{x,y} = 0 is two rows PLUS p_z = 0 -- three equalities in total."""
    block = flat_contact()
    assert block.n_eq == 3
    # 7b: 8 smooth rows per axis (2 signs of p x 4 sign cases inside ^b xi_a),
    # rather than 2 rows using a non-differentiable fabs. See frames.py.
    assert block.n_ineq == 16


# =============================================================================
# Eq. 7c / 7d -- contact wrench
# =============================================================================
def wrench_block(f, kappa, xi_a=(0.10, 0.05), mu=0.7, mu_t=0.05):
    return contact_wrench(
        ca.DM(np.asarray(f, dtype=float).reshape(3, 1)),
        ca.DM(np.asarray(kappa, dtype=float).reshape(3, 1)),
        ca.DM(list(xi_a)), mu=mu, mu_torsional=mu_t,
    )


def test_7c_pure_normal_force_is_admissible():
    assert val(wrench_block([0, 0, 100.0], [0, 0, 0]).ineq).max() <= 0


def test_7c_unilaterality_rejects_pulling():
    """f_z >= 0: contacts push, never pull."""
    block = wrench_block([0, 0, -10.0], [0, 0, 0])
    values = val(block.ineq)
    assert values[block.ineq_labels.index("7c:f_z>=0")] == pytest.approx(10.0)


def test_7c_friction_is_pyramidal_not_conic():
    """The paper says "a pyramidal approximation of Coulomb friction".

    The distinguishing case: f_x = f_y = mu*f_z is INSIDE the pyramid (each
    component is individually within bound) but OUTSIDE the quadratic cone, whose
    limit would be sqrt(2)*mu*f_z. Asserting it is admissible pins the pyramid.
    """
    mu, fz = 0.7, 100.0
    at_corner = wrench_block([mu * fz, mu * fz, fz], [0, 0, 0])
    assert val(at_corner.ineq).max() <= 1e-9

    # And it must still reject a component beyond mu*f_z.
    beyond = wrench_block([1.01 * mu * fz, 0.0, fz], [0, 0, 0])
    assert val(beyond.ineq).max() > 0


def test_7d_torsional_friction_bounds_kappa_z():
    mu_t, fz = 0.05, 200.0
    assert val(wrench_block([0, 0, fz], [0, 0, mu_t * fz], mu_t=mu_t).ineq).max() <= 1e-9
    assert val(wrench_block([0, 0, fz], [0, 0, 1.5 * mu_t * fz], mu_t=mu_t).ineq).max() > 0


def test_7d_centre_of_pressure_uses_the_SWAPPED_half_extent():
    """Paper: |[kappa_x; kappa_y]| <= f_z [(^a xi_a)_y; (^a xi_a)_x].

    kappa_x is bounded by the half-extent in Y (and vice versa), because a moment
    about x comes from the normal force acting at a lever arm along y. This test
    exists because "tidying" the subscripts to match is a plausible-looking bug that
    would silently mis-size the support polygon.
    """
    xi_a, fz = (0.10, 0.02), 100.0  # deliberately very asymmetric

    # kappa_x is limited by xi_y = 0.02 -> 2.0 Nm.
    assert val(wrench_block([0, 0, fz], [1.9, 0, 0], xi_a=xi_a).ineq).max() <= 0
    assert val(wrench_block([0, 0, fz], [2.1, 0, 0], xi_a=xi_a).ineq).max() > 0

    # kappa_y is limited by xi_x = 0.10 -> 10.0 Nm, i.e. five times larger.
    assert val(wrench_block([0, 0, fz], [0, 9.9, 0], xi_a=xi_a).ineq).max() <= 0
    assert val(wrench_block([0, 0, fz], [0, 10.1, 0], xi_a=xi_a).ineq).max() > 0


def test_7d_cop_bound_scales_with_normal_force():
    """A lightly loaded contact tolerates less moment -- the bound is f_z-scaled."""
    xi_a = (0.10, 0.05)
    assert val(wrench_block([0, 0, 100.0], [0, 4.0, 0], xi_a=xi_a).ineq).max() <= 0
    assert val(wrench_block([0, 0, 10.0], [0, 4.0, 0], xi_a=xi_a).ineq).max() > 0


def test_contact_full_is_the_union_of_7a_7b_7c_7d():
    block = contact_full(
        ca.DM(pin.rpy.rpyToMatrix(np.pi, 0, 0)), ca.DM(np.zeros((3, 1))),
        ca.DM(np.eye(3)), ca.DM(np.zeros((3, 1))),
        ca.DM([0.08, 0.03]), ca.DM([1.0, 1.0]),
        ca.DM([0.0, 0.0, 50.0]), ca.DM(np.zeros((3, 1))),
        mu=0.7, mu_torsional=0.05,
    )
    # 7a -> 3 eq.  7b -> 4 ineq.  7c -> 5 ineq (|f_x|, |f_y| as 2 rows each, f_z>=0).
    # 7d -> 6 ineq (|kappa_z|, |kappa_x|, |kappa_y| as 2 rows each).
    assert block.n_eq == 3
    assert block.n_ineq == 16 + 5 + 6   # 7b (smooth expansion) + 7c + 7d
    assert np.allclose(val(block.eq), 0.0, atol=1e-12)
    assert val(block.ineq).max() <= 1e-9


# =============================================================================
# Eq. 8 / 16 -- no-slip
# =============================================================================
def test_8_satisfied_when_contact_does_not_move():
    R_a = ca.DM(pin.rpy.rpyToMatrix(np.pi, 0, 0))
    R_b, p = ca.DM(np.eye(3)), ca.DM(np.zeros((3, 1)))
    block = no_slip(R_a, p, R_b, p, R_a, p, R_b, p)
    assert np.allclose(val(block.eq), 0.0, atol=1e-12)
    assert block.n_eq == 3


def test_8_detects_in_plane_sliding():
    """A contact that translates in-plane between steps must violate Eq. 8.

    Residuals are expressed in patch b's FLIPPED frame (Rx(pi) maps y -> -y), so the
    y row carries the opposite sign to the world-frame slide. That is harmless --
    Eq. 8 is an equality constrained to zero, so only the magnitude is meaningful --
    but it is asserted explicitly here so the flip's effect is documented rather
    than discovered later as a "sign bug".
    """
    R_a = ca.DM(pin.rpy.rpyToMatrix(np.pi, 0, 0))
    R_b, p0 = ca.DM(np.eye(3)), ca.DM(np.zeros((3, 1)))
    slid = ca.DM(np.array([[0.02], [-0.01], [0.0]]))
    eq = val(no_slip(R_a, p0, R_b, p0, R_a, slid, R_b, p0).eq)
    assert eq[0] == pytest.approx(0.02)          # x is unaffected by the flip
    assert abs(eq[1]) == pytest.approx(0.01)     # y magnitude; sign flipped by Rx(pi)
    assert np.linalg.norm(eq[:2]) == pytest.approx(np.hypot(0.02, 0.01))


def test_8_detects_spinning_about_the_normal():
    R_b, p = ca.DM(np.eye(3)), ca.DM(np.zeros((3, 1)))
    R_a = ca.DM(pin.rpy.rpyToMatrix(np.pi, 0, 0))
    R_a_spun = ca.DM(pin.rpy.rpyToMatrix(np.pi, 0, 0) @ rot_z(0.15))
    eq = val(no_slip(R_a, p, R_b, p, R_a_spun, p, R_b, p).eq)
    assert abs(eq[2]) == pytest.approx(0.15, abs=1e-9)


def test_8_is_relative_so_a_carried_contact_does_not_slip():
    """A hand stuck to a box satisfies Eq. 8 while both sweep through space.

    Eq. 8 constrains the RELATIVE pose, so rigidly transporting both patches
    together must leave the residual at zero.
    """
    R_a0 = pin.rpy.rpyToMatrix(np.pi, 0, 0)
    R_b0, p0 = np.eye(3), np.zeros((3, 1))

    transport_R = pin.rpy.rpyToMatrix(0.2, -0.1, 0.4)
    offset = np.array([[0.3], [-0.2], [0.7]])

    block = no_slip(
        ca.DM(R_a0), ca.DM(p0), ca.DM(R_b0), ca.DM(p0),
        ca.DM(transport_R @ R_a0), ca.DM(offset), ca.DM(transport_R @ R_b0), ca.DM(offset),
    )
    assert np.allclose(val(block.eq), 0.0, atol=1e-9)


def test_16_persistence_and_release_trigger_no_slip():
    """Eq. 16a persistent contact, 16b release."""
    assert applies_between("floor", "floor")      # (16a)
    assert applies_between("floor", None)         # (16b) no sliding while separating


def test_16_acquisition_and_free_do_not_trigger_no_slip():
    """Acquiring a contact must NOT be constrained.

    There is no previous contact location to preserve, and constraining one would
    forbid the robot from choosing where to place a new foot.
    """
    assert not applies_between(None, "floor")     # acquisition
    assert not applies_between(None, None)        # free throughout
    assert not applies_between("floor", "tabletop")  # switched support entirely


# =============================================================================
# Eq. 9 -- collision
# =============================================================================
def test_9_separated_bodies_satisfy_the_constraint():
    witness = WitnessData(
        p_A=np.array([-0.5, 0.0, 0.0]), p_B=np.array([0.5, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]), distance=2.0,
    )
    block = collision_avoidance(
        ca.DM(np.eye(3)), ca.DM(np.array([[3.0], [0.0], [0.0]])),
        ca.DM(np.eye(3)), ca.DM(np.zeros((3, 1))),
        witness,
    )
    # sd = 1 . ((3 - 0.5) - 0.5) = 2.0  -> ineq = -2.0
    assert val(block.ineq)[0] == pytest.approx(-2.0)


def test_9_penetration_violates_the_constraint():
    witness = WitnessData(
        p_A=np.array([-0.5, 0.0, 0.0]), p_B=np.array([0.5, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]), distance=0.0,
    )
    block = collision_avoidance(
        ca.DM(np.eye(3)), ca.DM(np.array([[0.5], [0.0], [0.0]])),
        ca.DM(np.eye(3)), ca.DM(np.zeros((3, 1))), witness,
    )
    assert val(block.ineq)[0] > 0


def test_9_margin_shifts_the_boundary():
    witness = WitnessData(
        p_A=np.zeros(3), p_B=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]), distance=0.05,
    )
    args = (ca.DM(np.eye(3)), ca.DM(np.array([[0.0], [0.0], [0.05]])),
            ca.DM(np.eye(3)), ca.DM(np.zeros((3, 1))), witness)
    assert val(collision_avoidance(*args, margin=0.0).ineq)[0] == pytest.approx(-0.05)
    assert val(collision_avoidance(*args, margin=0.10).ineq)[0] == pytest.approx(0.05)


def test_9_rejects_non_unit_normal():
    """Eq. 9 projects onto n; a non-unit normal silently rescales the distance."""
    with pytest.raises(ValueError, match="unit vector"):
        WitnessData(np.zeros(3), np.zeros(3), np.array([0.0, 0.0, 2.0]), 0.0)


def test_9_witness_query_against_coal_matches_known_geometry():
    """End-to-end GJK query: two unit boxes 3 m apart have a 2 m gap."""
    try:
        import coal
    except ImportError:
        import hppfcl as coal

    tf_A, tf_B = coal.Transform3s(), coal.Transform3s()
    tf_A.setTranslation(np.array([3.0, 0.0, 0.0]))
    witness = query_witness(coal.Box(1.0, 1.0, 1.0), tf_A, coal.Box(1.0, 1.0, 1.0), tf_B)

    assert witness.distance == pytest.approx(2.0, abs=1e-6)
    assert np.linalg.norm(witness.normal) == pytest.approx(1.0)

    # And Eq. 9 built from it must reproduce that same distance.
    block = collision_avoidance(
        ca.DM(np.eye(3)), ca.DM(np.array([[3.0], [0.0], [0.0]])),
        ca.DM(np.eye(3)), ca.DM(np.zeros((3, 1))), witness,
    )
    assert val(block.ineq)[0] == pytest.approx(-2.0, abs=1e-6)


# =============================================================================
# Eq. 10 -- robot dynamics
# =============================================================================
def test_10b_momentum_rate_in_freefall_is_pure_gravity():
    """No contacts -> hdot = (0, 0, -mg, 0, 0, 0)."""
    hdot = val(centroidal_momentum_rate([], [], [], ca.DM(np.zeros((3, 1))), mass=35.0))
    assert hdot[:3] == pytest.approx([0.0, 0.0, -35.0 * 9.81])
    assert hdot[3:] == pytest.approx([0.0, 0.0, 0.0])


def test_10b_contact_force_supporting_weight_gives_zero_linear_rate():
    mass = 35.0
    f = ca.DM([[0.0], [0.0], [mass * 9.81]])
    hdot = val(centroidal_momentum_rate(
        [f], [ca.DM(np.zeros((3, 1)))], [ca.DM(np.zeros((3, 1)))],
        ca.DM(np.zeros((3, 1))), mass=mass,
    ))
    assert hdot[:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_10b_moment_arm_is_measured_from_the_com():
    """The angular row uses (p_e - c), not p_e.

    A vertical force offset in +x from the CoM produces a moment about -y. Using the
    world origin instead of the CoM would give a different answer whenever the robot
    is not standing at the origin -- which is essentially always.
    """
    f = ca.DM([[0.0], [0.0], [100.0]])
    p = ca.DM([[0.2], [0.0], [0.0]])
    com = ca.DM([[0.5], [0.0], [1.0]])
    hdot = val(centroidal_momentum_rate([f], [ca.DM(np.zeros((3, 1)))], [p], com, mass=0.0))
    # (p - c) x f = (-0.3, 0, -1) x (0, 0, 100) = (0*100 - (-1)*0, (-1)*0 - (-0.3)*100, 0)
    assert hdot[3:] == pytest.approx([0.0, 30.0, 0.0], abs=1e-9)


def test_10b_contact_moment_adds_directly():
    kappa = ca.DM([[1.0], [2.0], [3.0]])
    hdot = val(centroidal_momentum_rate(
        [ca.DM(np.zeros((3, 1)))], [kappa], [ca.DM(np.zeros((3, 1)))],
        ca.DM(np.zeros((3, 1))), mass=0.0,
    ))
    assert hdot[3:] == pytest.approx([1.0, 2.0, 3.0])


# =============================================================================
# Eq. 11 -- object dynamics
# =============================================================================
def test_11_spatial_inertia_is_angular_first():
    """G = diag(I, m*Id3) with ANGULAR first, matching the twist ordering.

    Pinocchio's own spatial types are linear-first, so a silent block swap here
    would give a box the wrong rotational behaviour while still looking plausible.
    """
    I = np.diag([0.1, 0.2, 0.3])
    G = spatial_inertia(2.0, I)
    assert np.allclose(G[:3, :3], I)
    assert np.allclose(G[3:, 3:], 2.0 * np.eye(3))
    assert np.allclose(G[:3, 3:], 0.0) and np.allclose(G[3:, :3], 0.0)


def test_11b_static_object_needs_wrench_equal_to_inertia_times_acceleration():
    """At rest (V = 0) the Coriolis term vanishes and Eq. 11b reduces to W = G Vdot."""
    G = spatial_inertia(2.0, np.diag([0.1, 0.2, 0.3]))
    Vdot = ca.DM(np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]]))
    W = ca.DM(G) @ Vdot
    block = object_newton_euler(ca.DM(np.zeros((6, 1))), Vdot, W, G)
    assert np.allclose(val(block.eq), 0.0, atol=1e-12)


def test_11b_coriolis_term_is_nonzero_for_a_spinning_object():
    """-[ad_V]^T G V must not vanish when the object spins about a non-principal axis.

    This term is what makes a tumbling box precess; dropping it is invisible at low
    speed and wrong for the paper's dynamic tasks (Toss to Table, Juggle).
    """
    G = spatial_inertia(2.0, np.diag([0.1, 0.5, 0.9]))  # asymmetric on purpose
    V = ca.DM(np.array([[1.0], [2.0], [3.0], [0.0], [0.0], [0.0]]))
    assert np.linalg.norm(val(adjoint_transpose_term(V, G))) > 1e-6


def test_11b_coriolis_vanishes_for_rotation_about_a_principal_axis():
    """A body spinning about a principal axis of a symmetric inertia does not precess."""
    G = spatial_inertia(2.0, np.diag([0.4, 0.4, 0.4]))
    V = ca.DM(np.array([[0.0], [0.0], [5.0], [0.0], [0.0], [0.0]]))
    assert np.allclose(val(adjoint_transpose_term(V, G)), 0.0, atol=1e-12)


def test_skew_matches_cross_product():
    a, b = np.array([1.0, -2.0, 3.0]), np.array([0.5, 4.0, -1.0])
    got = val(skew(ca.DM(a.reshape(3, 1))) @ ca.DM(b.reshape(3, 1)))
    assert got == pytest.approx(np.cross(a, b))


# =============================================================================
# Eqs. 12 / 13 -- limits
# =============================================================================
def test_13a_within_limits_is_satisfied_and_outside_is_not():
    q = ca.DM(np.array([[0.0], [0.5]]))
    block = joint_position_limits(q, [-1.0, -1.0], [1.0, 1.0])
    assert val(block.ineq).max() <= 0

    over = joint_position_limits(ca.DM(np.array([[0.0], [1.5]])), [-1.0, -1.0], [1.0, 1.0])
    values = val(over.ineq)
    assert values.max() == pytest.approx(0.5)
    assert over.ineq_labels[int(np.argmax(values))] == "13a:q<=max[1]"


def test_13b_velocity_limits():
    block = joint_velocity_limits(ca.DM(np.array([[2.0]])), [-1.0], [1.0])
    assert val(block.ineq).max() == pytest.approx(1.0)


def test_13c_full_torque_available_at_zero_speed():
    """The envelope gives exactly tau_max when the joint is at rest."""
    block = torque_speed_limits(
        ca.DM([[100.0]]), ca.DM([[0.0]]), tau_max=[100.0], v_tau_max=[10.0]
    )
    assert val(block.ineq).max() == pytest.approx(0.0, abs=1e-12)


def test_13c_torque_budget_shrinks_affinely_with_speed():
    """|tau| + (tau_max/v_tau_max)|v| <= tau_max.

    At half of v_tau_max only half the torque is available -- the defining property
    of the envelope.
    """
    tau_max, v_max = 100.0, 10.0
    ok = torque_speed_limits(ca.DM([[50.0]]), ca.DM([[5.0]]), tau_max=[tau_max], v_tau_max=[v_max])
    assert val(ok.ineq).max() == pytest.approx(0.0, abs=1e-12)

    over = torque_speed_limits(ca.DM([[60.0]]), ca.DM([[5.0]]), tau_max=[tau_max], v_tau_max=[v_max])
    assert val(over.ineq).max() == pytest.approx(10.0)


def test_13c_is_symmetric_in_both_signs():
    """All four sign combinations of (tau, v) are covered, so braking is bounded too."""
    for tau, v in [(80.0, 5.0), (-80.0, 5.0), (80.0, -5.0), (-80.0, -5.0)]:
        block = torque_speed_limits(
            ca.DM([[tau]]), ca.DM([[v]]), tau_max=[100.0], v_tau_max=[10.0]
        )
        assert val(block.ineq).max() == pytest.approx(30.0)


def test_13c_uses_smooth_linear_rows_not_fabs():
    """4 rows per joint, one per sign combination.

    fabs() is non-differentiable at zero, which is exactly where joints sit at rest,
    so the absolute values are expanded into linear rows instead.
    """
    block = torque_speed_limits(
        ca.DM(np.zeros((3, 1))), ca.DM(np.zeros((3, 1))),
        tau_max=[1.0, 1.0, 1.0], v_tau_max=[1.0, 1.0, 1.0],
    )
    assert block.n_ineq == 4 * 3


def test_actuated_slice_skips_the_floating_base():
    """q_j / v_j are the ACTUATED joints; the 6 base DoF have no limits."""
    from faro.robots.robot_model import RobotModel

    robot = RobotModel.from_config("g1")
    assert actuated_slice(robot.model) == slice(7, 36)
    assert actuated_slice(robot.model, velocity=True) == slice(6, 35)
    # 29 actuated joints either way.
    assert len(range(*actuated_slice(robot.model).indices(robot.nq))) == 29


# =============================================================================
# Block plumbing
# =============================================================================
def test_block_rejects_mismatched_labels():
    """Labels must line up with rows, or debug output misattributes violations."""
    with pytest.raises(ValueError, match="labels"):
        ConstraintBlock(name="x", eq=ca.SX.zeros(3), eq_labels=["only-one"])


def test_merge_concatenates_and_namespaces_labels():
    a = ConstraintBlock("a", eq=ca.SX.zeros(2), eq_labels=["p", "q"])
    b = ConstraintBlock("b", ineq=ca.SX.zeros(1), ineq_labels=["r"])
    merged = merge("both", [a, b])
    assert merged.n_eq == 2 and merged.n_ineq == 1
    assert merged.eq_labels == ["a/p", "a/q"]
    assert merged.ineq_labels == ["b/r"]


def test_rx_pi_is_a_proper_rotation():
    R = np.array(ca.DM(RX_PI))
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(R @ np.array([0, 0, 1.0]), [0, 0, -1.0])


# =============================================================================
# Eq. 12 -- the actuated-torque DEFINITION
#
# This had no coverage at all until a review noticed it. It is not a limit, it is
# what 13c consumes: get it wrong and 13c rejects perfectly good motions, or admits
# impossible ones, with nothing pointing at the cause.
# =============================================================================
def test_12_free_flight_torque_is_just_the_inverse_dynamics():
    """With no contacts, tau_j is simply S (M vdot + b)."""
    from faro.constraints.limits import actuated_torque

    nv = 9
    M_vdot = np.arange(1.0, nv + 1.0).reshape(-1, 1)
    b = np.full((nv, 1), 0.5)
    tau = actuated_torque(ca.DM(M_vdot), ca.DM(b), [], [], slice(6, nv))

    assert val(tau) == pytest.approx((M_vdot + b).ravel()[6:nv])


def test_12_selection_drops_the_six_floating_base_rows():
    """S exists because the base is UNACTUATED -- no motor produces those 6 rows.

    Including them would let the optimizer "pay" for base acceleration with torque
    that no hardware can supply, which is the whole underactuation the paper is about.
    """
    from faro.constraints.limits import actuated_slice, actuated_torque
    from faro.robots.robot_model import RobotModel

    model = RobotModel.from_config("g1").model
    nv = model.nv
    total = np.arange(float(nv)).reshape(-1, 1)
    tau = actuated_torque(ca.DM(total), ca.DM(np.zeros((nv, 1))), [], [],
                          actuated_slice(model, velocity=True))

    assert tau.shape[0] == nv - 6 == 29
    assert val(tau) == pytest.approx(total.ravel()[6:])


def test_12_contact_wrench_offloads_the_actuators():
    """THE sign that matters: -J^T lambda, because contact carries load for you.

    Standing still, the ground holds the robot up and the legs need less torque, not
    more. Flip this sign and the apparent torque roughly doubles, so Eq. 13c starts
    rejecting motions that are perfectly feasible -- and the failure appears in the
    TO, five modules away from the cause.
    """
    from faro.constraints.limits import actuated_torque

    nv = 8
    M_vdot = np.zeros((nv, 1))
    J = np.zeros((6, nv))
    J[2, 6] = 1.0                       # joint 6 feels the vertical contact force
    wrench = np.zeros((6, 1))
    wrench[2] = 100.0                   # 100 N pushing up

    tau = actuated_torque(ca.DM(M_vdot), ca.DM(M_vdot), [ca.DM(J)], [ca.DM(wrench)],
                          slice(6, nv))
    assert val(tau)[0] == pytest.approx(-100.0), "contact must SUBTRACT from tau"


def test_12_multiple_end_effectors_sum():
    """sum_e J_e^T lambda_e -- every contact contributes."""
    from faro.constraints.limits import actuated_torque

    nv = 7
    J1, J2 = np.zeros((6, nv)), np.zeros((6, nv))
    J1[2, 6] = 1.0
    J2[2, 6] = 2.0
    w = np.zeros((6, 1))
    w[2] = 10.0

    tau = actuated_torque(ca.DM(np.zeros((nv, 1))), ca.DM(np.zeros((nv, 1))),
                          [ca.DM(J1), ca.DM(J2)], [ca.DM(w), ca.DM(w)], slice(6, nv))
    assert val(tau)[0] == pytest.approx(-(10.0 + 20.0))


def test_12_feeds_13c_end_to_end():
    """Eq. 12 defines the tau that Eq. 13c bounds; check they compose.

    A contact that offloads the joint must WIDEN the admissible speed range, since
    13c's envelope trades torque against speed.
    """
    from faro.constraints.limits import actuated_torque, torque_speed_limits

    nv = 7
    M_vdot = np.zeros((nv, 1))
    M_vdot[6] = 80.0                    # the joint would need 80 Nm unaided
    J = np.zeros((6, nv))
    J[2, 6] = 1.0
    w = np.zeros((6, 1))
    w[2] = 60.0                         # a contact supplying 60 Nm of it

    unaided = actuated_torque(ca.DM(M_vdot), ca.DM(np.zeros((nv, 1))), [], [], slice(6, nv))
    aided = actuated_torque(ca.DM(M_vdot), ca.DM(np.zeros((nv, 1))),
                            [ca.DM(J)], [ca.DM(w)], slice(6, nv))
    assert val(unaided)[0] == pytest.approx(80.0)
    assert val(aided)[0] == pytest.approx(20.0)

    speed = ca.DM([[5.0]])
    worst = lambda tau: max(val(torque_speed_limits(
        tau, speed, tau_max=[100.0], v_tau_max=[10.0]).ineq))
    assert worst(unaided) > 0.0, "80 Nm at half speed must exceed the envelope"
    assert worst(aided) < 0.0, "with the contact offloading it, the same motion fits"
