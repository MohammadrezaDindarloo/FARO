"""Symbolic SE(3) helpers and the patch-frame convention.

Small, boring, and load-bearing: every contact constraint is built from these, so
a sign error here would propagate silently into Eqs. 7, 8, 14, 15 and 17 at once.

THE CONVENTION QUESTION (paper Section II-C 1)
----------------------------------------------
The paper states only that "the patch normals are aligned with their local z-axes",
and then writes Eq. 7a as

    log3(R)_{x,y} = 0,   p_z = 0

where R is the rotation of patch `a` relative to patch `b`. `log3(R)_{x,y} = 0`
means R is a *pure yaw* -- so under the paper's convention two patches in contact
have PARALLEL z-axes.

FARO (this repo) instead defines every patch's +z as its OUTWARD normal, pointing
away from the body it belongs to (see `faro.core.patches`). That is uniform,
physically meaningful, and directly visible in the Meshcat viewer. Under it, two
patches in face-to-face contact have ANTI-parallel z-axes.

The two are reconciled by one fixed rotation: flip patch b's frame by pi about its
x-axis before taking the relative rotation. `relative_patch_transform` does exactly
that, and it is the only place in the codebase where the choice is encoded.

Why keep the outward convention rather than adopting the paper's implicitly?
Because the paper's convention is not uniform across patches. Eq. 7b's containment
condition (a inside b) makes the pair asymmetric: an end-effector patch is always
`a`, a support surface always `b`. A box face is `b` when a hand presses it, but
the box's bottom is `a` when it rests on the floor -- so the same object would need
opposite z conventions on different faces. Outward normals stay uniform, and the
asymmetry is handled once, here.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio.casadi as cpin

# Rotation by pi about x. Maps z -> -z and y -> -y, so it turns an outward normal
# into an inward one while keeping the frame right-handed (det = +1).
RX_PI = ca.DM([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def relative_transform(R_a, p_a, R_b, p_b):
    """Pose of frame a relative to frame b, expressed in frame b.

    Returns (R, p) with

        R = R_b^T R_a,      p = R_b^T (p_a - p_b)

    which is the paper's "relative position and rotation of patch a with respect to
    patch b, expressed in frame b" (Section II-C 1).
    """
    R = R_b.T @ R_a
    p = R_b.T @ (p_a - p_b)
    return R, p


def relative_patch_transform(R_a, p_a, R_b, p_b, flip_b: bool = True):
    """Relative transform between two CONTACT PATCHES, in the paper's convention.

    With FARO's outward-normal convention, mating patches have anti-parallel z-axes,
    so b's frame is flipped by pi about x first. After the flip, a correct contact
    gives a pure-yaw R and p_z = 0 -- exactly what Eq. 7a tests.

    Set `flip_b=False` to disable the flip, i.e. to interpret the patch frames
    directly in the paper's implicit parallel-z convention.
    """
    R_b_eff = R_b @ RX_PI if flip_b else R_b
    return relative_transform(R_a, p_a, R_b_eff, p_b)


def as_sx(value):
    """Promote a numeric value to constant SX, leaving symbols untouched.

    Every `pinocchio.casadi` entry point binds only the SX overload and rejects DM
    outright. Promoting here lets one implementation be evaluated numerically in a
    test and symbolically in an NLP, with no second copy to keep in sync -- which is
    the only reason the tests can cross-check against numeric Pinocchio at all.
    """
    if isinstance(value, (ca.DM, np.ndarray, list, float, int)):
        return ca.SX(ca.DM(value))
    return value


def log3(R):
    """SO(3) logarithm, the paper's log3(.), as a differentiable CasADi expression.

    Provided by pinocchio.casadi, so it matches the numeric `pin.log3` exactly --
    which is what lets the tests cross-check symbolic against numeric.

    Smooth everywhere except at a rotation angle of pi, where the axis is no longer
    unique. Eq. 7a only reaches that if two patches are face-on-backwards, which is
    far outside any feasible contact -- but Eq. 8's relative-yaw row could approach it
    for a contact that spins half a turn between timesteps. Worth watching at
    Milestone 4 rather than guarding blindly here.
    """
    return cpin.log3(as_sx(R))


def normal_alignment_rows(R, form: str = "log3"):
    """Eq. 7a's normal-alignment condition, in either of two EQUIVALENT forms.

    The paper writes `log3(R)_{x,y} = 0`: the relative rotation's axis lies along z,
    so the patch normals coincide and only the in-plane yaw is free. That is the
    right condition, and `form="log3"` is it, verbatim.

    THE PROBLEM WITH WRITING IT THAT WAY IN A SOLVER -- MEASURED, NOT EXPLAINED
    ---------------------------------------------------------------------------
    Inside Eq. 14 the log3 form makes Ipopt fail with `Invalid_Number_Detected`: a
    NaN in the Lagrangian Hessian, reported at the box and base quaternion entries.
    What is established:

      * it is reproducible -- the grasp and place modes and the grasp->front edge all
        fail with it and all solve without it (tests/test_mode_edge.py);
      * it is not an artifact of row order: shuffling the five contact blocks fails
        under the log3 form in every ordering tried and succeeds under this one in
        every ordering tried;
      * where both forms converge they reach the same optimum to 1e-6, which is the
        equivalence showing up end to end.

    An attempt to localize it further, by putting the log3 form on one contact pair at
    a time, appeared to blame the pairs already aligned at the nominal pose -- and
    then did not replicate under a different pair ordering. So no per-pair attribution
    is claimed.

    What is NOT established is the mechanism. `log3` was probed in isolation at
    theta = 0, at theta = pi, and on the non-orthogonal matrices a non-unit quaternion
    produces, and its value, Jacobian and Hessian were finite in every case; random
    probing of the assembled rows and of the full Lagrangian Hessian did not
    reproduce it either. So the NaN needs the whole expression graph and the solver's
    own trial points, and the honest statement is that the log3 form is empirically
    fragile here for a reason not yet pinned down. Recorded as an open question in
    docs/ambiguities.md #23 rather than written up as a theory.

    THE EQUIVALENT SMOOTH FORM
    --------------------------
    "The patch normals are aligned with their local z-axes" (Section II-C 1), so
    alignment is a statement about ONE COLUMN of R -- its third:

        R[0,2] = 0,   R[1,2] = 0,   and   R[2,2] >= 0.

    The two equalities say the a-frame's z-axis has no component along b's x or y;
    orthonormality then forces R[2,2] = +-1, and the inequality picks +1 (aligned)
    over -1 (back-to-back). Same feasible set as the log3 form -- `tests/
    test_mode_edge.py` checks that on random rotations -- but every row is LINEAR in
    the entries of R, so there is nothing in it that can produce a NaN anywhere in
    SO(3) -- which is why it is the default for the solver stages even without a
    diagnosis of what the log3 form does.

    The cost is one extra row and a departure from the paper's literal notation, which
    is why both forms are kept and the choice is explicit. See docs/ambiguities.md #23.

    Returns `(eq_rows, eq_labels, ineq_rows, ineq_labels)`; the log3 form returns
    empty inequality lists.
    """
    if form == "log3":
        omega = log3(R)
        return [omega[0], omega[1]], ["7a:log3(R)_x", "7a:log3(R)_y"], [], []
    if form == "column":
        # `-R[2,2] <= 0` in FARO's `ineq <= 0` convention.
        return (
            [R[0, 2], R[1, 2]], ["7a:R_xz", "7a:R_yz"],
            [-R[2, 2]], ["7a:R_zz>=0"],
        )
    raise ValueError(f"unknown alignment form {form!r}; expected 'log3' or 'column'")


def rotated_half_extents(R, xi_a):
    """Half-extents of patch a expressed in frame b -- the paper's `^b xi_a` (Eq. 7b).

    A rectangle with half-extents (xi_x, xi_y) in frame a, viewed from frame b, is
    no longer axis-aligned. Its half-width along b's x-axis is the support of the
    rotated rectangle in that direction:

        ^b xi_a,x = |R_00| xi_ax + |R_01| xi_ay
        ^b xi_a,y = |R_10| xi_ax + |R_11| xi_ay

    (the maximum of |(R c)_x| over the four corners c = (+-xi_ax, +-xi_ay, 0)).

    This is the tight axis-aligned bound, and it reduces to the familiar
    xi_x|cos t| + xi_y|sin t| when R is a yaw by t -- the case that actually occurs
    once Eq. 7a holds. Written generally so the expression stays valid mid-solve,
    when Eq. 7a is not yet satisfied and R is some arbitrary rotation.

    NOT DIFFERENTIABLE -- see `rotated_half_extent_rows`
    ------------------------------------------------------
    The `|.|` here is a genuine kink, and it sits exactly where it hurts most: when
    the patches are axis-aligned, R[0,1] = R[1,0] = 0, so BOTH absolute values are at
    their kinks. That is the nominal pose, i.e. the initial guess of every solve.
    CasADi reports a derivative of 0 there while the true one-sided slopes are -/+
    xi_ay, so a solver would linearize Eq. 7b as insensitive to relative yaw when it
    is not.

    Use this function for EVALUATION and diagnostics. For anything handed to a
    solver, use `rotated_half_extent_rows`, which is exact and smooth.
    """
    xi_ax, xi_ay = xi_a[0], xi_a[1]
    ext_x = ca.fabs(R[0, 0]) * xi_ax + ca.fabs(R[0, 1]) * xi_ay
    ext_y = ca.fabs(R[1, 0]) * xi_ax + ca.fabs(R[1, 1]) * xi_ay
    return ca.vertcat(ext_x, ext_y)


# The four sign combinations of (R-entry, R-entry) used to expand `^b xi_a`, and the
# two signs of p. Their product gives 8 smooth rows per axis.
_SIGNS = ((+1.0, +1.0), (+1.0, -1.0), (-1.0, +1.0), (-1.0, -1.0))


def rotated_half_extent_rows(R, p_rel, xi_a, xi_b, axis: int):
    """Eq. 7b along one axis, as smooth rows -- the solver-facing form.

        |p_i| <= (xi_b)_i - ( |R_i0| xi_ax + |R_i1| xi_ay )

    Every `|x|` is expanded into its sign cases rather than written with `fabs`,
    exactly as Eq. 7c's friction rows and Eq. 13c's torque-speed envelope already
    are. Since `max_s (s x) = |x|` and the three absolute values are independent,

        |p_i| + |R_i0| xi_ax + |R_i1| xi_ay - (xi_b)_i <= 0
          <=>  s_p p_i + s_0 R_i0 xi_ax + s_1 R_i1 xi_ay - (xi_b)_i <= 0
               for all 8 sign combinations,

    which is EXACT -- not a smoothing approximation, so there is no epsilon to tune
    and the feasible set is unchanged. Each row is bilinear in (R, xi) and linear in
    p, hence twice differentiable everywhere, which is what Ipopt requires.

    Costs 8 rows per axis instead of 2. They are cheap linear rows, and the
    alternative is a kink sitting on the initial guess of every solve.
    """
    rows, labels = [], []
    axis_name = "xy"[axis]
    for s_p in (+1.0, -1.0):
        for s_0, s_1 in _SIGNS:
            rows.append(
                s_p * p_rel[axis]
                + s_0 * R[axis, 0] * xi_a[0]
                + s_1 * R[axis, 1] * xi_a[1]
                - xi_b[axis]
            )
            labels.append(
                f"7b:{'+' if s_p > 0 else '-'}p_{axis_name}"
                f"[{'+' if s_0 > 0 else '-'}{'+' if s_1 > 0 else '-'}]"
            )
    return ca.vertcat(*rows), labels


def se3_to_rt(placement) -> tuple[np.ndarray, np.ndarray]:
    """Split a numeric pinocchio.SE3 into (rotation, translation) arrays."""
    return np.asarray(placement.rotation), np.asarray(placement.translation)


# =============================================================================
# 6-VECTOR ORDERING -- the one convention this codebase does NOT unify, on purpose
# =============================================================================
# Two orderings are in use, because the paper itself uses both and each half is
# checked against a different reference:
#
#   LINEAR-FIRST  [force; moment] / [linear; angular]
#       Eq. 10b as the paper writes it (`hdot = [mg + sum f ; sum (p-c) x f + kappa]`),
#       and every Pinocchio spatial quantity -- `Ag` from `computeCentroidalMap`,
#       `pin.Force`, and the frame Jacobians whose transpose Eq. 12 uses. Verified:
#       `Ag @ v` for a pure +z base translation lands in rows 0:3.
#
#   ANGULAR-FIRST [moment; force] / [angular; linear]
#       Eq. 11b, which the paper takes from Lynch & Park (its ref [23]); `G = diag(I,
#       m I3)`, `ad_V = [[w] 0; [v] [w]]` and `W = G Vdot - [ad_V]^T G V` are only
#       correct together in that ordering. Verified: the term reduces to w x (I w).
#
# Rewriting either to match the other would mean silently departing from its source,
# so instead the boundary is made explicit. Anything crossing it -- and a contact
# wrench DOES cross it, entering the robot's Eq. 10b and the object's Eq. 11b with
# opposite signs -- must go through `swap_spatial_ordering`. Blocks swapped by
# accident produce a plausible-looking answer with torques where forces belong.
def swap_spatial_ordering(w):
    """Convert a 6-vector between [linear; angular] and [angular; linear].

    The map is its own inverse, so one function serves both directions; name the
    call site, not the function, when it matters:

        W_object = swap_spatial_ordering(lambda_e)   # Eq. 10b wrench -> Eq. 11b wrench
    """
    if w.shape[0] != 6:
        raise ValueError(f"expected a 6-vector, got shape {w.shape}")
    return ca.vertcat(w[3:6], w[0:3])


def skew(v):
    """3x3 skew-symmetric matrix, so that skew(a) @ b == cross(a, b)."""
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0),
    )
