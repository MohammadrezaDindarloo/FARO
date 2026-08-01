"""Patch-to-patch contact constraints -- paper Eq. 7 (Section II-C 1).

Verbatim from the paper:

    Let p, R, f, and kappa denote the relative position, rotation, force, and moment
    of contact patch a with respect to contact patch b, expressed in frame b. The
    patch normals are aligned with their local z-axes. The vector xi denotes the
    half-extents of a rectangular patch, so ^b xi_a denotes the half-extents of
    patch a expressed in frame b.

        log3(R)_{x,y} = 0,   p_z = 0                                        (7a)
        |p_{x,y}| <= (^b xi_b)_{x,y} - (^b xi_a)_{x,y}                      (7b)
        |f_{x,y}| <= mu f_z,   f_z >= 0                                     (7c)
        |kappa_z| <= mu_r f_z,  |[kappa_x; kappa_y]| <= f_z [(^a xi_a)_y;
                                                             (^a xi_a)_x]   (7d)

Which stages use what:
  * Eq. 14 (mode/edge) and Eq. 15 (KSO) use only (7a) and (7b) -- there are no
    force variables in a kinematic problem.
  * Eq. 17 (TO) uses all of (7).

Hence the split into `contact_kinematic` and `contact_wrench` below.

ASYMMETRY: (7b) is a CONTAINMENT condition -- patch a must fit inside patch b. So
`a` is the smaller, "contacting" patch (a sole, a palm, a box bottom) and `b` is
the larger supporting surface (floor, tabletop, box face). The paper never says
this outright; it follows from the sign of the right-hand side. `contact_kinematic`
warns when the ordering makes the bound infeasible from the outset.
"""

from __future__ import annotations

import casadi as ca

from faro.constraints.block import ConstraintBlock
from faro.constraints.frames import (
    log3,
    relative_patch_transform,
    rotated_half_extent_rows,
)


def contact_kinematic(
    R_a, p_a, R_b, p_b, xi_a, xi_b, *, name: str = "contact", flip_b: bool = True
) -> ConstraintBlock:
    """Eqs. 7a + 7b: the purely geometric part of patch-to-patch contact.

    Parameters
    ----------
    R_a, p_a : world rotation (3x3) and position (3x1) of patch a's frame.
    R_b, p_b : same for patch b.
    xi_a, xi_b : half-extents (2-vectors) of each patch, in its own frame.
    flip_b : apply FARO's outward-normal reconciliation (see `frames.py`).

    Returns
    -------
    ConstraintBlock with 3 equalities (7a) and 16 inequalities (7b).

    Eq. 7a contributes THREE equalities, not two: `log3(R)_{x,y} = 0` is two rows
    (normal alignment, leaving yaw free) plus `p_z = 0` (zero separation).
    Eq. 7b's absolute values expand to EIGHT smooth rows per axis (two signs of
    p times four sign combinations inside `^b xi_a`), hence sixteen inequalities --
    see `rotated_half_extent_rows` for why `fabs` is not used.
    """
    R_rel, p_rel = relative_patch_transform(R_a, p_a, R_b, p_b, flip_b=flip_b)

    # --- (7a) -----------------------------------------------------------------
    # log3(R)_{x,y} = 0 forces the relative rotation to be a pure yaw: the patch
    # normals align while the in-plane yaw stays FREE (an unconstrained foot can
    # still rotate about the contact normal). p_z = 0 is zero normal separation.
    omega = log3(R_rel)
    eq = ca.vertcat(omega[0], omega[1], p_rel[2])
    eq_labels = ["7a:log3(R)_x", "7a:log3(R)_y", "7a:p_z"]

    # --- (7b) -----------------------------------------------------------------
    # |p_{x,y}| <= ^b xi_b - ^b xi_a.  ^b xi_b is just b's own half-extents (it is
    # already expressed in its own frame); ^b xi_a needs a's rectangle rotated into
    # b's frame, which is what `rotated_half_extents` computes.
    # Both `|p_i|` and the `|R_ij|` inside `^b xi_a` are expanded into their sign
    # cases rather than written with `fabs`. That is EXACT, not a smoothing: the
    # feasible set is identical. It matters because `fabs`'s kink sits precisely at
    # R_01 = R_10 = 0, i.e. axis-aligned patches -- the nominal pose, and therefore
    # the initial guess of every solve. See `rotated_half_extent_rows`.
    rows_x, labels_x = rotated_half_extent_rows(R_rel, p_rel, xi_a, xi_b, 0)
    rows_y, labels_y = rotated_half_extent_rows(R_rel, p_rel, xi_a, xi_b, 1)
    ineq = ca.vertcat(rows_x, rows_y)
    ineq_labels = labels_x + labels_y

    return ConstraintBlock(name=name, eq=eq, ineq=ineq, eq_labels=eq_labels, ineq_labels=ineq_labels)


def contact_wrench(
    f, kappa, xi_a, *, mu: float, mu_torsional: float, name: str = "contact_wrench",
    R_rel=None, p_rel=None,
) -> ConstraintBlock:
    """Eqs. 7c + 7d: unilaterality, friction, torsional friction, centre of pressure.

    Parameters
    ----------
    f : 3-vector, contact force of a w.r.t. b, expressed in frame b.
    kappa : 3-vector, contact moment, same frame, taken about b's origin.
    xi_a : half-extents of patch a IN ITS OWN FRAME (the paper's `^a xi_a`).
    mu : Coulomb friction coefficient.
    mu_torsional : torsional friction coefficient (the paper's mu_r).
    R_rel, p_rel : the relative transform of a w.r.t. b from `relative_patch_transform`.
        Needed to express Eq. 7d in patch a's frame -- see below. Omitting them
        asserts the patches are aligned (zero relative yaw, coincident origins);
        that is exactly the case where it makes no difference.

    All 9 rows are inequalities; there are no equalities here.

    WHICH FRAME EQ. 7d LIVES IN  (paper ambiguity #15)
    ---------------------------------------------------
    The paper's prose says p, R, f and kappa are all "expressed in frame b", but Eq.
    7d bounds the moment by `^a xi_a` -- a's half-extents in A'S OWN frame, and the
    superscript is deliberate, since Eq. 7b writes `^b xi_a` for the other one.

    Those two statements conflict as soon as the patches differ by a yaw, which Eq.
    7a explicitly leaves FREE. Eq. 7d is a centre-of-pressure bound: the CoP must lie
    inside patch a's rectangle. Read literally in frame b it instead confines the CoP
    to a rectangle of a's dimensions but aligned with B -- a different set. With a
    G1 sole (85 x 30 mm) at 45 deg of yaw, that both rejects contacts whose CoP is
    comfortably inside the foot and accepts ones 3x outside it.

    So the moment is transported into a's frame before the CoP rows are formed,
    which is what makes `^a xi_a` mean what it says. Eq. 7c is deliberately NOT
    transported: a friction PYRAMID is not rotationally symmetric, so it is only
    well defined once a frame is chosen, and the paper's choice is b (the support
    surface). That asymmetry is intentional, not an oversight here.
    """
    fx, fy, fz = f[0], f[1], f[2]

    # Eq. 7d in patch a's frame: kappa_a = R^T (kappa_b - p x f_b), the moment taken
    # about A's origin and resolved on A's axes. Under Eq. 7a the relative rotation is
    # a pure yaw, so the normal component f_z is unchanged -- but the in-plane pair
    # (kappa_x, kappa_y) rotates, and that is precisely what the CoP bound reads.
    if R_rel is None:
        kappa_a, fz_a = kappa, fz
    else:
        moment_about_a = kappa if p_rel is None else kappa - ca.cross(p_rel, f)
        kappa_a = R_rel.T @ moment_about_a
        fz_a = (R_rel.T @ f)[2]

    kx, ky, kz = kappa_a[0], kappa_a[1], kappa_a[2]

    # --- (7c) -----------------------------------------------------------------
    # PYRAMIDAL friction (the paper says so explicitly), i.e. componentwise
    # |f_x| <= mu f_z AND |f_y| <= mu f_z. This is deliberately NOT the quadratic
    # cone sqrt(f_x^2 + f_y^2) <= mu f_z: the pyramid keeps the constraint LINEAR
    # in the force variables, which is a real advantage for the NLP.
    # f_z >= 0 is unilaterality: contacts push, never pull.
    ineq = ca.vertcat(
        fx - mu * fz, -fx - mu * fz,
        fy - mu * fz, -fy - mu * fz,
        -fz,
    )
    labels = ["7c:+f_x", "7c:-f_x", "7c:+f_y", "7c:-f_y", "7c:f_z>=0"]

    # --- (7d) -----------------------------------------------------------------
    # Torsional friction about the contact normal, at patch a's centre.
    ineq = ca.vertcat(ineq, kz - mu_torsional * fz_a, -kz - mu_torsional * fz_a)
    labels += ["7d:+kappa_z", "7d:-kappa_z"]

    # Centre-of-pressure bounds. NOTE THE x<->y SWAP, which is in the paper and is
    # correct physics, not a typo: a moment about x is produced by the normal force
    # acting at a lever arm along y, so kappa_x is bounded by the half-extent in Y.
    # "Tidying" this into matching subscripts is a silent, plausible-looking bug.
    ineq = ca.vertcat(
        ineq,
        kx - fz_a * xi_a[1], -kx - fz_a * xi_a[1],
        ky - fz_a * xi_a[0], -ky - fz_a * xi_a[0],
    )
    labels += ["7d:+kappa_x", "7d:-kappa_x", "7d:+kappa_y", "7d:-kappa_y"]

    return ConstraintBlock(name=name, ineq=ineq, ineq_labels=labels)


def contact_full(
    R_a, p_a, R_b, p_b, xi_a, xi_b, f, kappa, *,
    mu: float, mu_torsional: float, name: str = "contact", flip_b: bool = True,
) -> ConstraintBlock:
    """All of Eq. 7 -- what TO (Eq. 17) enforces at every knot point."""
    from faro.constraints.block import merge
    from faro.constraints.frames import relative_patch_transform

    kin = contact_kinematic(R_a, p_a, R_b, p_b, xi_a, xi_b, name="kin", flip_b=flip_b)
    # Eq. 7d needs the relative transform to reach patch a's frame; pass it rather
    # than letting `contact_wrench` assume alignment, since Eq. 7a leaves yaw free.
    R_rel, p_rel = relative_patch_transform(R_a, p_a, R_b, p_b, flip_b=flip_b)
    wrench = contact_wrench(f, kappa, xi_a, mu=mu, mu_torsional=mu_torsional,
                            name="wrench", R_rel=R_rel, p_rel=p_rel)
    return merge(name, [kin, wrench])


def patch_containment_is_possible(xi_a, xi_b) -> bool:
    """Can patch a ever fit inside patch b? A numeric sanity check for scene setup.

    Eq. 7b is only satisfiable if a is no larger than b in both directions (in the
    best case, zero relative yaw). If a scene pairs a large patch onto a small one,
    every contact using it is infeasible for a reason that has nothing to do with
    the robot -- worth catching when the scene is built, not inside a solver.
    """
    return float(xi_a[0]) <= float(xi_b[0]) + 1e-12 and float(xi_a[1]) <= float(xi_b[1]) + 1e-12
