"""Sticking / no-slip contact -- paper Eq. 8 (Section II-C 1).

    For a sticking contact that remains active across adjacent timesteps s and
    s + 1, we also have the no-slip condition,

        (p^{s+1} - p^s)_{x,y} = 0,   log3( (R^s)^T R^{s+1} )_z = 0        (8)

    which is enforced by constraining both the in-plane position and the in-plane
    rotation across timesteps.

Note the complementarity with Eq. 7a. Eq. 7a pins the two out-of-plane rotation
components and the normal offset, deliberately LEAVING the in-plane position and
yaw free. Eq. 8 then pins exactly those remaining three degrees of freedom, but
only *relative to the previous timestep* -- so a sticking contact may sit anywhere
admissible, it just may not move once established.

WHEN IT APPLIES (paper Eq. 16, used by KSO Eq. 15): for interface a with contact
partner b_s at step s and b_{s+1} at step s+1, enforce Eq. 8 if either

    b_s = b_{s+1} != empty          (16a)  persistent contact
    b_s != empty AND b_{s+1} = empty (16b)  contact release

(16b) is the subtle one: on release, q_{s+1} is "the boundary configuration at
which contact is broken, so the sticking constraint is enforced up to the instant
of separation". A contact must not slide on its way out.

`applies_between` implements Eq. 16 so the rule lives in one place.
"""

from __future__ import annotations

import casadi as ca

from faro.constraints.block import ConstraintBlock
from faro.constraints.frames import log3, relative_patch_transform


def no_slip(
    R_a_s, p_a_s, R_b_s, p_b_s,
    R_a_next, p_a_next, R_b_next, p_b_next,
    *, name: str = "no_slip", flip_b: bool = True, yaw: str = "column",
) -> ConstraintBlock:
    """Eq. 8: the relative in-plane pose of a contact is unchanged across a step.

    Takes both patches at BOTH timesteps, because the quantity held fixed is the
    *relative* pose p, R of patch a w.r.t. patch b -- not either patch's world pose.
    That distinction matters whenever the support itself moves: a hand sticking to
    the box while the box is carried satisfies Eq. 8 even though both patches sweep
    through space.

    Returns 3 equalities: two in-plane position, one relative yaw (plus, for the
    default `yaw` form, one inequality selecting the correct branch).

    THE YAW ROW HAS TWO FORMS, FOR THE SAME REASON EQ. 7a DOES
    ----------------------------------------------------------
    The paper writes `log3((R^s)^T R^{s+1})_z = 0`, and `yaw="log3"` is that verbatim.
    It is also unusable with an exact-Hessian solver here: `cpin.log3`'s SECOND
    derivative returns NaN for some inputs, and Ipopt reports `nlp_hess_l failed: NaN
    detected` followed by `Invalid_Number_Detected`. First derivatives are clean --
    which is why this went unnoticed through Milestone 2's finite-difference checks
    and through Eq. 14, whose default 7a form avoids log3 entirely.

    What is established: the KSO fails this way with `yaw="log3"` and solves with
    `yaw="column"`; the NaN is in the Hessian, not the value or the Jacobian; and it
    does not reproduce as a clean function of rotation angle or axis. What is NOT
    established is the mechanism inside pinocchio -- `cpin.log3` branches on the
    rotation angle, and second-order AD through a branch evaluates BOTH sides, where
    a `0/0` in the inactive one propagates as `NaN * 0 = NaN`. That is a plausible
    account, not a demonstrated one. See docs/ambiguities.md #23.

    `yaw="column"` uses `(R^s)^T R^{s+1}` directly:

        R_rel[1,0] = 0     (the sine of the relative yaw)
        R_rel[0,0] >= 0    (its cosine, selecting yaw = 0 over yaw = pi)

    Equivalent as a set wherever Eq. 7a holds, because 7a already forces both relative
    rotations to be pure yaws, and a pure yaw with zero sine and non-negative cosine is
    the identity. Exactly the same argument, and the same two-branch structure, as
    `normal_alignment_rows`.
    """
    if yaw not in {"column", "log3"}:
        raise ValueError(f"yaw must be 'column' or 'log3', got {yaw!r}")

    R_s, p_s = relative_patch_transform(R_a_s, p_a_s, R_b_s, p_b_s, flip_b=flip_b)
    R_n, p_n = relative_patch_transform(R_a_next, p_a_next, R_b_next, p_b_next, flip_b=flip_b)

    # In-plane translation is unchanged (the z component is already pinned by 7a).
    d_p = p_n - p_s

    # The rotation FROM step s TO step s+1.
    R_rel = R_s.T @ R_n

    if yaw == "log3":
        return ConstraintBlock(
            name=name,
            eq=ca.vertcat(d_p[0], d_p[1], log3(R_rel)[2]),
            eq_labels=["8:dp_x", "8:dp_y", "8:dyaw"],
        )

    return ConstraintBlock(
        name=name,
        eq=ca.vertcat(d_p[0], d_p[1], R_rel[1, 0]),
        eq_labels=["8:dp_x", "8:dp_y", "8:dyaw"],
        ineq=ca.vertcat(-R_rel[0, 0]),
        ineq_labels=["8:cos_dyaw>=0"],
    )


def applies_between(partner_s, partner_next) -> bool:
    """Eq. 16: should Eq. 8 be enforced for this interface across s -> s+1?

    Parameters
    ----------
    partner_s, partner_next : the interface's contact partner at each step, or
        None for `free` (the paper's empty set).

    Returns True for persistent contact (16a) or release (16b), False otherwise.

    Deliberately NOT applied on contact *acquisition* (None -> partner): there is
    no previous contact location to preserve, and constraining one would forbid the
    robot from choosing where to place a new contact.
    """
    if partner_s is None:
        return False                       # newly acquired (or still free): nothing to hold
    if partner_next == partner_s:
        return True                        # (16a) persistent contact
    if partner_next is None:
        return True                        # (16b) release: no sliding on the way out
    return False                           # partner switched: a genuinely new contact
