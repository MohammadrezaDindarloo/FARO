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
    *, name: str = "no_slip", flip_b: bool = True,
) -> ConstraintBlock:
    """Eq. 8: the relative in-plane pose of a contact is unchanged across a step.

    Takes both patches at BOTH timesteps, because the quantity held fixed is the
    *relative* pose p, R of patch a w.r.t. patch b -- not either patch's world pose.
    That distinction matters whenever the support itself moves: a hand sticking to
    the box while the box is carried satisfies Eq. 8 even though both patches sweep
    through space.

    Returns 3 equalities: two in-plane position, one relative yaw.
    """
    R_s, p_s = relative_patch_transform(R_a_s, p_a_s, R_b_s, p_b_s, flip_b=flip_b)
    R_n, p_n = relative_patch_transform(R_a_next, p_a_next, R_b_next, p_b_next, flip_b=flip_b)

    # In-plane translation is unchanged (the z component is already pinned by 7a).
    d_p = p_n - p_s

    # Relative yaw is unchanged. log3(R_s^T R_n) is the rotation FROM step s TO
    # step s+1; its z component is the in-plane (about-normal) part.
    d_omega = log3(R_s.T @ R_n)

    eq = ca.vertcat(d_p[0], d_p[1], d_omega[2])
    return ConstraintBlock(
        name=name,
        eq=eq,
        eq_labels=["8:dp_x", "8:dp_y", "8:dyaw"],
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
