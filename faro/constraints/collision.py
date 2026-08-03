"""Collision avoidance -- paper Eq. 9 (Section II-C 2).

    All bodies in the robot and scene are modeled using convex geometries, and
    collision avoidance is imposed with the signed-distance approximation of
    Schulman et al. [20]. For each collision pair A and B, the collision is queried
    using the GJK algorithm [21] as implemented in [22] (coal). The query returns
    witness points p_A and p_B, expressed in the local frames of the two bodies,
    together with a separating normal n. T^w_A(x) and T^w_B(x) represent the
    transformation of body A and B in the world frame. The resulting constraint
    based on the signed-distance approximation is

        0 <= sd_AB(x) ~= n . ( T^w_A(x) p_A  -  T^w_B(x) p_B )               (9)

HOW THIS IS DIFFERENTIABLE
--------------------------
GJK is combinatorial and not differentiable, so it is not inside the optimization.
Instead, at a given linearization point:

  1. run the GJK query ONCE numerically -> witness points p_A, p_B (in local
     frames) and separating normal n;
  2. FREEZE those three quantities as constants;
  3. the constraint expression keeps only the body transforms T^w_A(x), T^w_B(x)
     as functions of the decision variables.

The result is smooth and cheap to differentiate. This is exactly Schulman et al.'s
approximation: the witness points are the current closest-point pair, and the
expression measures how far apart *those material points* are along the frozen
normal. It is a first-order model, so it is valid near the linearization point --
which is why the witness data must be REFRESHED as the solution moves. The
`refresh` workflow below makes that explicit rather than implicit.

The paper writes `0 <= sd`, i.e. no safety margin. A small positive margin usually
helps convergence by keeping the iterates off the constraint boundary, so it is
exposed as a parameter defaulting to the paper's 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np


@dataclass(frozen=True)
class WitnessData:
    """One GJK query result, frozen for use as constants in Eq. 9.

    Attributes
    ----------
    p_A, p_B : witness points in the LOCAL frames of bodies A and B.
    normal : separating normal n, a unit vector in the WORLD frame.
    distance : the signed distance the query reported, kept for diagnostics.
    """

    p_A: np.ndarray
    p_B: np.ndarray
    normal: np.ndarray
    distance: float

    def __post_init__(self) -> None:
        norm = float(np.linalg.norm(self.normal))
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise ValueError(
                f"separating normal must be a unit vector, got |n| = {norm}. "
                "Eq. 9 projects onto it, so a non-unit normal silently rescales the "
                "signed distance."
            )


def signed_distance_expr(R_A, p_A_origin, R_B, p_B_origin, witness: WitnessData):
    """The right-hand side of Eq. 9: n . (T^w_A p_A - T^w_B p_B).

    R_*, p_*_origin are the world placements of the two BODIES (functions of the
    decision variables); `witness` holds the frozen GJK output.
    """
    world_A = R_A @ ca.DM(witness.p_A.reshape(3, 1)) + p_A_origin
    world_B = R_B @ ca.DM(witness.p_B.reshape(3, 1)) + p_B_origin
    return ca.dot(ca.DM(witness.normal.reshape(3, 1)), world_A - world_B)


def collision_avoidance(
    R_A, p_A_origin, R_B, p_B_origin, witness: WitnessData,
    *, margin: float = 0.0, name: str = "collision",
):
    """Eq. 9 as an inequality in FARO's `<= 0` convention.

    The paper's `0 <= sd` becomes `margin - sd <= 0`.

    `margin` defaults to 0.0 to match the paper exactly. Raising it to a few
    millimetres keeps iterates off the boundary and typically helps Ipopt converge;
    it is a tuning knob, not a modelling change.
    """
    from faro.constraints.block import ConstraintBlock

    sd = signed_distance_expr(R_A, p_A_origin, R_B, p_B_origin, witness)
    return ConstraintBlock(
        name=name,
        ineq=ca.vertcat(margin - sd),
        ineq_labels=["9:sd>=margin"],
    )


def query_witness(geom_A, tf_A, geom_B, tf_B) -> WitnessData:
    """Run a GJK distance query with coal and package the result for Eq. 9.

    Parameters
    ----------
    geom_A, geom_B : coal collision geometries (convex).
    tf_A, tf_B : coal Transform3s world placements at the linearization point.

    The returned witness points are converted into each body's LOCAL frame, as
    Eq. 9 requires -- coal reports them in the world frame, and using them
    unconverted would freeze the bodies in place and make the constraint constant.
    """
    try:
        import coal
    except ImportError:  # pragma: no cover - older stacks
        import hppfcl as coal

    request, result = coal.DistanceRequest(), coal.DistanceResult()
    # Without this, penetrating pairs report an unsigned distance and Eq. 9 loses the
    # only information it exists to carry. It defaults to True in coal 3.x; set
    # explicitly so a version or backend change cannot flip it silently.
    request.enable_signed_distance = True
    distance = coal.distance(geom_A, tf_A, geom_B, tf_B, request, result)

    w_A = np.asarray(result.getNearestPoint1(), dtype=float)
    w_B = np.asarray(result.getNearestPoint2(), dtype=float)

    R_A, t_A = np.asarray(tf_A.getRotation()), np.asarray(tf_A.getTranslation())
    R_B, t_B = np.asarray(tf_B.getRotation()), np.asarray(tf_B.getTranslation())
    local_A = R_A.T @ (w_A - t_A)
    local_B = R_B.T @ (w_B - t_B)

    # Direction from B's witness point toward A's.
    delta = w_A - w_B
    norm = float(np.linalg.norm(delta))
    if norm >= 1e-9:
        normal = delta / norm
    else:
        # TOUCHING EXACTLY. The witness points coincide, so their difference carries no
        # direction -- and this is not a corner case here, it is the normal state of
        # affairs: every contact the mode asserts sits at sd = 0.000000 by construction,
        # a sole on the floor and a box bottom on a platform included.
        #
        # Fall back in order of how much each source knows:
        #   1. coal's own separating normal, which is correct for exact face-on-face
        #      contact between convex bodies and is what it reports there;
        #   2. failing that, centre to centre. Crude, but always defined for distinct
        #      bodies and pointing the right way for two convex sets in contact -- and
        #      Eq. 9 is a first-order model that the refresh loop re-linearizes anyway,
        #      so a slightly wrong direction at a touching pair costs an iteration, not
        #      correctness.
        #
        # Raising instead, which is what this used to do, means one degenerate pair out
        # of ~700 aborts the whole solve and reports nothing about the mode.
        # `np.isfinite` as well as the norm: coal returns a NaN normal for some exactly
        # touching pairs, and NaN fails `< 1e-9`, so a norm test alone lets it straight
        # through into Eq. 9. It surfaces much later as `Invalid_Number_Detected` with
        # nothing pointing back here.
        normal = np.asarray(result.normal, dtype=float)
        n_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal).all() or n_norm < 1e-9:
            normal = t_A - t_B
            n_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal).all() or n_norm < 1e-9:
            raise RuntimeError(
                "GJK returned coincident witness points, a degenerate normal, AND "
                "coincident body origins; Eq. 9 has no direction to work with here. "
                "Two bodies are exactly co-located, which is a scene definition error."
            )
        normal = normal / n_norm

    # ORIENT THE NORMAL BY THE SIGN OF THE DISTANCE. This is the whole reason Eq. 9
    # is a *signed* distance, and getting it wrong is silent and inverted:
    #
    #   When the bodies overlap, the witness points swap sides, so `w_A - w_B` points
    #   back the way it came. Projecting onto it then returns +|penetration| -- a
    #   6 cm interpenetration reported as 6 cm of CLEARANCE. `0 <= sd` would be
    #   satisfied most comfortably exactly where the bodies are most deeply merged,
    #   and the solver would be rewarded for driving them together.
    #
    # coal reports the sign correctly in `distance`; only the direction built from the
    # witness points needs flipping to agree with it.
    if distance < 0.0:
        normal = -normal

    return WitnessData(p_A=local_A, p_B=local_B, normal=normal, distance=float(distance))
