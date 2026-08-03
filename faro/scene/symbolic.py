"""Symbolic configuration variables for a scene -- the `q` of Eqs. 14, 15 and 17.

Eq. 14 optimizes over

    q = (q^r, q^o1, ..., q^ono)

    "the contact, collision, and limit constraints are defined in section II-C,
     while q = (q^r, q^o1, ..., q^ono) represents the robot and object
     configurations"   -- Section II-D

so the decision vector is the robot configuration STACKED WITH every movable
object's pose. This module builds that vector symbolically, resolves any patch's
world placement as a function of it, and hands back the pieces the NLP needs. It
does no optimizing, so KSO (Eq. 15) can build K+1 of these and TO (Eq. 17) one per
knot without either importing from mode/edge.

THE QUATERNION CONSTRAINT THE PAPER DOES NOT WRITE
--------------------------------------------------
Eq. 6a says q^r in SE(3) x R^n and Eq. 6b says q^o in SE(3) -- manifolds. But an
NLP optimizes over R^m, and both Pinocchio and this code parameterize SE(3) with a
UNIT quaternion: 7 numbers for 6 degrees of freedom. Those 7 numbers are only a
rigid transform when the quaternion has unit norm.

Nothing in Eq. 14 says so. Left out, the solver is free to shrink or stretch the
quaternion, and it will -- a non-unit quaternion's rotation matrix is scaled by
|q|^2, which lets the optimizer cheat the cost by shrinking the robot's effective
rotation instead of moving the robot. So `unit_quaternion_rows` adds one equality
per pose. It is not a modelling choice; it is what makes the 7-vector mean what
Eq. 6 says it means. Recorded as docs/ambiguities.md #21.

The alternative -- optimizing in the 6-dimensional tangent space around a fixed
reference, so no constraint is needed -- is what a Lie-group solver does, and it is
how acados/Hippo would likely be set up. It is left for later because it changes
what the variables ARE, which would make every residual in Milestone 2 harder to
compare against the paper's own indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np
import pinocchio as pin

from faro.constraints.block import ConstraintBlock
from faro.core.patches import Attachment, ContactPatch
from faro.scene.scene import Scene


def quaternion_to_rotation(quat):
    """Rotation matrix from a CasADi quaternion in Pinocchio's (x, y, z, w) order.

    Written out rather than delegated to `cpin` because it must accept the raw
    decision variables, which are NOT guaranteed unit-norm during the solve -- the
    unit-norm equality is a constraint, satisfied only at convergence. Pinocchio's
    quaternion type normalizes or asserts, either of which would hide from the
    optimizer exactly the deviation `unit_quaternion_rows` is there to penalize.
    """
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        ca.horzcat(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        ca.horzcat(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


@dataclass
class SymbolicScene:
    """The scene's configuration variables, and everything derived from them.

    Attributes
    ----------
    scene : the scene these variables describe.
    robot_q : SX of length `robot.nq` -- the paper's q^r.
    object_q : {object_name: SX of length 7} -- the paper's q^o_l, as
        (translation, quaternion xyzw), matching Pinocchio's layout so a slice can
        be handed straight to `pin.SE3`.
    suffix : appended to every variable name. KSO builds K+1 of these in one NLP,
        and CasADi identifies variables by name -- without a suffix the second
        SymbolicScene's variables would be indistinguishable from the first's in
        every error message the solver produces.
    """

    scene: Scene
    suffix: str = ""
    robot_q: ca.SX = field(init=False)
    object_q: dict[str, ca.SX] = field(init=False)

    def __post_init__(self) -> None:
        tag = f"_{self.suffix}" if self.suffix else ""
        self.robot_q = ca.SX.sym(f"q_r{tag}", self.scene.robot.nq)
        self.object_q = {
            name: ca.SX.sym(f"q_{name}{tag}", 7) for name in sorted(self.scene.objects)
        }
        self._cmodel = self.scene.robot.casadi_model()
        self._cdata = self._cmodel.createData()
        self._placements_cached = False

    # ------------------------------------------------------------- the vector
    @property
    def variables(self) -> ca.SX:
        """The stacked decision vector q, robot first then objects in name order."""
        return ca.vertcat(self.robot_q, *[self.object_q[n] for n in sorted(self.object_q)])

    @property
    def n_var(self) -> int:
        return int(self.variables.shape[0])

    def layout(self) -> list[tuple[str, slice]]:
        """`[(name, slice), ...]` into the stacked vector -- for unpacking results."""
        out = [("robot", slice(0, self.scene.robot.nq))]
        offset = self.scene.robot.nq
        for name in sorted(self.object_q):
            out.append((name, slice(offset, offset + 7)))
            offset += 7
        return out

    def split(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Unpack a solved vector into `{"robot": q_r, "<object>": q_o}`."""
        x = np.asarray(x, dtype=float).ravel()
        return {name: x[sl].copy() for name, sl in self.layout()}

    def nominal(self) -> np.ndarray:
        """q_nom: the robot's nominal posture and each object's initial pose.

        This is BOTH the regularization target of Eq. 14 and the initial guess. The
        paper never says what it is beyond "the nominal configuration"; see
        docs/ambiguities.md #22.

        Uses `Scene.nominal_configuration`, which grounds the robot on its soles --
        the raw `robot.q_nominal` leaves the pelvis at z = 0 and puts seven links
        inside the floor slab, which Eq. 9 sees even though Eq. 7 does not.
        """
        parts = [np.asarray(self.scene.nominal_configuration(), dtype=float)]
        for name in sorted(self.object_q):
            pose = self.scene.objects[name].initial_pose
            parts.append(
                np.concatenate([pose.translation, pin.Quaternion(pose.rotation).coeffs()])
            )
        return np.concatenate(parts)

    # -------------------------------------------------------------- kinematics
    def _ensure_placements(self) -> None:
        """Run symbolic forward kinematics once; every patch reads from the result."""
        if self._placements_cached:
            return
        import pinocchio.casadi as cpin

        cpin.forwardKinematics(self._cmodel, self._cdata, self.robot_q)
        cpin.updateFramePlacements(self._cmodel, self._cdata)
        self._placements_cached = True

    def joint_placement(self, joint_id: int):
        """(R, p) of a robot JOINT frame in the world.

        Eq. 9 needs this rather than `patch_placement`: a collision geometry is
        attached to a joint by `GeometryObject.placement`, not to a named frame.
        """
        self._ensure_placements()
        placement = self._cdata.oMi[int(joint_id)]
        return placement.rotation, placement.translation

    def object_placement(self, name: str):
        """(R, p) of an object's body frame, as expressions in its 7 variables."""
        q = self.object_q[name]
        return quaternion_to_rotation(q[3:7]), q[0:3]

    def patch_placement(self, patch: ContactPatch | str):
        """(R, p) of a patch's frame in the world, as expressions in the variables.

        The symbolic counterpart of `Scene.patch_world_placement`, resolving the same
        three attachment kinds. Keeping the two in step matters: Milestone 2's
        scenarios evaluate constraints through the numeric path and the NLP evaluates
        them through this one, so a divergence would make every scenario a check of
        code the solver never runs.
        """
        if isinstance(patch, str):
            patch = self.scene.patches[patch]

        offset_R = ca.DM(patch.placement.rotation)
        offset_p = ca.DM(patch.placement.translation.reshape(3, 1))

        if patch.attachment is Attachment.ENVIRONMENT:
            # Fixed in the world: constant, and independent of every variable.
            return offset_R, offset_p

        if patch.attachment is Attachment.OBJECT:
            R_body, p_body = self.object_placement(patch.parent)
        else:
            self._ensure_placements()
            frame = self._cdata.oMf[self.scene.robot.frame_id(patch.parent)]
            R_body, p_body = frame.rotation, frame.translation

        return R_body @ offset_R, R_body @ offset_p + p_body

    # ------------------------------------------------------------- constraints
    def unit_quaternion_rows(self) -> ConstraintBlock:
        """|quat|^2 - 1 = 0 for the robot base and every object. See module docstring.

        Squared norm, not the norm: `sqrt` has an infinite derivative at 0, and the
        optimizer can pass arbitrarily close to a zero quaternion on its way
        somewhere. The squared form is a polynomial with the same zero set.
        """
        rows, labels = [], []
        rows.append(ca.sumsqr(self.robot_q[3:7]) - 1.0)
        labels.append("unit_quat:robot")
        for name in sorted(self.object_q):
            rows.append(ca.sumsqr(self.object_q[name][3:7]) - 1.0)
            labels.append(f"unit_quat:{name}")
        return ConstraintBlock(name="manifold", eq=ca.vertcat(*rows), eq_labels=labels)
