"""Robot loading and Pinocchio wrapping.

The only module that knows how a robot gets off disk and into memory. Everything
downstream receives a `RobotModel` and never touches a URDF path again -- that is
what makes the robot swappable from `configs/robots/*.yaml`.

Floating base: FARO targets humanoid loco-manipulation, so the root joint is a
free-flyer. A fixed-base load silently drops the 6 base DoF and would make the
whole formulation wrong rather than merely inaccurate, so it is asserted, not
assumed (see `_check_floating_base`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pinocchio as pin

from faro.utils.config import load_config
from faro.utils.paths import example_robot_data_share, resolve_asset


@dataclass
class RobotModel:
    """A loaded robot: kinematic model, geometry, and its nominal posture.

    Attributes
    ----------
    name : identifier from the config, e.g. "g1_29dof".
    model : the Pinocchio kinematic/dynamic model (free-flyer root).
    visual_model, collision_model : Pinocchio geometry models.
    q_nominal : the reference posture. Used as the regularization target in the
        mode/edge NLP (Eq. 14) and as an initial guess everywhere -- so it matters
        for convergence, not just for looks.
    urdf_path : kept for provenance and for backends (e.g. MuJoCo) that re-parse it.
    """

    name: str
    model: pin.Model
    visual_model: pin.GeometryModel
    collision_model: pin.GeometryModel
    q_nominal: np.ndarray
    urdf_path: Path

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_config(cls, name_or_cfg: str | dict) -> RobotModel:
        """Load from `configs/robots/<name>.yaml` (or an already-loaded dict)."""
        cfg = load_config("robots", name_or_cfg) if isinstance(name_or_cfg, str) else name_or_cfg

        urdf_path = resolve_asset(cfg["urdf"])
        package_dirs = str(example_robot_data_share())

        # JointModelFreeFlyer = 6-DoF floating base, quaternion-parameterised.
        model = pin.buildModelFromUrdf(urdf_path.as_posix(), pin.JointModelFreeFlyer())
        visual_model = pin.buildGeomFromUrdf(
            model, urdf_path.as_posix(), pin.GeometryType.VISUAL, package_dirs=package_dirs
        )
        collision_model = pin.buildGeomFromUrdf(
            model, urdf_path.as_posix(), pin.GeometryType.COLLISION, package_dirs=package_dirs
        )

        _check_floating_base(model, urdf_path)

        robot = cls(
            name=cfg.get("name", urdf_path.stem),
            model=model,
            visual_model=visual_model,
            collision_model=collision_model,
            q_nominal=pin.neutral(model),  # replaced below
            urdf_path=urdf_path,
        )
        robot.q_nominal = robot.build_configuration(cfg.get("nominal_configuration", {}))
        return robot

    # --------------------------------------------------------------- properties
    @property
    def nq(self) -> int:
        """Configuration dimension (7 for the floating base + 1 per revolute joint)."""
        return self.model.nq

    @property
    def nv(self) -> int:
        """Velocity/tangent dimension. nq - nv == 1 because of the base quaternion."""
        return self.model.nv

    @property
    def n_actuated(self) -> int:
        """Actuated joints = everything except the 6 unactuated floating-base DoF.

        This underactuation is the reason FARO needs contact forces at all: the base
        can only be moved through contact.
        """
        return self.model.nv - 6

    @cached_property
    def data(self) -> pin.Data:
        """Working buffer for numeric algorithms. Reused; not thread-safe."""
        return self.model.createData()

    @cached_property
    def actuated_joint_names(self) -> list[str]:
        return [self.model.names[i] for i in range(1, self.model.njoints) if self.model.names[i] != "root_joint"]

    # ------------------------------------------------------------------- frames
    def frame_id(self, frame_name: str) -> int:
        """Look up a frame, with an error message that actually helps.

        Pinocchio's `getFrameId` returns `model.nframes` for an unknown name instead
        of raising, which turns a typo into silent nonsense far downstream.
        """
        if not self.model.existFrame(frame_name):
            available = [f.name for f in self.model.frames if f.type == pin.FrameType.BODY]
            raise KeyError(
                f"robot {self.name!r} has no frame {frame_name!r}. "
                f"Body frames available: {available}"
            )
        return self.model.getFrameId(frame_name)

    def frame_placement(self, q: np.ndarray, frame_name: str) -> pin.SE3:
        """World placement of a frame at configuration q (numeric forward kinematics)."""
        pin.forwardKinematics(self.model, self.data, np.asarray(q))
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id(frame_name)].copy()

    # ------------------------------------------------------------ configuration
    def build_configuration(self, joint_values: dict[str, float], base: pin.SE3 | None = None) -> np.ndarray:
        """Assemble a configuration vector from named joint angles.

        Writing postures as `{"left_knee_joint": 0.3}` instead of raw indices keeps
        configs readable and survives a robot swap.
        """
        q = pin.neutral(self.model)
        for joint_name, value in joint_values.items():
            if not self.model.existJointName(joint_name):
                raise KeyError(
                    f"robot {self.name!r} has no joint {joint_name!r}. "
                    f"Available: {self.actuated_joint_names}"
                )
            jid = self.model.getJointId(joint_name)
            idx = self.model.joints[jid].idx_q
            nq_joint = self.model.joints[jid].nq
            if nq_joint != 1:
                raise ValueError(f"joint {joint_name!r} has nq={nq_joint}; only 1-DoF joints supported here")
            q[idx] = float(value)
        if base is not None:
            q[:3] = base.translation
            q[3:7] = pin.Quaternion(base.rotation).coeffs()  # (x, y, z, w)
        return q

    def place_on_ground(self, q: np.ndarray, patch_frames: list[str], ground_z: float = 0.0) -> np.ndarray:
        """Shift the floating base vertically so the lowest given frame sits at ground_z.

        Convenience for building a plausible standing posture without hand-tuning the
        base height. Purely kinematic -- it does NOT check static balance.
        """
        q = np.asarray(q, dtype=float).copy()
        lowest = min(self.frame_placement(q, f).translation[2] for f in patch_frames)
        q[2] += ground_z - lowest
        return q

    # ------------------------------------------------------------------ limits
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) configuration limits, paper Eq. 12.

        Returned over the full nq, including the meaningless floating-base entries;
        callers that enforce Eq. 12 must slice off the first 7. Kept whole here so
        the indexing matches q exactly and nobody has to guess an offset.
        """
        return self.model.lowerPositionLimit.copy(), self.model.upperPositionLimit.copy()

    # ---------------------------------------------------------------- symbolic
    def casadi_model(self):
        """The CasADi mirror of this model, for symbolic kinematics.

        Not used in Milestone 1, but exposed now so the interface is settled: every
        later stage (Eqs. 14/15/17) builds its constraints from this, and its
        existence is what the Milestone 0 check verified.
        """
        import pinocchio.casadi as cpin

        return cpin.Model(self.model)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"RobotModel(name={self.name!r}, nq={self.nq}, nv={self.nv}, "
            f"actuated={self.n_actuated})"
        )


def _check_floating_base(model: pin.Model, urdf_path: Path) -> None:
    """Fail loudly if the model is not floating-base.

    `nq - nv == 1` is the signature of a quaternion-parameterised free-flyer root:
    the quaternion uses 4 configuration entries but only 3 tangent entries.
    """
    if model.nq - model.nv != 1:
        raise RuntimeError(
            f"{urdf_path.name} did not load with a floating base "
            f"(nq={model.nq}, nv={model.nv}; expected nq - nv == 1). "
            "FARO models humanoid loco-manipulation and requires a free-flyer root."
        )
    if model.names[1] != "root_joint":
        raise RuntimeError(f"expected 'root_joint' as joint 1, found {model.names[1]!r}")
