"""Contact patches and interfaces -- FARO's shared vocabulary (paper Section II-A, II-C 1).

This module deliberately contains NO optimization code. It is the common language
that mode/edge (Eq. 14), KSO (Eq. 15), TO (Eq. 17) and the tree search (Alg. 1) all
speak, so none of them has to import from another.

Paper grounding, Section II-C 1, quoted:

    "all end-effectors and environment contact interfaces are modeled as rectangular
     patches with patch-to-patch unilateral contact"
    "The patch normals are aligned with their local z-axes."
    "The vector h denotes the half-extents of a rectangular patch"

So a patch is fully described by: a parent frame, a rigid placement within that
frame, and two half-extents. Nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pinocchio as pin


class Attachment(str, Enum):
    """What a patch is rigidly attached to.

    This determines how the patch's world placement is computed, and therefore which
    decision variables its constraints depend on:

      ROBOT       -> a link frame; placement depends on the robot configuration q
      OBJECT      -> a movable object; placement depends on that object's pose
      ENVIRONMENT -> fixed in the world; placement is constant
    """

    ROBOT = "robot"
    OBJECT = "object"
    ENVIRONMENT = "environment"


@dataclass(frozen=True)
class ContactPatch:
    """A rectangular planar contact patch.

    Attributes
    ----------
    name : unique identifier, e.g. "left_foot_sole".
    attachment : ROBOT / OBJECT / ENVIRONMENT (see Attachment).
    parent : name of the thing it is attached to -- a robot frame name for ROBOT,
        an object name for OBJECT, ignored for ENVIRONMENT.
    placement : the patch frame expressed in the parent frame. Its z-axis is the
        patch normal (paper Section II-C 1); its x/y axes span the patch plane and
        define the directions the half-extents refer to.
    half_extents : (hx, hy), the paper's `h`. Half-width along the patch frame's
        local x and y. Used by Eq. 7b (patch overlap) and Eq. 7d (centre-of-pressure
        bounds), which is why they must be real dimensions, not guesses.

    NORMAL SIGN CONVENTION (ours, not the paper's -- the paper never states it):
        A patch's local +z points OUTWARD, i.e. away from the body it belongs to and
        into the free space it can touch. So the floor's normal is +z (up), and a
        foot sole's normal points down out of the foot.

        Consequence: two patches in contact are face-to-face, meaning their z-axes
        are ANTI-parallel. Eq. 7a's alignment residual must account for that flip.
        This is deliberately isolated here and re-examined in Milestone 2; see
        docs/ambiguities.md.
    """

    name: str
    attachment: Attachment
    parent: str
    placement: pin.SE3
    half_extents: tuple[float, float]

    def __post_init__(self) -> None:
        hx, hy = self.half_extents
        if hx <= 0 or hy <= 0:
            raise ValueError(f"patch {self.name!r}: half-extents must be positive, got {(hx, hy)}")

    @property
    def normal_local(self) -> np.ndarray:
        """The patch normal in the PARENT frame (the placement's z-axis)."""
        return self.placement.rotation[:, 2].copy()

    def corners_local(self) -> np.ndarray:
        """The four rectangle corners in the parent frame, shape (4, 3).

        Only used for visualization and for sanity checks -- the optimization uses
        the half-extents algebraically (Eq. 7b), never the explicit corners.
        """
        hx, hy = self.half_extents
        local = np.array([[+hx, +hy, 0.0], [-hx, +hy, 0.0], [-hx, -hy, 0.0], [+hx, -hy, 0.0]])
        return np.array([self.placement.act(c) for c in local])

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> ContactPatch:
        """Build from a `configs/scenes/*.yaml` entry.

        Expected keys:
            attachment: robot | object | environment
            parent: frame or object name
            half_extents: [hx, hy]
            translation: [x, y, z]         (optional, default origin)
            rpy_deg: [roll, pitch, yaw]    (optional, default no rotation)

        Rotation is given in DEGREES and named `rpy_deg` explicitly, because a
        silently-misread radians/degrees swap is a classic and very confusing bug.
        """
        translation = np.asarray(cfg.get("translation", [0.0, 0.0, 0.0]), dtype=float)
        rpy = np.deg2rad(np.asarray(cfg.get("rpy_deg", [0.0, 0.0, 0.0]), dtype=float))
        placement = pin.SE3(pin.rpy.rpyToMatrix(*rpy), translation)
        hx, hy = cfg["half_extents"]
        return cls(
            name=name,
            attachment=Attachment(cfg["attachment"]),
            parent=cfg.get("parent", ""),
            placement=placement,
            half_extents=(float(hx), float(hy)),
        )


@dataclass
class Interface:
    """A named contact site, plus which other sites it may touch (paper Table IV).

    An interface owns exactly one patch. The `allowed` list is the discrete action
    set that the tree search (Alg. 1) branches over, and the paper's Table IV is
    precisely this data:

        Left hand  -> box left, box front, free
        Left foot  -> floor, free
        ...

    "free" is represented by the absence of a partner, i.e. `None`, and is always
    an implicit option -- Section II-A defines a contact state as a pair marking an
    interface either in unilateral contact with another interface, or free.
    """

    name: str
    patch: ContactPatch
    allowed: list[str] = field(default_factory=list)

    def options(self) -> list[str | None]:
        """Every discrete contact choice for this interface, including `free`."""
        return [None, *self.allowed]

    def branching_factor(self) -> int:
        return len(self.allowed) + 1
