"""Movable rigid objects (paper Eq. 5b, Eq. 11).

Objects are NOT part of the robot's Pinocchio model. The paper gives each object its
own state -- pose and body velocity (Eq. 5b) -- and its own rigid-body dynamics
(Eq. 11), separate from the robot's centroidal dynamics (Eq. 10). Modelling the box
as an extra floating joint on the robot model would fuse two things the paper keeps
apart, so they stay separate here.

Milestone 1 only needs the geometry and the face patches. Mass and inertia are
carried along now because Eq. 11 will need them at Milestone 5, and getting the
convention right once is cheaper than retrofitting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from faro.core.patches import Attachment, ContactPatch

# Face naming convention, in the WORLD frame at the object's nominal orientation:
#   front = +x, rear = -x, left = +y, right = -y, top = +z, bottom = -z
#
# This is exactly what makes the paper's Table IV consistent:
#   "Left hand -> box left, box front"   (the +y and +x faces, on the robot's left)
#   "Right hand -> box right, box rear"  (the -y and -x faces)
#
# Each entry maps a face name to (outward axis, rotation taking local +z to it).
# The patch's local +z is its OUTWARD normal -- see ContactPatch's docstring.
_FACE_ROTATIONS: dict[str, tuple[int, int, np.ndarray]] = {
    #        axis, sign, rotation matrix (columns = images of local x, y, z)
    "front":  (0, +1, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])),   # Ry(+90)
    "rear":   (0, -1, np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])),   # Ry(-90)
    "left":   (1, +1, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])),   # Rx(-90)
    "right":  (1, -1, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])),   # Rx(+90)
    "top":    (2, +1, np.eye(3)),
    "bottom": (2, -1, np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])),  # Rx(180)
}


@dataclass
class BoxObject:
    """A movable rectangular box.

    Attributes
    ----------
    name : identifier, e.g. "box".
    half_extents : (ax, ay, az), half-sizes along the object's own x/y/z.
    mass : kg. Needed by Eq. 11 (object dynamics) at Milestone 5.
    initial_pose : pose in the world at scene setup.
    """

    name: str
    half_extents: tuple[float, float, float]
    mass: float
    initial_pose: pin.SE3 = field(default_factory=pin.SE3.Identity)

    @property
    def size(self) -> np.ndarray:
        """Full side lengths (2 * half-extents), which is what viewers want."""
        return 2.0 * np.asarray(self.half_extents, dtype=float)

    @property
    def inertia(self) -> np.ndarray:
        """Body-frame rotational inertia about the centre of mass, for Eq. 11.

        Solid cuboid: I_xx = m/12 * (h^2 + d^2), and cyclically. Written from the
        full side lengths, hence the factor on `size`.
        """
        sx, sy, sz = self.size
        return (self.mass / 12.0) * np.diag(
            [sy**2 + sz**2, sx**2 + sz**2, sx**2 + sy**2]
        )

    def face_patch(self, face: str, name: str | None = None) -> ContactPatch:
        """Build the contact patch for one face.

        Generated rather than hand-written in YAML: six consistent SE3s with correct
        outward normals and correctly-paired half-extents are easy to get subtly
        wrong by hand, and a flipped normal would produce a contact constraint that
        looks fine but holds the box the wrong way round.
        """
        if face not in _FACE_ROTATIONS:
            raise KeyError(f"unknown box face {face!r}; expected one of {sorted(_FACE_ROTATIONS)}")
        axis, sign, rotation = _FACE_ROTATIONS[face]

        translation = np.zeros(3)
        translation[axis] = sign * self.half_extents[axis]

        # The patch's in-plane half-extents are the box's two OTHER half-extents,
        # ordered to match the patch frame's local x and y axes.
        local_x_axis = int(np.argmax(np.abs(rotation[:, 0])))
        local_y_axis = int(np.argmax(np.abs(rotation[:, 1])))
        half_extents = (self.half_extents[local_x_axis], self.half_extents[local_y_axis])

        return ContactPatch(
            name=name or f"{self.name}_{face}",
            attachment=Attachment.OBJECT,
            parent=self.name,
            placement=pin.SE3(rotation, translation),
            half_extents=half_extents,
        )

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> BoxObject:
        """Build from a `configs/scenes/*.yaml` entry.

        Expected keys: `half_extents: [ax, ay, az]`, `mass`, and optionally
        `initial_position: [x, y, z]` / `initial_rpy_deg: [r, p, y]`.
        """
        translation = np.asarray(cfg.get("initial_position", [0.0, 0.0, 0.0]), dtype=float)
        rpy = np.deg2rad(np.asarray(cfg.get("initial_rpy_deg", [0.0, 0.0, 0.0]), dtype=float))
        ax, ay, az = cfg["half_extents"]
        return cls(
            name=name,
            half_extents=(float(ax), float(ay), float(az)),
            mass=float(cfg["mass"]),
            initial_pose=pin.SE3(pin.rpy.rpyToMatrix(*rpy), translation),
        )
