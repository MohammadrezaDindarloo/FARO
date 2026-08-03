"""Meshcat overlays that make a constraint's status visible.

The point of this module is that a number like `7d:+kappa_y = +0.4` means nothing
until you can see the foot tipping over its toe edge at that instant. It draws:

  * patches recoloured GREEN (satisfied) / RED (violated) live,
  * the contact force as an arrow,
  * the friction pyramid of Eq. 7c as four edges, so you can watch the force vector
    leave it,
  * the centre-of-pressure bound of Eq. 7d as a rectangle on the patch.

Everything here is presentation only. The physics lives in `faro.scenarios`, which
is headless and independently tested -- so a graphics problem can never be mistaken
for a constraint problem.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from faro.viz.meshcat_viz import SceneVisualizer

COLOR_OK = 0x22DD88          # satisfied
COLOR_VIOLATED = 0xFF3355    # violated
COLOR_FORCE = 0xFFCC00       # contact force arrow
COLOR_PYRAMID = 0x44AAFF     # friction pyramid edges (Eq. 7c)
COLOR_COP = 0xFF88DD         # centre-of-pressure rectangle (Eq. 7d)
COLOR_WEIGHT = 0xFF5533      # gravity, the mg term of Eqs. 10b / 11b
COLOR_MOMENT = 0xAA66FF      # a moment vector (Eq. 10b angular row, Eq. 11b)
COLOR_COM = 0xFFFFFF         # centre of mass -- what Eq. 10b's moment arm is about

# meshcat/three.js cylinders run along their local Y axis.
_Y_TO_Z = pin.SE3(pin.rpy.rpyToMatrix(np.pi / 2.0, 0.0, 0.0), np.zeros(3))

FORCE_SCALE = 1.0 / 400.0    # metres per newton, so 200 N draws as a 0.5 m arrow

# The overlay is an ANNOTATION layer, not scene geometry: it must never be hidden by
# the robot it is annotating. Eq. 7d's CoP rectangle lies exactly in the sole plane,
# so with normal depth testing it renders *inside* the G1's shoe mesh and is simply
# invisible -- leaving the CoP dot with no visible bound to be inside or outside of,
# which is the entire content of the constraint. `depthTest=False` is passed through
# meshcat's GenericMaterial **kwargs to three.js, which draws these last regardless
# of what is in front. Set to True to see the overlay occluded like real geometry.
_ON_TOP = dict(depthTest=False, depthWrite=False)

# Metres per newton-metre, for the Eq. 7d torsion rings. 1/100 puts the default
# capacity (mu_r f_z = 0.05 * 200 = 10 Nm) at a 0.10 m radius -- foot-sized, so
# "inside the ring" is judged against something the same scale as the sole.
MOMENT_SCALE = 1.0 / 100.0


def _circle(radius: float, span: float = 2.0 * np.pi, segments: int = 72) -> np.ndarray:
    """Consecutive-pair vertices for a LineSegments arc in the z = 0 plane.

    `span` may be negative, which sweeps the arc the other way round -- that is what
    makes the sign of a torsional moment visible rather than just its magnitude.
    """
    angles = np.linspace(0.0, span, segments + 1)
    points = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)], axis=1
    )
    return np.repeat(points, 2, axis=0)[1:-1].astype(np.float32)

# Rx(pi): our patch normals are uniformly OUTWARD, so a foot sole's +z points DOWN.
_FLIP = pin.SE3(np.diag([1.0, -1.0, -1.0]), np.zeros(3))


def contact_frame(patch_placement: pin.SE3) -> pin.SE3:
    """The frame Eq. 7's wrench actually lives in, given patch `a`'s placement.

    The paper writes f and kappa "expressed in frame b" -- the SUPPORTING patch --
    where `f_z >= 0` (7c) is the ground pushing UP on the foot. Our convention gives
    every patch a uniformly outward normal, so a sole's own +z points *down*, and
    `relative_patch_transform` reconciles the two with one Rx(pi) flip.

    Drawing in the patch's raw frame therefore renders the force arrow, the friction
    pyramid and the CoP rectangle BELOW the floor, buried and invisible -- which is
    exactly the bug this function exists to prevent. Applying the same Rx(pi) the
    constraint uses puts them above the floor, pointing the way physics does.

    Anchored to patch `a` rather than `b` on purpose: Eq. 7a leaves the relative yaw
    free, and 7d's CoP rectangle is the *foot's* outline, so it must follow the foot.
    """
    return patch_placement * _FLIP


class ConstraintOverlay:
    """Draws constraint status on top of a `SceneVisualizer`."""

    def __init__(self, vis: SceneVisualizer):
        self.vis = vis
        self.viewer = vis.viewer
        self._active: set[str] = set()

    # ------------------------------------------------------------------ status
    def set_patch_status(self, patch_names, violated: bool) -> None:
        """Recolour the patches involved in the constraint being demonstrated."""
        import meshcat.geometry as g

        color = COLOR_VIOLATED if violated else COLOR_OK
        for name in patch_names:
            patch = self.vis.scene.patches.get(name)
            if patch is None:
                continue
            hx, hy = patch.half_extents
            self.viewer[f"patches/{name}/plate"].set_object(
                g.Box([2 * hx, 2 * hy, 0.006]),
                g.MeshLambertMaterial(color=color, opacity=0.85),
            )

    def set_object_status(self, object_name: str, violated: bool) -> None:
        """Tint a whole scene object green/red.

        Eqs. 9 and 11 constrain an OBJECT, not a contact patch, so `set_patch_status`
        has nothing to colour for them and they used to show no violation at all --
        the box drove straight through the platform and stayed its normal colour.
        """
        import meshcat.geometry as g

        obj = self.vis.scene.objects.get(object_name)
        if obj is None:
            return
        self.viewer[f"objects/{object_name}"].set_object(
            g.Box(list(obj.size)),
            g.MeshLambertMaterial(color=COLOR_VIOLATED if violated else COLOR_OK,
                                  opacity=0.85),
        )

    def draw_status(self, position, violated: bool, *, radius: float = 0.045,
                    path: str = "overlay/status") -> None:
        """A green/red bead at the place the constraint actually lives.

        The universal fallback. Eqs. 13a/13b/13c constrain a JOINT -- there is no patch
        and no object to colour -- so without this a limit violation was visible only in
        the terminal. Put on the joint itself, it answers "where" as well as "whether".
        """
        import meshcat.geometry as g

        self._active.add(path)
        self.viewer[path].set_object(
            g.Sphere(radius),
            g.MeshLambertMaterial(color=COLOR_VIOLATED if violated else COLOR_OK,
                                  **_ON_TOP),
        )
        self.viewer[path].set_transform(
            pin.SE3(np.eye(3), np.asarray(position, dtype=float)).homogeneous)

    def draw_probe(self, pose: pin.SE3, size, *, color: int = 0x8899AA,
                   path: str = "overlay/probe") -> None:
        """A scenario-owned body, drawn because the scene's own object cannot show it.

        Eq. 11b's gyroscopic term vanishes identically for an isotropic body, and the
        scene's box is a CUBE, so `11-spin` has to bring its own elongated body. Drawing
        it -- rather than spinning the cube and asserting it is something else -- is the
        difference between a demonstration and a claim.
        """
        import meshcat.geometry as g

        self._active.add(path)
        self.viewer[path].set_object(
            g.Box(list(size)), g.MeshLambertMaterial(color=color, opacity=0.9))
        self.viewer[path].set_transform(pose.homogeneous)

    # ------------------------------------------------------------------- force
    def draw_force(self, placement: pin.SE3, force_local, path: str = "overlay/force") -> None:
        """Draw the contact force as an arrow rooted at the patch centre.

        `force_local` is expressed in the patch frame (the frame Eq. 7 uses), so it
        is rotated into the world here. The arrow visibly leans over as the
        tangential component grows -- and leaves the pyramid at the friction limit.
        """
        import meshcat.geometry as g

        f_world = placement.rotation @ np.asarray(force_local, dtype=float)
        magnitude = float(np.linalg.norm(f_world))
        self._active.add(path)
        if magnitude < 1e-9:
            self.viewer[path].delete()
            return

        length = magnitude * FORCE_SCALE
        direction = f_world / magnitude

        shaft = max(length * 0.8, 1e-4)
        head = max(length * 0.2, 1e-4)

        self.viewer[f"{path}/shaft"].set_object(
            g.Cylinder(shaft, 0.006), g.MeshLambertMaterial(color=COLOR_FORCE, **_ON_TOP)
        )
        self.viewer[f"{path}/head"].set_object(
            g.Cylinder(head, radiusTop=0.0, radiusBottom=0.018),
            g.MeshLambertMaterial(color=COLOR_FORCE, **_ON_TOP),
        )

        # Build a frame whose +z is the force direction, then place shaft and head.
        frame = pin.SE3(_rotation_from_z(direction), np.asarray(placement.translation))
        self.viewer[path].set_transform(frame.homogeneous)
        self.viewer[f"{path}/shaft"].set_transform(
            (_Y_TO_Z * pin.SE3(np.eye(3), np.array([0.0, shaft / 2.0, 0.0]))).homogeneous
        )
        self.viewer[f"{path}/head"].set_transform(
            (_Y_TO_Z * pin.SE3(np.eye(3), np.array([0.0, shaft + head / 2.0, 0.0]))).homogeneous
        )

    # ------------------------------------------------------- world-frame vectors
    def draw_vector(self, origin, vector, *, color: int = COLOR_FORCE,
                    scale: float = FORCE_SCALE, path: str = "overlay/vector") -> None:
        """An arrow given directly in WORLD coordinates.

        `draw_force` takes a patch placement and a force in that patch's frame, which
        suits Eq. 7. Eqs. 10 and 11 are written about the centre of mass and the
        object body frame, not about a contact patch, so their terms -- gravity at the
        CoM, a support force at a sliding contact point, a gyroscopic moment -- have
        no patch to hang off. Without this they had nothing drawn at all, and the
        dynamics scenarios were visually static.

        `scale` is per-arrow because the magnitudes differ by an order of magnitude:
        the robot's weight is 324 N, the box's is 19.6 N.
        """
        import meshcat.geometry as g

        origin = np.asarray(origin, dtype=float).ravel()
        vector = np.asarray(vector, dtype=float).ravel()
        magnitude = float(np.linalg.norm(vector))
        self._active.add(path)
        if magnitude < 1e-9:
            self.viewer[path].delete()
            return

        length = magnitude * scale
        shaft, head = max(length * 0.8, 1e-4), max(length * 0.2, 1e-4)
        self.viewer[f"{path}/shaft"].set_object(
            g.Cylinder(shaft, 0.006), g.MeshLambertMaterial(color=color, **_ON_TOP))
        self.viewer[f"{path}/head"].set_object(
            g.Cylinder(head, radiusTop=0.0, radiusBottom=0.018),
            g.MeshLambertMaterial(color=color, **_ON_TOP))
        self.viewer[path].set_transform(
            pin.SE3(_rotation_from_z(vector / magnitude), origin).homogeneous)
        self.viewer[f"{path}/shaft"].set_transform(
            (pin.SE3(np.eye(3), np.array([0.0, 0.0, shaft / 2.0])) * _Y_TO_Z).homogeneous)
        self.viewer[f"{path}/head"].set_transform(
            (pin.SE3(np.eye(3), np.array([0.0, 0.0, shaft + head / 2.0])) * _Y_TO_Z).homogeneous)

    def draw_point(self, position, *, color: int = COLOR_COM, radius: float = 0.02,
                   path: str = "overlay/point") -> None:
        """A small sphere -- used for the centre of mass, Eq. 10b's moment reference."""
        import meshcat.geometry as g

        self._active.add(path)
        self.viewer[path].set_object(
            g.Sphere(radius), g.MeshLambertMaterial(color=color, **_ON_TOP))
        self.viewer[path].set_transform(
            pin.SE3(np.eye(3), np.asarray(position, dtype=float).ravel()).homogeneous)

    # ---------------------------------------------------------------- pyramid
    def draw_friction_pyramid(
        self, placement: pin.SE3, mu: float, f_z: float, path: str = "overlay/pyramid"
    ) -> None:
        """Eq. 7c's friction pyramid, drawn as four edges from the contact point.

        The apex is the contact point and the four upper corners are at
        (+-mu f_z, +-mu f_z, f_z), scaled by the same factor as the force arrow. A
        force vector inside this pyramid satisfies Eq. 7c.

        Note it is a PYRAMID, not a cone: the corners stick out further than the
        edge midpoints, which is exactly why f_x = f_y = mu f_z is admissible.
        """
        import meshcat.geometry as g

        self._active.add(path)
        if f_z <= 0:
            self.viewer[path].delete()
            return

        limit = mu * f_z * FORCE_SCALE
        height = f_z * FORCE_SCALE
        corners = [
            [+limit, +limit, height], [-limit, +limit, height],
            [-limit, -limit, height], [+limit, -limit, height],
        ]

        segments = []
        for corner in corners:                    # apex -> each corner
            segments += [[0.0, 0.0, 0.0], corner]
        for i in range(4):                        # rim
            segments += [corners[i], corners[(i + 1) % 4]]

        self.viewer[path].set_object(
            g.LineSegments(
                g.PointsGeometry(np.asarray(segments, dtype=np.float32).T),
                g.LineBasicMaterial(color=COLOR_PYRAMID, **_ON_TOP),
            )
        )
        self.viewer[path].set_transform(placement.homogeneous)

    # -------------------------------------------------------------------- CoP
    def draw_cop_bounds(self, placement: pin.SE3, half_extents, path: str = "overlay/cop") -> None:
        """Eq. 7d's centre-of-pressure rectangle: the patch outline itself.

        The CoP must stay inside this rectangle, which is why the moment bounds are
        f_z times the half-extents -- and why the long and short directions differ.
        """
        import meshcat.geometry as g

        hx, hy = float(half_extents[0]), float(half_extents[1])
        corners = [[+hx, +hy, 0.0], [-hx, +hy, 0.0], [-hx, -hy, 0.0], [+hx, -hy, 0.0]]
        segments = []
        for i in range(4):
            segments += [corners[i], corners[(i + 1) % 4]]

        self._active.add(path)
        self.viewer[path].set_object(
            g.LineSegments(
                g.PointsGeometry(np.asarray(segments, dtype=np.float32).T),
                g.LineBasicMaterial(color=COLOR_COP, linewidth=3.0, **_ON_TOP),
            )
        )
        self.viewer[path].set_transform(placement.homogeneous)

    def draw_cop_marker(
        self, placement: pin.SE3, force_local, moment_local, path: str = "overlay/cop_point"
    ) -> None:
        """Where the centre of pressure actually is, given (f, kappa).

        From the standard relation kappa_x = -f_z * y_cop, kappa_y = +f_z * x_cop.
        Watching this dot slide to the patch edge as the moment grows is the clearest
        possible statement of what Eq. 7d bounds.
        """
        import meshcat.geometry as g

        f = np.asarray(force_local, dtype=float)
        kappa = np.asarray(moment_local, dtype=float)
        self._active.add(path)
        if abs(f[2]) < 1e-9:
            self.viewer[path].delete()
            return

        x_cop = float(kappa[1] / f[2])
        y_cop = float(-kappa[0] / f[2])

        self.viewer[path].set_object(
            g.Sphere(0.012), g.MeshLambertMaterial(color=COLOR_COP, **_ON_TOP)
        )
        marker = placement * pin.SE3(np.eye(3), np.array([x_cop, y_cop, 0.0]))
        self.viewer[path].set_transform(marker.homogeneous)

    # --------------------------------------------------------------- torsion
    def draw_torsion(
        self,
        placement: pin.SE3,
        moment_local,
        mu_r: float,
        force_local,
        path: str = "overlay/torsion",
    ) -> None:
        """Eq. 7d's TORSIONAL row, `|kappa_z| <= mu_r f_z`, as two circles.

        This row is invisible in every other overlay, and that is not an oversight of
        the drawing -- it is a property of the quantity. A moment about the contact
        normal changes neither the force (so the arrow and pyramid are unmoved) nor
        the centre of pressure (`x_cop = kappa_y/f_z`, `y_cop = -kappa_x/f_z`, neither
        of which contains kappa_z). Sweeping it, the entire scene is static and only
        the patch colour flips. So it gets its own indicator:

          * a BLUE ring of radius `mu_r * f_z` -- the torsional capacity, the limit;
          * a YELLOW arc of radius `|kappa_z|` -- the applied twist, swept in the
            direction of its sign so the drilling direction is visible.

        Read exactly like the force arrow against its pyramid: yellow inside blue is
        satisfied, yellow growing past blue is the violation. Scaled by MOMENT_SCALE
        so the limit ring is a foot-sized 0.10 m for the default 0.05 * 200 N.
        """
        import meshcat.geometry as g

        kappa_z = float(np.asarray(moment_local, dtype=float)[2])
        f_z = float(np.asarray(force_local, dtype=float)[2])
        ring_path, arc_path = f"{path}/limit", f"{path}/applied"
        self._active.update({path, ring_path, arc_path})

        if f_z <= 0.0:
            self.viewer[path].delete()
            return

        self.viewer[ring_path].set_object(
            g.LineSegments(
                g.PointsGeometry(_circle(mu_r * f_z * MOMENT_SCALE).T),
                g.LineBasicMaterial(color=COLOR_PYRAMID, **_ON_TOP),
            )
        )
        self.viewer[ring_path].set_transform(placement.homogeneous)

        if abs(kappa_z) < 1e-9:
            self.viewer[arc_path].delete()
            return

        # 300 degrees rather than a full turn, so the gap shows which way it twists.
        span = np.sign(kappa_z) * np.deg2rad(300.0)
        self.viewer[arc_path].set_object(
            g.LineSegments(
                g.PointsGeometry(_circle(abs(kappa_z) * MOMENT_SCALE, span).T),
                g.LineBasicMaterial(color=COLOR_FORCE, linewidth=3.0, **_ON_TOP),
            )
        )
        self.viewer[arc_path].set_transform(placement.homogeneous)

    # ----------------------------------------------------------------- cleanup
    def clear(self) -> None:
        """Remove every overlay, so scenarios do not leave debris behind."""
        for path in sorted(self._active):
            self.viewer[path].delete()
        self._active.clear()
        # Restore the standard patch AND object colours from Milestone 1, and put
        # every object back at its scene pose, so nothing a scenario did bleeds into
        # the next one -- neither a red tint nor a box left floating in mid-air.
        self.vis._draw_patches()
        self.vis.reset_object_poses()
        self.vis._draw_objects()


def _rotation_from_z(direction: np.ndarray) -> np.ndarray:
    """Any rotation whose +z axis is `direction` (the roll about it is arbitrary)."""
    z = np.asarray(direction, dtype=float)
    z = z / np.linalg.norm(z)
    # Pick a reference that is not parallel to z, so the cross products are stable.
    reference = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(reference, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
