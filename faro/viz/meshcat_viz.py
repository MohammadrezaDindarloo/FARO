"""Meshcat visualization of a FARO scene.

Meshcat runs a small web server on the cluster node and renders in a browser on
your laptop through VS Code port forwarding, which is why it -- not an OpenGL
window -- is the visualization path for Milestones 1-4. See SETUP.md section 4.

The important thing this module draws is the CONTACT PATCHES. Patch placements and
normal directions are the single easiest thing to get silently wrong in the whole
project: a flipped normal or a mis-signed half-extent produces constraints that
solve happily and mean the wrong thing. Drawing them makes the convention visible
and checkable by eye before any solver depends on it.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from faro.core.patches import Attachment, ContactPatch
from faro.scene.scene import Scene

# Colors chosen so attachment type is readable at a glance.
COLOR_ROBOT_PATCH = 0x22DD88       # green  -- moves with q
COLOR_OBJECT_PATCH = 0xFFAA22      # orange -- moves with the object pose
COLOR_ENV_PATCH = 0x4488FF         # blue   -- fixed in the world
COLOR_NORMAL = 0xFF3355            # red    -- the patch's outward +z
COLOR_BOX = 0xEEEEEE               # the paper's "white box"
COLOR_PLATFORM = 0x8899AA

_PATCH_THICKNESS = 0.004           # visual only; patches are mathematically planar
_NORMAL_LENGTH = 0.08
_NORMAL_RADIUS = 0.0035

# meshcat/three.js cylinders run along their local Y axis, so this rotation is
# needed to point one along +Z. Forgetting it is a classic silent 90-degree bug.
_Y_TO_Z = pin.SE3(pin.rpy.rpyToMatrix(np.pi / 2.0, 0.0, 0.0), np.zeros(3))


class SceneVisualizer:
    """Draws a `Scene` -- robot, objects, environment, and contact patches."""

    def __init__(self, scene: Scene, open_browser: bool = False, show_patches: bool = True):
        self.scene = scene
        self.show_patches = show_patches
        # The pose currently on screen. Robot patches are redrawn against it,
        # not against q_nominal -- see `_initial_world_placement`.
        self._last_q = None
        # The scene's own object poses, captured before anything can move them.
        # `update_object_pose` MUTATES `obj.initial_pose`, so without this copy the
        # "default" drifts to wherever the last scenario left the object and there is
        # nothing left to restore to.
        self._home_poses = {n: o.initial_pose.copy() for n, o in scene.objects.items()}

        self.viz = MeshcatVisualizer(
            scene.robot.model, scene.robot.collision_model, scene.robot.visual_model
        )
        self.viz.initViewer(open=open_browser)
        self.viz.loadViewerModel(rootNodeName="robot")
        self.viewer = self.viz.viewer

        self._draw_environment()
        self._draw_objects()
        if show_patches:
            self._draw_patches()

    # ------------------------------------------------------------------- public
    @property
    def url(self) -> str:
        return self.viewer.url()

    def display(self, q: np.ndarray) -> None:
        """Show configuration q and move every q-dependent patch with it."""
        q = np.asarray(q)
        self._last_q = q.copy()
        self.viz.display(q)
        if self.show_patches:
            self._update_robot_patches(q)

    def update_object_pose(self, object_name: str, pose: pin.SE3) -> None:
        """Move a movable object (and its face patches) to a new pose.

        Objects carry their own pose (Eq. 5b), so this is separate from `display(q)`.
        """
        self.scene.objects[object_name].initial_pose = pose
        self.viewer[f"objects/{object_name}"].set_transform(pose.homogeneous)
        if self.show_patches:
            for patch in self.scene.patches.values():
                if patch.attachment is Attachment.OBJECT and patch.parent == object_name:
                    self._set_patch_transform(patch, pose * patch.placement)

    # ------------------------------------------------------------------ drawing
    def _draw_environment(self) -> None:
        import meshcat.geometry as g

        for patch in self.scene.patches_by_attachment(Attachment.ENVIRONMENT):
            hx, hy = patch.half_extents
            # Draw environment patches as slabs extending downward from the surface,
            # so a floor looks like ground rather than a floating sheet.
            thickness = 0.02 if patch.name == "floor" else patch.placement.translation[2]
            thickness = max(float(thickness), 0.02)
            color = COLOR_PLATFORM if patch.name != "floor" else 0x303538

            self.viewer[f"environment/{patch.name}"].set_object(
                g.Box([2 * hx, 2 * hy, thickness]),
                g.MeshLambertMaterial(color=color, opacity=1.0),
            )
            # The patch plane is the slab's TOP face, so drop the centre by half.
            slab = patch.placement * pin.SE3(np.eye(3), np.array([0.0, 0.0, -thickness / 2.0]))
            self.viewer[f"environment/{patch.name}"].set_transform(slab.homogeneous)

    def reset_object_poses(self) -> None:
        """Put every object back where the SCENE says it lives.

        Scenarios that move an object leave it moved, because `update_object_pose`
        mutates the scene. Most scenarios never mention the box at all, so without
        this they inherit whatever the previous one left -- and the viewer state
        depends on the order you happened to run things in.

        The visible symptom: `11-hold` parks the box in mid-air at z = 0.45, so
        `11-spin`, which runs next and draws its own probe body at z = 0.60,
        appeared to intersect it. Running `-s 11-spin` alone showed no such overlap,
        which is exactly the signature of leaked state.
        """
        for name, pose in self._home_poses.items():
            self.update_object_pose(name, pose.copy())

    def _draw_objects(self) -> None:
        import meshcat.geometry as g

        for name, obj in self.scene.objects.items():
            self.viewer[f"objects/{name}"].set_object(
                g.Box(list(obj.size)),
                g.MeshLambertMaterial(color=COLOR_BOX, opacity=0.85),
            )
            self.viewer[f"objects/{name}"].set_transform(obj.initial_pose.homogeneous)

    def _draw_patches(self) -> None:
        import meshcat.geometry as g

        for patch in self.scene.patches.values():
            hx, hy = patch.half_extents
            color = {
                Attachment.ROBOT: COLOR_ROBOT_PATCH,
                Attachment.OBJECT: COLOR_OBJECT_PATCH,
                Attachment.ENVIRONMENT: COLOR_ENV_PATCH,
            }[patch.attachment]

            node = self.viewer[f"patches/{patch.name}"]
            node["plate"].set_object(
                g.Box([2 * hx, 2 * hy, _PATCH_THICKNESS]),
                g.MeshLambertMaterial(color=color, opacity=0.65),
            )

            # The outward normal, drawn as a stub along the patch's local +z.
            # This is the thing to eyeball: every normal must point AWAY from the
            # body it belongs to.
            node["normal"].set_object(
                g.Cylinder(_NORMAL_LENGTH, _NORMAL_RADIUS),
                g.MeshLambertMaterial(color=COLOR_NORMAL, opacity=0.95),
            )
            node["normal"].set_transform(
                (_Y_TO_Z * pin.SE3(np.eye(3), np.array([0.0, _NORMAL_LENGTH / 2.0, 0.0]))).homogeneous
            )

            self._set_patch_transform(patch, self._initial_world_placement(patch))

    # ------------------------------------------------------------------ helpers
    def _initial_world_placement(self, patch: ContactPatch) -> pin.SE3:
        """Where a patch goes when the whole set is (re)drawn.

        Robot patches must follow the pose CURRENTLY on screen, not `q_nominal`.
        `q_nominal` places the pelvis at z = 0, so its feet hang 0.78 m below the
        floor -- `standing_configuration` is what drops the robot onto the ground.
        Redrawing against the nominal therefore detached every green robot patch from
        the robot and left it floating underground until the next `display()`, which
        is precisely what a viewer sees while the runner waits for a keypress.
        """
        if patch.attachment is not Attachment.ROBOT:
            return self.scene.patch_world_placement(patch, None)
        q = self._last_q if self._last_q is not None else self.scene.robot.q_nominal
        return self.scene.patch_world_placement(patch, q)

    def _set_patch_transform(self, patch: ContactPatch, world: pin.SE3) -> None:
        self.viewer[f"patches/{patch.name}"].set_transform(world.homogeneous)

    def _update_robot_patches(self, q: np.ndarray) -> None:
        """Robot patches depend on q, so they must be re-placed on every display."""
        robot = self.scene.robot
        pin.forwardKinematics(robot.model, robot.data, np.asarray(q))
        pin.updateFramePlacements(robot.model, robot.data)
        for patch in self.scene.patches_by_attachment(Attachment.ROBOT):
            frame = robot.data.oMf[robot.frame_id(patch.parent)]
            self._set_patch_transform(patch, frame * patch.placement)
