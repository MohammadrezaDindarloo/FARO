"""Scene assembly (paper Fig. 2, Table IV).

    "The scene defines the robot, movable objects, environment geometry, and
     available contact interfaces from which candidate contact-mode sequences are
     generated."   -- Fig. 2 caption

A `Scene` is therefore the single object that later stages consume:
  * mode/edge (Eq. 14) and KSO (Eq. 15) read patches to build contact constraints,
  * the tree search (Alg. 1) reads `interfaces` to enumerate its discrete actions.

It holds no optimization state and does no solving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from faro.core.patches import Attachment, ContactPatch, Interface
from faro.robots.robot_model import RobotModel
from faro.scene.objects import BoxObject
from faro.utils.config import load_config


@dataclass
class Scene:
    """Robot + movable objects + environment patches + the allowed-contact graph."""

    name: str
    robot: RobotModel
    objects: dict[str, BoxObject] = field(default_factory=dict)
    patches: dict[str, ContactPatch] = field(default_factory=dict)
    interfaces: dict[str, Interface] = field(default_factory=dict)
    gravity: float = 9.81

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_config(cls, name_or_cfg: str | dict) -> Scene:
        """Build from `configs/scenes/<name>.yaml`."""
        cfg = load_config("scenes", name_or_cfg) if isinstance(name_or_cfg, str) else name_or_cfg

        robot = RobotModel.from_config(cfg["robot"])
        scene = cls(name=cfg.get("name", "scene"), robot=robot, gravity=float(cfg.get("gravity", 9.81)))

        # --- movable objects, each contributing its own face patches -----------
        for obj_name, obj_cfg in (cfg.get("objects") or {}).items():
            obj = BoxObject.from_config(obj_name, obj_cfg)
            scene.objects[obj_name] = obj
            for face in obj_cfg.get("faces", ["front", "rear", "left", "right", "top", "bottom"]):
                patch = obj.face_patch(face)
                scene.patches[patch.name] = patch

        # --- explicitly declared patches: robot end-effectors + environment ----
        for patch_name, patch_cfg in (cfg.get("patches") or {}).items():
            scene.patches[patch_name] = ContactPatch.from_config(patch_name, patch_cfg)

        # --- interfaces (Table IV) --------------------------------------------
        for iface_name, iface_cfg in (cfg.get("interfaces") or {}).items():
            patch_name = iface_cfg["patch"]
            if patch_name not in scene.patches:
                raise KeyError(
                    f"interface {iface_name!r} references unknown patch {patch_name!r}. "
                    f"Known patches: {sorted(scene.patches)}"
                )
            scene.interfaces[iface_name] = Interface(
                name=iface_name,
                patch=scene.patches[patch_name],
                allowed=list(iface_cfg.get("allowed", [])),
            )

        scene.validate()
        return scene

    # --------------------------------------------------------------- validation
    def validate(self) -> None:
        """Catch scene-definition mistakes here rather than inside a solver.

        A typo in an `allowed` list would otherwise surface as a mysteriously
        infeasible NLP many milestones later.
        """
        for patch in self.patches.values():
            if patch.attachment is Attachment.ROBOT:
                self.robot.frame_id(patch.parent)  # raises with a helpful message
            elif patch.attachment is Attachment.OBJECT and patch.parent not in self.objects:
                raise KeyError(
                    f"patch {patch.name!r} is attached to unknown object {patch.parent!r}. "
                    f"Known objects: {sorted(self.objects)}"
                )

        for iface in self.interfaces.values():
            for target in iface.allowed:
                if target not in self.patches:
                    raise KeyError(
                        f"interface {iface.name!r} allows contact with unknown patch "
                        f"{target!r}. Known patches: {sorted(self.patches)}"
                    )
            if iface.patch.name in iface.allowed:
                raise ValueError(f"interface {iface.name!r} lists its own patch as a contact target")

    def containment_report(self) -> list[str]:
        """Allowed pairs that Eq. 7b can never satisfy, whatever the robot does.

        Eq. 7b is a CONTAINMENT condition, |p_{x,y}| <= (^b xi_b) - (^b xi_a): the
        interface's own patch `a` must fit inside its target `b`. If it cannot, every
        contact using that pair is infeasible for a reason that has nothing to do
        with the robot -- and inside a tree search it would look like an unlucky
        branch rather than a scene bug.

        Returns human-readable descriptions; empty means every pair is satisfiable.
        Not raised from `validate()` because an equal-sized pair is legitimate at the
        boundary, and because a scene may deliberately include a pair it never uses.
        """
        from faro.constraints.contact import patch_containment_is_possible

        problems = []
        for iface in self.interfaces.values():
            for target in iface.allowed:
                a, b = iface.patch, self.patches[target]
                if not patch_containment_is_possible(a.half_extents, b.half_extents):
                    problems.append(
                        f"{iface.name}: patch {a.name} {a.half_extents} does not fit "
                        f"inside {b.name} {b.half_extents} -- Eq. 7b can never hold"
                    )
        return problems

    # ------------------------------------------------------------------ queries
    def branching_factor(self) -> int:
        """Size of the discrete action set at one tree-search node (Alg. 1).

        The product over interfaces of (number of allowed partners + 1 for `free`).
        The paper reports 108 for its box-placement task with the Table IV
        interfaces, which makes this a direct, checkable reproduction target.

        NOTE: this is the raw combinatorial count. It ignores the consistency
        condition of Section II-A (if A touches B then B touches A), so the paper's
        108 is the number *after* whatever consistency filtering they apply. Treat a
        mismatch as a question about consistency handling, not a bug here.
        """
        total = 1
        for iface in self.interfaces.values():
            total *= iface.branching_factor()
        return total

    def patch_world_placement(self, patch: ContactPatch, q: np.ndarray | None = None) -> pin.SE3:
        """World placement of a patch.

        This is the one function that resolves the three attachment kinds, and it is
        the numeric counterpart of what the symbolic constraint code will do in
        Milestone 2:

          ROBOT       -> forward kinematics of the parent frame at q, then the patch offset
          OBJECT      -> the object's current pose, then the patch offset
          ENVIRONMENT -> the patch placement is already a world placement
        """
        if patch.attachment is Attachment.ENVIRONMENT:
            return patch.placement.copy()
        if patch.attachment is Attachment.OBJECT:
            return self.objects[patch.parent].initial_pose * patch.placement
        if q is None:
            raise ValueError(f"patch {patch.name!r} is robot-attached; a configuration q is required")
        return self.robot.frame_placement(q, patch.parent) * patch.placement

    def patches_by_attachment(self, attachment: Attachment) -> list[ContactPatch]:
        return [p for p in self.patches.values() if p.attachment is attachment]

    def summary(self) -> str:
        """Human-readable description, printed by the demo scripts."""
        lines = [
            f"Scene: {self.name}",
            f"  robot      : {self.robot.name}  nq={self.robot.nq} nv={self.robot.nv} "
            f"actuated={self.robot.n_actuated}",
            f"  objects    : {', '.join(sorted(self.objects)) or '(none)'}",
            f"  patches    : {len(self.patches)}",
        ]
        for attachment in Attachment:
            names = sorted(p.name for p in self.patches_by_attachment(attachment))
            if names:
                lines.append(f"    {attachment.value:<12}: {', '.join(names)}")
        lines.append(f"  interfaces : {len(self.interfaces)}  "
                     f"(raw branching factor {self.branching_factor()})")
        for iface in self.interfaces.values():
            allowed = ", ".join(iface.allowed) or "-"
            lines.append(f"    {iface.name:<12}-> free, {allowed}")
        return "\n".join(lines)
