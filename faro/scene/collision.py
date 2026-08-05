"""The scene's collision bodies and pair set -- the input side of Eq. 9.

    "All bodies in the robot and scene are modeled using convex geometries, and
     collision avoidance is imposed with the signed-distance approximation of
     Schulman et al. For each collision pair A and B, the collision is queried using
     the GJK algorithm as implemented in coal."   -- Section II-C 2

`faro/constraints/collision.py` holds the MATH of Eq. 9 -- freeze a GJK query, build
a differentiable expression from the witness points. This module holds everything
that has to happen before that: which bodies exist, what convex shape each one is,
where it is as a function of the decision variables, and which pairs get a row.

CONVEX, AS THE PAPER SAYS -- AND THE G1 IS NOT
----------------------------------------------
The G1's URDF collision geometry is 24 triangle meshes (`BVHModelOBBRSS`), 8 spheres
and 4 cylinders. Only the last 12 are convex. coal will happily run a distance query
against a BVH, but the Schulman approximation is a statement about convex sets: it
freezes ONE witness pair and treats the distance as linear in the body transforms.
On a non-convex mesh the witness pair can jump between concave features, and the
frozen model then describes a different pocket of geometry than the one the solver is
moving toward.

So the meshes are replaced by their convex hulls -- `buildConvexRepresentation` --
which is what the paper's sentence asks for and what makes the query cheap enough to
run for every pair at every refresh.

THE PAIR SET (docs/ambiguities.md #19)
--------------------------------------
The paper says "for each collision pair A and B" and never defines the set. We take
every pair, with three exclusions that are structural rather than tuning:

  * SAME LINK. Two geometries on one rigid body cannot move relative to each other,
    so their signed distance is a constant. The row is either trivially satisfied or
    permanently violated, and it can never be influenced by q.
  * ADJACENT LINKS (parent-child in the kinematic tree). These touch by construction
    -- that is what a joint is. Measured on the G1: 12 pairs interpenetrate at the
    nominal pose and all 12 are adjacent. Keeping them makes the nominal posture
    itself infeasible.
  * ENVIRONMENT-ENVIRONMENT. Both static; the distance does not depend on q at all.

Everything else is in: robot self-collision across the tree, robot against the box,
robot against the floor and platform, and the box against the floor and platform.
Both exclusions are flags, so the "literally every pair" set can be built and
measured rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from faro.constraints.block import ConstraintBlock
from faro.constraints.collision import collision_avoidance, query_witness
from faro.core.patches import Attachment
from faro.scene.scene import Scene


def _coal():
    try:
        import coal
    except ImportError:  # pragma: no cover - older stacks
        import hppfcl as coal
    return coal


@dataclass(frozen=True)
class CollisionBody:
    """One convex body, plus how to find it in the world.

    `kind` decides which decision variables its placement depends on, mirroring
    `Attachment` for patches:

        robot       -> `joint` and the robot configuration q^r
        object      -> `owner` and that object's pose q^o
        environment -> fixed; `placement` IS the world placement
    """

    name: str
    geometry: object
    kind: str
    placement: pin.SE3
    joint: int = -1
    owner: str = ""

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"CollisionBody({self.name!r}, {self.kind})"


# =============================================================================
# Pair classes -- the vocabulary a config selects pairs with
# =============================================================================
# Eq. 9 says "for each collision pair A and B" and defines no set, so which pairs
# exist is ours to decide (docs/ambiguities.md #19). Rather than expose that as a
# fixed policy, every body gets a GROUP name and every pair a CLASS built from the
# two groups. A config can then say "robot-box" or "box" and mean something exact.
#
# The groups on `box_placement` are: robot, box, floor, tabletop -- giving classes
# robot-robot (579 pairs), box-robot (36), floor-robot (36), robot-tabletop (36),
# box-floor (1), box-tabletop (1).
def body_group(body: CollisionBody) -> str:
    """The name a config uses for this body: `robot`, or the object/patch's own name.

    Objects and environment surfaces get their OWN name rather than a shared
    `object`/`environment` label, because the whole point is to be able to say
    "always watch the box" without also meaning some other object.
    """
    if body.kind == "robot":
        return "robot"
    if body.kind == "object":
        return body.owner
    return body.name.split("/", 1)[-1]


def pair_class(a: CollisionBody, b: CollisionBody) -> str:
    """Canonical class name for a pair, e.g. `box-robot`.

    Alphabetically ordered so `robot-box` and `box-robot` are the same class and a
    config cannot half-select one by writing it the other way round.
    """
    return "-".join(sorted((body_group(a), body_group(b))))


def select_pairs(bodies, pairs, selectors) -> set[tuple[int, int]]:
    """Resolve config selectors to a set of pair indices.

    A selector is one of:

      * a GROUP    -- `box` means every pair with the box on either side;
      * a CLASS    -- `robot-box` (either order) means exactly that class;
      * `all`      -- every pair.

    `None` or an empty list selects NOTHING. That asymmetry with `all` is deliberate:
    the two callers want opposite defaults (`include_pairs` defaults to everything,
    `always_active` to nothing), and a single value meaning both would make one of
    them a silent surprise.

    An unrecognized selector RAISES, listing what is available. A typo that quietly
    selected nothing would look exactly like a working config while disabling the
    protection it was written to add.
    """
    if selectors is None:
        return set()
    if isinstance(selectors, str):
        selectors = [selectors]

    groups = {body_group(b) for b in bodies}
    hyphenated = sorted(g for g in groups if "-" in g)
    if hyphenated:
        raise ValueError(
            f"collision group names must not contain '-', got {hyphenated}. The "
            f"character separates the two halves of a pair class, so a name using it "
            f"makes `a-b` ambiguous. Rename the object or environment patch."
        )
    classes = {pair_class(bodies[i], bodies[j]) for i, j in pairs}

    chosen: set[tuple[int, int]] = set()
    for raw in selectors:
        key = str(raw).strip()
        if key == "all":
            return set(pairs)
        if key in groups:
            chosen |= {(i, j) for i, j in pairs
                       if key in (body_group(bodies[i]), body_group(bodies[j]))}
            continue
        canonical = "-".join(sorted(key.split("-")))
        if canonical in classes:
            chosen |= {(i, j) for i, j in pairs
                       if pair_class(bodies[i], bodies[j]) == canonical}
            continue
        raise ValueError(
            f"unknown collision selector {key!r}. Use a group "
            f"{sorted(groups)}, a pair class {sorted(classes)}, or 'all'."
        )
    return chosen


# =============================================================================
# Building the bodies
# =============================================================================
def _convexified(geometry):
    """The convex hull of a mesh; anything already convex is returned unchanged."""
    if hasattr(geometry, "buildConvexRepresentation"):
        try:
            geometry.buildConvexRepresentation(False)
            if getattr(geometry, "convex", None) is not None:
                return geometry.convex
        except Exception:  # pragma: no cover - shape has no hull to build
            pass
    return geometry


def robot_bodies(scene: Scene) -> list[CollisionBody]:
    """One body per entry of the robot's Pinocchio `GeometryModel`."""
    return [
        CollisionBody(
            name=geom.name,
            geometry=_convexified(geom.geometry),
            kind="robot",
            placement=geom.placement.copy(),
            joint=int(geom.parentJoint),
        )
        for geom in scene.robot.collision_model.geometryObjects
    ]


def object_bodies(scene: Scene) -> list[CollisionBody]:
    """One box per movable object, centred on its body frame."""
    coal = _coal()
    return [
        CollisionBody(
            name=f"object/{name}",
            geometry=coal.Box(*obj.size),
            kind="object",
            placement=pin.SE3.Identity(),
            owner=name,
        )
        for name, obj in sorted(scene.objects.items())
    ]


def environment_bodies(scene: Scene, *, floor_thickness: float = 2.0) -> list[CollisionBody]:
    """A solid slab under every environment patch.

    The slab hangs BELOW the patch plane, so the patch surface is its top face and
    the volume a foot must not enter is underneath -- the same construction the
    visualizer draws, deliberately, so what you see is what Eq. 9 tests.

    A patch is a rectangle with no thickness and no interior, so it cannot be a
    collision body: GJK against a zero-volume set gives a distance that flips sign
    for no physical reason as a body crosses the plane.

    THE SLAB MUST BE DEEP, NOT JUST NON-ZERO. `floor_thickness` was 0.05 m, and a
    robot dropped 0.30 m through the floor audited as 3.5 mm of penetration -- because
    it had passed clean THROUGH the slab and out the bottom, where the nearest surface
    is once again close. Tunnelling like that reads as "almost fine" to every distance
    query and to the constraint built from it. 2.0 m is deeper than anything in this
    scene can travel in a solve. The platform's thickness is its own height instead:
    it stands on the floor, so it is already solid all the way down.
    """
    coal = _coal()
    bodies = []
    for patch in scene.patches_by_attachment(Attachment.ENVIRONMENT):
        hx, hy = patch.half_extents
        height = float(patch.placement.translation[2])
        thickness = max(height, floor_thickness)
        centre = patch.placement * pin.SE3(np.eye(3), np.array([0.0, 0.0, -thickness / 2.0]))
        bodies.append(
            CollisionBody(
                name=f"environment/{patch.name}",
                geometry=coal.Box(2.0 * hx, 2.0 * hy, thickness),
                kind="environment",
                placement=centre,
            )
        )
    return bodies


# =============================================================================
# The model
# =============================================================================
@dataclass
class SceneCollisionModel:
    """Every collision body in the scene, and the pairs Eq. 9 is applied to."""

    scene: Scene
    bodies: list[CollisionBody] = field(default_factory=list)
    pairs: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def cached(cls, scene: Scene) -> SceneCollisionModel:
        """The default model for a scene, built once and reused.

        Building is not cheap -- 24 convex hulls out of the G1's meshes -- and it is
        pure function of the scene, so rebuilding it per solve is wasted work that
        dominates the small solves. Attached to the scene rather than kept in a module
        dict so two scenes cannot collide and nothing outlives its scene.
        """
        model = getattr(scene, "_collision_model", None)
        if model is None:
            model = cls.build(scene, include=scene.collision.get("include_pairs", "all"))
            object.__setattr__(scene, "_collision_model", model)
        return model

    @classmethod
    def build(
        cls,
        scene: Scene,
        *,
        exclude_same_link: bool = True,
        exclude_adjacent: bool = True,
        exclude_static: bool = True,
        include: str | list[str] | None = "all",
    ) -> SceneCollisionModel:
        bodies = robot_bodies(scene) + object_bodies(scene) + environment_bodies(scene)
        model = scene.robot.model

        def adjacent(a: CollisionBody, b: CollisionBody) -> bool:
            if a.kind != "robot" or b.kind != "robot":
                return False
            return model.parents[a.joint] == b.joint or model.parents[b.joint] == a.joint

        pairs = []
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                a, b = bodies[i], bodies[j]
                if exclude_same_link and a.kind == "robot" and b.kind == "robot" and a.joint == b.joint:
                    continue
                if exclude_adjacent and adjacent(a, b):
                    continue
                if exclude_static and a.kind == "environment" and b.kind == "environment":
                    continue
                pairs.append((i, j))

        # `include` removes whole CLASSES of pair from the problem -- a modelling
        # decision, not a speed knob, and a different thing from `activation_distance`.
        # Dropping `robot-robot` says "I do not care whether this robot passes through
        # itself"; the cutoff only ever says "this pair cannot become active in one
        # step". The audit still measures every pair that is in the set, so a class
        # removed here is genuinely unwatched -- which is why it is spelled out in the
        # config rather than inferred.
        if include not in (None, "all"):
            keep = select_pairs(bodies, pairs, include)
            pairs = [p for p in pairs if p in keep]

        return cls(scene=scene, bodies=bodies, pairs=pairs)

    # ------------------------------------------------------------ pair classes
    def always_active_pairs(self) -> set[tuple[int, int]]:
        """Pairs the scene config exempts from `activation_distance`.

        WHY THIS EXISTS. The cutoff rests on an assumption -- that a pair currently
        far apart cannot be driven into contact within one convex subproblem -- and
        that assumption is about how far a body can TRAVEL per step. It holds well for
        robot links, which are bounded by kinematics. It does not hold for a movable
        object, whose pose is a free decision variable in `q`: the solver can translate
        and rotate the box arbitrarily in a single pass.

        Measured on the ten-target tour: the worst-penetration pair was one involving
        the box or the platform in 9 of 9 feasible entries. Not once was it a
        robot-robot self-collision.
        """
        return select_pairs(self.bodies, self.pairs,
                            self.scene.collision.get("always_active"))

    def class_counts(self) -> dict[str, int]:
        """`{pair class: count}` for the pairs actually in this model."""
        from collections import Counter

        return dict(Counter(pair_class(self.bodies[i], self.bodies[j])
                            for i, j in self.pairs))

    # ------------------------------------------------------------ placements
    def world_placement(self, body: CollisionBody, q: np.ndarray,
                        object_poses: dict[str, pin.SE3] | None = None) -> pin.SE3:
        """Numeric world placement, for running the GJK query."""
        if body.kind == "environment":
            return body.placement
        if body.kind == "object":
            poses = object_poses or {}
            base = poses.get(body.owner) or self.scene.objects[body.owner].initial_pose
            return base * body.placement

        robot = self.scene.robot
        pin.forwardKinematics(robot.model, robot.data, np.asarray(q, dtype=float))
        return robot.data.oMi[body.joint] * body.placement

    def symbolic_placement(self, body: CollisionBody, sym):
        """(R, p) as expressions in the decision variables -- the `T^w(x)` of Eq. 9."""
        import casadi as ca

        offset_R = ca.DM(body.placement.rotation)
        offset_p = ca.DM(body.placement.translation.reshape(3, 1))

        if body.kind == "environment":
            return offset_R, offset_p
        if body.kind == "object":
            R_body, p_body = sym.object_placement(body.owner)
        else:
            R_body, p_body = sym.joint_placement(body.joint)
        return R_body @ offset_R, R_body @ offset_p + p_body

    # ------------------------------------------------------------- the rows
    def witnesses(self, q: np.ndarray, object_poses: dict[str, pin.SE3] | None = None):
        """Run GJK on every pair at one linearization point.

        Returns `[(i, j, WitnessData), ...]`. This is the step Eq. 9 freezes: after
        it, the witness points and normal are constants and the only thing left
        varying is the body transforms.
        """
        coal = _coal()
        placements = [self.world_placement(b, q, object_poses) for b in self.bodies]
        transforms = [coal.Transform3s(p.rotation, p.translation) for p in placements]

        return [
            (i, j, query_witness(self.bodies[i].geometry, transforms[i],
                                 self.bodies[j].geometry, transforms[j]))
            for i, j in self.pairs
        ]

    def bodies_for_patch(self, patch_name: str) -> set[int]:
        """Body indices belonging to whatever a contact patch is mounted on.

        A patch is a rectangle; a collision body is a solid. They are not the same
        object and the mapping is not one-to-one -- the left hand patch is a 27x30 mm
        window on the wrist cut face, while `left_wrist_yaw_link_0` is the whole
        forearm hull. That gap is exactly why excluding a contact pair from Eq. 9 is
        not free: it stops constraining the WHOLE link, not just the patch.
        """
        patch = self.scene.patches[patch_name]
        if patch.attachment is Attachment.ENVIRONMENT:
            return {k for k, b in enumerate(self.bodies) if b.name == f"environment/{patch_name}"}
        if patch.attachment is Attachment.OBJECT:
            return {k for k, b in enumerate(self.bodies) if b.name == f"object/{patch.parent}"}

        model = self.scene.robot.model
        joint = int(model.frames[self.scene.robot.frame_id(patch.parent)].parentJoint)
        return {k for k, b in enumerate(self.bodies) if b.kind == "robot" and b.joint == joint}

    def contact_pairs(self, target) -> set[tuple[int, int]]:
        """Body pairs that `target`'s contacts hold in contact -- Eq. 7's own pairs.

        `target` is a `ContactMode` or `ContactEdge`; both expose `active_pairs`.
        """
        out = set()
        for a_name, b_name in target.active_pairs(self.scene):
            for i in self.bodies_for_patch(a_name):
                for j in self.bodies_for_patch(b_name):
                    out.add((min(i, j), max(i, j)))
        return out

    def blocks(self, sym, q: np.ndarray, object_poses: dict[str, pin.SE3] | None = None,
               *, margin: float = 0.0, activation_distance: float | None = None,
               relaxed: set[tuple[int, int]] | None = None,
               relaxed_margin: float = -1.0e-3,
               always_active: set[tuple[int, int]] | None = None) -> list[ConstraintBlock]:
        """Eq. 9, linearized at `(q, object_poses)`, for the pairs that can matter.

        `activation_distance` is Schulman et al.'s construction: a pair whose bodies
        are already far apart cannot be driven into contact within one convex
        subproblem, so it gets no row. It is not an approximation of the constraint --
        every row that is generated is exactly Eq. 9 -- it is a statement about which
        rows can be active, and it is what makes the method tractable. TrajOpt sets the
        equivalent threshold from its safety margin plus a buffer for how far a body can
        travel in a step.

        Measured on `box_placement`: 689 pairs, of which 53 are within 0.10 m at the
        nominal pose and 52-64 at the solutions. So this is a ~12x cut in rows, and the
        rows it drops are ones whose constraint is slack by 10 cm.

        THE PART THAT MAKES IT SAFE
        ---------------------------
        Two things, and neither is optional:

          * the pair set is RE-SELECTED at every refresh pass, not chosen once. A pair
            that was far at q_nom and close after the first solve gets its row on the
            second pass, which is why the refresh loop is what licenses the cutoff;
          * the answer is re-checked against ALL pairs at the end. `check` reports
            `max_penetration` from the full set, so a cutoff that was too small shows up
            as a number rather than as a robot quietly standing inside the platform.

        Passing `None` restores every pair, which is what the comparison in
        `tests/test_collision_model.py` uses to show the cutoff changes no verdict.

        `relaxed` names the pairs the CONTACT MODE holds in contact (see
        `contact_pairs`). They keep their row, at `relaxed_margin` -- a small NEGATIVE
        margin, i.e. `sd >= -1 mm` -- rather than being dropped.

        Why not simply drop them, which is the usual advice? Because a patch is not a
        body. The left hand patch is a 27x30 mm window on the wrist cut face, while
        `left_wrist_yaw_link_0` is the whole forearm hull, so dropping the pair stops
        constraining the entire forearm against the box. Measured: with the pairs
        dropped, `lift` -- which holds the box in mid-air by both hands -- came back
        feasible with the forearm 12.8 cm INSIDE the box.

        Why not leave them at `margin = 0`, which is exact? Because `sd = 0` is exactly
        where they sit, so round-off alone reads as a hair of penetration, the
        audit-driven refresh loop never calls the answer clean, and every solve runs to
        its cap. That cost 2.7x.

        The relaxed row is the middle: 1 mm of slack is enough that the loop converges,
        and 1 mm is not enough for a forearm to disappear into a box.
        """
        out = []
        for i, j, witness in self.witnesses(q, object_poses):
            in_contact = bool(relaxed) and (i, j) in relaxed
            exempt = bool(always_active) and (i, j) in always_active
            if (activation_distance is not None
                    and witness.distance > activation_distance
                    and not exempt):
                continue
            R_A, p_A = self.symbolic_placement(self.bodies[i], sym)
            R_B, p_B = self.symbolic_placement(self.bodies[j], sym)
            out.append(
                collision_avoidance(
                    R_A, p_A, R_B, p_B, witness,
                    margin=relaxed_margin if in_contact else margin,
                    name=f"collision[{self.bodies[i].name}|{self.bodies[j].name}]",
                )
            )
        return out

    def worst_penetration(self, q: np.ndarray, object_poses=None) -> tuple[float, str]:
        """Deepest interpenetration over ALL pairs, and which pair. 0.0 if clear.

        Deliberately ignores `activation_distance`: this is the independent audit of
        an answer produced with a cutoff, so using the same reduced set would make it
        agree by construction.
        """
        worst, where = 0.0, ""
        for i, j, w in self.witnesses(q, object_poses):
            if w.distance < worst:
                worst, where = w.distance, f"{self.bodies[i].name} | {self.bodies[j].name}"
        return worst, where

    # ---------------------------------------------------------------- report
    def report(self, q: np.ndarray, object_poses=None, *, limit: int = 10) -> str:
        """Which pairs are in collision at a given state. For deciding, not solving."""
        found = [
            (w.distance, self.bodies[i].name, self.bodies[j].name)
            for i, j, w in self.witnesses(q, object_poses) if w.distance < 0.0
        ]
        found.sort()
        lines = [
            f"{len(self.bodies)} bodies, {len(self.pairs)} pairs, "
            f"{len(found)} interpenetrating"
        ]
        lines += [f"    {d:+.5f} m  {a} | {b}" for d, a, b in found[:limit]]
        if len(found) > limit:
            lines.append(f"    ... and {len(found) - limit} more")
        return "\n".join(lines)

    # ------------------------------------------- Eq. 9 with parameterized witnesses
    def fixed_pairs(self, q: np.ndarray, object_poses=None, *,
                    activation_distance: float | None = None,
                    always_active: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
        """The pair list a parametric problem is built against, chosen ONCE.

        A parametric Eq. 9 needs a FIXED pair set: the parameter vector is
        `9 * len(pairs)` numbers, and a set that changed between refreshes would change
        the size of the solver's parameter block -- which for a code-generating solver
        means regenerating the very C we parameterized in order to reuse.

        So the cutoff is applied once, at the point given, and the resulting list is
        frozen for the life of the solver. That is a real difference from the Ipopt
        path, where the active set is re-selected every pass: a pair that starts beyond
        `activation_distance` and closes later gets no row here. `always_active`
        exists to hold exactly the pairs where that matters -- anything touching a
        movable object, whose pose is a free variable and can therefore move
        arbitrarily far in one solve.

        Passing `activation_distance=None` freezes the complete set, which is the
        conservative choice and what the paper's own `0 <= sd` implies.
        """
        if activation_distance is None:
            return list(self.pairs)
        keep = set(always_active or ())
        for i, j, witness in self.witnesses(q, object_poses):
            if witness.distance <= activation_distance:
                keep.add((i, j))
        return [p for p in self.pairs if p in keep]

    def parametric_blocks(self, sym, pairs, params, *, margin: float = 0.0,
                          relaxed: set[tuple[int, int]] | None = None,
                          relaxed_margin: float = -1.0e-3) -> list[ConstraintBlock]:
        """Eq. 9 over `pairs`, reading the witness data from `params`.

        The expression depends on `pairs` only through the body placements, so it is
        built once per knot and re-linearized by writing new values into `params`.
        `witness_values` produces those values in the matching order.
        """
        from faro.constraints.collision import collision_avoidance_parametric

        out = []
        for index, (i, j) in enumerate(pairs):
            R_A, p_A = self.symbolic_placement(self.bodies[i], sym)
            R_B, p_B = self.symbolic_placement(self.bodies[j], sym)
            in_contact = bool(relaxed) and (i, j) in relaxed
            out.append(collision_avoidance_parametric(
                R_A, p_A, R_B, p_B, params, index,
                margin=relaxed_margin if in_contact else margin,
                name=f"collision[{self.bodies[i].name}|{self.bodies[j].name}]",
            ))
        return out

    def witness_values(self, pairs, q: np.ndarray, object_poses=None) -> np.ndarray:
        """Numeric witness data for `pairs`, in the order `parametric_blocks` expects.

        Queried per pair rather than by filtering `witnesses()`, so the ORDER is the
        pair list's and cannot drift from the expression's indexing.
        """
        from faro.constraints.collision import query_witness, witness_parameter_values

        coal = _coal()
        placements = {}
        for i, j in pairs:
            for k in (i, j):
                if k not in placements:
                    pose = self.world_placement(self.bodies[k], q, object_poses)
                    placements[k] = coal.Transform3s(pose.rotation, pose.translation)
        return witness_parameter_values([
            query_witness(self.bodies[i].geometry, placements[i],
                          self.bodies[j].geometry, placements[j])
            for i, j in pairs
        ])
