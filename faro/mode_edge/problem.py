"""Contact mode and edge feasibility -- paper Eq. 14 (Section II-D).

    "In order to prune kinematically infeasible contact modes and transitions
     between them, we formulate the following inverse kinematics optimization
     problem (14). A contact mode c is said to be infeasible if the resulting
     nonlinear program does not converge within a number of maximum iterations.
     To test whether two modes c1 and c2 can be connected via an edge, c is
     replaced by c1 u c2, representing the instantaneous transition between the
     two modes."

        min_q  (q - q_nom)^T W (q - q_nom)
        s.t.   contact(q, c)  <= 0     (7a), (7b)
               collision(q)   <= 0     (9)
               limits(q)      <= 0     (13a)                                  (14)

WHAT IS AND IS NOT IN HERE
--------------------------
Eq. 14 is an INVERSE KINEMATICS problem. It cites 7a and 7b but NOT 7c or 7d; it
cites 13a but not 13b or 13c. That is not an oversight, it is the whole point of
the hierarchy in Fig. 3: there are no forces and no time in this problem, so
friction, torque and velocity have nothing to act on. Adding them would make the
cheap filter expensive and would reject modes that a full TO could realize -- the
false negatives the paper's Table III measures.

So the constraint list here is exactly four things: patch alignment and containment,
collision avoidance, joint position limits, and the unit-quaternion rows that make
the 7-vector parameterization mean what Eq. 6 says (see `faro/scene/symbolic.py`).

WHAT "INFEASIBLE" MEANS
-----------------------
The verdict is the solver's, not a residual threshold of ours -- Section II-D
defines it as non-convergence within a maximum iteration count. `faro/solvers/nlp.py`
preserves the distinction between `Infeasible_Problem_Detected` and
`Maximum_Iterations_Exceeded`, because only the first is a statement about the
geometry; the second is a statement about our initial guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import casadi as ca
import numpy as np

from faro.constraints.block import ConstraintBlock, merge
from faro.constraints.contact import contact_kinematic, patch_containment_is_possible
from faro.constraints.limits import actuated_slice, joint_position_limits
from faro.core.modes import ContactEdge, ContactMode
from faro.scene.scene import Scene
from faro.scene.symbolic import SymbolicScene
from faro.solvers.nlp import NLPProblem


@dataclass(frozen=True)
class RegularizationWeights:
    """The diagonal of Eq. 14's W, grouped by what each block of q means.

    The paper gives W a name and never a value:

        "q_nom represents the nominal configuration used for regularization with
         weight W."

    so the values live in the scene config, not here. The groups exist because a
    single number for all of q is not what you want once you start looking at the
    poses: pulling the base POSITION toward nominal fights walking, while pulling the
    base ORIENTATION toward nominal is what keeps the torso upright.

    `base_orientation` is the one that is not 1.0, and the reason is worth stating.
    Under W = I a torso tilted 78 degrees costs the same as one finger joint being off
    by the same number, so Eq. 14 -- which has no gravity, no balance and no CoM
    condition -- happily returns the robot diving over the box with its knees at their
    hyperextension limit. That answer satisfies every constraint; it is simply not the
    one anyone wants out of a kinematic filter. Weighting the torso is the "stay
    upright" prior the paper never writes down. Measured on the grasp mode: weight
    1 -> 73 deg of lean, 10 -> 48 deg, 100 -> 16 deg. Recorded as
    docs/ambiguities.md #22.

    Note on the quaternion entries. `(q - q_nom)` is plain subtraction, including on
    the quaternion -- and that IS legitimate here, which is worth stating because it
    looks like a manifold error. For unit quaternions,

        |quat - quat_nom|^2 = 2 - 2 <quat, quat_nom> = 2 (1 - cos(theta/2)),

    a strictly increasing function of the geodesic angle theta. So the chordal
    distance the paper writes ranks rotations in exactly the same order the geodesic
    distance would; it is a valid rotation metric, not an approximation of one. This
    holds only because `unit_quaternion_rows` pins the norm -- without that
    constraint the expression is not a metric on anything.
    """

    base_position: float = 1.0
    base_orientation: float = 1.0
    joints: float = 1.0
    object_position: float = 1.0
    object_orientation: float = 1.0
    #: Per-group joint weights, `{name_fragment: weight}`. The KEY IS A SUBSTRING of
    #: the joint name, which is what lets one number cover a left/right pair --
    #: `knee` matches `left_knee_joint` and `right_knee_joint` and nothing else. See
    #: `weight_for_joint` for how overlaps resolve. Anything unmatched falls back to
    #: `joints`.
    joint_groups: dict = field(default_factory=dict)

    def weight_for_joint(self, name: str) -> float:
        """The weight for one named joint, resolving `joint_groups` by name.

        MOST SPECIFIC WINS, measured by the length of the matching fragment. So

            hip: 1.0
            hip_pitch: 5.0

        gives the four hip roll/yaw joints 1.0 and the two hip pitch joints 5.0, which
        is the reading anyone would expect from writing those two lines. It also means
        a full joint name is a legal group of one -- `left_knee_joint: 3.0` weights
        exactly that joint -- so the granularity goes from "all 29" down to a single
        joint without a second mechanism.

        A TIE IS AN ERROR, not a coin flip. Two fragments of equal length both matching
        one joint have no defensible winner, and picking one silently would make the
        weights depend on dict ordering.
        """
        matches = [(frag, w) for frag, w in self.joint_groups.items() if frag in name]
        if not matches:
            return self.joints

        longest = max(len(frag) for frag, _ in matches)
        best = [(frag, w) for frag, w in matches if len(frag) == longest]
        if len(best) > 1:
            raise ValueError(
                f"joint {name!r} matches several regularization groups of equal "
                f"specificity: {sorted(frag for frag, _ in best)}. Rename one, or make "
                f"one of them more specific -- FARO will not pick for you."
            )
        return float(best[0][1])

    def per_joint(self, scene: Scene) -> dict[str, float]:
        """`{joint_name: weight}` for every actuated joint. For checking an edit."""
        return {n: self.weight_for_joint(n) for n in scene.robot.actuated_joint_names}

    @classmethod
    def from_scene(cls, scene: Scene) -> RegularizationWeights:
        """Read W from the scene's `regularization:` block, falling back to defaults.

        A config read rather than a constructor default, so a sweep over W is a YAML
        edit and two scenes can disagree about it.
        """
        raw = dict(scene.regularization or {})
        groups = dict(raw.pop("joint_groups", None) or {})

        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(
                f"unknown regularization weights {sorted(unknown)}; "
                f"expected some of {sorted(known - {'joint_groups'})} plus joint_groups"
            )

        # A group fragment that matches NOTHING is a typo, and a silent one: the joints
        # it was meant for quietly keep the `joints` fallback and the config reads as
        # though it took effect. Same reason the unknown-key check above exists.
        names = scene.robot.actuated_joint_names
        empty = [frag for frag in groups if not any(frag in n for n in names)]
        if empty:
            raise KeyError(
                f"regularization joint_groups {sorted(empty)} match no joint on "
                f"{scene.robot.name!r}. Available joints: {names}"
            )

        return cls(joint_groups={k: float(v) for k, v in groups.items()},
                   **{k: float(v) for k, v in raw.items()})

    def diagonal(self, scene: Scene) -> np.ndarray:
        """Expand to a full diagonal over the stacked q, in `SymbolicScene` order."""
        model = scene.robot.model
        robot = np.empty(scene.robot.nq)
        robot[:3] = self.base_position
        robot[3:7] = self.base_orientation

        # Placed by each joint's OWN idx_q rather than by counting from 7. The two
        # agree for a chain of 1-DoF joints, and stop agreeing the moment one is not --
        # at which point counting would shift every later joint's weight by one slot
        # and nothing would report it.
        for name in scene.robot.actuated_joint_names:
            joint = model.joints[model.getJointId(name)]
            robot[joint.idx_q:joint.idx_q + joint.nq] = self.weight_for_joint(name)

        per_object = np.concatenate([
            np.full(3, self.object_position),
            np.full(4, self.object_orientation),
        ])
        return np.concatenate([robot, *[per_object] * len(scene.objects)])


def contact_blocks(
    sym: SymbolicScene, pairs: list[tuple[str, str]], *, alignment: str = "column"
) -> list[ConstraintBlock]:
    """Eqs. 7a + 7b for every active contact pair of the mode (or edge).

    The pair is ordered `(a, b)` with `a` the interface's own patch and `b` its
    partner, which is the order Eq. 7b needs: it reads

        |p_{x,y}| <= (^b xi_b)_{x,y} - (^b xi_a)_{x,y}

    i.e. patch `a` must fit INSIDE patch `b`. Feet inside the floor, hands inside a
    box face, the box bottom inside the tabletop -- every pair in Table IV runs
    end-effector-into-surface, so the ordering falls out of the table rather than
    being a convention we picked. It is checked rather than assumed, because a scene
    where it fails produces a right-hand side that is negative in every
    configuration: an NLP that is infeasible for a reason having nothing to do with
    the robot, reported as a mode being unreachable.

    `alignment` defaults to `"column"` here, NOT to the `"log3"` form the paper writes
    and `contact_kinematic` defaults to. The two are equivalent as sets -- and reach
    the same optimum wherever both converge -- but the log3 form makes Ipopt return
    `Invalid_Number_Detected` on the grasp and place modes of this very scene. That is
    a measured result whose mechanism is NOT diagnosed; `normal_alignment_rows` says
    what was and was not established. Milestone 2's scenarios keep the literal form,
    because interrogating the paper's own expression is their job.
    See docs/ambiguities.md #23.
    """
    blocks = []
    for a_name, b_name in pairs:
        patch_a, patch_b = sym.scene.patches[a_name], sym.scene.patches[b_name]
        if not patch_containment_is_possible(patch_a.half_extents, patch_b.half_extents):
            raise ValueError(
                f"Eq. 7b cannot be satisfied for {a_name} inside {b_name}: half-extents "
                f"{patch_a.half_extents} do not fit in {patch_b.half_extents}. Either the "
                f"pair is reversed or the scene geometry is wrong."
            )
        R_a, p_a = sym.patch_placement(patch_a)
        R_b, p_b = sym.patch_placement(patch_b)
        blocks.append(
            contact_kinematic(
                R_a, p_a, R_b, p_b,
                ca.DM(list(patch_a.half_extents)), ca.DM(list(patch_b.half_extents)),
                name=f"contact[{a_name}->{b_name}]",
                alignment=alignment,
            )
        )
    return blocks


def limit_block(sym: SymbolicScene) -> ConstraintBlock:
    """Eq. 13a on the actuated joints.

    The floating base is sliced off: `model.lowerPositionLimit` carries entries for
    the base's 7 quaternion-parameterized numbers, and they are not limits on
    anything -- constraining them would bound the robot's position and orientation
    to whatever the URDF happened to leave in those slots.
    """
    model = sym.scene.robot.model
    lower, upper = sym.scene.robot.joint_limits()
    sl = actuated_slice(model)
    return joint_position_limits(sym.robot_q[sl], lower[sl], upper[sl], name="limits")


def build_problem(
    scene: Scene,
    target: ContactMode | ContactEdge,
    *,
    weights: RegularizationWeights | None = None,
    q0: np.ndarray | None = None,
    collision_blocks: list[ConstraintBlock] | None = None,
    alignment: str = "column",
    sym: SymbolicScene | None = None,
    name: str | None = None,
) -> tuple[NLPProblem, SymbolicScene]:
    """Assemble Eq. 14 for a contact mode or a mode transition.

    `target` is a `ContactMode` (the mode filter M of Section III) or a
    `ContactEdge` (the edge filter E). Both expose `active_pairs`, and this function
    does not branch on which it got -- that is the entire reason `ContactEdge`
    exists as a separate type with a matching method.

    Returns the problem AND the `SymbolicScene` that owns its variables, because the
    caller needs the latter to unpack the answer into a robot configuration and
    object poses.

    `collision_blocks` is passed in rather than built here so that the choice of
    collision pairs -- which the paper never specifies, see docs/ambiguities.md #19
    -- is made by one module with a testable policy, instead of being hardcoded
    inside the objective of every stage that needs it.

    `sym` MUST be the same `SymbolicScene` those blocks were built against. CasADi
    expressions are tied to the specific symbols they were written with, so blocks
    from one `SymbolicScene` dropped into a problem built around another are not
    merely mismatched -- they reference variables the solver never sees, and the
    resulting NLP is silently wrong rather than an error. Passing the object through
    is what makes that impossible to get wrong.
    """
    target.validate(scene)
    weights = weights if weights is not None else RegularizationWeights.from_scene(scene)

    sym = sym if sym is not None else SymbolicScene(scene)
    q_nom = sym.nominal()
    q = sym.variables

    # --- Eq. 14's objective ---------------------------------------------------
    delta = q - ca.DM(q_nom.reshape(-1, 1))
    cost = ca.dot(delta, ca.DM(weights.diagonal(scene).reshape(-1, 1)) * delta)

    blocks = [
        *contact_blocks(sym, target.active_pairs(scene), alignment=alignment),
        limit_block(sym),
        sym.unit_quaternion_rows(),
        *(collision_blocks or []),
    ]

    label = name or (target.label() if hasattr(target, "label") else "mode")
    return (
        NLPProblem(
            name=f"eq14[{label}]",
            variables=q,
            cost=cost,
            block=merge(f"eq14[{label}]", blocks),
            # q_nom doubles as the initial guess. The paper only calls it the
            # regularization target, but starting anywhere else would make the
            # first iterate's cost gradient point away from where the constraints
            # were linearized -- and Section IV-C attributes its false negatives
            # precisely to initialization.
            x0=q_nom if q0 is None else np.asarray(q0, dtype=float),
        ),
        sym,
    )
