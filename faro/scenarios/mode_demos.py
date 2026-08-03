"""A curated tour of Eq. 14 -- modes and edges worth looking at.

The same idea as `constraint_demos.py`, one level up. There, each scenario swept a
parameter and checked a constraint failed where the paper says it must. Here each
entry poses a mode or a transition, states what it SHOULD come back as and why, and
`scripts/03_mode_feasibility.py` shows the configuration Eq. 14 actually returned.

Looking at the configuration is the point, and it is not decoration. A feasibility
filter that says "feasible" has told you almost nothing -- the verdict is one bit,
and a bug that drops a constraint makes MORE things feasible, not fewer. The way to
catch that is to look at the pose it produced and see whether it is the pose the mode
described.

TWO ENTRIES RECORD PREDICTIONS THAT TURNED OUT WRONG
----------------------------------------------------
`far-reach` and `false-negative` are kept with their histories rather than quietly
rewritten, because both corrections say something the correct answer alone does not:

  * `far-reach` -- we argued from the G1's 0.315 m arm reach that the platform at
    x = 0.9 was unreachable. The argument forgot that the base is a decision variable
    and the robot can simply walk over. Reach arguments about a floating-base robot
    have to include the base.
  * `false-negative` -- reported infeasible under the original settings, and we said
    we did not believe it. Raising `max_iter` and adding restarts turned it feasible.
    It was a false negative all along, of exactly the kind Section IV-C describes:

        "The few observed false negatives were primarily associated with poor
         initialization or insufficient solver iterations in the KSO."

A filter's failure modes are worth seeing before it is wired into a tree search that
will trust it thousands of times.
"""

from __future__ import annotations

from dataclasses import dataclass

from faro.core.modes import ContactEdge, ContactMode
from faro.scene.scene import Scene


@dataclass(frozen=True)
class ModeDemo:
    """One mode or edge, with the verdict it is expected to produce and why."""

    key: str
    title: str
    question: str
    #: "feasible" or "infeasible" -- the verdict Eq. 14 is expected to return.
    expect: str
    why: str
    build: object  # Callable[[Scene], ContactMode | ContactEdge]

    def target(self, scene: Scene):
        return self.build(scene)


def _mode(**kwargs) -> ContactMode:
    base = {"left_hand": None, "right_hand": None,
            "left_foot": None, "right_foot": None, "box_bottom": None}
    base.update(kwargs)
    return ContactMode.from_dict(base)


STAND = _mode(left_foot="floor", right_foot="floor", box_bottom="floor")
GRASP = _mode(left_foot="floor", right_foot="floor", box_bottom="floor",
              left_hand="box_left", right_hand="box_right")
LIFT = _mode(left_foot="floor", right_foot="floor",
             left_hand="box_left", right_hand="box_right")
PLACE = _mode(left_foot="floor", right_foot="floor", box_bottom="tabletop",
              left_hand="box_left", right_hand="box_right")
RELEASE = _mode(left_foot="floor", right_foot="floor", box_bottom="tabletop")
FRONT_REAR = _mode(left_foot="floor", right_foot="floor", box_bottom="floor",
                   left_hand="box_front", right_hand="box_rear")


ALL_DEMOS: list[ModeDemo] = [
    ModeDemo(
        key="stand",
        title="Both feet on the floor, box on the floor",
        question="Does Eq. 14 produce a pose, or just a verdict?",
        expect="feasible",
        why="The nominal posture already satisfies this mode, so the cost has nothing "
            "to trade against the constraints and the solve should finish in a handful "
            "of iterations. Watch the SOLES: they must sit flat at z = 0, which is "
            "Eq. 7a's `p_z = 0` and `log3(R)_{x,y} = 0` doing their job.",
        build=lambda scene: STAND,
    ),
    ModeDemo(
        key="grasp",
        title="Hands on the box's left and right faces",
        question="Can the robot reach the box without moving it?",
        expect="feasible",
        why="The interesting part is what has to move. The hand patches are on the "
            "WRIST CUT FACE with the normal running along the forearm (ambiguity #7b), "
            "so the robot cannot pinch with palms -- it has to bring both forearms in "
            "and press the box between them. If the arms end up somewhere that does not "
            "look like that, the patch frames are wrong, not the solver.",
        build=lambda scene: GRASP,
    ),
    ModeDemo(
        key="lift",
        title="Box held by both hands, off the floor",
        question="Do the object's own variables actually move?",
        expect="feasible",
        why="`box_bottom` is free, so nothing holds the box down and nothing holds it "
            "up either -- Eq. 14 has no gravity and no forces. A box floating in mid-air "
            "is the CORRECT answer here, and it is worth seeing, because it is exactly "
            "what the kinematic filter cannot rule out and what the TO of Eq. 17 exists "
            "to check.",
        build=lambda scene: LIFT,
    ),
    ModeDemo(
        key="place",
        title="Box on the platform, hands still on it",
        question="Does the box end up on the platform, or does only the robot move?",
        expect="feasible",
        why="Eq. 14 optimizes q = (q^r, q^o), so the box's pose is a decision variable. "
            "Its bottom must land on the tabletop at z = 0.55 (platform 0.40 + half-height "
            "0.15). If the box stays on the floor while the verdict says feasible, the "
            "object variables are not reaching the constraints.",
        build=lambda scene: PLACE,
    ),
    ModeDemo(
        key="release",
        title="Box on the platform, hands free",
        question="What does a free interface cost?",
        expect="feasible",
        why="Nothing: `b = null` contributes no rows at all. The robot should relax back "
            "toward its nominal posture, since with the hands unconstrained the only "
            "thing acting on the arms is the regularization term. Compare the arms here "
            "against `place`.",
        build=lambda scene: RELEASE,
    ),
    ModeDemo(
        key="front-rear",
        title="Hands on the front and rear faces instead",
        question="Is a different grasp on the same box a different problem?",
        expect="feasible",
        why="Same box, same feet, same everything except which two faces the hands claim. "
            "The robot has to reach around the box's near and far ends rather than its "
            "sides. Both grasps being feasible is what makes the tree search's branching "
            "meaningful -- if only one worked, there would be nothing to search.",
        build=lambda scene: FRONT_REAR,
    ),
    ModeDemo(
        key="edge-reach",
        title="EDGE: standing to grasping",
        question="Can the transition happen at an instant?",
        expect="feasible",
        why="Section II-D replaces `c` with `c1 u c2`, so this problem carries the "
            "contacts of BOTH modes at once -- feet down, box down, and both hands on "
            "the box. That is satisfiable, so the transition is admissible. Note the "
            "edge is strictly harder than either endpoint: it is the union.",
        build=lambda scene: ContactEdge(STAND, GRASP),
    ),
    ModeDemo(
        key="edge-regrasp",
        title="EDGE: sliding the left hand from the side face to the front face",
        question="Does the edge filter catch anything the mode filter misses?",
        expect="infeasible",
        why="THE ENTRY THAT JUSTIFIES HAVING AN EDGE FILTER AT ALL. Both endpoint modes "
            "are individually feasible -- `grasp` and `front-rear` are both in this tour "
            "and both pass. But the union asks one flat patch to lie against two "
            "PERPENDICULAR faces at the same instant, and Eq. 7a cannot align a single "
            "normal with both. A search that only checked modes would happily walk "
            "through this transition.",
        build=lambda scene: ContactEdge(GRASP, FRONT_REAR),
    ),
    ModeDemo(
        key="far-reach",
        title="Box on the platform, both feet down, one hand on the box",
        question="Can the robot reach the platform at all?",
        expect="feasible",
        why="A PREDICTION OF OURS THAT WAS WRONG, kept because the correction is the "
            "useful part. We argued this had to be infeasible: the platform is at "
            "x = 0.9 and the G1's shoulder-to-patch reach is 0.315 m. The filter said "
            "feasible, and the filter is right -- the argument silently assumed the "
            "robot stays where it starts. The floor patch is 6 x 6 m and the base is a "
            "free variable, so the robot simply stands next to the platform. Reach "
            "arguments about a floating-base robot have to include the base.",
        build=lambda scene: _mode(left_foot="floor", box_bottom="tabletop",
                                  left_hand="box_left", right_foot="floor"),
    ),
    ModeDemo(
        key="false-negative",
        title="One foot down, both hands on the box",
        question="Is every 'infeasible' a fact about the scene?",
        expect="feasible",
        why="NO -- and this entry is the proof, because we watched it flip. With "
            "max_iter = 300 and a single solve from q_nom it came back "
            "`Maximum_Iterations_Exceeded`, and we said at the time that we did not "
            "believe it: nothing about the geometry forbids standing on one foot while "
            "holding the box, since Eq. 14 has no balance requirement to violate. "
            "Raising the cap and adding seeded restarts turned it into "
            "`Solve_Succeeded`. The mode was reachable the whole time and the filter "
            "was wrong about it -- a false negative, of exactly the kind Section IV-C "
            "attributes to 'poor initialization or insufficient solver iterations'. "
            "A tree search would have pruned a perfectly good branch.",
        build=lambda scene: _mode(left_foot="floor", box_bottom="floor",
                                  left_hand="box_left", right_hand="box_right"),
    ),
]


def get_demo(key: str) -> ModeDemo:
    for demo in ALL_DEMOS:
        if demo.key == key:
            return demo
    raise KeyError(f"unknown demo {key!r}; available: {[d.key for d in ALL_DEMOS]}")
