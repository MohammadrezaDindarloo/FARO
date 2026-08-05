"""A curated tour of Eq. 15 -- contact SEQUENCES worth looking at.

`mode_demos.py` asks whether one mode is reachable. This asks whether a whole plan
holds together, which is a different question with a different failure mode: every
step can be individually fine and the sequence still impossible, because Eq. 8 says a
contact may not slide once established.

WHAT TO LOOK FOR, AND IT IS NOT THE VERDICT
-------------------------------------------
Step through the configurations and watch the CONTACTS THAT PERSIST. A foot that is
`floor` in two adjacent steps must be in the same place in both -- not merely on the
floor in both. That is the whole content of Eq. 15 over Eq. 14, and it is invisible in
the feasible/infeasible bit. `scripts/04_kso_demo.py --drift` measures it directly.

Two things Eq. 15 will NOT do, worth knowing before reading a verdict:

  * IT DOES NOT CHECK BALANCE. There are no forces in Eq. 15 (Section IV-A: the KSO
    "retains 70.0% of the constraint types present in the TO", omitting the dynamic
    ones). A sequence of head-first dives over the box is perfectly acceptable here.
    That is the TO's job, and until Milestone 5 exists nothing in this repo rejects it.

  * IT DOES NOT ORDER TIME. Eq. 15 is kinematic, so there is no velocity and no
    duration. "The box moves from the floor to the tabletop" is expressed entirely by
    which patches are in contact at each step, never by anything moving.
"""

from __future__ import annotations

from dataclasses import dataclass

from faro.core.modes import ContactMode, ContactSequence
from faro.scene.scene import Scene


@dataclass(frozen=True)
class SequenceDemo:
    """One contact sequence, the verdict it should produce, and why."""

    key: str
    title: str
    question: str
    #: "feasible" or "infeasible" -- what Eq. 15 is expected to return.
    expect: str
    why: str
    steps: tuple[dict, ...]

    def sequence(self, scene: Scene) -> ContactSequence:
        return ContactSequence(tuple(ContactMode.from_dict(s) for s in self.steps))


def _mode(**kwargs) -> dict:
    base = {"left_hand": None, "right_hand": None,
            "left_foot": None, "right_foot": None, "box_bottom": None}
    base.update(kwargs)
    return base


# The five states the box-placement task passes through (Table IV's vocabulary).
STAND = _mode(left_foot="floor", right_foot="floor", box_bottom="floor")
GRASP = _mode(left_foot="floor", right_foot="floor", box_bottom="floor",
              left_hand="box_left", right_hand="box_right")
LIFT = _mode(left_foot="floor", right_foot="floor",
             left_hand="box_left", right_hand="box_right")
PLACE = _mode(left_foot="floor", right_foot="floor", box_bottom="tabletop",
              left_hand="box_left", right_hand="box_right")
RELEASE = _mode(left_foot="floor", right_foot="floor", box_bottom="tabletop")

# Same grasp, opposite faces. Used to build a sequence that must fail.
FRONT_REAR = _mode(left_foot="floor", right_foot="floor", box_bottom="floor",
                   left_hand="box_front", right_hand="box_rear")


ALL_DEMOS: list[SequenceDemo] = [
    SequenceDemo(
        key="reach",
        title="Stand, then grasp",
        question="Can the robot take hold of the box without moving its feet?",
        expect="feasible",
        why="The shortest sequence that has anything to say. Both feet and the box "
            "bottom persist on the floor across the transition, so Eq. 16a fires "
            "three times and Eq. 8 pins all three in place; only the hands are newly "
            "acquired, and acquisition is deliberately unconstrained (there is no "
            "previous contact location to preserve). Watch the feet: they must be at "
            "the SAME spot in both steps, not merely both on the floor.",
        steps=(STAND, GRASP),
    ),
    SequenceDemo(
        key="pick",
        title="Stand, grasp, lift",
        question="Does releasing the box from the floor stay consistent?",
        expect="feasible",
        why="The box bottom goes floor -> floor -> free, so the last transition is "
            "Eq. 16b: a RELEASE. The paper is explicit that the sticking condition "
            "still applies there -- q_{s+1} is 'the boundary configuration at which "
            "contact is broken' -- so the box may not slide on its way up. The hands "
            "persist through both transitions and are pinned to the box faces.",
        steps=(STAND, GRASP, LIFT),
    ),
    SequenceDemo(
        key="pick-place",
        title="The full task: stand, grasp, lift, place, release",
        question="Is the whole box-placement plan kinematically consistent?",
        expect="feasible",
        why="This is the sequence the paper's tree search is searching FOR (Fig. 4). "
            "Five modes, four transitions, and every kind of Eq. 16 event in one "
            "problem: persistence (feet throughout), release (box off the floor), "
            "acquisition (hands, then box onto the tabletop), and release again "
            "(hands off at the end). If any single thing in Milestone 4 is worth "
            "looking at, it is this one.",
        steps=(STAND, GRASP, LIFT, PLACE, RELEASE),
    ),
    SequenceDemo(
        key="regrasp",
        title="Grasp the sides, then the front and rear",
        question="Can the hands switch faces without ever letting go?",
        expect="infeasible",
        why="The intended NEGATIVE, and it only works under the PAPER'S Eq. 15. Each "
            "mode is individually feasible -- Eq. 14 says so for both. What rejects the "
            "sequence is the union: Eq. 15 constrains q_s by `c_{s-1} U c_s`, so the "
            "middle configuration must put the left hand flat on box_left AND box_front "
            "at the same time. Eq. 7a cannot do that.\n\n"
            "Worth knowing as history: an earlier build of this repo used "
            "`contact(q_s, c_s)` -- the current mode only -- and this demo came back "
            "FEASIBLE, with the edge filter disagreeing. That was not a subtlety about "
            "which filter checks what; it was a strictly weaker problem than the paper's, "
            "and it would have passed sequences whose transitions cannot happen.",
        steps=(GRASP, FRONT_REAR),
    ),
    SequenceDemo(
        key="teleport",
        title="Lift, then place -- with no hands",
        question="Can the box cross the room while nothing is holding it?",
        expect="feasible",
        why="PREDICTION CORRECTED. Written expecting infeasible; it is feasible, and "
            "it should be. The box is unsupported in step 0 and on the tabletop in "
            "step 1 with both hands free throughout, so nothing touches it at either "
            "step -- and Eq. 15 has no dynamics, no gravity and no notion that an "
            "unsupported box must fall. Eq. 7b is satisfied at both steps and Eq. 8 "
            "never applies, because the box bottom goes free -> tabletop, an "
            "ACQUISITION. So the box teleports across the room and the KSO is right to "
            "say yes: this is a true statement about Eq. 15 and a useless one about the "
            "world. Keep it. It is the sharpest reminder that these filters are "
            "NECESSARY conditions, and that a KSO pass means only that the TO is worth "
            "attempting.",
        steps=(_mode(left_foot="floor", right_foot="floor"),
               _mode(left_foot="floor", right_foot="floor", box_bottom="tabletop")),
    ),
]


def get_demo(key: str) -> SequenceDemo:
    for demo in ALL_DEMOS:
        if demo.key == key:
            return demo
    raise KeyError(f"unknown sequence demo {key!r}. Known: {[d.key for d in ALL_DEMOS]}")
