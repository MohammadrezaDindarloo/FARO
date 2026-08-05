"""Kinematic Sequence Optimization -- paper Eq. 15 (Section II-E).

    "These optimization problems ... share a common set of contact, collision,
     dynamics, and limit constraints, while differing in the variables they optimize
     and the level of feasibility they enforce."   -- Section II

Eq. 15 is the middle filter of Fig. 3. Where Eq. 14 asks "does a configuration exist
for THIS mode?", Eq. 15 asks the same question of a whole SEQUENCE at once, and adds
the one constraint that only makes sense across time:

    min_{q_0:K}  sum_{s=0}^{K} (q_s - q_nom)^T W (q_s - q_nom)
    s.t.  q_0 = q_init,   q_K in Q_goal,
          for all s in {1, ..., K}:
              contact(q_s, c_{s-1} U c_s) <= 0    (7a), (7b)
              collision(q_s)              <= 0    (9)
              limits(q_s)                 <= 0    (13a)
          for all s in {0, ..., K-1}:
              for all a in I satisfying (16a) or (16b):
                  contact(q_s, q_{s+1}, c_s) = 0  (8)                        (15)

with `c_K := c_{K-1}`, so K modes give K+1 configurations. Transcribed in
docs/eq15_kso.md; where this file and that disagree, that one is right.

THE UNION IS NOT A DETAIL. `c_{s-1} U c_s` is the EDGE of Section II-D -- the same
union filter E tests. So Eq. 15 embeds the edge check at every knot: q_s must satisfy
the contacts of the mode it is LEAVING and the one it is ENTERING simultaneously. An
earlier version of this file used `contact(q_s, c_s)` and was a strictly weaker filter:
it passed sequences whose transitions are impossible, e.g. a hand moving from box_left
to box_front without releasing.

WHAT IS STILL ABSENT, AND WHY THAT MATTERS MORE THAN IT SOUNDS
--------------------------------------------------------------
Eq. 15 has NO forces and NO dynamics. Section IV-A measures exactly this: the KSO
"retains 70.0% of the constraint types present in the TO", and the ones it omits are
the dynamic ones. So 7c, 7d, 10 and 11 are not skipped for speed -- there are no
force, velocity or acceleration variables for them to act on, exactly as in Eq. 14.

The consequence is worth stating plainly because it is easy to expect otherwise: the
KSO does NOT check balance. A configuration whose centre of mass lies outside the
support polygon is as acceptable to Eq. 15 as to Eq. 14. Static equilibrium arrives
only at the TO (Eq. 17), with 10a/10b and 7c/7d. What the KSO adds over Eq. 14 is
CONSISTENCY ACROSS TIME -- that the same contact does not teleport along a surface
between steps -- not physical realizability.

`faro/constraints/README.md` holds the stage-by-stage constraint table, and
`tests/test_kso.py::test_eq_15_adds_no_slip_and_nothing_dynamic` pins this on the row
labels so it cannot be satisfied by a comment.

WHY THE SEQUENCE IS ONE NLP AND NOT K SOLVES
--------------------------------------------
Eq. 8 couples adjacent configurations, so the steps cannot be solved independently:
a foot placed at one spot in step 0 constrains where it may be in step 1. Solving
mode-by-mode and hoping the answers line up is precisely what Eq. 8 exists to
prevent, and it is the whole difference between the `KSO` and `M,E` rows of Table II.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from faro.constraints.block import ConstraintBlock, merge
from faro.constraints.no_slip import applies_between, no_slip
from faro.core.modes import ContactEdge, ContactSequence
from faro.mode_edge.problem import RegularizationWeights, contact_blocks, limit_block
from faro.scene.scene import Scene
from faro.scene.symbolic import SymbolicScene
from faro.solvers.nlp import NLPProblem


def _quat_xyzw(rotation) -> np.ndarray:
    """Rotation matrix -> (x, y, z, w), the layout `SymbolicScene` uses for a pose."""
    import pinocchio as pin

    q = pin.Quaternion(np.asarray(rotation))
    return np.array([q.x, q.y, q.z, q.w])


@dataclass(frozen=True)
class SequenceWeights:
    """Eq. 15's cost, which the paper names and never writes down.

    RESOLVED (docs/eq15_kso.md). The paper states it in full:

        min_{q_0:K}  sum_{s=0}^{K} (q_s - q_nom)^T W (q_s - q_nom)

    so `regularization` below IS Eq. 15's objective, not a reading of it.

      * `regularization` -- the direct generalization of Eq. 14: sum the same
        weighted nominal-distance over every step. This is the default because it
        is the smallest assumption that makes Eq. 15 look like the problem Eq. 14
        generalizes, and because it needs no new weights.

      * `smoothness` -- additionally penalize `q_{s+1} - q_s`. NOT IN THE PAPER.
        Eq. 15's objective is exactly the sum above and nothing else, so this stays
        at 0.0 and the delivered problem is the paper's. It is kept only because a
        smoothing term is the first thing anyone reaches for when a sequence looks
        jumpy, and having it as a measured switch is better than having it added
        by hand later without a record.

    Turning smoothness on changes which answer you get, never which answers are
    admissible -- it is a cost term, not a constraint. It matters anyway, because
    Alg. 1 warm-starts the TO from KSO output.

    Note both terms use the SAME `RegularizationWeights` block structure, so a scene
    that has already tuned W for Eq. 14 does not need a second set of numbers.
    """

    regularization: RegularizationWeights = field(default_factory=RegularizationWeights)
    smoothness: float = 0.0

    @classmethod
    def from_scene(cls, scene: Scene) -> SequenceWeights:
        """Read `kso:` from the scene config, falling back to Eq. 14's own W."""
        cfg = dict(getattr(scene, "kso", None) or {})
        smoothness = float(cfg.pop("smoothness", 0.0))
        unknown = set(cfg) - {"regularization"}
        if unknown:
            raise ValueError(
                f"unknown keys in the scene's `kso:` block: {sorted(unknown)}. "
                f"Known: ['regularization', 'smoothness']."
            )
        return cls(
            regularization=RegularizationWeights.from_scene(scene),
            smoothness=smoothness,
        )


def mode_at(sequence: ContactSequence, s: int):
    """`c_s` with the paper's `c_K := c_{K-1}` convention.

    "For notational convenience we define c_K := c_{K-1}, so that the terminal
     configuration can be handled using the same indexing as the intermediate
     configurations."   -- Section II-E
    """
    return sequence[min(s, len(sequence) - 1)]


def edge_at(sequence: ContactSequence, s: int) -> ContactEdge:
    """`c_{s-1} U c_s`, the contact set q_s must satisfy (Eq. 15, s in 1..K).

    At s = K this is `c_{K-1} U c_{K-1}`, i.e. the last mode alone -- which is exactly
    what the `c_K := c_{K-1}` convention is for.
    """
    return ContactEdge(mode_at(sequence, s - 1), mode_at(sequence, s))


def n_knots(sequence: ContactSequence) -> int:
    """K + 1 configurations for K modes (docs/ambiguities.md #27, resolved)."""
    return len(sequence) + 1


def symbolic_knots(scene: Scene, count: int) -> list[SymbolicScene]:
    """One `SymbolicScene` per step, each with distinct variable names.

    The suffix is not cosmetic. CasADi identifies variables by name, so K knots built
    without one would collide into a single set of symbols and the "sequence" would
    silently be K copies of one configuration -- an NLP that solves, reports success,
    and answers a different question than the one asked.
    """
    return [SymbolicScene(scene, suffix=f"k{s}") for s in range(count)]


def no_slip_blocks(scene: Scene, sequence: ContactSequence,
                   syms: list[SymbolicScene], *, yaw: str = "column") -> list[ConstraintBlock]:
    """Eq. 8 at every transition where Eq. 16 says it applies.

    Eq. 16 is consulted PER INTERFACE, not per step: within one transition a foot may
    be persisting (16a) while a hand is releasing (16b) and another hand is newly
    acquiring (no constraint). Applying a single decision to the whole step would
    either freeze a contact that is being established or let an established one slide.
    """
    blocks = []
    # s in {0, ..., K-1}: K transitions over K+1 knots. `mode_at` supplies c_K := c_{K-1},
    # so the final transition compares the last mode with itself -- every contact it
    # holds persists into the terminal configuration, which is what Eq. 16a then pins.
    for s in range(len(sequence)):
        mode, nxt = mode_at(sequence, s), mode_at(sequence, s + 1)
        for interface in sorted(scene.interfaces):
            partner_s = mode.partner(interface)
            partner_next = nxt.partner(interface)
            if not applies_between(partner_s, partner_next):
                continue

            patch_a = scene.interfaces[interface].patch
            # Eq. 8 holds the pose of `a` RELATIVE TO ITS PARTNER, so the partner is
            # the one from step s -- the contact being preserved is the one that
            # already exists. On release (16b) there is no partner at s+1 to measure
            # against, and using one would be measuring a contact that is over.
            patch_b = scene.patches[partner_s]

            R_a_s, p_a_s = syms[s].patch_placement(patch_a)
            R_b_s, p_b_s = syms[s].patch_placement(patch_b)
            R_a_n, p_a_n = syms[s + 1].patch_placement(patch_a)
            R_b_n, p_b_n = syms[s + 1].patch_placement(patch_b)

            blocks.append(
                no_slip(
                    R_a_s, p_a_s, R_b_s, p_b_s,
                    R_a_n, p_a_n, R_b_n, p_b_n,
                    name=f"no_slip[{s}->{s + 1}][{interface}->{partner_s}]",
                    yaw=yaw,
                )
            )
    return blocks


def build_problem(
    scene: Scene,
    sequence: ContactSequence,
    *,
    weights: SequenceWeights | None = None,
    q0: np.ndarray | None = None,
    collision_blocks: list[list[ConstraintBlock]] | None = None,
    alignment: str = "column",
    yaw: str = "column",
    q_init: np.ndarray | None = None,
    goal: object | None = None,
    syms: list[SymbolicScene] | None = None,
    name: str | None = None,
) -> tuple[NLPProblem, list[SymbolicScene]]:
    """Assemble Eq. 15 for a contact sequence.

    Returns the problem AND the per-step `SymbolicScene`s, because the caller needs
    them to unpack the stacked answer back into configurations.

    `collision_blocks` is a list-of-lists indexed by step, built outside for the same
    reason as in Eq. 14: the pair policy is one module's testable decision rather
    than something hardcoded into every stage. `syms` MUST be the scenes those blocks
    were written against -- CasADi expressions are bound to specific symbols, so
    mixing them produces an NLP that is silently wrong rather than an error.
    """
    sequence.validate(scene)
    weights = weights if weights is not None else SequenceWeights.from_scene(scene)
    syms = syms if syms is not None else symbolic_knots(scene, n_knots(sequence))

    if len(syms) != n_knots(sequence):
        raise ValueError(
            f"got {len(syms)} symbolic knots for a sequence of {len(sequence)} modes. "
            f"Eq. 15 optimizes K+1 configurations for K modes (c_K := c_{{K-1}})."
        )
    if collision_blocks is not None and len(collision_blocks) != n_knots(sequence):
        raise ValueError(
            f"got {len(collision_blocks)} collision block lists for "
            f"{n_knots(sequence)} knots. Each knot is linearized at its own "
            f"configuration, so they cannot be shared."
        )

    diagonal = ca.DM(weights.regularization.diagonal(scene).reshape(-1, 1))
    nominal = [sym.nominal() for sym in syms]

    cost = ca.SX(0)
    blocks: list[ConstraintBlock] = []
    for s, sym in enumerate(syms):
        # The objective runs over ALL of s = 0..K, including the pinned q_0. That row
        # is a constant once q_0 = q_init holds, but it is in the paper's sum and
        # dropping it would change the reported cost for no reason.
        delta = sym.variables - ca.DM(nominal[s].reshape(-1, 1))
        cost = cost + ca.dot(delta, diagonal * delta)

        if s == 0:
            # Eq. 15 applies contact/collision/limits for s in {1..K} only. q_0 is
            # fully determined by q_0 = q_init, so constraining it again is redundant
            # at best and, if q_init is slightly off the constraint manifold, an
            # infeasible problem describing nothing.
            continue

        blocks += [
            # THE UNION, c_{s-1} U c_s -- see the module docstring.
            *contact_blocks(sym, edge_at(sequence, s).active_pairs(scene),
                            alignment=alignment),
            limit_block(sym),
            sym.unit_quaternion_rows(),
        ]
        if collision_blocks is not None:
            blocks += collision_blocks[s]

    # Ours, not the paper's; 0.0 by default so it contributes nothing (ambiguity #28).
    if weights.smoothness:
        for s in range(len(syms) - 1):
            step = syms[s + 1].variables - syms[s].variables
            cost = cost + weights.smoothness * ca.dot(step, diagonal * step)

    # --- q_0 = q_init ---------------------------------------------------------------
    # An EQUALITY in the paper, not an option. It pins the WHOLE configuration -- robot
    # and every object -- not just object poses, which is what an earlier version of
    # this file got wrong (docs/ambiguities.md #29, resolved).
    if q_init is None:
        q_init = nominal[0]
    q_init = np.asarray(q_init, dtype=float)
    if q_init.size != syms[0].n_var:
        raise ValueError(
            f"q_init has {q_init.size} entries but a configuration has {syms[0].n_var}."
        )
    blocks.append(ConstraintBlock(
        name="q_init",
        eq=syms[0].variables - ca.DM(q_init.reshape(-1, 1)),
        eq_labels=[f"15:q_0=q_init[{k}]" for k in range(q_init.size)],
    ))

    # --- q_K in Q_goal --------------------------------------------------------------
    # The paper names Q_goal and does not define it (docs/eq15_kso.md). Represented
    # here as a set of contact pairs the terminal configuration must satisfy, because
    # that is how the box-placement task states its goal -- "the box is on the
    # platform" is a contact condition, not a fixed configuration. A caller wanting a
    # literal target configuration can pass a goal of `None` and pin it via q_init-style
    # equality instead.
    if goal is not None:
        blocks += contact_blocks(syms[-1], goal.active_pairs(scene), alignment=alignment)

    blocks += no_slip_blocks(scene, sequence, syms, yaw=yaw)

    label = name or f"{len(sequence)} modes"
    variables = ca.vertcat(*[sym.variables for sym in syms])
    x0 = np.concatenate(nominal) if q0 is None else np.asarray(q0, dtype=float)
    if x0.size != int(variables.shape[0]):
        raise ValueError(
            f"initial guess has {x0.size} entries but Eq. 15 has "
            f"{int(variables.shape[0])} variables ({len(syms)} steps x "
            f"{syms[0].n_var}). Pass one stacked configuration per step."
        )

    return (
        NLPProblem(
            name=f"eq15[{label}]",
            variables=variables,
            cost=cost,
            block=merge(f"eq15[{label}]", blocks),
            x0=x0,
        ),
        syms,
    )
