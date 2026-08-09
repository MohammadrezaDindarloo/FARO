"""Algorithm 1 -- the feasibility-guided contact-mode tree search.

    Algorithm 1 Feasibility-Guided Contact-Mode Tree Search
    Require: initial mode c_0, time budget B, feasibility filters F
     1: T <- INITIALIZETREE(c_0)
     2: S_goal <- {}
     3: K_feas, K_infeas <- {}, {}
     4: while ELAPSEDTIME() < B do
     5:     v <- SELECT(T)                  # cost-based UCT with progressive widening
     6:     c+ <- PROPOSESUCCESSOR(v)
     7:     C+ <- C(v) + c+
     8:     (feasible, J) <- VERIFY(C+, F, K_feas, K_infeas)
     9:     if not feasible then
    10:         continue
    11:     end if
    12:     ADDCHILD(T, v, c+, J)
    13:     if GOAL(C+) then
    14:         if TO in F then
    15:             goalFeasible <- true
    16:         else
    17:             goalFeasible <- TRAJECTORYOPT.(C+)
    18:         end if
    19:         if goalFeasible then
    20:             S_goal <- S_goal + {C+}
    21:         end if
    22:     end if
    23: end while
    24: return S_goal

Transcribed literally, including line 14's structure: if TO is already one of the
filters then VERIFY has run it and the goal sequence is known dynamically feasible;
otherwise Alg. 1 pays for TO once, here, only on sequences that actually reach the
goal. That is the whole point of the F1/F2 variants in Table II --

    "Filters that exclude TO during node expansion explore substantially faster and
     invoke TO only after reaching the discrete goal."   -- Section IV-B

TO is Milestone 6. Until it exists, `trajectory_opt` is None and a goal-reaching
sequence is recorded with `to_verified = False` rather than being silently counted
as dynamically feasible -- the distinction Table II reports in its "No. sol." row.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from faro.core.modes import ContactMode, ContactSequence
from faro.scene.scene import Scene
from faro.search.tree import Goal, Node, iter_nodes, make_root
from faro.search.uct import (
    DEFAULT_ALPHA, DEFAULT_C, DEFAULT_K, propose_successor, select,
)
from faro.search.verify import Caches, VerifyResult, normalize_filters, verify


@dataclass
class Solution:
    """One entry of `S_goal` (Alg. 1 line 20)."""

    sequence: ContactSequence
    cost: float
    #: Seconds from the start of the search to the moment it was found -- Table II's
    #: "Time to first sol." row is the minimum of these.
    found_at: float
    #: False when TO was not available to confirm dynamic feasibility.
    to_verified: bool = False


@dataclass
class SearchStats:
    """Table II's rows, so a run can be compared against the paper directly."""

    iterations: int = 0
    nodes: int = 1                     # the root exists from line 1
    rejected: int = 0
    attempts: dict[str, int] = field(default_factory=dict)
    filter_time: dict[str, float] = field(default_factory=dict)
    depth_reached: int = 0
    exhausted: int = 0                 # SELECT returned a node with nothing to propose
    elapsed: float = 0.0

    def record(self, result: VerifyResult) -> None:
        for name, seconds in result.times.items():
            self.attempts[name] = self.attempts.get(name, 0) + 1
            self.filter_time[name] = self.filter_time.get(name, 0.0) + seconds

    def summary(self, solutions: list[Solution]) -> str:  # pragma: no cover - display
        lines = [
            f"  iterations        {self.iterations}",
            f"  total tree nodes  {self.nodes}",
            f"  solutions         {len(solutions)}",
            f"  rejected          {self.rejected}",
            f"  max depth         {self.depth_reached}",
            f"  elapsed           {self.elapsed:.1f} s",
        ]
        if solutions:
            lines.append(f"  time to first sol {min(s.found_at for s in solutions):.1f} s")
        for name in sorted(self.attempts):
            total = self.filter_time.get(name, 0.0)
            count = self.attempts[name]
            lines.append(
                f"  {name:<3s} attempts      {count:4d}   {total:8.1f} s "
                f"({total / max(count, 1):.2f} s each)"
            )
        return "\n".join(lines)


@dataclass
class SearchResult:
    solutions: list[Solution]
    root: Node
    stats: SearchStats
    caches: Caches


def search(
    scene: Scene,
    root_mode: ContactMode,
    goal: Goal,
    *,
    budget: float,
    filters=("M", "E"),
    caches: Caches | None = None,
    C: float = DEFAULT_C,
    k: float = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    max_depth: int = 5,
    successor_policy: str = "all",
    seed: int = 0,
    trajectory_opt=None,
    eq14_kwargs: dict | None = None,
    kso_kwargs: dict | None = None,
    on_event=None,
    verify_fn=None,
) -> SearchResult:
    """Run Alg. 1 for `budget` seconds and return `S_goal`.

    `seed` controls PROPOSESUCCESSOR's shuffle and nothing else, which is what
    Section IV-B varies:

        "Each variant is evaluated over five runs with different seeds controlling
         the random successor selection."

    Everything else is deterministic given the seed -- Eq. 18 ties break on
    insertion order, and `faro/utils/determinism.py` pins the BLAS so Ipopt's
    verdicts do not vary between runs. That last one is not optional here: Alg. 1
    CACHES verdicts, so a filter that flips under thread scheduling does not produce
    occasional noise, it produces a permanent arbitrary answer.

    The budget is checked at the top of each iteration, so a single long filter call
    may overrun it. Interrupting a solve mid-flight would leave a partial verdict
    that is not cacheable, which is worse than finishing late.

    `verify_fn` replaces VERIFY entirely, taking `(scene, sequence)` and returning a
    `VerifyResult`. It exists so the UCT arithmetic can be exercised without solving
    anything -- `scripts/05_tree_search.py --dry-run` and most of
    `tests/test_search.py` use it. Injected rather than monkeypatched on purpose:
    this module imports `verify` by name, so patching the module attribute would
    silently have no effect here and the "dry" run would quietly solve NLPs.
    """
    filters = normalize_filters(filters)
    goal.validate(scene)
    root_mode.validate(scene)

    caches = caches if caches is not None else Caches()
    rng = np.random.default_rng(seed)
    root = make_root(root_mode)
    solutions: list[Solution] = []
    stats = SearchStats()

    started = time.perf_counter()
    while time.perf_counter() - started < budget:
        stats.iterations += 1

        # --- line 5: v <- SELECT(T) ---------------------------------------------
        node = select(root, C=C, k=k, alpha=alpha, max_depth=max_depth)

        # --- line 6: c+ <- PROPOSESUCCESSOR(v) ----------------------------------
        if node.depth >= max_depth:
            # Section IV-B's "max depth of 5 switches". The node was still visited,
            # so its count went up and Eq. 18 will look elsewhere next time.
            stats.exhausted += 1
            continue
        successor = propose_successor(node, scene, rng, policy=successor_policy)
        if successor is None:
            stats.exhausted += 1
            continue

        # --- line 7: C+ <- C(v) + c+ --------------------------------------------
        candidate = ContactSequence(tuple(m.mode for m in node.path()) + (successor,))

        # --- line 8: VERIFY ------------------------------------------------------
        if verify_fn is not None:
            result = verify_fn(scene, candidate)
        else:
            result = verify(scene, candidate, filters=filters, caches=caches,
                            eq14_kwargs=eq14_kwargs, kso_kwargs=kso_kwargs)
        stats.record(result)

        if on_event is not None:
            on_event(node, successor, candidate, result)

        # --- lines 9-11 ----------------------------------------------------------
        if not result.feasible:
            stats.rejected += 1
            continue

        # --- line 12: ADDCHILD ---------------------------------------------------
        child = Node(mode=successor, parent=node, depth=node.depth + 1, cost=result.cost)
        node.children.append(child)
        stats.nodes += 1
        stats.depth_reached = max(stats.depth_reached, child.depth)

        # --- lines 13-22: GOAL ---------------------------------------------------
        if goal.reached(candidate):
            if "TO" in filters:
                goal_feasible, to_verified = True, True
            elif trajectory_opt is not None:
                goal_feasible = bool(trajectory_opt(scene, candidate))
                to_verified = True
            else:
                # Milestone 6 is not here. Record the sequence -- it IS a discrete
                # goal-reaching plan that passed every active filter -- but do not
                # claim TO confirmed it.
                goal_feasible, to_verified = True, False
            if goal_feasible:
                solutions.append(Solution(
                    sequence=candidate, cost=result.cost,
                    found_at=time.perf_counter() - started, to_verified=to_verified,
                ))

    stats.elapsed = time.perf_counter() - started
    stats.nodes = sum(1 for _ in iter_nodes(root))
    return SearchResult(solutions=solutions, root=root, stats=stats, caches=caches)
