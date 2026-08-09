#!/usr/bin/env python
"""Milestone 5 -- watch Alg. 1 search the space of contact-mode sequences.

The tree is rooted at c_0 and grown by cost-based UCT (Eq. 18) with progressive
widening (Eq. 19). Every proposed successor is put through the filters of Section
III before it is allowed in, and the mode/edge verdicts are cached, which is what
Table II attributes the method's advantage to.

WHAT TO WATCH, AND IT IS NOT THE SOLUTION COUNT
-----------------------------------------------
Watch the REJECTIONS and the CACHE. A tree search that admits everything is not
searching, it is enumerating, and the whole claim of the paper is that cheap filters
prune enough to make the expensive ones affordable:

    "Since contact modes and transitions are repeatedly revisited during search,
     cached infeasibility results allow previously checked queries to be rejected
     without resolving the corresponding optimization problems."   -- Section IV-B

The live log prints one line per iteration: which filter rejected the candidate, or
the cost it was admitted with, and whether the verdict came from the cache. Once the
cache starts hitting, those lines get much faster, and that is the effect to see.

Run:
    conda activate faro
    python scripts/05_tree_search.py --budget 300           # five minutes
    python scripts/05_tree_search.py --budget 300 --quiet   # totals only
    python scripts/05_tree_search.py --dry-run              # UCT only, no solving
    python scripts/05_tree_search.py --budget 7200 --seed 1 # the paper's budget

`--dry-run` replaces the filters with "everything passes". It solves nothing, runs
in a second, and is there to show the tree SHAPE that Eqs. 18 and 19 produce --
which is the part that is pure arithmetic and worth seeing on its own before it is
mixed with a solver's verdicts.
"""

from __future__ import annotations

import argparse
import sys
import time

import faro  # noqa: F401  -- pins the BLAS before numpy/pinocchio load

from faro.search.config import SearchConfig
from faro.search.search import search
from faro.search.tree import iter_nodes
from faro.search.verify import Caches, VerifyResult
from faro.scene.scene import Scene

_WIDTH = 78


def _rule(char: str = "-") -> str:
    return char * _WIDTH


def make_logger(quiet: bool):
    """One line per Alg. 1 iteration."""
    state = {"n": 0, "t0": time.perf_counter()}

    def log(node, successor, candidate, result: VerifyResult) -> None:
        state["n"] += 1
        if quiet:
            return
        elapsed = time.perf_counter() - state["t0"]
        spent = sum(result.times.values())
        tag = "cache" if result.cached else f"{spent:5.1f}s"
        if result.feasible:
            verdict = f"ADMIT  J={result.cost:8.3f}"
        else:
            verdict = f"reject by {result.rejected_by:<3s} {result.status[:26]}"
        print(f"  {state['n']:5d} {elapsed:7.1f}s  d{len(candidate) - 1} "
              f"{tag:>7s}  {verdict}", flush=True)

    return log


def dry_run(scene, cfg, budget: float) -> None:
    """Grow the tree with a filter that always passes -- Eq. 18/19 shape only."""

    def always_pass(scene, sequence) -> VerifyResult:
        # Cost stands in for J. Fewer active contacts is "cheaper", which is
        # arbitrary -- it exists only so Eq. 18's argmin has something to order by.
        return VerifyResult(feasible=True, cost=float(5 - sequence[-1].n_active()),
                            times={"M": 0.0}, status="dry-run")

    result = search(scene, cfg.root, cfg.goal, verify_fn=always_pass,
                    **{**cfg.as_kwargs(), "budget": budget})

    depths: dict[int, int] = {}
    for node in iter_nodes(result.root):
        depths[node.depth] = depths.get(node.depth, 0) + 1

    print(f"\n{_rule('=')}\n  DRY RUN -- no solving, {budget:.0f} s of pure UCT\n{_rule('=')}")
    print(result.stats.summary(result.solutions))
    print("\n  nodes by depth:")
    for depth in sorted(depths):
        print(f"    d{depth}  {depths[depth]:6d}  {'#' * min(depths[depth], 60)}")
    print(f"\n  branching at the root: {len(result.root.children)} children "
          f"from {result.root.visits} visits (Eq. 19 caps it at "
          f"floor(k*N^alpha) = {int(cfg.k * result.root.visits ** cfg.alpha)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="box_placement")
    parser.add_argument("--config", default="default", help="configs/search/<name>.yaml")
    parser.add_argument("--budget", type=float, help="override the config's budget, seconds")
    parser.add_argument("--filters", nargs="*", help="override, e.g. --filters M E")
    parser.add_argument("--seed", type=int, help="override the config's seed")
    parser.add_argument("--quiet", action="store_true", help="totals only, no per-iteration log")
    parser.add_argument("--dry-run", action="store_true",
                        help="grow the tree with no solving -- Eq. 18/19 shape only")
    args = parser.parse_args()

    scene = Scene.from_config(args.scene)
    cfg = SearchConfig.from_config(scene, args.config)
    if args.budget is not None:
        cfg.budget = args.budget
    if args.filters:
        cfg.filters = tuple(args.filters)
    if args.seed is not None:
        cfg.seed = args.seed

    if args.dry_run:
        dry_run(scene, cfg, min(cfg.budget, 5.0))
        return 0

    print()
    print(_rule("="))
    print(f"  Alg. 1 on scene {scene.name!r}, config {cfg.name!r}")
    print(_rule("="))
    print(f"  root c_0        {cfg.root.label()}")
    print(f"  goal            {cfg.goal.label()}")
    print(f"  filters F       {list(cfg.filters)}")
    print(f"  budget          {cfg.budget:.0f} s      max depth {cfg.max_depth} switches")
    print(f"  UCT             C={cfg.C}  k={cfg.k}  alpha={cfg.alpha}   "
          f"(Section III's stated values)")
    print(f"  successors      {cfg.successor_policy}")
    print(f"  seed            {cfg.seed}")
    print(_rule())
    if not args.quiet:
        print(f"  {'iter':>5s} {'elapsed':>8s} {'d':>2s} {'cost':>7s}  verdict")

    caches = Caches()
    result = search(scene, cfg.root, cfg.goal, caches=caches,
                    on_event=make_logger(args.quiet), **cfg.as_kwargs())

    print()
    print(_rule("="))
    print("  RESULT")
    print(_rule("="))
    print(result.stats.summary(result.solutions))
    print(f"\n  {caches.summary()}")

    if result.solutions:
        print(f"\n  S_goal -- {len(result.solutions)} goal-reaching sequence(s):\n")
        for i, solution in enumerate(sorted(result.solutions, key=lambda s: s.cost)[:10]):
            flag = "" if solution.to_verified else "   (TO not run -- Milestone 6)"
            print(f"    {i + 1}. J={solution.cost:8.3f} at {solution.found_at:6.1f} s"
                  f"  {len(solution.sequence)} modes{flag}")
            for s, mode in enumerate(solution.sequence):
                print(f"         c_{s}  {mode.label()}")
            print()
    else:
        print("\n  S_goal is EMPTY. That is a real outcome, not necessarily a bug --")
        print("  Table II reports 0.0 solutions for the TO-only baseline on the hard")
        print("  task. Check the rejection counts above: if almost everything is")
        print("  rejected by M, the goal may be unreachable under these filters.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
