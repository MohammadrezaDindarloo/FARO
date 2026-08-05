#!/usr/bin/env python
"""Milestone 4 -- watch Eq. 15 decide whether a whole contact SEQUENCE holds together.

Eq. 14 asked whether one mode is reachable. Eq. 15 asks the same of a plan, and adds
the one constraint that only exists across time:

    (p^{s+1} - p^s)_{x,y} = 0,   log3( (R^s)^T R^{s+1} )_z = 0        (8)

a contact that persists may not SLIDE. Every step of a sequence can be individually
feasible and the sequence still impossible, which is exactly what makes Eq. 15 worth
more than K calls to Eq. 14.

WHAT TO LOOK AT
---------------
The verdict is one bit and it is the least interesting output. Watch instead:

  * the FEET across a transition. Both steps have them on the floor either way; Eq. 8
    is what makes them be in the SAME PLACE. `--drift` prints that in millimetres.
  * step 0 is PINNED. Eq. 15 states `q_0 = q_init` as a hard equality, so the first
    configuration is the scene's own initial state and never moves. Steps 1..K are the
    ones being solved for.

TWO THINGS EQ. 15 DOES NOT DO, so do not read them into the answer:

  * it does not check BALANCE -- there are no forces (Section IV-A). A sequence of
    head-first dives is perfectly acceptable here; that is the TO's job.
  * it does not order TIME -- there is no velocity and no duration, only which patches
    touch at each step.

Run:
    conda activate faro
    python scripts/04_kso_demo.py --list
    python scripts/04_kso_demo.py                    # the tour, animated in Meshcat
    python scripts/04_kso_demo.py -s pick-place      # just one
    python scripts/04_kso_demo.py --drift            # contact drift table, no viz
    python scripts/04_kso_demo.py --warm             # start from per-mode Eq. 14 answers

Navigation is script 03's, single keypress:
    ENTER  next entry     SPACE  replay the animation     p  previous     q  quit
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pinocchio as pin

from faro.kso.feasibility import KSOCache, check, warm_start
from faro.scenarios.kso_demos import ALL_DEMOS, get_demo
from faro.scene.scene import Scene
from faro.viz.keyboard import NEXT, PREVIOUS, QUIT, REPEAT, read_navigation, stdin_is_interactive

_WIDTH = 78


def _rule(char: str = "-") -> str:
    return char * _WIDTH


def _wrap(text: str, indent: str = "  ") -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=_WIDTH - len(indent),
                                   initial_indent=indent, subsequent_indent=indent))


def _pose(state: dict, name: str) -> pin.SE3:
    q = state[name]
    return pin.SE3(pin.Quaternion(q[6], q[3], q[4], q[5]).toRotationMatrix(), q[:3])


def _patch_world(scene: Scene, patch, state: dict) -> pin.SE3:
    if patch.attachment.value == "robot":
        return scene.patch_world_placement(patch, state["robot"])
    if patch.attachment.value == "object":
        return _pose(state, patch.parent) * patch.placement
    return patch.placement.copy()


def contact_drift(scene: Scene, demo, report) -> list[tuple[int, str, float]]:
    """How far each PERSISTING contact slid, measured the way Eq. 8 defines it.

    RELATIVE, not world. Eq. 8 constrains the pose of patch `a` **with respect to its
    partner `b`**, not with respect to the world:

        (p^{s+1} - p^s)_{x,y} = 0    with p the relative position

    Measuring world motion instead reports a violation whenever the SUPPORT moves --
    a hand sticking to the box while the box is carried travels half a metre in the
    world and slides not at all. The first version of this function did exactly that
    and reported 506 mm of "drift" on `pick`, which was the box being lifted.

    So this measures `a` in `b`'s frame at both steps and returns the in-plane
    distance between them. For every persisting contact, on a moving support or not,
    this must be ~0 or Eq. 8 is not doing its job.
    """
    from faro.kso.problem import mode_at

    sequence = demo.sequence(scene)
    rows = []
    # K transitions over K+1 knots, using c_K := c_{K-1} -- the same indexing Eq. 15
    # and `no_slip_blocks` use. `sequence.transitions()` gives K-1 and would silently
    # skip the last one.
    for s in range(len(sequence)):
        mode, nxt = mode_at(sequence, s), mode_at(sequence, s + 1)
        for interface in sorted(scene.interfaces):
            partner = mode.partner(interface)
            if partner is None or nxt.partner(interface) != partner:
                continue
            patch_a = scene.interfaces[interface].patch
            patch_b = scene.patches[partner]
            here, there = report.configurations[s], report.configurations[s + 1]
            rel0 = _patch_world(scene, patch_b, here).inverse() * _patch_world(scene, patch_a, here)
            rel1 = _patch_world(scene, patch_b, there).inverse() * _patch_world(scene, patch_a, there)
            moved = float(np.linalg.norm((rel1.translation - rel0.translation)[:2]))
            rows.append((s, f"{interface}->{partner}", moved))
    return rows


#: Alg. 1's `K_feas` / `K_infeas` for the Eq. 14 filters. Module-level so it survives
#: across demos in one run: Section II-D caches mode/edge results precisely because the
#: same edges recur, and the tour's five sequences share most of theirs.
EDGE_CACHE = None


def goal_for(scene: Scene, demo):
    """`Q_goal` for a demo (Eq. 15's terminal condition).

    The paper names Q_goal and does not define it (docs/eq15_kso.md). For box-placement
    the task goal is a CONTACT condition -- "the box is on the platform" -- so it is
    represented as the terminal mode's own contact set when the sequence already ends
    in the goal state, and None otherwise. Passing None makes q_K unconstrained beyond
    c_K, which is the honest reading for a sequence that is not a complete task.
    """
    terminal = demo.sequence(scene)[-1]
    return terminal if terminal.partner("box_bottom") == "tabletop" else None


def show(demo, scene: Scene, viz, cache: KSOCache, args) -> bool:
    sequence = demo.sequence(scene)

    print()
    print(_rule("="))
    print(f"  {demo.title}")
    print(f"  {demo.question}")
    print(_rule("="))
    for s, mode in enumerate(sequence):
        print(f"    step {s}: {mode.label()}")
    print()
    print(_wrap(demo.why))
    print()

    # SECTION III'S FILTER ORDER, AND IT IS NOT COSMETIC.
    # VERIFY applies M, then E, then KSO, "ordered from cheaper to more expensive",
    # and a candidate only reaches the next filter by passing the previous one. So a
    # sequence that fails M or E is one the paper NEVER hands to Eq. 15 -- running the
    # KSO on it anyway asks the solver a question the method never asks, and whatever
    # it answers says nothing about our KSO. `regrasp` is exactly such a sequence.
    from faro.mode_edge.feasibility import FeasibilityCache
    from faro.mode_edge.feasibility import check as check_eq14

    from faro.kso.problem import edge_at

    global EDGE_CACHE
    if EDGE_CACHE is None:
        EDGE_CACHE = FeasibilityCache()

    print(f"  filter M: {len(sequence)} modes with Eq. 14 ... ", end="", flush=True)
    mode_lines, passed = [], True
    for s_index, mode in enumerate(sequence):
        r = check_eq14(scene, mode, cache=EDGE_CACHE)
        passed &= r.feasible
        mode_lines.append(f"       M  c_{s_index}         "
                          f"{'ok    ' if r.feasible else 'FAILS '} {r.result.status}")
    print("done")

    print(f"  filter E: {len(sequence)} edges with Eq. 14 ... ", end="", flush=True)
    edge_lines = []
    for s_index in range(1, len(sequence) + 1):
        edge_report = check_eq14(scene, edge_at(sequence, s_index), cache=EDGE_CACHE)
        passed &= edge_report.feasible
        edge_lines.append(
            f"       E  c_{s_index - 1} u c_{s_index}   "
            f"{'ok    ' if edge_report.feasible else 'FAILS '} {edge_report.result.status}"
        )
    print(f"{EDGE_CACHE.hits} cache hits / {EDGE_CACHE.hits + EDGE_CACHE.misses} lookups")

    print()
    print("     Filters M and E (Eq. 14) -- Section III runs these BEFORE the KSO")
    print("\n".join(mode_lines))
    print("\n".join(edge_lines))

    if not passed:
        print()
        print(_wrap(
            "M or E rejected this sequence, so Section III's VERIFY stops here and "
            "Eq. 15 is never invoked. That IS the verdict: the tree search prunes "
            "this branch on the cheap filters and never pays for a KSO."))
        ok = demo.expect == "infeasible"
        print(f"\n     expected: {demo.expect}   -> {'MATCHES' if ok else 'DOES NOT MATCH'}")
        return ok

    # Alg. 1 has already solved every edge by now, so the warm start is free here --
    # and knot s must be seeded from c_{s-1} u c_s, the contact set Eq. 15 gives it.
    print(f"  filter KSO: Eq. 15 on {len(sequence) + 1} configurations "
          f"via acados SQP ... ", end="", flush=True)
    q0 = warm_start(scene, sequence, cache=EDGE_CACHE) if args.warm else None
    report = check(scene, sequence, cache=None if args.warm else cache,
                   q0=q0, goal=goal_for(scene, demo))
    print(f"{report.wall_time:.1f} s")

    mark = "FEASIBLE  " if report.feasible else "INFEASIBLE"
    print(f"  -> {mark} {report.result.status:28s} {report.wall_time:6.2f} s total")
    print(f"     backend {report.result.backend}, {report.result.iterations} SQP "
          f"iteration(s)")

    ok = report.feasible == (demo.expect == "feasible")
    print(f"     expected: {demo.expect}   -> {'MATCHES' if ok else 'DOES NOT MATCH'}")
    if not report.feasible:
        print(f"     worst row: {report.result.worst_row or 'n/a'}")

    if report.feasible:
        print()
        print(f"     penetration: {-report.max_penetration * 1e3:.2f} mm"
              + (f"  (step {report.penetrating_step}, {report.penetrating_pair})"
                 if report.max_penetration < 0 else ""))

    # SHOWN EVEN WHEN INFEASIBLE, and that is the point of a diagnostic demo. The
    # solver returns configurations either way, and the iterate it gave up on is the
    # evidence for WHY it gave up -- which knot is contorted, which contact slid.
    # Hiding it behind `feasible` leaves an 11-minute run with an empty viewer and
    # nothing to look at.
    if report.configurations:
        print()
        if not report.feasible:
            print("     the iterate the solver stopped at (NOT a solution):")
        print(f"     {'step':<6}{'base xyz':<26}{'box xyz':<26}{'lean'}")
        for s, state in enumerate(report.configurations):
            q = state["robot"]
            R = pin.Quaternion(q[6], q[3], q[4], q[5]).toRotationMatrix()
            lean = np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
            box = np.round(state["box"][:3], 3) if "box" in state else "-"
            print(f"     {s:<6}{str(np.round(q[:3], 3)):<26}{str(box):<26}{lean:5.1f} deg")

        drift = contact_drift(scene, demo, report)
        if drift:
            print()
            print("     Eq. 8 check -- how far each PERSISTING contact moved:")
            for s, label, moved in drift:
                flag = "" if moved < 1e-3 else "   <-- SLID"
                print(f"       {s}->{s + 1}  {label:<28} {moved * 1e3:8.3f} mm{flag}")

    if viz is not None and report.configurations:
        animate(viz, scene, report, args.dwell)

    return ok


def animate(viz, scene: Scene, report, dwell: float) -> None:
    """Put each step on screen in order. No interpolation -- and that is the point.

    Eq. 15 has no time and no velocity, so there is nothing BETWEEN two steps to draw.
    Tweening them would invent motion the optimization never claimed existed and would
    make an unreachable jump look like a smooth transition. What you are watching is a
    sequence of poses, which is exactly what the KSO returns.
    """
    for s, state in enumerate(report.configurations):
        for name in sorted(scene.objects):
            viz.update_object_pose(name, _pose(state, name))
        viz.display(state["robot"])
        print(f"       [viz] step {s}", end="\r", flush=True)
        time.sleep(dwell)
    print(" " * 30, end="\r")


def run_drift_table(scene: Scene, args) -> None:
    """Every demo, verdict + worst persisting-contact drift. The Eq. 8 summary."""
    print()
    print(f"  {'demo':<20}{'verdict':>10}{'pen(mm)':>9}{'max drift':>11}{'t(s)':>8}")
    print(f"  {_rule('-')[:60]}")
    cache = KSOCache()
    for demo in ALL_DEMOS:
        sequence = demo.sequence(scene)
        report = check(scene, sequence, cache=cache, goal=goal_for(scene, demo))
        verdict = "FEAS" if report.feasible else "infeas"
        if report.feasible:
            drift = contact_drift(scene, demo, report)
            worst = max((m for _, _, m in drift), default=0.0) * 1e3
            pen = f"{-report.max_penetration * 1e3:9.2f}"
            drift_s = f"{worst:9.3f}mm"
        else:
            pen, drift_s = "        -", "         -"
        print(f"  {demo.key:<20}{verdict:>10}{pen}{drift_s:>11}{report.wall_time:8.2f}")
    print()
    print(_wrap("Drift is the quantity Eq. 8 exists to control, and it is the only "
                "column here that Eq. 14 could not have produced. Measured RELATIVE to "
                "the contact partner, which is how Eq. 8 defines it -- so a hand riding "
                "a lifted box reads 0, not the half-metre it travels in the world. "
                "Every persisting contact should read ~0; anything else means Eq. 8 is "
                "not being applied where Eq. 16 says it should be.", "  "))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--scenario", help="run a single entry by key")
    parser.add_argument("--list", action="store_true", help="list the entries and exit")
    parser.add_argument("--no-viz", action="store_true", help="no Meshcat, print only")
    parser.add_argument("--auto", action="store_true", help="play through without prompting")
    parser.add_argument("--drift", action="store_true",
                        help="contact-drift table over every entry (implies --no-viz)")
    # ON by default: Alg. 1 has already solved every edge through filter E before the
    # KSO runs, so seeding knot s from c_{s-1} u c_s costs nothing and is what the
    # method actually provides. `--cold` starts from q_nom, which is a deliberately
    # poor guess kept only so the difference stays measurable (Section IV-C names
    # initialization as the source of the paper's false negatives).
    parser.add_argument("--cold", dest="warm", action="store_false",
                        help="start from q_nom instead of the Eq. 14 edge answers")
    parser.set_defaults(warm=True)
    parser.add_argument("--dwell", type=float, default=1.2,
                        help="seconds to hold each step on screen (default 1.2)")
    parser.add_argument("--scene", default="box_placement")
    args = parser.parse_args()

    if args.list:
        print(f"\n  {len(ALL_DEMOS)} sequences:\n")
        for demo in ALL_DEMOS:
            print(f"    {demo.key:20s} [{demo.expect:10s}] {len(demo.steps)} steps  {demo.title}")
        print()
        return 0

    scene = Scene.from_config(args.scene)

    if args.drift:
        run_drift_table(scene, args)
        return 0

    viz = None
    if not args.no_viz:
        from urllib.parse import urlparse

        from faro.viz.connection import print_connection_help
        from faro.viz.meshcat_viz import SceneVisualizer

        viz = SceneVisualizer(scene, open_browser=False)
        print_connection_help(urlparse(viz.url).port or 7000)

    demos = [get_demo(args.scenario)] if args.scenario else list(ALL_DEMOS)
    cache = KSOCache()
    interactive = stdin_is_interactive() and not args.auto

    index, mismatches = 0, []
    while 0 <= index < len(demos):
        demo = demos[index]
        if not show(demo, scene, viz, cache, args):
            mismatches.append(demo.key)

        if not interactive:
            index += 1
            continue

        action = read_navigation("\n  [ENTER] next   [SPACE] replay   [p] previous   [q] quit  ")
        if action == QUIT:
            break
        if action == PREVIOUS:
            index = max(0, index - 1)
        elif action == REPEAT:
            continue
        else:
            index += 1

    print()
    print(_rule("="))
    print(f"  {len(demos) - len(mismatches)}/{len(demos)} entries matched their predicted verdict.")
    if mismatches:
        print(f"  Did not match: {', '.join(mismatches)}")
    print(f"  {cache.summary()}")
    print(_rule("="))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
