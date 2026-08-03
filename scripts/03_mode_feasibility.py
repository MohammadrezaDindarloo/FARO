#!/usr/bin/env python
"""Milestone 3 -- watch Eq. 14 decide whether a contact mode is reachable.

Each entry poses a contact mode (or a transition between two), solves the inverse
kinematics problem of Eq. 14, and puts the resulting configuration on screen. The
verdict is printed next to what the paper's hierarchy says it should be.

LOOK AT THE POSE, NOT JUST THE VERDICT. "Feasible" is one bit, and the dangerous
bugs make MORE things feasible, not fewer -- a dropped constraint, a patch offset in
the wrong frame, an object whose variables never reach the solver. None of those show
up in the verdict. All of them show up the moment you look at where the robot and the
box actually ended up.

Run:
    conda activate faro
    python scripts/03_mode_feasibility.py --list
    python scripts/03_mode_feasibility.py                  # the tour, in Meshcat
    python scripts/03_mode_feasibility.py -s grasp         # just one
    python scripts/03_mode_feasibility.py --no-viz         # headless table
    python scripts/03_mode_feasibility.py --sweep          # ALL 108 modes, no viz

`--sweep` is the one that shows what the filter is for: it runs the entire discrete
action space of Table IV through Eq. 14 and reports how much of it survives, which is
the pruning Alg. 1 is built on.

Navigation is the same as the Milestone 2 playground -- single keypress, no ENTER:

    ENTER   next        SPACE   re-solve this one        p  previous        q  quit

SETUP.md section 4 covers the browser side; the connection banner is printed for you.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from faro.core.modes import ContactEdge, enumerate_modes
from faro.mode_edge.feasibility import FeasibilityCache, check
from faro.scenarios.mode_demos import ALL_DEMOS, get_demo
from faro.scene.scene import Scene
from faro.viz.keyboard import NEXT, PREVIOUS, QUIT, REPEAT, read_navigation, stdin_is_interactive

_WIDTH = 78


def _rule(char: str = "-") -> str:
    return char * _WIDTH


def _wrap(text: str, indent: str = "  ") -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=_WIDTH - len(indent),
                                   initial_indent=indent, subsequent_indent=indent))


def _verdict_line(report) -> str:
    mark = "FEASIBLE  " if report.feasible else "INFEASIBLE"
    return (
        f"  -> {mark} {report.result.status:28s} "
        f"{report.result.iterations:4d} iters   {report.result.wall_time * 1e3:6.0f} ms"
    )


def _agrees(demo, report) -> bool:
    return report.feasible == (demo.expect == "feasible")


def show(demo, scene: Scene, viz, cache: FeasibilityCache) -> bool:
    target = demo.target(scene)

    print()
    print(_rule("="))
    print(f"  {demo.title}")
    print(f"  {demo.question}")
    print(_rule("="))
    print(f"  mode: {target.label()}")
    print()
    print(_wrap(demo.why))
    print()

    report = check(scene, target, cache=cache)
    print(_verdict_line(report))
    if not report.feasible:
        print(f"     worst row: {report.result.worst_row or 'n/a'}")

    ok = _agrees(demo, report)
    print(f"     expected: {demo.expect}   -> {'MATCHES' if ok else 'DOES NOT MATCH'}")

    # Where things actually ended up. This is the part worth reading.
    q = report.configurations["robot"]
    print(f"     base    : xyz = {np.round(q[:3], 3)}")
    for name in sorted(scene.objects):
        print(f"     {name:8s}: xyz = {np.round(report.configurations[name][:3], 3)}")
    for patch_name in ("left_foot_sole", "right_foot_sole", "left_hand_patch", "right_hand_patch"):
        placement = scene.patch_world_placement(scene.patches[patch_name], q)
        print(f"     {patch_name:17s} at z = {placement.translation[2]:+.4f}")

    if viz is not None:
        for name in sorted(scene.objects):
            pose = report.configurations[name]
            import pinocchio as pin

            rotation = pin.Quaternion(pose[6], pose[3], pose[4], pose[5]).toRotationMatrix()
            viz.update_object_pose(name, pin.SE3(rotation, pose[:3]))
        viz.display(q)

    return ok


def run_sweep(scene: Scene) -> None:
    """Every mode Table IV allows, through Eq. 14. The pruning Alg. 1 stands on."""
    from collections import Counter

    modes = enumerate_modes(scene)
    print(f"\n  Running all {len(modes)} modes of Table IV through Eq. 14 ...\n")

    cache = FeasibilityCache()
    started = time.perf_counter()
    reports = [(m, check(scene, m, cache=cache)) for m in modes]
    elapsed = time.perf_counter() - started

    feasible = sum(1 for _, r in reports if r.feasible)
    print(_rule("="))
    print(f"  {len(modes)} modes in {elapsed:.1f} s  ({elapsed / len(modes) * 1e3:.0f} ms each)")
    print(f"  feasible: {feasible}    infeasible: {len(modes) - feasible}")
    print(_rule("="))
    for status, count in Counter(r.result.status for _, r in reports).most_common():
        print(f"    {status:30s} {count:4d}")

    print("\n  The modes that did NOT pass:\n")
    for mode, report in reports:
        if not report.feasible:
            print(f"    {report.result.status:28s} {mode.label()}")
    print(
        "\n  Read the two statuses differently. `Infeasible_Problem_Detected` is a claim\n"
        "  about the scene -- Ipopt proved no configuration exists.\n"
        "  `Maximum_Iterations_Exceeded` is a claim about this solve, and Section IV-C\n"
        "  names it as where the paper's own false negatives come from.\n"
    )
    print(f"  {cache.summary()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--scenario", help="run a single entry by key")
    parser.add_argument("--list", action="store_true", help="list the entries and exit")
    parser.add_argument("--no-viz", action="store_true", help="no Meshcat, print only")
    parser.add_argument("--auto", action="store_true", help="play through without prompting")
    parser.add_argument("--sweep", action="store_true",
                        help="run all 108 modes of Table IV (implies --no-viz)")
    parser.add_argument("--scene", default="box_placement")
    args = parser.parse_args()

    if args.list:
        print(f"\n  {len(ALL_DEMOS)} entries:\n")
        for demo in ALL_DEMOS:
            print(f"    {demo.key:16s} [{demo.expect:14s}] {demo.title}")
        print()
        return 0

    scene = Scene.from_config(args.scene)

    if args.sweep:
        run_sweep(scene)
        return 0

    viz = None
    if not args.no_viz:
        from faro.viz.connection import print_connection_help
        from faro.viz.meshcat_viz import SceneVisualizer

        viz = SceneVisualizer(scene, open_browser=False)
        from urllib.parse import urlparse

        print_connection_help(urlparse(viz.url).port or 7000)

    demos = [get_demo(args.scenario)] if args.scenario else list(ALL_DEMOS)
    cache = FeasibilityCache()
    interactive = stdin_is_interactive() and not args.auto

    index, mismatches = 0, []
    while 0 <= index < len(demos):
        demo = demos[index]
        if not show(demo, scene, viz, cache):
            mismatches.append(demo.key)

        if not interactive:
            index += 1
            continue

        action = read_navigation("\n  [ENTER] next   [SPACE] re-solve   [p] previous   [q] quit  ")
        if action == QUIT:
            break
        if action == PREVIOUS:
            index = max(0, index - 1)
        elif action != REPEAT:
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
