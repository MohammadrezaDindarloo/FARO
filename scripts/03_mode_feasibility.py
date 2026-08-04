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
from faro.mode_edge.feasibility import FeasibilityCache, _object_poses, check
from faro.scenarios.mode_demos import ALL_DEMOS, get_demo
from faro.scene.scene import Scene
from faro.scene.symbolic import SymbolicScene
from faro.utils.paths import CONFIG_DIR
from faro.viz.keyboard import NEXT, PREVIOUS, QUIT, REPEAT, read_navigation, stdin_is_interactive

_WIDTH = 78


def _rule(char: str = "-") -> str:
    return char * _WIDTH


def _wrap(text: str, indent: str = "  ") -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width=_WIDTH - len(indent),
                                   initial_indent=indent, subsequent_indent=indent))


def _verdict_line(report) -> str:
    """The verdict, and the cost of GETTING it.

    Both times are printed because they are not the same number and the difference is
    not small. `result.wall_time` is one call to Ipopt -- the LAST refresh pass, which
    is by construction the cheapest of the sequence: it starts from the previous pass's
    answer and only has to confirm it. `report.wall_time` is what you actually waited
    for. On `grasp` those are 0.23 s and 33 s, and pass 1 alone is 87% of it (772
    iterations, thrown away as an intermediate iterate against stale witness data).

    Printing only the first number is what made the screen say 302 ms while the scene
    took a minute to update.
    """
    mark = "FEASIBLE  " if report.feasible else "INFEASIBLE"
    passes = f"{report.refreshes} pass" + ("" if report.refreshes == 1 else "es")
    return (
        f"  -> {mark} {report.result.status:28s} {report.wall_time:6.2f} s total\n"
        f"     {passes:>10s} of solve+relinearize; the last took "
        f"{report.result.iterations} iters / {report.result.wall_time * 1e3:.0f} ms"
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


def print_weights(scene: Scene) -> None:
    """Show W as the config resolved it -- per group AND per joint.

    Worth having because `joint_groups` matches by SUBSTRING with most-specific-wins,
    so what a given YAML block actually assigns is not always obvious by eye. Reading
    it back beats re-deriving it in your head, and it catches a fragment that matched
    fewer joints than you meant.
    """
    from collections import defaultdict

    from faro.mode_edge.problem import RegularizationWeights

    weights = RegularizationWeights.from_scene(scene)
    diagonal = weights.diagonal(scene)
    nq = scene.robot.nq

    print(f"\n  Eq. 14's W for scene {scene.name!r}: "
          f"{diagonal.size} x {diagonal.size} diagonal\n")
    print(f"    {'indices':<10}{'block':<30}{'weight':>8}")
    print(f"    {'-' * 48}")
    for lo, hi, label in [(0, 3, "base position (x,y,z)"),
                          (3, 7, "base quaternion (x,y,z,w)"),
                          (7, nq, "actuated joints (see below)")]:
        shown = f"{diagonal[lo]:.4g}" if len(set(diagonal[lo:hi])) == 1 else "varies"
        print(f"    {lo}-{hi - 1:<8}{label:<30}{shown:>8}")
    offset = nq
    for name in sorted(scene.objects):
        print(f"    {offset}-{offset + 2:<8}{name + ' position':<30}{diagonal[offset]:>8.4g}")
        print(f"    {offset + 3}-{offset + 6:<8}{name + ' quaternion':<30}{diagonal[offset + 3]:>8.4g}")
        offset += 7

    print(f"\n  Joints, grouped by the weight each resolved to "
          f"(fallback `joints` = {weights.joints:g}):\n")
    by_weight = defaultdict(list)
    for joint, value in weights.per_joint(scene).items():
        by_weight[value].append(joint)
    for value in sorted(by_weight, reverse=True):
        members = by_weight[value]
        print(f"    {value:8.4g}   {len(members):2d} joints")
        for joint in members:
            print(f"               {joint}")
    print()


def run_scenarios(scene_name: str, names: list[str]) -> None:
    """Run the tour under several `configs/collision/*.yaml` policies, side by side.

    WHAT TO COMPARE, in order of what actually matters:

      1. `pen`  -- worst interpenetration, audited over every pair IN THAT MODEL. A
         policy that drops a pair class cannot be caught by its own audit, so read
         this column together with the pair count, never alone.
      2. `lean` -- torso pitch. Eq. 14 has no balance term, so this is the only
         readout of whether the returned pose is one you would want the KSO
         warm-started from. Row count and pose quality are NOT monotonically related:
         all 689 pairs gives a WORSE `grasp` pose than the 0.10 cutoff does.
      3. `t`    -- seconds. The cheapest of the three to care about.
    """
    import numpy as np
    import pinocchio as pin

    from faro.scene.collision import SceneCollisionModel
    from faro.utils.config import load_config

    print()
    for name in names:
        cfg = load_config("collision", name)
        scene = Scene.from_config(scene_name)
        scene.collision = {**scene.collision,
                           **{k: v for k, v in cfg.items() if k != "name"}}
        model = SceneCollisionModel.cached(scene)
        always = model.always_active_pairs()

        print(_rule("="))
        print(f"  {name}")
        print(f"    pairs in model : {len(model.pairs):4d}   "
              f"{', '.join(f'{k} {v}' for k, v in sorted(model.class_counts().items()))}")
        print(f"    always active  : {len(always):4d}   "
              f"(exempt from activation_distance = {scene.collision['activation_distance']})")
        print(_rule("="))
        print(f"    {'demo':16s} {'verdict':>9s} {'pen(mm)':>8s} {'lean':>7s} "
              f"{'rows':>6s} {'t(s)':>7s}")

        cache, total = FeasibilityCache(), 0.0
        for demo in ALL_DEMOS:
            report = check(scene, demo.target(scene), cache=cache)
            q = report.configurations["robot"]
            rotation = pin.Quaternion(q[6], q[3], q[4], q[5]).toRotationMatrix()
            lean = np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
            # Rows the FINAL pass actually built, at the answer it returned.
            rows = sum(
                b.n_ineq for b in model.blocks(
                    SymbolicScene(scene), q, _object_poses(scene, report.configurations),
                    margin=float(scene.collision["margin"]),
                    activation_distance=scene.collision["activation_distance"],
                    always_active=always,
                )
            )
            total += report.wall_time
            agrees = "" if _agrees(demo, report) else "  <- DIFFERS from expected"
            print(f"    {demo.key:16s} "
                  f"{('FEAS' if report.feasible else 'infeas'):>9s} "
                  f"{-report.max_penetration * 1e3:>8.2f} {lean:>7.1f} "
                  f"{rows:>6d} {report.wall_time:>7.2f}{agrees}")
        print(f"    {'TOTAL':16s} {'':>9s} {'':>8s} {'':>7s} {'':>6s} {total:>7.2f}\n")

    print(_wrap(
        "Lower `pen` is better and lower `lean` is better, but neither is a function "
        "of row count -- more rows can land Ipopt in a worse local minimum, which is "
        "why this sweep exists instead of a rule of thumb. A policy whose "
        "`include_pairs` drops a class is not audited on that class at all.", "  "))
    print()


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
    parser.add_argument("--weights", action="store_true",
                        help="print Eq. 14's W as your config resolved it, and exit")
    parser.add_argument("--scenarios", nargs="*", metavar="NAME",
                        help="compare configs/collision/*.yaml policies on the tour "
                             "(no names = all of them). Implies --no-viz.")
    parser.add_argument("--scene", default="box_placement")
    args = parser.parse_args()

    if args.list:
        print(f"\n  {len(ALL_DEMOS)} entries:\n")
        for demo in ALL_DEMOS:
            print(f"    {demo.key:16s} [{demo.expect:14s}] {demo.title}")
        print()
        return 0

    scene = Scene.from_config(args.scene)

    if args.weights:
        print_weights(scene)
        return 0

    if args.scenarios is not None:
        names = args.scenarios or sorted(
            p.stem for p in (CONFIG_DIR / "collision").glob("*.yaml"))
        run_scenarios(args.scene, names)
        return 0

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
