#!/usr/bin/env python
"""Milestone 2 -- see each constraint of Eqs. 7-13 hold, and break.

Every scenario sweeps ONE physical parameter and shows the constraint responding:
the patches turn from green to red, a force arrow leaves its friction pyramid, a
centre-of-pressure dot slides off the edge of the foot. The terminal prints the
watched rows live, and each scenario reports where the constraint ACTUALLY started
failing next to where the paper's equation says it MUST.

That last comparison is the point. Milestones 3-5 assemble these same expressions
into NLPs with thousands of rows, where a swapped axis or a sign error surfaces
only as "infeasible". Here each one is checked alone, against a number derived by
hand from the equation.

The paper's own statement of each equation is printed above its animation, so the
algebra and the motion are on screen together.

Run:
    conda activate faro
    python scripts/02_constraint_playground.py --list
    python scripts/02_constraint_playground.py --kill-stale       # all, in Meshcat
    python scripts/02_constraint_playground.py -s 7d-pitch        # just one
    python scripts/02_constraint_playground.py --no-viz           # headless table
    python scripts/02_constraint_playground.py --auto             # no prompts

After each scenario you choose what happens next -- single keypress, no ENTER
needed:

    ENTER   next scenario
    SPACE   replay this one (watch it again against the printed equation)
    p       previous scenario
    q       quit

Piped or non-interactive runs play straight through, so `--no-viz` still works in
scripts and CI.

SETUP.md section 4 covers the browser side; the connection banner is printed for you.
"""

from __future__ import annotations

import argparse
import time
from urllib.parse import urlparse

import numpy as np

from faro.scenarios.constraint_demos import (
    ALL_SCENARIOS,
    Scenario,
    SweepResult,
    get_scenario,
    paper_equation,
    resolve_predictions,
    run_sweep,
)
from faro.scene.scene import Scene
from faro.viz.keyboard import NEXT, PREVIOUS, QUIT, REPEAT, read_navigation, stdin_is_interactive

_TTY = True


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


GREEN, RED, YELLOW, BOLD, DIM, CYAN = "32", "31", "33", "1", "2", "36"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scene", default="box_placement")
    p.add_argument("-s", "--scenario", action="append", default=None,
                   help="scenario key(s) to run; repeatable. Default: all")
    p.add_argument("--list", action="store_true", help="list scenarios and exit")
    p.add_argument("--no-viz", action="store_true", help="headless: table only, no Meshcat")
    p.add_argument("--fps", type=float, default=25.0, help="animation rate")
    p.add_argument("--pause", type=float, default=2.0, help="seconds to hold at the end of a sweep")
    p.add_argument("--kill-stale", action="store_true",
                   help="terminate orphaned meshcat servers first")
    p.add_argument("--no-wait", action="store_true",
                   help="do not wait for ENTER before starting the animation")
    p.add_argument("--auto", action="store_true",
                   help="play all scenarios straight through without the navigation prompt")
    return p.parse_args()


# =============================================================================
def print_header(scenario: Scenario, index: int, total: int) -> None:
    print()
    print(c("=" * 78, DIM))
    print(c(f"  [{index}/{total}]  Eq. {scenario.equation}  --  {scenario.title}", BOLD))
    print(c("=" * 78, DIM))

    # The equation itself, exactly as the paper writes it, so the algebra and the
    # animation are on screen together.
    print(c("  THE PAPER SAYS:", BOLD))
    for line in paper_equation(scenario.equation).splitlines():
        print(c("      " + line, CYAN))
    print()

    print(c("  QUESTION  ", CYAN) + scenario.question)
    print(c("  EXPECT    ", CYAN) + _wrap(scenario.expect, indent=12))
    print(c("  PREDICTS  ", CYAN) + _wrap(scenario.prediction_note, indent=12))
    if scenario.predicted_crossing is None:
        print(c("  CROSSING  ", CYAN) + "never -- this constraint should NOT care")
    else:
        print(c("  CROSSING  ", CYAN) + f"{scenario.predicted_crossing:.4f}  ({scenario.param_label})")
    print()


def _wrap(text: str, indent: int, width: int = 76) -> str:
    import textwrap

    lines = textwrap.wrap(text, width=width - indent)
    pad = " " * indent
    return ("\n" + pad).join(lines)


def report(result: SweepResult) -> bool:
    """Print the verdict for one sweep. Returns True if it matched the prediction."""
    scenario = result.scenario
    measured = result.measured_crossing
    predicted = scenario.predicted_crossing

    if predicted is None:
        ok = not result.ever_violated
        if ok:
            print(c("  RESULT    never violated, as predicted", GREEN))
        else:
            print(c(f"  RESULT    violated at {measured} -- but it should NOT", RED))
    elif measured is None:
        ok = False
        print(c(f"  RESULT    never violated, but should have at {predicted:.4f}", RED))
    else:
        error = abs(measured - predicted)
        ok = error <= scenario.tolerance
        colour = GREEN if ok else RED
        print(c(f"  RESULT    predicted {predicted:.4f}   measured {measured:.4f}   "
                f"error {error:.2e}", colour))

    print(c(f"  {'MATCHES THE PAPER' if ok else 'DOES NOT MATCH -- INVESTIGATE'}",
            BOLD + ";" + (GREEN if ok else RED)))
    return ok


def print_navigation_hint(index: int, total: int) -> None:
    """The one-line key legend shown after each scenario."""
    keys = [
        (c("ENTER", BOLD), "next" if index + 1 < total else "finish"),
        (c("SPACE", BOLD), "replay this one"),
    ]
    if index > 0:
        keys.append((c("p", BOLD), "previous"))
    keys.append((c("q", BOLD), "quit"))
    print("\n  " + c("|", DIM).join(f"  {key} {label}  " for key, label in keys))


def format_rows(rows: dict[str, float], violated: bool) -> str:
    parts = [f"{label}={value:+.4f}" for label, value in rows.items()]
    status = c("VIOLATED", RED) if violated else c("ok      ", GREEN)
    return f"{status}  " + "  ".join(parts)


# =============================================================================
def animate(result: SweepResult, scene: Scene, vis, overlay, args) -> None:
    """Play the sweep in Meshcat with a live terminal readout."""
    # Imported here, not at module scope: `constraint_viz` pulls in meshcat, and
    # `--no-viz` must stay importable on a machine without it.
    from faro.viz.constraint_viz import (
        COLOR_FORCE, COLOR_MOMENT, COLOR_PYRAMID, COLOR_WEIGHT, FORCE_SCALE,
        contact_frame,
    )

    _VECTOR_COLORS = {"force": COLOR_FORCE, "weight": COLOR_WEIGHT,
                      "moment": COLOR_MOMENT, "capacity": COLOR_PYRAMID}
    scenario = result.scenario
    dt = 1.0 / max(args.fps, 1e-3)
    last_violated = None

    for step in result.steps:
        state = step.state

        if "q" in state:
            vis.display(state["q"])
        if "box_pose" in state:
            vis.update_object_pose("box", state["box_pose"])

        patches = state.get("patches", ())
        if patches:
            overlay.set_patch_status(patches, step.violated)

        # Every scenario must show its violation SOMEWHERE. Patches are the usual
        # signal, but Eqs. 9, 11 and 13 constrain an object or a joint and have none --
        # those declare an object to tint or a point to mark instead. A scenario with
        # no indicator at all is a bug, and `tests/test_constraint_viz.py` enforces it.
        for name in state.get("highlight_objects", ()):
            overlay.set_object_status(name, step.violated)
        if "status_at" in state:
            overlay.draw_status(state["status_at"], step.violated)
        if "probe" in state:
            overlay.draw_probe(state["probe"]["pose"], state["probe"]["size"])

        # Eqs. 10/11 are written about the CoM and the object body frame, not about a
        # contact patch, so their terms are drawn as world-frame arrows.
        for spec in state.get("vectors", ()):
            overlay.draw_vector(
                spec["origin"], spec["vector"],
                color=_VECTOR_COLORS.get(spec.get("color", "force"), COLOR_FORCE),
                scale=spec.get("scale", FORCE_SCALE),
                path=spec.get("path", "overlay/vector"),
            )
        if "com" in state:
            overlay.draw_point(state["com"], path="overlay/com")

        # Wrench scenarios also get the force arrow, friction pyramid and CoP dot.
        if "force" in state:
            anchor = scene.patches[state["anchor"]]
            # `contact_frame` applies the same Rx(pi) the constraint uses, so the
            # wrench is drawn in the frame Eq. 7 writes it in. Without it a sole's
            # outward (downward) normal buries all of this under the floor.
            placement = contact_frame(scene.patch_world_placement(anchor, state.get("q")))
            overlay.draw_force(placement, state["force"])
            overlay.draw_friction_pyramid(placement, state["mu"], float(state["force"][2]))
            overlay.draw_cop_bounds(placement, anchor.half_extents)
            overlay.draw_cop_marker(placement, state["force"], state["moment"])
            if "mu_r" in state:
                overlay.draw_torsion(
                    placement, state["moment"], state["mu_r"], state["force"]
                )

        # Print only when the status flips, plus a periodic sample, so the terminal
        # stays readable instead of scrolling 120 near-identical lines.
        flipped = step.violated != last_violated
        if flipped or step is result.steps[0] or step is result.steps[-1]:
            marker = c(" <-- CROSSES HERE", YELLOW) if flipped and last_violated is not None else ""
            print(f"    {scenario.param_label} = {step.param:+8.4f}   "
                  f"{format_rows(step.rows, step.violated)}{marker}")
            last_violated = step.violated

        time.sleep(dt)

    time.sleep(args.pause)
    overlay.clear()


def tabulate(result: SweepResult) -> None:
    """Headless: print a sampled table of the sweep."""
    steps = result.steps
    stride = max(1, len(steps) // 10)
    indices = set(range(0, len(steps), stride)) | {len(steps) - 1}
    # Always include the two steps that straddle the crossing -- the whole point.
    crossings = {i for i in range(len(steps) - 1) if steps[i].violated != steps[i + 1].violated}
    indices |= crossings | {i + 1 for i in crossings}

    for i in sorted(indices):
        step = steps[i]
        marker = c("  <-- CROSSES HERE", YELLOW) if i - 1 in crossings else ""
        print(f"    {result.scenario.param_label} = {step.param:+8.4f}   "
              f"{format_rows(step.rows, step.violated)}{marker}")


# =============================================================================
def main() -> None:
    args = parse_args()

    if args.list:
        print(c("\nConstraint scenarios\n", BOLD))
        for scenario in ALL_SCENARIOS:
            print(f"  {c(scenario.key, BOLD):<24} Eq. {scenario.equation:<4} {scenario.title}")
        print()
        return

    print(f"Loading scene {args.scene!r} ...")
    scene = Scene.from_config(args.scene)
    resolve_predictions(scene)

    keys = args.scenario or [s.key for s in ALL_SCENARIOS]
    scenarios = [get_scenario(k) for k in keys]

    vis = overlay = None
    if not args.no_viz:
        from faro.viz.connection import print_connection_help
        from faro.viz.constraint_viz import ConstraintOverlay
        from faro.viz.meshcat_ports import kill_stale_servers, preflight_report
        from faro.viz.meshcat_viz import SceneVisualizer

        if args.kill_stale:
            for server in kill_stale_servers():
                print(f"  killed orphaned meshcat server pid={server.pid}")
            time.sleep(1.0)
        warning = preflight_report()
        if warning:
            print("\n" + warning)

        vis = SceneVisualizer(scene)
        vis.display(scene.robot.q_nominal)
        overlay = ConstraintOverlay(vis)

        print()
        print_connection_help(urlparse(vis.url).port or 7000)
        if not args.no_wait:
            try:
                input(c("\n  Open the viewer, then press ENTER to start ...", BOLD))
            except EOFError:
                # Not attached to a terminal (piped, or run from a job script).
                print(c("\n  stdin is not a terminal -- starting immediately.", DIM))

    # Navigate rather than march straight through: a scenario is far more useful
    # watched two or three times against the printed equation than once.
    verdicts: dict[str, bool] = {}
    interactive = stdin_is_interactive() and not args.auto
    index = 0

    while 0 <= index < len(scenarios):
        scenario = scenarios[index]
        print_header(scenario, index + 1, len(scenarios))
        result = run_sweep(scenario, scene)

        if args.no_viz:
            tabulate(result)
        else:
            animate(result, scene, vis, overlay, args)

        verdicts[scenario.key] = report(result)

        if not interactive:
            index += 1
            continue

        print_navigation_hint(index, len(scenarios))
        action = read_navigation(allow_previous=index > 0)
        if action == NEXT:
            index += 1
        elif action == REPEAT:
            print(c("  replaying ...", DIM))
        elif action == PREVIOUS:
            index = max(0, index - 1)
        elif action == QUIT:
            print(c("\n  stopped early.", DIM))
            break

    passed = sum(verdicts.values())
    total = len(verdicts)
    print()
    print(c("=" * 78, DIM))
    colour = GREEN if passed == total and total else RED
    print(c(f"  {passed}/{total} constraints matched their predicted failure point",
            BOLD + ";" + colour))
    if total < len(scenarios):
        print(c(f"  ({len(scenarios) - total} not run)", DIM))
    print(c("=" * 78, DIM))
    print()

    if not args.no_viz:
        print("Viewer stays alive -- press Ctrl-C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nbye.")


if __name__ == "__main__":
    main()
