#!/usr/bin/env python
"""Look at q_init, and at the stance the grasp actually needs.

WHY THIS EXISTS
---------------
Eq. 15 pins `q_0 = q_init` and Eq. 8 then forbids every persisting contact to move.
So `q_init` decides where the feet are FOR THE WHOLE SEQUENCE -- and in FARO it is an
INPUT, the robot's actual configuration at the root of the tree search, not something
the method derives. We currently synthesize it by solving Eq. 14 on `c_0`, which
optimizes toward the nominal pose knowing nothing about the box that must be grasped
later. Measured on `reach`, that costs 298 mm of base travel and 76 deg of joint angle
between the stand pose and a stance that can actually reach -- and with Eq. 8 enforced
the KSO is infeasible as a result.

WHAT YOU ARE LOOKING AT
-----------------------
    1  mode c_0        the stand pose. This is what q_init is currently set from.
    2  edge c_0 u c_1  a configuration that satisfies BOTH modes at once, so it holds
                       the box while standing. Filter E proved this exists.

Watch the FEET between 1 and 2. Eq. 8 says they may not move, so if they are in
different places here, no configuration reachable from q_init can grasp the box --
which is the infeasibility, and it is a q_init problem, not an Eq. 15 problem.

    python scripts/05_qinit_inspect.py
    python scripts/05_qinit_inspect.py --no-viz     # numbers only
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import faro  # noqa: F401  -- pins the BLAS before numpy/pinocchio load
from faro.kso.problem import edge_at, mode_at
from faro.mode_edge.feasibility import FeasibilityCache, check
from faro.scenarios.kso_demos import get_demo
from faro.scene.scene import Scene


def _pose(state: dict, name: str):
    import pinocchio as pin

    q = state[name]
    return pin.SE3(pin.Quaternion(q[6], q[3], q[4], q[5]).toRotationMatrix(), q[:3])


def put_on_screen(viz, scene: Scene, report) -> None:
    """Objects first, then the robot -- `display` takes the ROBOT config alone (36),
    not the stacked scene vector (43)."""
    state = report.configurations
    for name in sorted(scene.objects):
        viz.update_object_pose(name, _pose(state, name))
    viz.display(state["robot"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--scenario", default="reach")
    parser.add_argument("--scene", default="box_placement")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--dwell", type=float, default=2.5)
    args = parser.parse_args()

    scene = Scene.from_config(args.scene)
    sequence = get_demo(args.scenario).sequence(scene)
    cache = FeasibilityCache()

    entries = [
        ("mode c_0  (q_init is set from this)", check(scene, mode_at(sequence, 0), cache=cache)),
        ("edge c_0 u c_1  (a stance that grasps)", check(scene, edge_at(sequence, 1), cache=cache)),
    ]

    print()
    for title, report in entries:
        q = report.configurations["robot"]
        print(f"  {title}")
        print(f"      {report.result.status}")
        print(f"      base xyz {np.round(q[:3], 4)}")

    a = entries[0][1].configurations["robot"]
    b = entries[1][1].configurations["robot"]
    print()
    print(f"  base travel between them : {np.linalg.norm(a[:3] - b[:3]) * 1e3:7.1f} mm")
    print(f"  largest joint difference : {np.degrees(np.max(np.abs(b[7:] - a[7:]))):7.1f} deg")
    print()
    print("  Eq. 8 forbids the persisting contacts (both feet, box bottom) from moving")
    print("  between these. If the feet are not already where the grasp needs them,")
    print("  q_init has decided the sequence is infeasible before Eq. 15 is even asked.")
    print()

    if args.no_viz:
        return 0

    import time

    from faro.viz.meshcat_viz import SceneVisualizer

    viz = SceneVisualizer(scene)
    print(f"  meshcat: {viz.url}\n")
    try:
        while True:
            for title, report in entries:
                print(f"  showing: {title}")
                put_on_screen(viz, scene, report)
                time.sleep(args.dwell)
    except KeyboardInterrupt:
        print("\n  stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
