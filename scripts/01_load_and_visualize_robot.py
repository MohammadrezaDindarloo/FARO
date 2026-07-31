#!/usr/bin/env python
"""Milestone 1 -- load the robot + scene and visualize it in Meshcat.

Paper grounding: this builds the scene of Fig. 2 and Table IV -- the Unitree G1,
a movable box, the floor and the goal platform, and the five contact interfaces
with their allowed contacts. No optimization yet; this is the substrate that
Eqs. 7-13 (Milestone 2) will be written against.

Run (see SETUP.md section 4 for the browser part):

    conda activate faro
    python scripts/01_load_and_visualize_robot.py                 # static pose
    python scripts/01_load_and_visualize_robot.py --mode sweep    # animate joints
    python scripts/01_load_and_visualize_robot.py --mode interactive

Then forward port 7000 in VS Code and open http://127.0.0.1:7000/static/
"""

from __future__ import annotations

import argparse
import time
from urllib.parse import urlparse

import numpy as np
import pinocchio as pin

from faro.scene.scene import Scene
from faro.viz.connection import open_browser_if_local, print_connection_help
from faro.viz.meshcat_ports import kill_stale_servers, preflight_report
from faro.viz.meshcat_viz import SceneVisualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="box_placement", help="config name under configs/scenes/")
    p.add_argument(
        "--mode",
        default="static",
        choices=["static", "sweep", "interactive"],
        help="static: hold the nominal pose | sweep: animate joints | interactive: type joint values",
    )
    p.add_argument("--no-patches", action="store_true", help="hide contact-patch overlays")
    p.add_argument(
        "--kill-stale",
        action="store_true",
        help="terminate ORPHANED meshcat servers first, so this run can claim port 7000",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="open the viewer in a browser automatically (only works when running locally)",
    )
    p.add_argument("--sweep-joints", nargs="*", default=None, help="joints to animate in sweep mode")
    p.add_argument("--amplitude", type=float, default=0.5, help="sweep amplitude [rad]")
    p.add_argument("--period", type=float, default=3.0, help="sweep period [s]")
    return p.parse_args()


def standing_configuration(scene: Scene) -> np.ndarray:
    """Nominal posture with the base dropped so the foot soles rest on the floor.

    Purely kinematic -- it places the robot, it does not check static balance.
    Balance is a dynamics question and only enters at TO (Eq. 17, Milestone 5).
    """
    from faro.core.patches import Attachment

    robot = scene.robot
    q = robot.q_nominal.copy()

    # Which robot patches are the ground-contact ones? Read it off the scene's
    # interface graph (Table IV) rather than pattern-matching on names, so this
    # keeps working after a robot or scene swap.
    ground_patches = [
        iface.patch
        for iface in scene.interfaces.values()
        if iface.patch.attachment is Attachment.ROBOT and "floor" in iface.allowed
    ]
    if not ground_patches:
        raise RuntimeError(f"scene {scene.name!r} has no robot patch allowed to touch the floor")

    # Lower the base until the lowest such patch reaches z = 0. Using the PATCH
    # placement rather than the ankle frame is what lands the feet exactly on the
    # floor plane, since the G1's sole sits 0.035 m below its ankle frame.
    lowest = min(scene.patch_world_placement(p, q).translation[2] for p in ground_patches)
    q[2] += -lowest
    return q


def robot_attachment():
    from faro.core.patches import Attachment

    return Attachment.ROBOT


def report_scene(scene: Scene, q: np.ndarray) -> None:
    """Print what was loaded, plus the numbers worth eyeballing."""
    print(scene.summary())
    print()

    robot = scene.robot
    print("Key frame placements at the standing pose (world):")
    for frame in ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link",
                  "left_rubber_hand", "right_rubber_hand"]:
        t = robot.frame_placement(q, frame).translation
        print(f"  {frame:<24} {np.array2string(t, precision=4, suppress_small=True)}")

    print("\nContact patch placements at the standing pose (world position, normal):")
    for patch in scene.patches.values():
        qq = q if patch.attachment is robot_attachment() else None
        placement = scene.patch_world_placement(patch, qq)
        normal = placement.rotation[:, 2]
        print(
            f"  {patch.name:<14} p={np.array2string(placement.translation, precision=3, suppress_small=True):<24}"
            f" n={np.array2string(normal, precision=2, suppress_small=True)}"
            f"  h={patch.half_extents}"
        )

    # The paper reports a maximum branching factor of 108 for this interface set.
    print(f"\nRaw branching factor: {scene.branching_factor()}  (paper Section IV-B reports 108)")


def run_sweep(vis: SceneVisualizer, scene: Scene, q0: np.ndarray, args: argparse.Namespace) -> None:
    """Animate a few joints so the kinematics are visibly alive."""
    robot = scene.robot
    joints = args.sweep_joints or [
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "waist_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
    ]
    indices = []
    for name in joints:
        if not robot.model.existJointName(name):
            print(f"  ! skipping unknown joint {name!r}")
            continue
        indices.append((name, robot.model.joints[robot.model.getJointId(name)].idx_q))

    print(f"\nSweeping {len(indices)} joints (Ctrl-C to stop):")
    for name, _ in indices:
        print(f"  - {name}")

    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            q = q0.copy()
            for k, (_, idx) in enumerate(indices):
                phase = 2.0 * np.pi * (t / args.period) + k * (np.pi / 4.0)
                q[idx] = q0[idx] + args.amplitude * np.sin(phase)
            vis.display(q)
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("\nstopped.")


def run_interactive(vis: SceneVisualizer, scene: Scene, q0: np.ndarray) -> None:
    """A tiny REPL for setting joint angles.

    Chosen over GUI sliders because it works unchanged over SSH + port forwarding,
    which browser-side widgets do not reliably do on a cluster.
    """
    robot = scene.robot
    q = q0.copy()
    vis.display(q)

    print(
        "\nInteractive mode. Commands:\n"
        "  <joint_name> <value>   set a joint angle [rad]\n"
        "  list [filter]          list joint names\n"
        "  reset                  back to the standing pose\n"
        "  quit\n"
    )
    while True:
        try:
            line = input("faro> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        parts = line.split()
        cmd = parts[0]

        if cmd in {"quit", "exit", "q"}:
            return
        if cmd == "reset":
            q = q0.copy()
            vis.display(q)
            print("  reset to standing pose")
            continue
        if cmd == "list":
            needle = parts[1] if len(parts) > 1 else ""
            for name in robot.actuated_joint_names:
                if needle in name:
                    idx = robot.model.joints[robot.model.getJointId(name)].idx_q
                    print(f"  {name:<32} = {q[idx]:+.3f}")
            continue

        if len(parts) != 2:
            print("  ? expected: <joint_name> <value>")
            continue
        name, value = parts
        if not robot.model.existJointName(name):
            print(f"  ? unknown joint {name!r} (try: list {name.split('_')[0]})")
            continue
        try:
            value = float(value)
        except ValueError:
            print(f"  ? {value!r} is not a number")
            continue

        idx = robot.model.joints[robot.model.getJointId(name)].idx_q
        lower, upper = robot.joint_limits()
        if not (lower[idx] <= value <= upper[idx]):
            # Eq. 12 is a hard constraint later, so flag violations now rather than
            # letting an out-of-range pose look acceptable in the viewer.
            print(f"  ! {value:+.3f} is outside joint limits [{lower[idx]:+.3f}, {upper[idx]:+.3f}] (Eq. 12)")
        q[idx] = value
        vis.display(q)
        print(f"  {name} = {value:+.3f}")


def main() -> None:
    args = parse_args()

    print(f"Loading scene {args.scene!r} ...")
    scene = Scene.from_config(args.scene)
    q = standing_configuration(scene)
    report_scene(scene, q)

    if args.kill_stale:
        killed = kill_stale_servers()
        if killed:
            for server in killed:
                print(f"  killed orphaned meshcat server pid={server.pid} (http port {server.http_port})")
            time.sleep(1.0)  # let the OS release the socket before meshcat scans
        else:
            print("  no orphaned meshcat servers found")

    # Meshcat scans upward from 7000 and cannot be told which port to use, so a
    # leftover server silently pushes this run onto 7001+. Warn BEFORE starting,
    # while the message is still near the top of the output.
    warning = preflight_report()
    if warning:
        print("\n" + warning)

    print("\nStarting Meshcat ...")
    vis = SceneVisualizer(scene, show_patches=not args.no_patches)
    vis.display(q)

    # Report the ACTUAL port -- telling you to forward 7000 when the viewer is on
    # 7001 sends you to a different (empty) scene, which looks like a broken script.
    port = urlparse(vis.url).port or 7000

    # Instructions are generated, not hard-coded: the Slurm node changes on every
    # allocation, and the right answer differs for local / SSH / VS Code Tunnels.
    print()
    session = print_connection_help(port)

    if port != 7000:
        print(f"\n  NOTE: port 7000 was taken, so this viewer is on {port}.")
        print("        Re-run with --kill-stale to clear orphaned viewers and get 7000 back.")

    if args.open and open_browser_if_local(f"http://127.0.0.1:{port}/static/", session):
        print("\n  Opened your browser automatically.")
    elif args.open and not session.browser_is_local:
        print("\n  --open ignored: this is a remote session, so there is no local browser.")

    print("  (SETUP.md section 4 has more detail.)\n")

    if args.mode == "sweep":
        run_sweep(vis, scene, q, args)
    elif args.mode == "interactive":
        run_interactive(vis, scene, q)
    else:
        print("Static mode -- press Ctrl-C to exit (the viewer dies with this process).")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nbye.")


if __name__ == "__main__":
    main()
