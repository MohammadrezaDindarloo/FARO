"""The wrench overlay must be drawn where it can actually be seen.

Found by eye, not by a test: every Eq. 7c / 7d scenario was drawing its force arrow,
friction pyramid and CoP rectangle BELOW the floor. Our patch normals are uniformly
outward, so a foot sole's +z points *down*; drawing the wrench in the patch's own
frame sent all of it underground, where it was invisible. The numbers were right the
whole time -- only the picture was wrong, which is worse for a module whose entire
job is building physical intuition.

    pytest tests/test_constraint_viz.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from faro.scenarios.constraint_demos import ALL_SCENARIOS, run_sweep
from faro.scene.scene import Scene
from faro.viz.constraint_viz import FORCE_SCALE, contact_frame


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_config("box_placement")


def test_contact_frame_flips_the_normal_to_point_away_from_the_support(scene):
    """A sole's outward normal points down; the contact normal points up."""
    q = scene.robot.q_nominal
    raw = scene.patch_world_placement(scene.patches["left_foot_sole"], q)
    assert raw.rotation[:, 2] == pytest.approx([0, 0, -1], abs=1e-9)

    frame = contact_frame(raw)
    assert frame.rotation[:, 2] == pytest.approx([0, 0, 1], abs=1e-9)
    # Same contact point -- only the orientation changes.
    assert frame.translation == pytest.approx(raw.translation, abs=1e-12)


def test_contact_frame_is_its_own_inverse(scene):
    """Rx(pi) applied twice is the identity, so the helper cannot drift."""
    raw = scene.patch_world_placement(scene.patches["left_foot_sole"], scene.robot.q_nominal)
    assert contact_frame(contact_frame(raw)).homogeneous == pytest.approx(
        raw.homogeneous, abs=1e-12
    )


@pytest.mark.parametrize(
    "key", [s.key for s in ALL_SCENARIOS if s.equation in ("7c", "7d")]
)
def test_wrench_overlays_are_drawn_above_the_floor(key, scene):
    """The regression itself: nothing may be rendered below the contact plane.

    Reproduces what the runner does -- resolve the anchor patch, map it through
    `contact_frame`, then place the force arrow -- and asserts the arrow tip ends up
    ABOVE the contact point for a positive normal load. Without the flip every one of
    these lands at z = -0.5 m, under the floor.
    """
    scenario = next(s for s in ALL_SCENARIOS if s.key == key)
    step = run_sweep(scenario, scene).steps[0]
    state = step.state
    assert "force" in state and "anchor" in state

    placement = contact_frame(
        scene.patch_world_placement(scene.patches[state["anchor"]], state.get("q"))
    )
    force_world = placement.rotation @ np.asarray(state["force"], dtype=float)
    tip = placement.translation + force_world * FORCE_SCALE

    f_z = float(state["force"][2])
    if f_z > 0:
        assert tip[2] > placement.translation[2], (
            f"{key}: a positive normal load must draw an arrow pointing UP; "
            f"tip is at z={tip[2]:.3f} vs contact z={placement.translation[2]:.3f}"
        )
        assert tip[2] > 0.0, f"{key}: arrow tip is below the floor plane"


def test_friction_pyramid_opens_upward_from_the_contact(scene):
    """Eq. 7c's pyramid has its apex at the contact and opens along +f_z.

    In the patch's raw frame that cone of admissible forces would open downward,
    which is not what Eq. 7c describes.
    """
    step = run_sweep(next(s for s in ALL_SCENARIOS if s.key == "7c-slip"), scene).steps[0]
    placement = contact_frame(
        scene.patch_world_placement(scene.patches[step.state["anchor"]], step.state["q"])
    )
    f_z = float(step.state["force"][2])
    apex_to_rim = placement.rotation @ np.array([0.0, 0.0, f_z * FORCE_SCALE])
    assert apex_to_rim[2] > 0.0, "the friction pyramid opens downward"


def test_the_foot_does_not_move_during_a_wrench_sweep(scene):
    """7c / 7d sweep the WRENCH, not the configuration -- so nothing should move.

    Worth asserting rather than assuming: it is the reason "slip" here shows up as
    the force arrow leaving the pyramid rather than the foot sliding. Milestone 2 has
    no integrator; turning a friction violation into motion needs Eq. 17.
    """
    for scenario in (s for s in ALL_SCENARIOS if s.equation in ("7c", "7d")):
        steps = run_sweep(scenario, scene).steps
        first, last = steps[0].state["q"], steps[-1].state["q"]
        assert np.allclose(first, last), f"{scenario.key}: q changed during a wrench sweep"


def test_overlay_materials_are_drawn_on_top_of_the_robot(scene):
    """Eq. 7d's CoP rectangle lies exactly in the sole plane.

    With normal depth testing it renders INSIDE the G1's shoe mesh and vanishes,
    leaving the CoP dot with no visible bound to be inside or outside of -- which is
    the whole content of the constraint. Every overlay material must therefore opt
    out of depth testing.
    """
    import meshcat.geometry as g

    from faro.viz.constraint_viz import _ON_TOP

    lowered = g.LineBasicMaterial(color=0xFF88DD, **_ON_TOP).lower({})
    assert lowered["depthTest"] is False, "meshcat dropped the depthTest kwarg"
    assert lowered["depthWrite"] is False

    source = (Path(__file__).resolve().parents[1] / "faro/viz/constraint_viz.py").read_text()
    material_lines = [
        line for line in source.splitlines()
        if "Material(" in line and "_ON_TOP" not in line and "def " not in line
    ]
    # `set_patch_status` colours real scene geometry, so it legitimately depth-tests.
    assert all("COLOR_OK" in l or "color=color" in l for l in material_lines), (
        f"overlay material without _ON_TOP: {material_lines}"
    )


# =============================================================================
# Eq. 7d's torsional row is the one with no other visual consequence
# =============================================================================
def test_torsion_changes_nothing_else_in_the_scene(scene):
    """Why `7d-torsion` needs its own indicator at all.

    A moment about the contact normal leaves the force untouched (so the arrow and
    friction pyramid are static) and does not appear in the CoP relations
    `x_cop = kappa_y/f_z`, `y_cop = -kappa_x/f_z` (so the dot does not move). Without
    a dedicated overlay the whole sweep is visually frozen and only the patch colour
    flips -- which is exactly what a user reported.
    """
    steps = run_sweep(next(s for s in ALL_SCENARIOS if s.key == "7d-torsion"), scene).steps

    forces = {tuple(s.state["force"]) for s in steps}
    assert len(forces) == 1, "the force must not change during a pure torsion sweep"

    cops = {
        (s.state["moment"][1] / s.state["force"][2], -s.state["moment"][0] / s.state["force"][2])
        for s in steps
    }
    assert len(cops) == 1 and cops == {(0.0, -0.0)}, "the CoP must not move"

    # ... and kappa_z really does vary, or the scenario tests nothing.
    assert len({s.state["moment"][2] for s in steps}) == len(steps)


def test_only_the_torsion_scenario_asks_for_the_rings(scene):
    """`mu_r` in the state is what switches the indicator on.

    The other wrench scenarios hold kappa_z at zero, so drawing a capacity ring on
    them would be clutter with nothing to compare against.
    """
    wrench = [s for s in ALL_SCENARIOS if s.equation in ("7c", "7d")]
    with_rings = [s.key for s in wrench if "mu_r" in run_sweep(s, scene).steps[0].state]
    assert with_rings == ["7d-torsion"]


def test_torsion_arc_crosses_the_limit_ring_exactly_at_the_violation(scene):
    """The rings must be readable the same way as the arrow against its pyramid.

    Radius encodes magnitude for both, so "yellow arc outside blue ring" has to
    coincide with the constraint row going positive -- otherwise the picture and the
    algebra would disagree.
    """
    from faro.viz.constraint_viz import MOMENT_SCALE

    scenario = next(s for s in ALL_SCENARIOS if s.key == "7d-torsion")
    for step in run_sweep(scenario, scene).steps:
        state = step.state
        arc = abs(state["moment"][2]) * MOMENT_SCALE
        ring = state["mu_r"] * state["force"][2] * MOMENT_SCALE
        assert (arc > ring + 1e-12) == step.violated, (
            f"kappa_z={state['moment'][2]}: arc {arc:.4f} vs ring {ring:.4f} "
            f"disagrees with violated={step.violated}"
        )


def test_torsion_arc_direction_follows_the_sign_of_the_moment():
    """A negative kappa_z must sweep the other way, or the drilling direction is lost."""
    from faro.viz.constraint_viz import _circle

    positive = _circle(0.1, np.deg2rad(300.0))
    negative = _circle(0.1, -np.deg2rad(300.0))
    assert positive[2][1] > 0 and negative[2][1] < 0, "the arc does not reverse with the sign"
    assert positive.shape == negative.shape


def test_circle_helper_produces_valid_line_segment_pairs():
    """LineSegments consumes vertices in consecutive PAIRS; an odd count would drop one."""
    from faro.viz.constraint_viz import _circle

    for span in (2 * np.pi, np.deg2rad(300.0), -np.deg2rad(300.0)):
        points = _circle(0.1, span, segments=12)
        assert len(points) % 2 == 0
        assert np.allclose(np.linalg.norm(points[:, :2], axis=1), 0.1)
        assert np.allclose(points[:, 2], 0.0), "the arc must lie in the contact plane"


# =============================================================================
# Every scenario must actually SHOW something
# =============================================================================
def _visual_signature(state, violated):
    """What a viewer would draw for one step, reduced to a comparable value.

    `violated` counts ONLY when there are patches to colour. Otherwise a scenario
    with `patches=()` looks like it changes -- the flag flips -- while absolutely
    nothing on screen does. That loophole let `13b-speed` through.
    """
    def _declared(key):
        value = state.get(key)
        if value is None:
            return False
        return value.size > 0 if isinstance(value, np.ndarray) else bool(value)

    indicators = ("patches", "highlight_objects", "status_at")
    parts = [violated if any(_declared(k) for k in indicators) else None]
    if "probe" in state:
        parts.append(tuple(np.round(state["probe"]["pose"].homogeneous.ravel(), 9)))
    if "q" in state:
        parts.append(tuple(np.round(np.asarray(state["q"], float), 9)))
    if "box_pose" in state:
        parts.append(tuple(np.round(state["box_pose"].homogeneous.ravel(), 9)))
    for key in ("force", "moment"):
        if key in state:
            parts.append(tuple(np.round(np.asarray(state[key], float), 9)))
    for spec in state.get("vectors", ()):
        parts.append((spec.get("path"),
                      tuple(np.round(np.asarray(spec["origin"], float), 9)),
                      tuple(np.round(np.asarray(spec["vector"], float), 9))))
    return tuple(parts)


@pytest.mark.parametrize("key", [s.key for s in ALL_SCENARIOS], ids=lambda k: k)
def test_every_scenario_changes_something_visible(key, scene):
    """A scenario whose picture never moves teaches nothing, however correct its numbers.

    This has now caught the same class of bug twice: `7d-torsion` (a moment about the
    normal changes neither the force nor the CoP, so only the patch colour flipped)
    and the Eq. 10/11 dynamics scenarios (which had no `q`, a fixed box pose and no
    patches -- literally nothing on screen changed, and the robot kept whatever pose
    the PREVIOUS scenario had left it in).
    """
    from faro.scenarios.constraint_demos import resolve_predictions

    resolve_predictions(scene)
    scenario = next(s for s in ALL_SCENARIOS if s.key == key)
    steps = run_sweep(scenario, scene).steps

    signatures = {_visual_signature(s.state, s.violated) for s in steps}
    assert len(signatures) > 1, (
        f"{key}: nothing a viewer draws changes across the sweep -- the scenario is "
        f"visually static and cannot demonstrate anything"
    )


def test_11_spin_actually_tumbles_the_body_at_the_swept_rate(scene):
    """Same rule as the knee: a scenario about an angular rate must contain one.

    `11-spin` used to hold the body at a fixed pose and grow two arrows, which is what
    made it unreadable -- there was no way to tell the gyroscopic moment apart from an
    arbitrary caption. The body now turns, and the rotation BETWEEN frames is measured
    on the manifold (`log3(R_i^T R_i+1)`, not a raw angle difference, which would wrap).

    Also checks the body is anisotropic: a cube has `omega x (I omega) = 0` for every
    omega, so this scenario is vacuous unless the probe is genuinely elongated.
    """
    import pinocchio as pin

    from faro.scenarios.constraint_demos import _FRAME_DT, _SPIN_INERTIA, _SPIN_SIZE

    steps = run_sweep(next(s for s in ALL_SCENARIOS if s.key == "11-spin"), scene).steps
    rotations = [step.state["probe"]["pose"].rotation for step in steps]
    rates = [step.param for step in steps]

    swept = [float(np.linalg.norm(pin.log3(rotations[i].T @ rotations[i + 1])))
             for i in range(len(rotations) - 1)]
    expected = [_FRAME_DT * 0.5 * (rates[i] + rates[i + 1]) for i in range(len(rates) - 1)]
    np.testing.assert_allclose(swept, expected, atol=1e-9)
    assert sum(swept) > 2 * np.pi, "the body should complete at least one full turn"

    assert len(set(np.round(_SPIN_SIZE, 6))) > 1, "an isotropic probe cannot show Eq. 11b's term"
    moments = np.diag(_SPIN_INERTIA)
    assert len(set(np.round(moments, 9))) > 1
    # And it must be a body that could exist: the triangle inequality is not optional.
    for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        assert moments[i] + moments[j] >= moments[k], (
            f"principal moments {moments} violate the triangle inequality -- no rigid "
            f"body has this inertia"
        )


def test_the_spin_probe_does_not_intersect_any_scene_object(scene):
    """`11-spin` draws its own body, so it must not be drawn inside the scene's.

    Reported from the viewer: the grey slab appeared to pass through the yellow box.
    It only did so when the scenarios were run in SEQUENCE -- `11-hold` parks the box
    in mid-air at z = 0.45 and `11-spin` inherited that, because most scenarios never
    mention the box and `update_object_pose` mutates the scene. Running `-s 11-spin`
    alone looked fine, which is the signature of leaked state rather than bad geometry.

    Checked against EVERY pose any scenario puts the box in, not just its home pose.
    Clearing only the home pose would have passed even at the original z = 0.60, since
    the collision was with the box where `11-hold` leaves it -- so that weaker check
    would have gone green on the exact bug it was written for.
    """
    from faro.scenarios.constraint_demos import (
        _SPIN_POSE, _SPIN_SIZE, resolve_predictions,
    )

    resolve_predictions(scene)      # several sweeps have no `values` until this runs
    box = scene.objects["box"]
    poses = [np.asarray(box.initial_pose.translation, float)]
    for scenario in ALL_SCENARIOS:
        for step in run_sweep(scenario, scene).steps:
            if "box_pose" in step.state:
                poses.append(np.asarray(step.state["box_pose"].translation, float))

    probe_c, probe_h = np.asarray(_SPIN_POSE, float), np.asarray(_SPIN_SIZE, float) / 2.0
    half = np.asarray(box.size, float) / 2.0
    for centre in poses:
        overlap = np.minimum(centre + half, probe_c + probe_h) - np.maximum(centre - half, probe_c - probe_h)
        assert not np.all(overlap > 0), (
            f"the 11-spin probe intersects the box at {np.round(centre, 3)}; "
            f"raise _SPIN_POSE clear of every pose a scenario uses"
        )


def test_object_poses_do_not_leak_between_scenarios(scene):
    """A scenario's picture must not depend on which scenario ran before it.

    `update_object_pose` mutates `scene.objects[...].initial_pose`, and 18 of the
    scenarios never declare a `box_pose` at all -- so before the fix the box simply
    stayed wherever the last one left it. `reset_object_poses` restores the scene's
    own poses, and `ConstraintOverlay.clear()` calls it after every sweep.
    """
    from faro.viz.meshcat_viz import SceneVisualizer

    vis = SceneVisualizer.__new__(SceneVisualizer)   # no meshcat server needed
    vis.scene = scene
    vis._home_poses = {n: o.initial_pose.copy() for n, o in scene.objects.items()}
    moved = []
    vis.update_object_pose = lambda name, pose: (
        moved.append(name), setattr(scene.objects[name], "initial_pose", pose))[0]

    home = scene.objects["box"].initial_pose.copy()
    scene.objects["box"].initial_pose = pin.SE3(np.eye(3), np.array([9.0, 9.0, 9.0]))

    # `clear()` is what runs between scenarios, so it -- not just `reset_object_poses`
    # -- has to do the restoring. Driving the reset directly would pass even with the
    # call missing from `clear()`, which is exactly how a first version of this test
    # went green on the bug it was written for.
    from faro.viz.constraint_viz import ConstraintOverlay

    overlay = ConstraintOverlay.__new__(ConstraintOverlay)
    overlay.vis = vis
    overlay._active = set()
    vis._draw_patches = lambda: None
    vis._draw_objects = lambda: None
    overlay.clear()

    assert moved == ["box"], "clear() must restore every object it knows about"
    np.testing.assert_allclose(
        scene.objects["box"].initial_pose.homogeneous, home.homogeneous, atol=1e-12,
        err_msg="the box was not returned to its scene pose",
    )


@pytest.mark.parametrize("key", [s.key for s in ALL_SCENARIOS], ids=lambda k: k)
def test_every_violating_scenario_shows_the_violation_on_screen(scene, key):
    """A constraint that fails must turn something red. Eight of them did not.

    `set_patch_status` is the usual signal, but Eqs. 9, 11 and 13 constrain an OBJECT
    or a JOINT and have no patch, so the failure existed only as a line of terminal
    text: the box drove clean through the platform and stayed its normal colour, and
    the knee blew past its speed limit with nothing on screen reacting. Every such
    scenario now declares `highlight_objects` (tint a body) or `status_at` (a bead at
    the joint), and this test requires one whenever the sweep can fail at all.
    """
    from faro.scenarios.constraint_demos import resolve_predictions

    resolve_predictions(scene)
    scenario = next(s for s in ALL_SCENARIOS if s.key == key)
    result = run_sweep(scenario, scene)
    if not result.ever_violated:
        pytest.skip(f"{key} never violates, so there is nothing to signal")

    state = next(step.state for step in result.steps if step.violated)
    assert state.get("patches") or state.get("highlight_objects") or "status_at" in state, (
        f"{key} violates but declares no visual indicator: a viewer sees the constraint "
        f"fail with nothing changing colour"
    )


@pytest.mark.parametrize("key", [s.key for s in ALL_SCENARIOS], ids=lambda k: k)
def test_every_scenario_places_the_robot(key, scene):
    """Each scenario must set `q`, even the ones that are not about the robot.

    Without it the viewer never calls `display`, so the robot keeps the pose left by
    whichever scenario ran before -- which is how `11-hold` came to be shown with the
    robot sunk into the floor from a previous sweep.
    """
    from faro.scenarios.constraint_demos import resolve_predictions

    resolve_predictions(scene)
    scenario = next(s for s in ALL_SCENARIOS if s.key == key)
    state = run_sweep(scenario, scene).steps[0].state
    assert "q" in state, f"{key}: no q, so the robot keeps the previous scenario's pose"
    assert len(state["q"]) == scene.robot.nq


def test_dynamics_scenarios_draw_both_sides_of_the_balance(scene):
    """Eqs. 10b and 11b are balances; showing only one side would be half a picture."""
    from faro.scenarios.constraint_demos import resolve_predictions

    resolve_predictions(scene)
    for key in ("10-weight", "10-moment", "11-hold"):
        scenario = next(s for s in ALL_SCENARIOS if s.key == key)
        vectors = run_sweep(scenario, scene).steps[-1].state["vectors"]
        up = [v for v in vectors if np.asarray(v["vector"], float)[2] > 1e-9]
        down = [v for v in vectors if np.asarray(v["vector"], float)[2] < -1e-9]
        assert up and down, f"{key}: only one side of the balance is drawn"


def test_drawn_arrows_stay_a_readable_length(scene):
    """Per-arrow `scale` exists because the magnitudes differ by 20x.

    The robot's weight is 324 N and the box's is 19.6 N; one metres-per-newton for
    both leaves half of them invisible or off-screen.
    """
    from faro.scenarios.constraint_demos import resolve_predictions
    from faro.viz.constraint_viz import FORCE_SCALE

    resolve_predictions(scene)
    for scenario in ALL_SCENARIOS:
        for step in run_sweep(scenario, scene).steps[::20]:
            for spec in step.state.get("vectors", ()):
                length = np.linalg.norm(spec["vector"]) * spec.get("scale", FORCE_SCALE)
                assert length < 1.2, (
                    f"{scenario.key}/{spec['path']}: {length:.2f} m arrow dwarfs the scene"
                )


# =============================================================================
# Patches must stay attached to the robot they belong to
# =============================================================================
def _placement_logic(scene, last_q):
    """Exercise `_initial_world_placement` without starting a Meshcat server.

    `__new__` skips `__init__` deliberately: the method under test depends on
    exactly two attributes, and constructing the real visualizer would open a
    browser-facing server just to check a transform.
    """
    from faro.viz.meshcat_viz import SceneVisualizer

    vis = SceneVisualizer.__new__(SceneVisualizer)
    vis.scene = scene
    vis.show_patches = True
    vis._last_q = last_q
    return vis


def test_redrawn_robot_patches_follow_the_pose_on_screen(scene):
    """The bug: green patches detached from the robot and floated underground.

    `overlay.clear()` redraws every patch, and that redraw used `q_nominal` --  whose
    pelvis sits at z = 0, so its feet hang 0.7779 m BELOW the floor. Only
    `standing_configuration` drops the robot onto the ground. Between one scenario
    ending and the next one's first `display()`, every robot patch was drawn three
    quarters of a metre under the floor, which is exactly the interval the runner
    spends waiting for a keypress.
    """
    from faro.scenarios.constraint_demos import standing_configuration

    standing = standing_configuration(scene)
    vis = _placement_logic(scene, standing)

    for name in ("left_foot_sole", "right_foot_sole", "left_hand_patch"):
        patch = scene.patches[name]
        drawn = vis._initial_world_placement(patch).translation
        actual = scene.patch_world_placement(patch, standing).translation
        assert drawn == pytest.approx(actual, abs=1e-12), f"{name} is detached"

    assert vis._initial_world_placement(scene.patches["left_foot_sole"]).translation[2] \
        == pytest.approx(0.0, abs=1e-12), "the sole should sit ON the floor"


def test_placement_falls_back_to_nominal_before_anything_is_displayed(scene):
    """With no pose shown yet there is nothing better to use, and it must not crash."""
    vis = _placement_logic(scene, None)
    patch = scene.patches["left_foot_sole"]
    assert vis._initial_world_placement(patch).translation == pytest.approx(
        scene.patch_world_placement(patch, scene.robot.q_nominal).translation, abs=1e-12
    )


def test_environment_patches_ignore_the_robot_pose(scene):
    """Floor and platform are world-fixed; a robot pose must not move them."""
    from faro.scenarios.constraint_demos import standing_configuration

    for last_q in (None, standing_configuration(scene)):
        vis = _placement_logic(scene, last_q)
        floor = vis._initial_world_placement(scene.patches["floor"])
        assert floor.translation == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_joint_scenarios_draw_demand_against_capacity(scene):
    """Eqs. 13b/13c live at a JOINT and have no contact patch, so they need arrows.

    Both follow the same applied-vs-capacity reading as Eq. 7c's arrow and pyramid:
    for 13b the swept rate grows past a fixed limit; for 13c the demand is fixed and
    the AVAILABLE torque shrinks, which is what makes it an envelope rather than a box.
    """
    from faro.scenarios.constraint_demos import resolve_predictions

    resolve_predictions(scene)
    for key in ("13b-speed", "13c-envelope"):
        scenario = next(s for s in ALL_SCENARIOS if s.key == key)
        steps = run_sweep(scenario, scene).steps
        for step in (steps[0], steps[-1]):
            paths = {v["path"] for v in step.state["vectors"]}
            assert len(paths) == 2, f"{key}: needs both a demand and a capacity arrow"

        applied = [np.linalg.norm(v["vector"]) for v in steps[0].state["vectors"]]
        final = [np.linalg.norm(v["vector"]) for v in steps[-1].state["vectors"]]
        assert applied != final, f"{key}: neither arrow changes across the sweep"

    # 13c specifically: capacity must SHRINK, which a box constraint would not do.
    envelope = next(s for s in ALL_SCENARIOS if s.key == "13c-envelope")
    steps = run_sweep(envelope, scene).steps
    cap = lambda st: next(np.linalg.norm(v["vector"]) for v in st.state["vectors"]
                          if "available" in v["path"])
    assert cap(steps[-1]) < cap(steps[0]), "the available torque must fall with speed"


def test_velocity_scenarios_actually_move_the_joint(scene):
    """A velocity demo must contain a velocity: finite-difference the DISPLAYED pose.

    This is the guard for a bug that shipped: 13b-speed held the knee at a constant
    0.42 rad for all 121 frames and merely drew an arrow whose LENGTH was the swept
    speed. Every assertion above still passed -- the arrows changed, the margin crossed
    at the right place -- because none of them ever looked at the robot. A wrong joint
    index in Eq. 13b would have been invisible.

    So this reads the angles the viewer is actually shown, differences consecutive
    frames, and demands the result reproduce the swept rate. The sweep is a linear ramp
    integrated with the trapezoid rule, which is exact for a ramp, so the agreement is
    to floating point rather than approximate.
    """
    from faro.scenarios.constraint_demos import (
        _FRAME_DT, _SLOWDOWN, _fold_into, resolve_predictions,
    )

    resolve_predictions(scene)
    model = scene.robot.model
    idx_q = model.joints[model.getJointId("left_knee_joint")].idx_q
    lo, hi = (float(limit[idx_q]) for limit in scene.robot.joint_limits())

    for key in ("13b-speed", "13c-envelope"):
        scenario = next(s for s in ALL_SCENARIOS if s.key == key)
        steps = run_sweep(scenario, scene).steps
        angles = np.array([step.state["q"][idx_q] for step in steps])
        speeds = np.array([step.param for step in steps])

        assert angles.std() > 0.1, f"{key}: the knee never moves -- there is no velocity here"

        # Per-frame travel, by this test's own quadrature over the swept rates -- the
        # trapezoid rule, exact for a linear ramp. Nothing here calls the implementation,
        # so agreement is evidence rather than a tautology. `_SLOWDOWN` is playback only:
        # it divides every frame equally, so the motion stays exactly proportional to the
        # swept rate and the crossing is untouched.
        expected = (_FRAME_DT / _SLOWDOWN) * 0.5 * (speeds[:-1] + speeds[1:])

        # Frames that straddle a turning point travel out and back, so their endpoints
        # cannot recover the distance. Locate those turns from the cumulative travel
        # instead of inferring them from the very differences under test.
        cumulative = np.concatenate([[0.0], np.cumsum(expected)])
        leg = np.floor((angles[0] - lo + cumulative) / (hi - lo)).astype(int)
        clean = leg[:-1] == leg[1:]
        assert clean.sum() > 0.7 * len(clean), f"{key}: too few frames to check"

        np.testing.assert_allclose(
            np.abs(np.diff(angles))[clean], expected[clean], atol=1e-9,
            err_msg=f"{key}: the knee's on-screen motion is not the swept velocity",
        )

        # And the joint stays inside its own position limits the whole time, so 13a is
        # never the thing that bites -- otherwise the two limits would be confounded.
        assert angles.min() >= lo - 1e-9 and angles.max() <= hi + 1e-9, \
            f"{key}: left its 13a range, so a 13b violation is no longer isolated"

    assert _fold_into(lo - 0.25, lo, hi) == pytest.approx(lo + 0.25), "fold must reflect"
    assert _fold_into(hi + 0.25, lo, hi) == pytest.approx(hi - 0.25), "fold must reflect"
