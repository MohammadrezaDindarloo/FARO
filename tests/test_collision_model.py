"""Milestone 3, part 3: Eq. 9 inside Eq. 14 -- the bodies, the pairs, and the loop.

Eq. 9 is the constraint most able to be wrong without looking wrong. A collision row
that is silently constant, attached to the wrong body, or built from a stale witness
still evaluates to a number, still has a gradient, and still lets the solver report
`Solve_Succeeded`. The failure shows up as a robot standing inside the platform.

So these tests measure the geometry rather than the residuals wherever they can.

    pytest tests/test_collision_model.py -v
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from faro.mode_edge.feasibility import check
from faro.mode_edge.problem import build_problem
from faro.scenarios.mode_demos import STAND
from faro.scene.collision import SceneCollisionModel, environment_bodies, robot_bodies
from faro.scene.scene import Scene
from faro.scene.symbolic import SymbolicScene


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_config("box_placement")


@pytest.fixture(scope="module")
def model(scene) -> SceneCollisionModel:
    return SceneCollisionModel.cached(scene)


# =============================================================================
# "All bodies ... are modeled using convex geometries"  -- Section II-C 2
# =============================================================================
def test_every_body_is_convex(scene, model):
    """The G1 ships 24 triangle MESHES, and Schulman's approximation needs convex sets.

    coal will answer a distance query against a BVH quite happily, so nothing crashes
    if this is skipped -- the witness pair simply jumps between concave features as the
    solve moves, and the frozen model then describes a different pocket of geometry
    than the one the solver is walking into. That is a wrong constraint that looks like
    a working one.
    """
    convex_types = {"Convex", "Box", "Sphere", "Cylinder", "Capsule", "Halfspace", "Plane"}
    for body in model.bodies:
        assert type(body.geometry).__name__ in convex_types, (
            f"{body.name} is a {type(body.geometry).__name__}, which is not convex"
        )


def test_the_robot_actually_had_non_convex_geometry_to_begin_with(scene):
    """Guards the test above from becoming vacuous.

    If a future URDF or Pinocchio version shipped convex collision geometry already,
    `test_every_body_is_convex` would pass without the hull step doing anything, and
    removing that step would go unnoticed. This asserts the raw model is NOT convex,
    so the two tests together say the conversion happened.
    """
    raw = [type(g.geometry).__name__ for g in scene.robot.collision_model.geometryObjects]
    assert "BVHModelOBBRSS" in raw, "raw robot geometry is already convex; the hull step is now a no-op"


def test_environment_bodies_are_solid_slabs_not_zero_thickness_patches(scene):
    """A patch is a rectangle with no interior, and GJK cannot use one.

    Against a zero-volume set the signed distance flips sign as a body crosses the
    plane with nothing in between, so `0 <= sd` would be satisfied on BOTH sides and a
    foot could pass straight through the floor. The slab is what gives "below the
    floor" a volume to be inside of.
    """
    for body in environment_bodies(scene):
        size = np.asarray(body.geometry.halfSide) * 2.0
        assert size.min() > 1e-3, f"{body.name} is effectively flat: {size}"

    floor = next(b for b in environment_bodies(scene) if b.name.endswith("floor"))
    top = floor.placement.translation[2] + float(floor.geometry.halfSide[2])
    assert top == pytest.approx(0.0, abs=1e-9), "the floor slab's TOP face must be the patch plane"


# =============================================================================
# The pair set (docs/ambiguities.md #19)
# =============================================================================
def test_the_pair_set_covers_robot_object_and_environment(scene, model):
    """"All bodies in the robot and scene" -- not just the robot against itself."""
    kinds = {b.kind for b in model.bodies}
    assert kinds == {"robot", "object", "environment"}

    seen = {tuple(sorted((model.bodies[i].kind, model.bodies[j].kind))) for i, j in model.pairs}
    for expected in [("robot", "robot"), ("object", "robot"),
                     ("environment", "robot"), ("environment", "object")]:
        assert tuple(sorted(expected)) in seen, f"no {expected} pairs in the set"


def test_the_exclusions_are_the_three_we_claim_and_nothing_else(scene):
    """Each flag removes exactly the class it names, measured by counting.

    Documented numbers, so a change in the URDF or the scene shows up here rather than
    as a mysteriously different feasibility rate later.
    """
    everything = SceneCollisionModel.build(
        scene, exclude_same_link=False, exclude_adjacent=False, exclude_static=False
    )
    default = SceneCollisionModel.build(scene)
    assert len(everything.pairs) == 741
    assert len(default.pairs) == 689
    assert set(default.pairs) < set(everything.pairs)


def test_adjacent_links_are_what_makes_the_nominal_pose_look_broken(scene):
    """The reason parent-child pairs are excluded, stated as a measurement.

    Connected links touch -- that is what a joint is. Keeping those pairs makes the
    nominal posture itself violate Eq. 9, so every mode would be reported infeasible
    for a reason that has nothing to do with the mode.
    """
    q = scene.nominal_configuration()
    everything = SceneCollisionModel.build(scene, exclude_adjacent=False, exclude_same_link=False)
    default = SceneCollisionModel.cached(scene)

    def penetrating(m):
        return [(m.bodies[i].name, m.bodies[j].name)
                for i, j, w in m.witnesses(q) if w.distance < -1e-9]

    assert len(penetrating(everything)) > len(penetrating(default))
    assert penetrating(default) == [], (
        f"the nominal pose must be collision-free under the default policy, "
        f"but these interpenetrate: {penetrating(default)}"
    )


def test_the_grounded_nominal_is_what_makes_it_collision_free(scene):
    """docs/ambiguities.md #25, pinned.

    `robot.q_nominal` leaves the floating base at the origin, which buries the robot.
    Nothing noticed until Eq. 9 existed. This checks that the raw nominal really is in
    collision and the grounded one really is not, so the fix cannot be quietly undone.
    """
    model = SceneCollisionModel.cached(scene)
    raw = [1 for _, _, w in model.witnesses(scene.robot.q_nominal) if w.distance < -1e-9]
    grounded = [1 for _, _, w in model.witnesses(scene.nominal_configuration()) if w.distance < -1e-9]
    assert len(raw) >= 5, "the raw nominal was expected to be buried in the floor"
    assert grounded == []


def test_contact_pairs_are_kept_and_sit_at_exactly_zero(scene, model):
    """Eq. 7a's `p_z = 0` and Eq. 9's `0 <= sd` meet at exactly sd = 0.

    We do NOT exclude the pairs a mode holds in contact: with the paper's `margin = 0`
    the two constraints are compatible, at the single point where the bodies touch.
    That also means coincident witness points are the ordinary case here rather than a
    degenerate corner, which is why `query_witness` must not raise on them.
    """
    q = scene.nominal_configuration()
    distances = {
        (model.bodies[i].name, model.bodies[j].name): w.distance
        for i, j, w in model.witnesses(q)
    }
    box_floor = distances[("object/box", "environment/floor")]
    assert box_floor == pytest.approx(0.0, abs=1e-6)
    assert box_floor >= -1e-9, "a resting contact must not read as penetration"


# =============================================================================
# The rows reach the NLP, and they depend on the variables
# =============================================================================
def test_eq_14_carries_one_collision_row_per_pair(scene, model):
    sym = SymbolicScene(scene)
    blocks = model.blocks(sym, scene.nominal_configuration(), None)
    assert len(blocks) == len(model.pairs)

    problem, _ = build_problem(scene, STAND, collision_blocks=blocks, sym=sym)
    labels = " ".join(problem.block.ineq_labels)
    assert "9:sd>=margin" in labels
    assert problem.block.n_ineq >= len(model.pairs)


def test_a_collision_row_moves_when_the_robot_moves(scene, model):
    """The row must be a function of q, not a frozen number.

    Eq. 9 freezes the witness POINTS and the normal; the body TRANSFORMS stay
    symbolic. Freeze one transform too many and the row becomes a constant that is
    always satisfied -- the whole constraint silently switches off, and every mode
    gets easier rather than harder, which is the direction no test of feasibility
    would flag.
    """
    import casadi as ca

    sym = SymbolicScene(scene)
    q = scene.nominal_configuration()
    # A robot-versus-environment pair, so lifting the base must change it.
    index = next(
        k for k, (i, j) in enumerate(model.pairs)
        if model.bodies[i].kind == "robot" and model.bodies[j].kind == "environment"
    )
    block = model.blocks(sym, q, None)[index]

    row = ca.Function("g", [sym.variables], [block.ineq])
    base = sym.nominal()
    lifted = base.copy()
    lifted[2] += 0.25

    assert abs(float(row(lifted)) - float(row(base))) > 1e-6


def test_object_pose_variables_reach_the_collision_rows(scene, model):
    """The box's own pose is a decision variable (Eq. 5b), so its rows must see it."""
    import casadi as ca

    sym = SymbolicScene(scene)
    index = next(
        k for k, (i, j) in enumerate(model.pairs)
        if "object/box" in (model.bodies[i].name, model.bodies[j].name)
    )
    block = model.blocks(sym, scene.nominal_configuration(), None)[index]

    row = ca.Function("g", [sym.variables], [block.ineq])
    base = sym.nominal()
    moved = base.copy()
    moved[scene.robot.nq + 2] += 0.30  # raise the box
    assert abs(float(row(moved)) - float(row(base))) > 1e-6


# =============================================================================
# The refresh loop (docs/ambiguities.md #12b)
# =============================================================================
def test_refreshing_changes_the_witness_data_after_the_robot_moves(scene, model):
    """Why one solve is not enough.

    Schulman's model is first-order around the frozen witnesses. Milestone 2 measured
    the cost of not refreshing under rotation; this is the same statement one level up:
    query at two different configurations and the frozen constants differ, so a single
    linearization at q_nom describes the nominal pose and not the answer.
    """
    q = scene.nominal_configuration()
    moved = q.copy()
    moved[2] += 0.10

    before = {(i, j): w for i, j, w in model.witnesses(q)}
    after = {(i, j): w for i, j, w in model.witnesses(moved)}
    changed = sum(
        1 for key in before
        if abs(before[key].distance - after[key].distance) > 1e-6
        or np.linalg.norm(before[key].normal - after[key].normal) > 1e-6
    )
    assert changed > 0, "no witness data changed when the robot moved; the query is frozen"


def test_a_solve_reports_how_many_refreshes_it_used(scene):
    report = check(scene, STAND)
    assert report.feasible, report.explain()
    assert 1 <= report.refreshes <= int(scene.collision["refresh_iterations"])


# =============================================================================
# The answer is collision-free, measured independently
# =============================================================================
def test_the_returned_configuration_is_collision_free(scene, model):
    """The point of adding Eq. 9, checked by re-querying GJK rather than the rows.

    Re-reading the residuals would only confirm Ipopt reported its own work. Running
    the queries again on the returned configuration is an independent measurement, and
    it is the one that catches a body whose symbolic transform was wired to the wrong
    joint.
    """
    report = check(scene, STAND)
    assert report.feasible, report.explain()

    state = report.configurations
    poses = {
        name: pin.SE3(
            pin.Quaternion(state[name][6], *state[name][3:6]).toRotationMatrix(),
            state[name][:3],
        )
        for name in scene.objects
    }
    worst = [
        (w.distance, model.bodies[i].name, model.bodies[j].name)
        for i, j, w in model.witnesses(state["robot"], poses)
        if w.distance < -1e-4
    ]
    assert worst == [], f"solution interpenetrates: {sorted(worst)[:5]}"


# =============================================================================
# Schulman's activation distance (docs/ambiguities.md #19b)
# =============================================================================
def test_the_cutoff_drops_only_slack_rows(scene, model):
    """A pair beyond the activation distance is slack by that distance, by definition.

    This is the property that makes the cutoff a statement about which rows can be
    ACTIVE rather than a change to what Eq. 9 says: every row that is generated is
    exactly Eq. 9, and every row that is dropped had a signed distance above the
    threshold at the linearization point.
    """
    sym = SymbolicScene(scene)
    q = scene.nominal_configuration()
    cutoff = 0.10

    kept = model.blocks(sym, q, None, activation_distance=cutoff)
    everything = model.blocks(sym, q, None)
    assert 0 < len(kept) < len(everything)

    dropped = {b.name for b in everything} - {b.name for b in kept}
    by_name = {
        f"collision[{model.bodies[i].name}|{model.bodies[j].name}]": w.distance
        for i, j, w in model.witnesses(q)
    }
    for name in dropped:
        assert by_name[name] > cutoff


def test_the_audit_measures_all_pairs_not_just_the_active_ones(scene, model):
    """`max_penetration` must ignore the cutoff, or it agrees by construction.

    The audit exists to tell us whether the activation distance was big enough. If it
    used the same reduced pair set the solve used, it could only ever report what the
    solve already believed, and a cutoff that let a body through would be invisible.
    """
    q = scene.nominal_configuration()
    buried = q.copy()
    buried[2] -= 0.30  # drive the robot into the floor

    worst, pair = model.worst_penetration(buried)
    assert worst < -0.10, "a robot 30 cm into the floor must audit as penetrating"
    assert pair, "the audit must name the offending pair"

    clear, _ = model.worst_penetration(q)
    assert clear >= -1e-6


def test_every_solve_reports_its_penetration_audit(scene):
    report = check(scene, STAND)
    assert report.feasible, report.explain()
    assert report.max_penetration >= -1e-4, (
        f"the standing answer interpenetrates by {report.max_penetration:.5f} m "
        f"at {report.penetrating_pair}"
    )


def test_contact_pairs_are_relaxed_and_not_dropped(scene, model):
    """A mode's contact pairs keep a row, at -1 mm, instead of being removed.

    Removing them is the standard advice and it is measurably wrong here, because a
    patch is not a body: the left hand patch is a 27x30 mm window on the wrist cut
    face, while `left_wrist_yaw_link_0` is the whole forearm hull. Drop that pair and
    nothing constrains the forearm against the box at all -- `lift`, which holds the
    box in mid-air by both hands, came back feasible with the arm 12.8 cm inside it.

    Keeping them at a hard `sd >= 0` is exact but costs 2.7x, because the pair sits at
    exactly `sd = 0` and round-off keeps the refresh loop from ever calling the answer
    clean. The relaxed row is the middle and this test pins it: the row must still be
    there, and it must be the relaxed one.
    """
    sym = SymbolicScene(scene)
    q = scene.nominal_configuration()
    contact = model.contact_pairs(STAND)
    assert contact, "the standing mode must hold some pairs in contact"

    blocks = model.blocks(sym, q, None, relaxed=contact, relaxed_margin=-1e-3)
    assert len(blocks) == len(model.pairs), "relaxing must not remove any row"

    names = {
        f"collision[{model.bodies[i].name}|{model.bodies[j].name}]" for i, j in contact
    }
    relaxed_blocks = [b for b in blocks if b.name in names]
    assert relaxed_blocks, "the contact pairs' rows went missing"

    # `margin - sd <= 0`, so a -1 mm margin shows up as a row shifted by -1e-3.
    import casadi as ca

    for block in relaxed_blocks:
        value = float(ca.DM(ca.Function("g", [sym.variables], [block.ineq])(sym.nominal())))
        assert value <= -5e-4, (
            f"{block.name} evaluates to {value:+.6f}; a relaxed contact row should be "
            f"slack by about the 1 mm margin at a resting contact"
        )


def test_the_relaxed_margin_still_bounds_penetration(scene):
    """The point of relaxing rather than dropping: 1 mm of slack, not 12.8 cm.

    `lift` is the case that exposed it -- the box held in mid-air by both hands, where
    the only thing between the forearm and the box interior is this row.
    """
    from faro.scenarios.mode_demos import LIFT

    report = check(scene, LIFT)
    assert report.feasible, report.explain()
    allowance = float(scene.collision["contact_pair_margin"]) - 1e-3
    assert report.max_penetration >= allowance, (
        f"lift interpenetrates by {report.max_penetration:.5f} m at "
        f"{report.penetrating_pair}, past the {allowance:.5f} m the relaxed rows allow"
    )


# =============================================================================
# Friction lives in config, not in code (docs/ambiguities.md #13)
# =============================================================================
def test_friction_coefficients_come_from_the_scene_config(scene):
    assert scene.friction["mu"] == pytest.approx(0.6)
    assert scene.friction["mu_torsional"] == pytest.approx(0.02)

    custom = Scene.from_config({
        **{"name": "t", "robot": "g1", "friction": {"mu": 0.25}},
    })
    assert custom.friction["mu"] == pytest.approx(0.25)
    # Merged onto the defaults, not replacing them: naming one coefficient must not
    # silently drop the other to nothing.
    assert custom.friction["mu_torsional"] == pytest.approx(0.02)


def test_friction_does_not_enter_eq_14(scene):
    """Eq. 14 cites 7a, 7b, 9, 13a -- there are no forces for friction to act on.

    Declaring the coefficients must not quietly pull 7c/7d into the kinematic filter;
    that would make the cheap check expensive and reject modes a full TO could realize.
    """
    model = SceneCollisionModel.cached(scene)
    sym = SymbolicScene(scene)
    blocks = model.blocks(sym, scene.nominal_configuration(), None)
    problem, _ = build_problem(scene, STAND, collision_blocks=blocks, sym=sym)
    labels = " ".join(problem.block.eq_labels + problem.block.ineq_labels)
    assert "7c:" not in labels and "7d:" not in labels
