"""Milestone 1 tests: robot loading, patch geometry, and scene assembly.

The theme here is *conventions*. Patch normals, half-extents and frame offsets are
the kind of thing that produces a solver which converges happily to the wrong
answer, so each convention is pinned down by a test now, before Eqs. 7-13 are
written on top of them in Milestone 2.

    pytest tests/test_milestone1_scene.py -v
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from faro.core.patches import Attachment, ContactPatch, Interface
from faro.robots.robot_model import RobotModel
from faro.scene.objects import BoxObject
from faro.scene.scene import Scene


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_config("box_placement")


@pytest.fixture(scope="module")
def robot(scene: Scene) -> RobotModel:
    return scene.robot


# ---------------------------------------------------------------- robot loading
def test_robot_is_floating_base(robot: RobotModel):
    """nq - nv == 1 is the signature of a quaternion free-flyer root."""
    assert robot.nq - robot.nv == 1
    assert robot.model.names[1] == "root_joint"


def test_g1_has_29_actuated_joints(robot: RobotModel):
    """The paper's robot; 29 actuated DoF on top of the 6-DoF floating base."""
    assert robot.n_actuated == 29
    assert (robot.nq, robot.nv) == (36, 35)


def test_frame_lookup_raises_on_typo(robot: RobotModel):
    """Pinocchio returns nframes for an unknown name; we must raise instead.

    Otherwise a typo'd frame silently becomes a garbage placement much later.
    """
    with pytest.raises(KeyError, match="no frame"):
        robot.frame_id("left_ankel_roll_link")  # deliberate typo


def test_build_configuration_rejects_unknown_joint(robot: RobotModel):
    with pytest.raises(KeyError, match="no joint"):
        robot.build_configuration({"not_a_joint": 0.0})


def test_nominal_configuration_respects_joint_limits(robot: RobotModel):
    """The Eq. 14 regularization target must itself satisfy Eq. 12."""
    lower, upper = robot.joint_limits()
    q = robot.q_nominal
    # Skip the 7 floating-base entries, whose "limits" are meaningless.
    assert np.all(q[7:] >= lower[7:] - 1e-9)
    assert np.all(q[7:] <= upper[7:] + 1e-9)


# ------------------------------------------------------------------ box patches
def test_box_face_normals_point_outward():
    """Every face patch's local +z must point away from the box centre.

    A flipped normal would make Eq. 7a hold the box by the wrong face while still
    solving, so this is checked explicitly rather than assumed.
    """
    box = BoxObject(name="b", half_extents=(0.1, 0.2, 0.3), mass=1.0)
    expected = {
        "front": [1, 0, 0], "rear": [-1, 0, 0],
        "left": [0, 1, 0], "right": [0, -1, 0],
        "top": [0, 0, 1], "bottom": [0, 0, -1],
    }
    for face, want in expected.items():
        patch = box.face_patch(face)
        assert np.allclose(patch.normal_local, want, atol=1e-12), face
        # The patch centre must lie on that face, i.e. displaced along the normal.
        assert np.dot(patch.placement.translation, want) > 0, face


def test_box_face_half_extents_match_the_other_two_dimensions():
    """A face's in-plane half-extents are the box's two OTHER half-extents."""
    box = BoxObject(name="b", half_extents=(0.1, 0.2, 0.3), mass=1.0)
    assert set(box.face_patch("top").half_extents) == {0.1, 0.2}
    assert set(box.face_patch("front").half_extents) == {0.2, 0.3}
    assert set(box.face_patch("left").half_extents) == {0.1, 0.3}


def test_box_face_patches_are_rigid_transforms():
    """Rotations must be proper (det = +1); a reflection would flip handedness."""
    box = BoxObject(name="b", half_extents=(0.1, 0.2, 0.3), mass=1.0)
    for face in ["front", "rear", "left", "right", "top", "bottom"]:
        R = box.face_patch(face).placement.rotation
        assert np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_box_inertia_is_solid_cuboid():
    """Eq. 11 needs this at Milestone 5; a cube's inertia must be isotropic."""
    box = BoxObject(name="b", half_extents=(0.15, 0.15, 0.15), mass=2.0)
    I = box.inertia
    expected = (2.0 / 12.0) * (0.3**2 + 0.3**2)
    assert np.allclose(np.diag(I), expected)
    assert np.allclose(I - np.diag(np.diag(I)), 0.0)


def test_patch_rejects_nonpositive_half_extents():
    with pytest.raises(ValueError, match="half-extents"):
        ContactPatch("bad", Attachment.ENVIRONMENT, "world", pin.SE3.Identity(), (0.0, 0.1))


def test_patch_corners_are_centred_and_sized():
    patch = ContactPatch("p", Attachment.ENVIRONMENT, "world", pin.SE3.Identity(), (0.2, 0.1))
    corners = patch.corners_local()
    assert corners.shape == (4, 3)
    assert np.allclose(corners.mean(axis=0), 0.0)          # centred on the patch origin
    assert np.allclose(np.abs(corners[:, 0]).max(), 0.2)   # hx
    assert np.allclose(np.abs(corners[:, 1]).max(), 0.1)   # hy
    assert np.allclose(corners[:, 2], 0.0)                 # planar


# ----------------------------------------------------------------------- scene
def test_scene_reproduces_paper_branching_factor(scene: Scene):
    """Paper Section IV-B: "yielding a maximum branching factor of 108".

    3 (left hand) x 3 (right hand) x 2 (left foot) x 2 (right foot) x 3 (box bottom)
    = 108, counting `free` as an option for each interface. Matching this exactly is
    strong evidence that our reading of Table IV is correct.
    """
    assert scene.branching_factor() == 108


def test_scene_interfaces_match_table_iv(scene: Scene):
    """Table IV, verbatim."""
    expected = {
        "left_hand": {"box_left", "box_front"},
        "right_hand": {"box_right", "box_rear"},
        "left_foot": {"floor"},
        "right_foot": {"floor"},
        "box_bottom": {"floor", "tabletop"},
    }
    assert set(scene.interfaces) == set(expected)
    for name, allowed in expected.items():
        assert set(scene.interfaces[name].allowed) == allowed


def test_free_is_always_an_option(scene: Scene):
    """Section II-A: an interface is either in contact with another, or free."""
    for iface in scene.interfaces.values():
        assert None in iface.options()
        assert len(iface.options()) == iface.branching_factor()


def test_scene_rejects_unknown_contact_target(scene: Scene):
    """A typo in an `allowed` list must fail here, not as a mystery infeasible NLP."""
    broken = Scene(
        name="broken",
        robot=scene.robot,
        objects=dict(scene.objects),
        patches=dict(scene.patches),
    )
    broken.interfaces["x"] = Interface(
        name="x", patch=scene.patches["floor"], allowed=["no_such_patch"]
    )
    with pytest.raises(KeyError, match="unknown patch"):
        broken.validate()


def test_scene_rejects_self_contact(scene: Scene):
    broken = Scene(
        name="broken",
        robot=scene.robot,
        objects=dict(scene.objects),
        patches=dict(scene.patches),
    )
    broken.interfaces["x"] = Interface(name="x", patch=scene.patches["floor"], allowed=["floor"])
    with pytest.raises(ValueError, match="own patch"):
        broken.validate()


# ------------------------------------------------- standing pose / world layout
def test_foot_soles_reach_the_floor_at_the_standing_pose(scene: Scene):
    """The sole patch -- not the ankle frame -- must land on z = 0.

    The G1's sole sits 0.035 m below its ankle frame (4 collision spheres of radius
    0.005 centred at z = -0.03), so placing the robot by the ankle frame would leave
    it hovering. This test locks in that the offset is applied.
    """
    q = scene.robot.q_nominal.copy()
    soles = [scene.patches[n] for n in ("left_foot_sole", "right_foot_sole")]
    lowest = min(scene.patch_world_placement(p, q).translation[2] for p in soles)
    q[2] += -lowest

    for patch in soles:
        placement = scene.patch_world_placement(patch, q)
        assert placement.translation[2] == pytest.approx(0.0, abs=1e-9)
        # Sole normal points DOWN, out of the foot and into the floor.
        assert placement.rotation[2, 2] < -0.99


def test_foot_sole_half_extents_match_the_urdf_collision_spheres(scene: Scene):
    """Derived from the G1 URDF, not invented: x in [-0.05, 0.12], y in [-0.03, 0.03]."""
    sole = scene.patches["left_foot_sole"]
    assert sole.half_extents == pytest.approx((0.085, 0.03))
    assert sole.placement.translation == pytest.approx([0.035, 0.0, -0.035])


def test_palms_face_each_other(scene: Scene):
    """For a two-handed grasp the palms must face inward, not outward.

    Left palm normal along -y, right palm along +y, in each hand's own frame.
    """
    assert scene.patches["left_palm"].normal_local == pytest.approx([0, -1, 0], abs=1e-12)
    assert scene.patches["right_palm"].normal_local == pytest.approx([0, 1, 0], abs=1e-12)


def test_box_rests_on_the_floor(scene: Scene):
    """box_bottom starts in contact with the floor -- the task's initial mode."""
    bottom = scene.patch_world_placement(scene.patches["box_bottom"])
    assert bottom.translation[2] == pytest.approx(0.0, abs=1e-12)
    assert bottom.rotation[2, 2] == pytest.approx(-1.0)


def test_environment_patches_have_upward_normals(scene: Scene):
    for name in ("floor", "tabletop"):
        placement = scene.patch_world_placement(scene.patches[name])
        assert placement.rotation[:, 2] == pytest.approx([0, 0, 1], abs=1e-12)


def test_box_is_narrower_than_the_nominal_hand_separation(scene: Scene):
    """A two-handed side grasp needs the box's +y/-y faces INSIDE the palms' span.

    If the box were wider than the arms rest apart, the palms would have to splay
    outward rather than close inward, and the inward-facing palm normals asserted in
    `test_palms_face_each_other` would be the wrong convention for this task.
    """
    box = scene.objects["box"]
    q = scene.robot.q_nominal
    left = scene.robot.frame_placement(q, "left_rubber_hand").translation[1]
    right = scene.robot.frame_placement(q, "right_rubber_hand").translation[1]
    half_span = abs(left - right) / 2.0
    assert box.half_extents[1] < half_span, (
        f"box half-width {box.half_extents[1]} exceeds hand half-separation {half_span:.3f}"
    )


def test_box_is_out_of_static_arm_reach_so_the_base_must_move(scene: Scene):
    """Documents a real property of this scene, and why Eq. 14 optimizes over the base.

    The G1's shoulder-to-hand distance is ~0.41 m, but the box's side face sits
    ~0.47 m from the shoulder at the nominal standing pose. So NO arm-only posture
    reaches the box: the mode/edge NLP (Eq. 14) must translate the floating base --
    which it can, because q includes the base. This is a feature of the paper's
    loco-manipulation setting, not a misconfigured scene.

    If someone later "fixes" this by moving the box closer, this test fails and
    forces the question of whether the task still exercises locomotion at all.
    """
    robot, q = scene.robot, scene.robot.q_nominal
    shoulder = robot.frame_placement(q, "left_shoulder_pitch_link").translation
    hand = robot.frame_placement(q, "left_rubber_hand").translation
    arm_length = float(np.linalg.norm(hand - shoulder))

    box_face = scene.patch_world_placement(scene.patches["box_left"]).translation
    reach_required = float(np.linalg.norm(box_face - shoulder))

    assert arm_length == pytest.approx(0.41, abs=0.05)
    assert reach_required > arm_length, "box is within static reach; the task no longer needs the base to move"
