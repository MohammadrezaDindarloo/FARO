"""The hand is cut off at the wrist -- reversibly, and without side effects.

The paper puts the contact patch on the wrist cut face (docs/ambiguities.md #7b).
Leaving the G1's 13 cm hand attached would push it 0.09 m PAST that patch, through
whatever the robot touches. These tests pin down that the removal:

  * takes out the mass, not just the picture;
  * changes nothing else -- nq, nv and every joint index must be untouched;
  * is REVERSIBLE, i.e. the original markup is still in the file;
  * refuses unsafe removals loudly rather than silently corrupting the model.

    pytest tests/test_urdf_surgery.py -v
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import pytest

from faro.robots.robot_model import RobotModel
from faro.robots.urdf_surgery import (
    HAND_LINKS,
    LinkRemovalError,
    comment_out_links,
    link_report,
)
from faro.utils.paths import example_robot_data_share, resolve_asset

SOURCE = "erd://g1_description/urdf/g1_29dof_rev_1_0.urdf"


@pytest.fixture(scope="module")
def source_xml() -> str:
    return resolve_asset(SOURCE).read_text()


@pytest.fixture(scope="module")
def robot() -> RobotModel:
    return RobotModel.from_config("g1")


# =============================================================================
# The model actually loses the hand
# =============================================================================
def test_hand_links_are_gone_from_the_loaded_model(robot):
    assert not any("rubber_hand" in f.name for f in robot.model.frames)
    assert not any("rubber" in g.name for g in robot.visual_model.geometryObjects)


def test_removal_subtracts_mass_not_just_geometry(robot, source_xml):
    """Hiding the mesh in the viewer would leave 0.34 kg in Eq. 10. This does not.

    The hand is on a FIXED joint, so Pinocchio merges its inertia into the parent
    body at parse time. Removing the link must therefore shrink the *wrist body's*
    mass, which is the check that a viewer-only fix could never pass.
    """
    full = pin.buildModelFromUrdf(
        resolve_asset(SOURCE).as_posix(), pin.JointModelFreeFlyer()
    )
    hand_mass = 0.170
    for side in ("left", "right"):
        jid = robot.model.getJointId(f"{side}_wrist_yaw_joint")
        assert robot.model.inertias[jid].mass == pytest.approx(
            full.inertias[full.getJointId(f"{side}_wrist_yaw_joint")].mass - hand_mass,
            abs=1e-9,
        )

    total = sum(i.mass for i in robot.model.inertias)
    assert total == pytest.approx(sum(i.mass for i in full.inertias) - 2 * hand_mass, abs=1e-9)


def test_nq_nv_and_joint_indices_are_untouched(robot):
    """A fixed leaf link carries no DoF, so removing it must renumber nothing.

    If it did, `q_nominal`, the Eq. 13a limits and every cached contact-mode result
    would silently refer to the wrong joints.
    """
    full = pin.buildModelFromUrdf(
        resolve_asset(SOURCE).as_posix(), pin.JointModelFreeFlyer()
    )
    assert (robot.model.nq, robot.model.nv) == (full.nq, full.nv) == (36, 35)
    assert list(robot.model.names) == list(full.names)
    for name in full.names[1:]:
        assert (robot.model.joints[robot.model.getJointId(name)].idx_q
                == full.joints[full.getJointId(name)].idx_q)
    assert np.allclose(robot.model.lowerPositionLimit, full.lowerPositionLimit)


def test_the_link_the_patch_hangs_off_still_exists(robot):
    """The contact patch is anchored to the wrist -- that must survive the surgery."""
    for side in ("left", "right"):
        assert robot.model.existFrame(f"{side}_wrist_yaw_link")


# =============================================================================
# Reversibility -- the whole point of commenting rather than deleting
# =============================================================================
def test_generated_urdf_still_contains_the_hand_markup(robot):
    """Commented out, not deleted: uncommenting must be enough to restore it."""
    text = robot.urdf_path.read_text()
    for link in HAND_LINKS:
        assert f'<link name="{link}"' in text, f"{link} markup was lost, not commented"
        assert f'{link}.STL' in text
    assert "HAND REMOVED AT THE WRIST" in text, "the explanatory note is missing"

    # ... and it really is inside comments, i.e. the parser does not see it.
    root = ET.fromstring(text)
    assert not [el for el in root.findall("link") if el.get("name") in HAND_LINKS]


def test_uncommenting_really_does_restore_the_original_model(source_xml):
    """The strongest reversibility check: strip the comment markers back off and
    confirm the result rebuilds a model identical to the untouched one.

    Note ElementTree's default parser DISCARDS comments, so the commented markup has
    to be recovered from the raw text -- which is also exactly what a human doing it
    by hand would do.
    """
    commented = comment_out_links(source_xml, HAND_LINKS, "a note")

    # Un-comment every block that holds hand markup, the way a human would.
    uncommented = re.sub(
        r"<!--\s*(<joint name=\"\w*hand_palm_joint\".*?</link>)\s*-->",
        r"\1",
        commented,
        flags=re.DOTALL,
    )
    assert "<!--" in commented and uncommented != commented, "nothing was uncommented"

    rebuilt = pin.buildModelFromXML(uncommented, pin.JointModelFreeFlyer())
    original = pin.buildModelFromXML(source_xml, pin.JointModelFreeFlyer())

    assert (rebuilt.nq, rebuilt.nv) == (original.nq, original.nv)
    assert sum(i.mass for i in rebuilt.inertias) == pytest.approx(
        sum(i.mass for i in original.inertias), abs=1e-12
    )
    for link in HAND_LINKS:
        assert rebuilt.existFrame(link), f"{link} did not come back"


# =============================================================================
# The guards -- these must fail loudly, never silently
# =============================================================================
def test_removing_an_unknown_link_raises(source_xml):
    with pytest.raises(LinkRemovalError, match="no such link"):
        comment_out_links(source_xml, ["left_rubber_hnad"])   # typo on purpose


def test_removing_a_link_with_children_raises(source_xml):
    """Would orphan the whole subtree below it."""
    with pytest.raises(LinkRemovalError, match="child links"):
        comment_out_links(source_xml, ["left_shoulder_pitch_link"])


def test_removing_a_link_on_a_movable_joint_raises(source_xml):
    """Would change nq/nv and renumber every configuration index.

    `*_ankle_roll_link` is the test case because it is a genuine LEAF on a revolute
    joint -- so it gets past the children guard and reaches the joint-type guard.
    """
    with pytest.raises(LinkRemovalError, match="not 'fixed'"):
        comment_out_links(source_xml, ["left_ankle_roll_link"])


def test_the_children_guard_fires_before_the_joint_type_guard(source_xml):
    """`left_wrist_yaw_link` is on a revolute joint AND has the hand hanging off it.

    Whichever guard fires, removal must be refused; asserting the message keeps the
    two failure modes distinguishable when one actually happens.
    """
    with pytest.raises(LinkRemovalError, match="child links"):
        comment_out_links(source_xml, ["left_wrist_yaw_link"])


def test_empty_removal_list_is_a_no_op(source_xml):
    assert comment_out_links(source_xml, []) is source_xml


def test_link_report_states_what_is_lost(source_xml):
    lines = link_report(source_xml, HAND_LINKS)
    assert len(lines) == 2
    for line in lines:
        assert "0.170 kg" in line
        assert "0 collision" in line, "these links never had collision geometry"
