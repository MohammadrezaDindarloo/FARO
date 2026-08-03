"""Milestone 3, part 1: the discrete vocabulary -- paper Eqs. 1-2 and Section II-D.

These tests are cheap and carry no solver. They exist because every later stage
takes a `ContactMode` on faith: if enumeration or validation is wrong, Eq. 14 will
happily solve a problem that is not the one anybody asked for, and the verdict will
look perfectly reasonable.

    pytest tests/test_modes.py -v
"""

from __future__ import annotations

import pytest

from faro.core.modes import ContactEdge, ContactMode, enumerate_modes, iter_successors
from faro.scene.scene import Scene


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_config("box_placement")


def mode(**kwargs) -> ContactMode:
    """A mode with everything free except what is named."""
    base = {
        "left_hand": None, "right_hand": None,
        "left_foot": None, "right_foot": None, "box_bottom": None,
    }
    base.update(kwargs)
    return ContactMode.from_dict(base)


# =============================================================================
# Eq. 1 / Table IV
# =============================================================================
def test_enumeration_reproduces_the_papers_branching_factor(scene):
    """Section IV-B: "yielding a maximum branching factor of 108".

    This is the one number in the paper that our scene YAML can be checked against
    directly. 108 = 3 x 3 x 2 x 2 x 3 factorizes over Table IV's rows, so a wrong
    `allowed` list on any single interface changes it -- which makes this a check on
    the whole table rather than on its total.
    """
    modes = enumerate_modes(scene)
    assert len(modes) == 108
    assert len(set(modes)) == 108, "enumeration produced duplicate modes"


def test_every_enumerated_mode_validates(scene):
    for m in enumerate_modes(scene):
        m.validate(scene)


def test_a_mode_missing_an_interface_is_rejected(scene):
    """Eq. 1 assigns a partner to EVERY interface ("for all a in I").

    An omitted interface is far more dangerous than a wrong one: it silently drops
    that interface's Eq. 7 rows, so the NLP solves a strictly easier problem and
    reports feasible. A foot quietly excused from touching the floor is exactly the
    kind of false POSITIVE the hierarchy is supposed not to produce.
    """
    incomplete = ContactMode.from_dict({"left_foot": "floor", "right_foot": "floor"})
    with pytest.raises(ValueError, match="Missing"):
        incomplete.validate(scene)


def test_a_contact_outside_table_iv_is_rejected(scene):
    with pytest.raises(ValueError, match="may not contact"):
        mode(left_hand="tabletop").validate(scene)


def test_free_interfaces_contribute_no_contact_pairs(scene):
    """`b = null` means no Eq. 7 rows -- but NOT no constraints at all.

    Collision (Eq. 9) still applies to a free hand, which is what stops it passing
    through the box on its way somewhere. That part is Eq. 14's business; here we
    only check that `active_pairs` reports the contacts and nothing else.
    """
    assert mode().active_pairs(scene) == []
    assert mode(left_foot="floor").active_pairs(scene) == [("left_foot_sole", "floor")]


def test_active_pairs_name_patches_not_interfaces(scene):
    """The mode speaks interfaces; the constraints need patches.

    `left_foot` is an interface, `left_foot_sole` is the patch it owns. Eq. 7 is
    written on patches, so the translation has to happen somewhere -- and doing it
    here rather than in each NLP builder is why mode/edge, KSO and TO cannot
    disagree about it.
    """
    pairs = mode(left_hand="box_left").active_pairs(scene)
    assert pairs == [("left_hand_patch", "box_left")]


# =============================================================================
# Section II-D -- the edge is the union
# =============================================================================
def test_edge_is_the_union_of_both_modes_contacts(scene):
    before = mode(left_foot="floor", right_foot="floor")
    after = mode(left_foot="floor", left_hand="box_left")
    pairs = ContactEdge(before, after).active_pairs(scene)
    assert set(pairs) == {
        ("left_foot_sole", "floor"),
        ("right_foot_sole", "floor"),
        ("left_hand_patch", "box_left"),
    }


def test_a_contact_held_across_the_transition_appears_once(scene):
    """Duplicating a persistent contact is not wrong -- it is worse than wrong.

    The duplicate rows are satisfied by exactly the same configurations, so the
    feasible set is unchanged and no test of the ANSWER would notice. What changes is
    the Jacobian: identical rows make it rank-deficient, and Ipopt gets a singular
    KKT system for a problem that is perfectly well posed. That shows up as slow or
    failed convergence attributed to the geometry.
    """
    held = mode(left_foot="floor", right_foot="floor")
    pairs = ContactEdge(held, held).active_pairs(scene)
    assert pairs == held.active_pairs(scene)
    assert len(pairs) == len(set(pairs))


def test_an_edge_that_keeps_a_contact_is_not_the_same_as_dropping_it(scene):
    """Releasing a contact leaves it in the union -- the edge is the INSTANT of change.

    A foot on the floor in c1 and free in c2 is still on the floor at the transition
    instant, so its rows belong in the edge problem. Getting this backwards would
    make every release edge trivially feasible.
    """
    stance = mode(left_foot="floor", right_foot="floor")
    swing = mode(right_foot="floor")
    assert ("left_foot_sole", "floor") in ContactEdge(stance, swing).active_pairs(scene)


# =============================================================================
# Usable as a cache key -- Section III's K_feas / K_infeas
# =============================================================================
def test_modes_are_hashable_and_order_independent(scene):
    """Two spellings of the same mode must be the SAME key.

    If they are not, Alg. 1's cache misses on every re-query and silently loses the
    entire benefit Table II attributes to it -- 814.6 nodes expanded versus 218.2 on
    the hard task -- while still passing every test of correctness.
    """
    a = ContactMode.from_dict({"left_foot": "floor", "right_foot": "floor",
                               "left_hand": None, "right_hand": None, "box_bottom": None})
    b = ContactMode.from_dict({"box_bottom": None, "right_hand": None, "left_hand": None,
                               "right_foot": "floor", "left_foot": "floor"})
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1


def test_edges_are_hashable_and_directional(scene):
    stand, grasp = mode(left_foot="floor"), mode(left_hand="box_left")
    assert ContactEdge(stand, grasp) == ContactEdge(stand, grasp)
    assert ContactEdge(stand, grasp) != ContactEdge(grasp, stand)
    assert len({ContactEdge(stand, grasp), ContactEdge(grasp, stand)}) == 2


# =============================================================================
# Successors -- what Alg. 1 branches over
# =============================================================================
def test_successors_change_exactly_one_interface(scene):
    start = mode(left_foot="floor", right_foot="floor", box_bottom="floor")
    successors = list(iter_successors(scene, start))

    for s in successors:
        differing = [n for n in scene.interfaces if s.partner(n) != start.partner(n)]
        assert len(differing) == 1, f"{s.label()} changed {len(differing)} interfaces"
        s.validate(scene)

    # One per alternative option per interface: sum over interfaces of (options - 1).
    expected = sum(len(i.options()) - 1 for i in scene.interfaces.values())
    assert len(successors) == expected
    assert start not in successors
