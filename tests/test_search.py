"""Milestone 5 -- Alg. 1, Eq. 18 and Eq. 19.

Almost everything here runs WITHOUT solving an NLP, by injecting `verify_fn`. That
is deliberate: the search policy is pure arithmetic, it is the part most likely to
be copied wrong from memory, and a test that needs a 20-second Ipopt call to check a
square root is a test nobody runs.
"""

from __future__ import annotations

import math

import pytest

from faro.core.modes import ContactMode, ContactSequence, enumerate_modes
from faro.scene.scene import Scene
from faro.search.config import SearchConfig
from faro.search.search import search
from faro.search.tree import Goal, Node, iter_nodes, make_root
from faro.search.uct import (
    best_child, propose_successor, select, successor_pool, uct_value, widening_limit,
)
from faro.search.verify import VerifyResult, normalize_filters, verify


@pytest.fixture(scope="module")
def scene():
    return Scene.from_config("box_placement")


def _mode(scene, **kwargs):
    return ContactMode.from_dict({name: kwargs.get(name) for name in scene.interfaces})


def _always(cost=1.0):
    def fn(scene, sequence):
        return VerifyResult(feasible=True, cost=cost, times={"M": 0.0})
    return fn


def _never():
    def fn(scene, sequence):
        return VerifyResult(feasible=False, rejected_by="M", times={"M": 0.0})
    return fn


# =============================================================================
# Eq. 19 -- progressive widening
# =============================================================================
@pytest.mark.parametrize("visits,expected", [
    (0, 1),      # max{1, .} floor: an unvisited node may still have one child
    (1, 1),      # 1.0 * 1^0.5 = 1
    (3, 1),      # sqrt(3) = 1.73 -> floor 1
    (4, 2),      # sqrt(4) = 2
    (9, 3),
    (100, 10),
])
def test_eq_19_widening_limit(visits, expected):
    """M(v) = max{1, k N(v)^alpha} at the paper's k = 1.0, alpha = 0.5."""
    assert widening_limit(visits) == expected


def test_eq_19_scales_with_k_and_alpha():
    assert widening_limit(100, k=2.0, alpha=0.5) == 20
    assert widening_limit(100, k=1.0, alpha=1.0) == 100


def test_eq_19_rejects_negative_visits():
    with pytest.raises(ValueError):
        widening_limit(-1)


# =============================================================================
# Eq. 18 -- cost-based UCT
# =============================================================================
def test_eq_18_is_cost_based_so_lower_is_better():
    """The paper takes an ARGMIN over J(u) - C sqrt(...).

    Standard UCT maximizes a reward and adds the bonus. Copying that sign here would
    make the search prefer the nodes it has already visited most, which is the exact
    opposite of exploration -- so the direction is pinned rather than the algebra.
    """
    parent = Node(mode=None, visits=10)
    cheap = Node(mode=None, cost=1.0, visits=5)
    dear = Node(mode=None, cost=9.0, visits=5)
    assert uct_value(cheap, parent) < uct_value(dear, parent)
    assert best_child(parent_with(parent, [cheap, dear])) is cheap


def test_eq_18_prefers_the_less_visited_child_at_equal_cost():
    parent = Node(mode=None, visits=50)
    fresh = Node(mode=None, cost=5.0, visits=0)
    stale = Node(mode=None, cost=5.0, visits=40)
    assert uct_value(fresh, parent) < uct_value(stale, parent)


def test_eq_18_matches_the_formula_exactly():
    parent = Node(mode=None, visits=7)
    child = Node(mode=None, cost=2.5, visits=3)
    expected = 2.5 - 3.0 * math.sqrt(math.log(7 + 1) / (3 + 1))
    assert uct_value(child, parent) == pytest.approx(expected)


def parent_with(parent, children):
    parent.children = list(children)
    return parent


def test_eq_18_has_no_argmin_over_an_empty_child_set(scene):
    with pytest.raises(ValueError):
        best_child(Node(mode=_mode(scene, left_foot="floor"), visits=1))


# =============================================================================
# The tree
# =============================================================================
def test_sequence_is_the_root_to_node_prefix(scene):
    """Section III: a node "together with its root-to-node prefix" IS C(v)."""
    a = _mode(scene, left_foot="floor")
    b = _mode(scene, left_foot="floor", right_foot="floor")
    c = _mode(scene, left_foot="floor", right_foot="floor", box_bottom="floor")

    root = make_root(a)
    mid = Node(mode=b, parent=root, depth=1)
    leaf = Node(mode=c, parent=mid, depth=2)

    assert leaf.sequence() == ContactSequence((a, b, c))
    assert root.sequence() == ContactSequence((a,))
    # depth d gives d + 1 modes, so "max depth 5" permits six-mode sequences
    assert len(leaf.sequence()) == leaf.depth + 1


def test_successor_pool_is_the_full_branching_factor(scene):
    """Section IV-B states a maximum branching factor of 108, i.e. every mode."""
    assert len(enumerate_modes(scene)) == 108
    current = _mode(scene, left_foot="floor", right_foot="floor", box_bottom="floor")
    pool = successor_pool(scene, current, policy="all")
    assert len(pool) == 107          # 108 minus the current mode
    assert current not in pool       # a self-transition burns a depth level for nothing


def test_successor_pool_single_changes_one_interface(scene):
    current = _mode(scene, left_foot="floor", right_foot="floor", box_bottom="floor")
    pool = successor_pool(scene, current, policy="single")
    assert 0 < len(pool) < 107
    for candidate in pool:
        differing = [n for n in scene.interfaces
                     if candidate.partner(n) != current.partner(n)]
        assert len(differing) == 1


def test_unknown_successor_policy_raises(scene):
    with pytest.raises(ValueError, match="unknown successor policy"):
        successor_pool(scene, _mode(scene), policy="whatever")


def test_propose_successor_never_repeats(scene):
    """"a random UNTESTED successor" -- so the pool must be drawn without replacement."""
    import numpy as np

    root = make_root(_mode(scene, left_foot="floor"))
    rng = np.random.default_rng(0)
    drawn = []
    while True:
        candidate = propose_successor(root, scene, rng)
        if candidate is None:
            break
        drawn.append(candidate)
    assert len(drawn) == 107
    assert len(set(drawn)) == 107


def test_select_increments_visits_along_the_path(scene):
    """Not stated by the paper (ambiguity #32), and both formulas are dead without it."""
    root = make_root(_mode(scene, left_foot="floor"))
    child = Node(mode=_mode(scene, right_foot="floor"), parent=root, depth=1)
    root.children.append(child)
    root.visits = 4                 # widening_limit(5) = 2 > 1 child, so it stops at root

    picked = select(root)
    assert picked is root
    assert root.visits == 5


def test_select_stops_at_max_depth(scene):
    """Section IV-B: "a max depth of 5 switches".

    Each node is given exactly one child and left at zero visits, so SELECT's
    increment takes it to N = 1 and Eq. 19 allows M = 1 -- already satisfied. That
    is what makes SELECT descend rather than widen, all the way to the depth cap.
    """
    node = root = make_root(_mode(scene, left_foot="floor"))
    for depth in range(1, 6):
        child = Node(mode=_mode(scene, right_foot="floor"), parent=node, depth=depth)
        node.children.append(child)
        node = child

    picked = select(root, max_depth=5)
    assert picked.depth == 5
    # every node on the path was visited, which is what makes Eqs. 18 and 19 move
    assert [n.visits for n in picked.path()] == [1, 1, 1, 1, 1, 1]


# =============================================================================
# GOAL
# =============================================================================
def test_goal_matches_only_the_terminal_mode(scene):
    goal = Goal(require={"box_bottom": "tabletop"})
    goal.validate(scene)
    placed = _mode(scene, left_foot="floor", box_bottom="tabletop")
    onfloor = _mode(scene, left_foot="floor", box_bottom="floor")
    assert goal.reached(ContactSequence((onfloor, placed)))
    assert not goal.reached(ContactSequence((placed, onfloor)))


def test_goal_released_requires_free_interfaces(scene):
    goal = Goal(require={"box_bottom": "tabletop"}, released=("left_hand", "right_hand"))
    holding = _mode(scene, box_bottom="tabletop", left_hand="box_left")
    letgo = _mode(scene, box_bottom="tabletop")
    assert not goal.reached(ContactSequence((holding,)))
    assert goal.reached(ContactSequence((letgo,)))


def test_goal_rejects_contacts_table_iv_forbids(scene):
    with pytest.raises(ValueError, match="Table IV"):
        Goal(require={"left_foot": "tabletop"}).validate(scene)


def test_empty_goal_raises():
    """A goal matching everything would report the root as a solution."""
    with pytest.raises(ValueError):
        Goal(require={})


# =============================================================================
# VERIFY
# =============================================================================
def test_filter_variants_match_eq_21():
    assert normalize_filters("F1") == ("KSO",)
    assert normalize_filters("F2") == ("M", "E", "KSO")
    assert normalize_filters("F3") == ("M", "E", "KSO", "TO")
    assert normalize_filters("F4") == ("TO",)


def test_empty_filter_set_raises():
    with pytest.raises(ValueError):
        normalize_filters([])


def test_unknown_filter_raises():
    with pytest.raises(ValueError, match="unknown filters"):
        normalize_filters(["M", "Q"])


def test_to_filter_raises_rather_than_passing_silently(scene):
    """A missing filter must not read as a pass -- F3 without TO is not F3."""
    sequence = ContactSequence((_mode(scene, left_foot="floor"),))
    with pytest.raises(NotImplementedError, match="Milestone 6"):
        verify(scene, sequence, filters=("TO",))


# =============================================================================
# Alg. 1
# =============================================================================
def test_search_respects_the_budget(scene):
    goal = Goal(require={"box_bottom": "tabletop"})
    root = _mode(scene, left_foot="floor", right_foot="floor", box_bottom="floor")
    result = search(scene, root, goal, budget=0.5, verify_fn=_always())
    assert 0.4 < result.stats.elapsed < 3.0
    assert result.stats.iterations > 0


def test_search_respects_max_depth(scene):
    goal = Goal(require={"box_bottom": "tabletop"})
    root = _mode(scene, left_foot="floor", right_foot="floor", box_bottom="floor")
    result = search(scene, root, goal, budget=1.0, max_depth=2, verify_fn=_always())
    assert max(n.depth for n in iter_nodes(result.root)) <= 2


def test_search_admits_nothing_when_every_candidate_is_rejected(scene):
    """Alg. 1 lines 9-11: a rejected candidate is not added and the loop continues."""
    goal = Goal(require={"box_bottom": "tabletop"})
    root = _mode(scene, left_foot="floor", right_foot="floor", box_bottom="floor")
    result = search(scene, root, goal, budget=0.5, verify_fn=_never())
    assert result.stats.nodes == 1          # the root only
    assert result.solutions == []
    assert result.stats.rejected > 0


def test_search_records_goal_sequences(scene):
    goal = Goal(require={"box_bottom": "tabletop"})
    root = _mode(scene, left_foot="floor", right_foot="floor", box_bottom="floor")
    result = search(scene, root, goal, budget=1.0, verify_fn=_always())
    assert result.solutions
    for solution in result.solutions:
        assert goal.reached(solution.sequence)
        # TO is Milestone 6, so nothing may claim dynamic feasibility yet.
        assert solution.to_verified is False


def test_search_is_reproducible_from_its_seed(scene):
    goal = Goal(require={"box_bottom": "tabletop"})
    root = _mode(scene, left_foot="floor", right_foot="floor", box_bottom="floor")
    shapes = []
    for _ in range(2):
        result = search(scene, root, goal, budget=0.4, seed=7, verify_fn=_always())
        shapes.append([c.mode for c in result.root.children][:5])
    assert shapes[0] == shapes[1]


# =============================================================================
# Config
# =============================================================================
def test_default_config_carries_the_papers_coefficients(scene):
    """Section III: "k = 1.0, alpha = 0.5 and C = 3"; Section IV-B: 2 h, depth 5."""
    cfg = SearchConfig.from_config(scene, "default")
    assert (cfg.C, cfg.k, cfg.alpha) == (3.0, 1.0, 0.5)
    assert cfg.budget == 7200.0
    assert cfg.max_depth == 5


def test_config_completes_partial_modes(scene):
    """Eq. 1 assigns EVERY interface; a partial YAML mode must be filled with free."""
    cfg = SearchConfig.from_config(scene, "default")
    cfg.root.validate(scene)
    assert set(dict(cfg.root.assignments)) == set(scene.interfaces)
    assert cfg.root.partner("left_hand") is None


def test_config_rejects_unknown_keys(scene):
    with pytest.raises(ValueError, match="unknown keys"):
        SearchConfig.from_config(scene, {"goal": {"require": {"box_bottom": "tabletop"}},
                                         "nonsense": 1})


def test_config_requires_a_goal(scene):
    with pytest.raises(KeyError, match="goal.require"):
        SearchConfig.from_config(scene, {"root": {}})
