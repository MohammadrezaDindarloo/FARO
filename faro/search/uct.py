"""SELECT and PROPOSESUCCESSOR -- paper Eqs. 18 and 19.

    "Node selection uses a cost-based variant of Upper Confidence bounds applied to
     Trees (UCT), which balances the exploitation of low-cost branches with the
     exploration of less frequently visited nodes.

         v* = argmin_{u in children(v)} [ J(u) - C sqrt( log(N(v)+1) / (N(u)+1) ) ]  (18)

     Expansion is controlled by progressive widening, with the maximum number of
     children M(v) increasing as node v is visited more often:

         M(v) = max{1, k N(v)^alpha}                                              (19)

     The coefficients are set as k = 1.0, alpha = 0.5 and C = 3. Given the selected
     parent node v, PROPOSESUCCESSOR returns a random untested successor mode c+."
                                                                    -- Section III

COST-BASED, SO argmin AND MINUS
-------------------------------
This is not the textbook UCT. Standard UCT maximizes a reward and ADDS the
exploration bonus; Eq. 18 minimizes a COST and SUBTRACTS it. Both say the same
thing -- "prefer good, and prefer unvisited" -- but the signs are easy to copy
wrong from memory, and a `+` here would make the search avoid the nodes it has
never tried. The unit tests pin the direction rather than the formula's appearance.
"""

from __future__ import annotations

import math

from faro.core.modes import ContactMode, enumerate_modes, iter_successors
from faro.scene.scene import Scene
from faro.search.tree import Node

#: Section III, verbatim: "The coefficients are set as k = 1.0, alpha = 0.5 and C = 3."
DEFAULT_C = 3.0
DEFAULT_K = 1.0
DEFAULT_ALPHA = 0.5


def widening_limit(visits: int, *, k: float = DEFAULT_K, alpha: float = DEFAULT_ALPHA) -> int:
    """Eq. 19: `M(v) = max{1, k N(v)^alpha}`.

    Floored to an integer, because it is compared against a count of children. The
    paper writes it as a real number and never says which way to round; flooring is
    the conservative reading -- it widens later rather than sooner -- and at the
    paper's own k = 1.0 the two differ only in when the second child is allowed.
    `max(1, .)` guarantees every selected node may have at least one child, which is
    what stops an unvisited node from being a dead end.
    """
    if visits < 0:
        raise ValueError(f"visit counts cannot be negative, got {visits}")
    return max(1, int(math.floor(k * (visits ** alpha))))


def uct_value(child: Node, parent: Node, *, C: float = DEFAULT_C) -> float:
    """The bracketed quantity of Eq. 18, for one child. Lower is better.

    `N(v) + 1` and `N(u) + 1` are the paper's, not a guard we added: they make the
    expression finite for an unvisited child, which is the case that matters most
    since a freshly admitted node has `N(u) = 0` and should look attractive.
    """
    exploration = math.sqrt(math.log(parent.visits + 1) / (child.visits + 1))
    return child.cost - C * exploration


def best_child(node: Node, *, C: float = DEFAULT_C) -> Node:
    """Eq. 18's `argmin` over `children(v)`.

    Ties break on the FIRST child in insertion order rather than randomly, so a run
    is reproducible from its seed alone. Insertion order is itself seeded, through
    the shuffled successor pool in `propose_successor`.
    """
    if not node.children:
        raise ValueError(f"Eq. 18 has no argmin over an empty child set (node {node.label()})")
    return min(node.children, key=lambda child: uct_value(child, node, C=C))


# =============================================================================
# The successor set
# =============================================================================
def successor_pool(scene: Scene, mode: ContactMode, *, policy: str = "all") -> list[ContactMode]:
    """The candidate modes PROPOSESUCCESSOR may draw from.

    AMBIGUITY (docs/ambiguities.md #31). The paper does not define the successor
    set. It gives one number that constrains it:

        "The allowed contact interfaces are listed in Table IV, yielding a maximum
         branching factor of 108."   -- Section IV-B

    108 is exactly the Cartesian product over Table IV (3 x 3 x 2 x 2 x 3), i.e. ALL
    modes -- not the modes reachable by changing one interface, of which there are
    8. So `all` is the paper-faithful default, and it means a single transition may
    change several contacts at once. That is not obviously wrong: filter E tests
    whether the transition can happen at an instant, so a physically impossible
    simultaneous change gets rejected there rather than being excluded by fiat.

    `single` restricts to one-interface changes (`iter_successors`). It makes each
    transition minimal and shrinks the branching factor by ~13x, at the cost of no
    longer being able to express a simultaneous two-handed release. Kept because
    which of the two the paper ran is genuinely unclear from the text, and the
    difference is measurable rather than a matter of taste.

    The current mode is excluded either way: a self-transition adds a mode to the
    sequence without changing any contact, so it burns a depth level to say nothing.
    """
    if policy == "all":
        candidates = enumerate_modes(scene)
    elif policy == "single":
        candidates = list(iter_successors(scene, mode))
    else:
        raise ValueError(
            f"unknown successor policy {policy!r}. Known: 'all' (every mode Table IV "
            f"allows -- the paper's branching factor of 108) and 'single' (one "
            f"interface changes at a time)."
        )
    return [c for c in candidates if c != mode]


def propose_successor(node: Node, scene: Scene, rng, *, policy: str = "all") -> ContactMode | None:
    """Alg. 1 line 6: `c+ <- PROPOSESUCCESSOR(v)`, "a random untested successor".

    The pool is built and shuffled ONCE per node, then popped. Drawing randomly and
    rejecting duplicates instead would re-draw ever more often as a node fills up --
    the classic coupon-collector blowup -- and "untested" would have to be checked
    against the child list on every draw.

    Returns None when the node has no untested successor left, which SELECT reads as
    "this node cannot be widened further" regardless of what Eq. 19 permits.
    """
    if not node.expanded:
        node.untried = successor_pool(scene, node.mode, policy=policy)
        rng.shuffle(node.untried)
        node.expanded = True
    return node.untried.pop() if node.untried else None


# =============================================================================
# SELECT -- Alg. 1 line 5
# =============================================================================
def select(root: Node, *, C: float = DEFAULT_C, k: float = DEFAULT_K,
           alpha: float = DEFAULT_ALPHA, max_depth: int = 5) -> Node:
    """Descend from the root to the node Alg. 1 should try to expand.

    At each level: if the node has room for another child under Eq. 19, stop and
    return it -- that is the node to expand. Otherwise take Eq. 18's argmin and
    descend. Visit counts are incremented along the way.

    WHERE `max_depth` BITS. Section IV-B runs "a max depth of 5 switches", and a
    node at that depth may not be expanded. It is returned anyway rather than
    skipped, and `search.py` treats a full-depth node as a failed iteration: the
    alternative -- descending past it or silently retrying -- would either violate
    the bound or spin.

    N(v) IS INCREMENTED ON THE PATH, which the paper does not state (ambiguity #32).
    Without it Eq. 19 never widens and Eq. 18's exploration term never decays, so
    the search would expand each node exactly once and degenerate into a random
    walk. Incrementing on the selection path is what makes both formulas move.
    """
    node = root
    while True:
        node.visits += 1
        if node.depth >= max_depth:
            return node
        if len(node.children) < widening_limit(node.visits, k=k, alpha=alpha):
            return node
        if not node.children:
            # Eq. 19 is satisfied but nothing was ever admitted here -- every
            # candidate this node proposed was rejected by VERIFY. There is no
            # argmin to take, so this branch is finished.
            return node
        node = best_child(node, C=C)
