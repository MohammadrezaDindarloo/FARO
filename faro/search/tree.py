"""The tree Alg. 1 grows -- nodes, their sequences, and the goal test.

    "The tree is rooted at an initial mode c_0, and each node v, together with its
     root-to-node prefix, represents a contact-mode sequence C(v)."   -- Section III

That sentence is the whole data structure. A node does NOT store a sequence; it
stores one mode and a parent pointer, and the sequence is read off the path. Storing
sequences per node would duplicate the prefix at every depth and, worse, allow two
nodes to disagree about what their shared prefix is.

This module holds no optimization and no search policy. `uct.py` decides which node
to visit, `verify.py` decides whether a candidate is admissible, and `search.py`
runs Alg. 1 over both. Splitting them that way is what lets the UCT arithmetic be
unit-tested without solving a single NLP -- see `tests/test_search.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from faro.core.modes import ContactMode, ContactSequence
from faro.scene.scene import Scene


@dataclass
class Node:
    """One node of Alg. 1's tree: a mode, its parent, and its search statistics.

    `cost` is Alg. 1 line 12's `J`:

        "If all active filters succeed, the candidate is added to the tree and
         assigned the cost J returned by the final filter in F."   -- Section III

    so it is set once, at admission, and never updated. That is the paper's literal
    reading and it is worth flagging, because it is unusual for a UCT: there is no
    backup step, and `J` therefore describes the node itself rather than the best
    thing found beneath it. See `uct.py` for what that costs and the switch that
    changes it.

    `visits` is Eq. 18's `N(.)`. The root is created with `visits = 0`; SELECT
    increments along the path it descends, which the paper does not spell out.
    """

    mode: ContactMode
    parent: "Node | None" = None
    depth: int = 0
    cost: float = 0.0
    visits: int = 0
    children: list["Node"] = field(default_factory=list)
    #: Successors not yet proposed from this node, shuffled once. PROPOSESUCCESSOR
    #: pops from here, which is what makes "a random UNTESTED successor" cheap and
    #: repeatable -- see `uct.propose_successor`.
    untried: list[ContactMode] = field(default_factory=list)
    #: Filled lazily, so a node that is never expanded never pays for its pool.
    expanded: bool = False

    # ------------------------------------------------------------------ queries
    def path(self) -> list["Node"]:
        """Root-to-node, inclusive."""
        chain, node = [], self
        while node is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))

    def sequence(self) -> ContactSequence:
        """`C(v)`: the root-to-node prefix as a contact-mode sequence (Eq. 2).

        A node at depth d yields d + 1 modes, because the root IS c_0. So "max depth
        5 switches" (Section IV-B) permits sequences of up to six modes.
        """
        return ContactSequence(tuple(n.mode for n in self.path()))

    def is_root(self) -> bool:
        return self.parent is None

    def label(self) -> str:
        return f"d{self.depth} N={self.visits} J={self.cost:.3f}  {self.mode.label()}"


def make_root(mode: ContactMode) -> Node:
    """Alg. 1 line 1: `T <- INITIALIZETREE(c_0)`."""
    return Node(mode=mode, parent=None, depth=0)


def iter_nodes(root: Node):
    """Every node in the tree, depth-first. Reporting and tests only."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children)


# =============================================================================
# GOAL -- Alg. 1 line 13
# =============================================================================
class Goal:
    """`GOAL(C+)`: does this sequence's LAST mode satisfy the task condition?

    AMBIGUITY (docs/ambiguities.md #30). The paper never defines GOAL. All it says
    of the task is:

        "In both tasks, the objective is to place the white box on the platform."
                                                                  -- Fig. 4 caption

    "The box is on the platform" is a CONTACT condition, not a configuration, so the
    goal is expressed here the same way a mode is: a partial assignment that the
    terminal mode must match. For `box_placement` that is `box_bottom -> tabletop`.

    `require` holds only the interfaces that must match; every other interface is
    free to be anything. That matters: demanding a fully specified terminal mode
    would also dictate where the hands are, and "place the box" says nothing about
    whether the robot is still holding it.

    `released` additionally requires the named interfaces to be FREE, which is how
    you ask for "placed AND let go" rather than "placed while still gripping". It is
    off by default because the paper's sentence does not ask for it.
    """

    def __init__(self, require: dict[str, str], released: tuple[str, ...] = ()):
        if not require:
            raise ValueError(
                "a goal with no required contacts matches every sequence, which would "
                "make Alg. 1 report the root as a solution."
            )
        self.require = dict(require)
        self.released = tuple(released)

    def validate(self, scene: Scene) -> None:
        for interface, partner in self.require.items():
            if interface not in scene.interfaces:
                raise ValueError(
                    f"goal names interface {interface!r}; scene declares "
                    f"{sorted(scene.interfaces)}"
                )
            allowed = scene.interfaces[interface].allowed
            if partner not in allowed:
                raise ValueError(
                    f"goal asks {interface!r} to contact {partner!r}, which Table IV "
                    f"does not allow. Allowed: {allowed + ['free']}"
                )
        for interface in self.released:
            if interface not in scene.interfaces:
                raise ValueError(f"goal names unknown interface {interface!r} in `released`")

    def reached(self, sequence: ContactSequence) -> bool:
        terminal = sequence[-1]
        for interface, partner in self.require.items():
            if terminal.partner(interface) != partner:
                return False
        return all(terminal.is_free(interface) for interface in self.released)

    def label(self) -> str:
        parts = [f"{k}->{v}" for k, v in sorted(self.require.items())]
        parts += [f"{k}->free" for k in sorted(self.released)]
        return ", ".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Goal({self.label()})"
