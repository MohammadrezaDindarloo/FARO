"""Contact modes and edges -- paper Eqs. 1-2 and Section II-D.

    "Let I be the set of all contact interfaces in the scene, indexed by integers.
     For each interface a in I, we represent its contact state by a pair (a, b),
     where b in I indicates unilateral contact with interface b, and b = null
     indicates that a is free. A contact mode is the complete assignment over all
     interfaces a unique contact pair:

         c = {(a, b) : for all a in I, exists! b in I u {null}}.               (1)

     Furthermore, the mode must be consistent, i.e. (a, b) in c <=> (b, a) in c.
     For a contact-mode time sequence C = (c_0, ..., c_{K-1}), each mode at time s
     is written as

         c_s = {(a, b_s) : a in I},   s = 0, ..., K - 1."                      (2)

This module holds NO optimization code and NO geometry. It is the discrete
vocabulary that Eq. 14 (mode/edge), Eq. 15 (KSO), Eq. 17 (TO) and Alg. 1 (tree
search) all speak. A `ContactMode` is hashable and totally ordered, which is what
lets Alg. 1's feasibility caches -- `K_feas` and `K_infeas` -- use one directly as
a key without inventing a serialization.

WHAT AN EDGE IS
---------------
Section II-D, on testing whether two modes can be connected:

    "To test whether two modes c1 and c2 can be connected via an edge, c is
     replaced by c1 u c2, representing the instantaneous transition between the
     two modes."

The union is a set union of contact pairs, so an interface in contact in EITHER
mode is in contact in the edge. `free` contributes no pair and therefore no
constraint, which is what makes the union well defined even though a foot that is
`floor` in c1 and `free` in c2 has two entries in the union: only the `floor` one
carries constraints.

That reading has teeth. If c1 puts the left hand on `box_left` and c2 puts it on
`box_front`, the edge demands both at once -- the hand pressed flat against two
perpendicular faces simultaneously. Eq. 7a cannot satisfy that, the NLP fails, and
the transition is correctly pruned. An edge is not a formality; it is the check
that a switch can happen at an instant rather than only in the limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from faro.scene.scene import Scene


# The paper writes `b = null` for a free interface (Eq. 1). None is that value.
FREE = None


@dataclass(frozen=True, order=True)
class ContactMode:
    """One contact mode c (Eq. 1): every interface assigned exactly one partner.

    Stored as a sorted tuple of `(interface_name, partner_patch_name_or_None)`
    rather than a dict, for three reasons that all matter later:

      * FROZEN AND HASHABLE, so Alg. 1's `K_feas` / `K_infeas` caches can key on the
        mode itself. A dict would force a serialization step, and two spellings of
        the same mode would then miss each other in the cache -- silently costing
        exactly the solves the cache exists to avoid.
      * SORTED, so mode identity does not depend on the order the interfaces were
        declared in the YAML. Same reason.
      * ORDERED (`order=True`), so modes sort reproducibly in reports and tests.

    `partner` names a PATCH, not an interface. The scene declares interfaces only
    for the sites Table IV branches over (hands, feet, box bottom); their partners
    -- `floor`, `tabletop`, `box_left` -- are patches. Eq. 1's consistency
    requirement `(a,b) in c <=> (b,a) in c` makes the reverse assignment implicit,
    and `active_pairs` is what materializes it.
    """

    assignments: tuple[tuple[str, str | None], ...]

    # ------------------------------------------------------------- construction
    @classmethod
    def from_dict(cls, mapping: dict[str, str | None]) -> ContactMode:
        """Build from `{interface_name: partner_patch_name_or_None}`."""
        return cls(tuple(sorted((str(k), v) for k, v in mapping.items())))

    @classmethod
    def all_free(cls, scene: "Scene") -> ContactMode:
        """The mode in which every interface is free -- Alg. 1's usual root c_0."""
        return cls.from_dict({name: FREE for name in scene.interfaces})

    # ------------------------------------------------------------------ queries
    def as_dict(self) -> dict[str, str | None]:
        return dict(self.assignments)

    def partner(self, interface: str) -> str | None:
        """Which patch `interface` touches, or None if it is free."""
        for name, value in self.assignments:
            if name == interface:
                return value
        raise KeyError(f"mode has no interface {interface!r}; has {[n for n, _ in self.assignments]}")

    def is_free(self, interface: str) -> bool:
        return self.partner(interface) is FREE

    def active_pairs(self, scene: "Scene") -> list[tuple[str, str]]:
        """The `(patch_a, patch_b)` pairs that carry contact constraints.

        This is the ONLY thing Eq. 14 / 15 / 17 need from a mode, and it is
        deliberately the same method name on `ContactEdge` -- so an NLP builder
        takes either without knowing which it got.

        Free interfaces are dropped: `b = null` means no contact, and no contact
        means no Eq. 7 rows. Note that it does NOT mean "no constraint at all" --
        collision avoidance (Eq. 9) still applies to a free hand, which is what
        stops it passing through the box on its way to somewhere else.
        """
        return [
            (scene.interfaces[name].patch.name, partner)
            for name, partner in self.assignments
            if partner is not FREE
        ]

    def n_active(self) -> int:
        return sum(1 for _, partner in self.assignments if partner is not FREE)

    # -------------------------------------------------------------- validation
    def validate(self, scene: "Scene") -> None:
        """Check against Eq. 1 and the scene's Table IV.

        Eq. 1 requires the assignment to be COMPLETE ("for all a in I") and UNIQUE
        ("exists! b"). Uniqueness is structural here -- one entry per interface --
        but completeness is not, so a mode that simply omits an interface would
        otherwise sail through and quietly drop that interface's constraints.
        """
        declared = set(scene.interfaces)
        named = [name for name, _ in self.assignments]

        if len(named) != len(set(named)):
            duplicated = sorted({n for n in named if named.count(n) > 1})
            raise ValueError(f"Eq. 1 requires exactly one partner per interface; duplicated: {duplicated}")

        missing = declared - set(named)
        if missing:
            raise ValueError(
                f"Eq. 1 assigns EVERY interface. Missing: {sorted(missing)}. "
                f"Use None for a free interface rather than omitting it -- an omitted "
                f"interface silently drops its contact constraints."
            )
        unknown = set(named) - declared
        if unknown:
            raise ValueError(f"unknown interfaces {sorted(unknown)}; scene declares {sorted(declared)}")

        for name, partner in self.assignments:
            if partner is FREE:
                continue
            allowed = scene.interfaces[name].allowed
            if partner not in allowed:
                raise ValueError(
                    f"interface {name!r} may not contact {partner!r}. "
                    f"Table IV allows: {allowed + ['free']}"
                )

    # ---------------------------------------------------------------- display
    def label(self) -> str:
        """Compact one-line form, e.g. `left_foot->floor, right_foot->floor`."""
        active = [f"{n}->{p}" for n, p in self.assignments if p is not FREE]
        return ", ".join(active) if active else "(all free)"

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"ContactMode({self.label()})"


@dataclass(frozen=True, order=True)
class ContactEdge:
    """The instantaneous transition c1 u c2 of Section II-D.

    Kept as its own type rather than collapsing into a `ContactMode`, because the
    union is NOT a mode: an interface in contact in c1 and free in c2 has two
    entries, which violates Eq. 1's `exists! b`. Pretending otherwise would mean
    either dropping one of the two contacts or failing `ContactMode.validate`, and
    both are wrong in a way that would be invisible from the outside.

    What survives the union is the SET OF CONTACT PAIRS, and that is exactly what
    the NLP consumes -- so `active_pairs` has the same signature here as on
    `ContactMode` and the Eq. 14 builder never needs to know the difference.
    """

    before: ContactMode
    after: ContactMode

    def active_pairs(self, scene: "Scene") -> list[tuple[str, str]]:
        """Union of both modes' contact pairs, deduplicated, order preserved.

        A contact held across the transition appears in both modes and must appear
        ONCE here -- duplicating it would double its Eq. 7 rows, which is not wrong
        in the feasible set but makes the Jacobian rank-deficient and hands Ipopt a
        singular KKT system for no reason.
        """
        seen: dict[tuple[str, str], None] = {}
        for pair in (*self.before.active_pairs(scene), *self.after.active_pairs(scene)):
            seen.setdefault(pair, None)
        return list(seen)

    def validate(self, scene: "Scene") -> None:
        self.before.validate(scene)
        self.after.validate(scene)

    def label(self) -> str:
        return f"{self.before.label()}  ==>  {self.after.label()}"

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"ContactEdge({self.label()})"


# =============================================================================
# Enumeration -- the discrete action space Alg. 1 branches over
# =============================================================================
def enumerate_modes(scene: "Scene") -> list[ContactMode]:
    """Every mode the scene allows: the Cartesian product over Table IV.

    For `box_placement` this is 3 x 3 x 2 x 2 x 3 = 108, which is the paper's
    stated maximum branching factor:

        "The allowed contact interfaces are listed in Table IV, yielding a maximum
         branching factor of 108."   -- Section IV-B

    That number matching is the check that our scene YAML reproduces Table IV, not
    just something shaped like it.

    Enumerating all 108 is fine for reporting and for tests. Alg. 1 does NOT do
    this -- `PROPOSESUCCESSOR` draws a random untested successor and progressive
    widening (Eq. 19) caps how many children a node may have, precisely so the
    search never has to materialize the full branching set.
    """
    names = sorted(scene.interfaces)
    choices = [scene.interfaces[name].options() for name in names]
    return [ContactMode(tuple(zip(names, combo))) for combo in product(*choices)]


def iter_successors(scene: "Scene", mode: ContactMode) -> Iterator[ContactMode]:
    """Modes reachable by changing exactly ONE interface's assignment.

    Not used by Eq. 14 itself; it is the natural successor set for Alg. 1 and lives
    here so the search never has to reimplement what a legal mode change is.
    """
    current = mode.as_dict()
    for name in sorted(scene.interfaces):
        for option in scene.interfaces[name].options():
            if option == current[name]:
                continue
            yield ContactMode.from_dict({**current, name: option})


@dataclass(frozen=True, order=True)
class ContactSequence:
    """A contact-mode time sequence C (Eq. 2), the input to the KSO (Eq. 15).

        "For a contact-mode time sequence C = (c_0, ..., c_{K-1}), each mode at time
         s is written as c_s = {(a, b_s) : a in I},  s = 0, ..., K - 1."   -- Eq. 2

    ONE CONFIGURATION PER MODE. The KSO asks for a configuration at every step of the
    sequence, and Eq. 16 is written in terms of an interface's partner "at step s and
    step s+1" -- so the knots are indexed by the modes themselves and `len(sequence)`
    is both the number of modes and the number of configurations. That sidesteps an
    indexing question the paper leaves open (whether C runs to c_{K-1} or c_K, i.e.
    whether there are K or K+1 configurations): however the paper counts, the code
    builds one configuration per mode it is given. See docs/ambiguities.md #27.

    Frozen and ordered for the same reason `ContactMode` is: Alg. 1 caches verdicts,
    and a sequence has to be usable as a cache key without a serialization step.
    """

    modes: tuple[ContactMode, ...]

    @classmethod
    def of(cls, *modes: ContactMode) -> ContactSequence:
        return cls(tuple(modes))

    @classmethod
    def from_dicts(cls, mappings) -> ContactSequence:
        """Build from `[{interface: partner}, ...]` -- the form a demo or an LLM emits."""
        return cls(tuple(ContactMode.from_dict(m) for m in mappings))

    def __len__(self) -> int:
        return len(self.modes)

    def __iter__(self):
        return iter(self.modes)

    def __getitem__(self, index):
        return self.modes[index]

    def transitions(self) -> list[tuple[int, ContactMode, ContactMode]]:
        """`(s, c_s, c_{s+1})` for every adjacent pair -- where Eq. 8 may apply."""
        return [(s, self.modes[s], self.modes[s + 1]) for s in range(len(self.modes) - 1)]

    def edges(self) -> list["ContactEdge"]:
        """The `c_s u c_{s+1}` edges of Section II-D, for pre-filtering with Eq. 14.

        Alg. 1 checks these BEFORE paying for a KSO: an edge that Eq. 14 rejects makes
        the whole sequence infeasible, and it costs one small solve instead of one
        large one.
        """
        return [ContactEdge(a, b) for _, a, b in self.transitions()]

    def validate(self, scene: "Scene") -> None:
        """Every mode must satisfy Eq. 1 and Table IV; the sequence must be non-empty."""
        if not self.modes:
            raise ValueError(
                "a contact sequence must contain at least one mode -- Eq. 15 optimizes "
                "one configuration per mode, so an empty sequence has no variables."
            )
        for s, mode in enumerate(self.modes):
            try:
                mode.validate(scene)
            except ValueError as exc:
                raise ValueError(f"mode {s} of the sequence is invalid: {exc}") from exc

    def label(self) -> str:
        return "  ->  ".join(mode.label() for mode in self.modes)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"ContactSequence({len(self.modes)} modes: {self.label()})"
