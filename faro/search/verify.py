"""VERIFY -- Alg. 1 line 8, the filter pipeline of Section III.

    "Before admission to the tree, VERIFY applies the selected feasibility filters
     F. We consider the following filters, ordered from cheaper to more expensive:
       1) contact-mode feasibility, M(c+), using (14),
       2) transition-edge feasibility, E(c(v), c+), using (14),
       3) kinematic sequence optimization, KSO(C+), using (15),
       4) trajectory optimization, TO(C+), using (17).
     ...
     Within each variant, the filters are applied from left to right and
     verification terminates at the first failed filter. Reusable outcomes are
     stored in the feasible and infeasible caches, K_feas and K_infeas, so that
     previously evaluated checks do not need to be solved again."   -- Section III

WHAT EACH FILTER IS GIVEN, WHICH IS EASY TO GET WRONG
------------------------------------------------------
The three filters do NOT all look at the same thing:

    M    the PROPOSED MODE alone, c+           -- one mode
    E    the TRANSITION, c(v) -> c+            -- the parent's terminal mode and c+
    KSO  the WHOLE CANDIDATE SEQUENCE, C+      -- root to c+

So M and E are local and cacheable across the entire tree, while KSO is keyed on a
sequence that is usually unique. Table II shows what that is worth: on the hard task
the M,E,KSO variant makes 1415 KSO attempts against 814.6 tree nodes, so nearly
every KSO call is a distinct sequence and the cache almost never hits for it. The
caching advantage the paper attributes to M and E --

    "Since contact modes and transitions are repeatedly revisited during search,
     cached infeasibility results allow previously checked queries to be rejected
     without resolving the corresponding optimization problems."   -- Section IV-B

-- is a statement about filters 1 and 2, and this module is where it is realized.

THE CACHE IS A MEMO, NOT A PHASE
--------------------------------
Alg. 1 line 3 initializes `K_feas, K_infeas <- {}, {}` INSIDE the search, so they
start empty on every run and fill during the budget. There is no separate
cache-building stage in the paper. `FeasibilityCache` and `KSOCache` are passed in
rather than created here precisely so a caller CAN keep one across runs -- but that
is our extension, and a run that wants the paper's numbers must start with fresh
caches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from faro.core.modes import ContactEdge, ContactMode, ContactSequence
from faro.kso.feasibility import KSOCache
from faro.kso.feasibility import check as check_kso
from faro.mode_edge.feasibility import FeasibilityCache
from faro.mode_edge.feasibility import check as check_eq14
from faro.scene.scene import Scene

#: The filter names Section III defines. TO is listed because F3 and F4 use it; it
#: is not implemented yet (Milestone 6) and asking for it raises rather than being
#: silently dropped from the pipeline.
KNOWN_FILTERS = ("M", "E", "KSO", "TO")

#: Section III's four variants, Eq. 21.
VARIANTS = {
    "F1": ("KSO",),
    "F2": ("M", "E", "KSO"),
    "F3": ("M", "E", "KSO", "TO"),
    "F4": ("TO",),
}


@dataclass
class VerifyResult:
    """The `(feasible, J)` of Alg. 1 line 8, plus why and at what cost."""

    feasible: bool
    #: Alg. 1 line 12's J -- "the cost returned by the final filter in F". Only
    #: meaningful when `feasible`.
    cost: float = 0.0
    #: Which filter rejected it, or "" when all passed.
    rejected_by: str = ""
    #: Per-filter wall time actually spent, cache hits included as 0.0.
    times: dict[str, float] = field(default_factory=dict)
    #: Which filters were served from cache rather than solved.
    cached: tuple[str, ...] = ()
    status: str = ""

    @property
    def solved_any(self) -> bool:
        return any(t > 0.0 for t in self.times.values())


@dataclass
class Caches:
    """Alg. 1's `K_feas` / `K_infeas`, one store per filter that has a key.

    Two stores rather than the paper's two SETS, for the reason `FeasibilityCache`
    documents: a verdict is a value, not a membership, and two independent sets can
    represent "in both caches", which is a contradiction that cannot be detected.

    There is no store for TO: it is the last filter, its result is never reused as a
    precondition for anything else, and Alg. 1 line 17 may call it outside VERIFY.
    """

    mode_edge: FeasibilityCache = field(default_factory=FeasibilityCache)
    kso: KSOCache = field(default_factory=KSOCache)

    def summary(self) -> str:  # pragma: no cover - display only
        return f"{self.mode_edge.summary()}\n  {self.kso.summary()}"


def normalize_filters(filters) -> tuple[str, ...]:
    """Accept a variant name (`"F2"`) or an explicit tuple (`("M", "E")`)."""
    if isinstance(filters, str):
        if filters not in VARIANTS:
            raise ValueError(
                f"unknown filter variant {filters!r}. Known: {sorted(VARIANTS)} "
                f"(Eq. 21), or pass an explicit sequence of {list(KNOWN_FILTERS)}."
            )
        return VARIANTS[filters]
    out = tuple(str(f) for f in filters)
    if not out:
        raise ValueError(
            "VERIFY with no filters admits every candidate, which is not one of the "
            "variants in Eq. 21 and would make the search a random walk."
        )
    unknown = [f for f in out if f not in KNOWN_FILTERS]
    if unknown:
        raise ValueError(f"unknown filters {unknown}; Section III defines {list(KNOWN_FILTERS)}")
    return out


def verify(
    scene: Scene,
    sequence: ContactSequence,
    *,
    filters=("M", "E"),
    caches: Caches | None = None,
    kso_kwargs: dict | None = None,
    eq14_kwargs: dict | None = None,
) -> VerifyResult:
    """Apply `filters` left to right, stopping at the first failure.

    `sequence` is the CANDIDATE `C+ = C(v) + c+`, so its last mode is `c+` and its
    second-to-last is `c(v)`. Both are read from the sequence rather than passed
    separately, so VERIFY cannot be handed an edge that does not belong to it.

    A root-only sequence has no transition, so filter E is skipped for it -- there is
    nothing to transition from. Alg. 1 never calls VERIFY on the root anyway (line 7
    always appends a proposed successor), but tests do.
    """
    filters = normalize_filters(filters)
    caches = caches if caches is not None else Caches()
    eq14_kwargs = dict(eq14_kwargs or {})
    kso_kwargs = dict(kso_kwargs or {})

    result = VerifyResult(feasible=True)
    cached: list[str] = []

    for name in filters:
        started = time.perf_counter()

        if name == "M":
            report = check_eq14(scene, sequence[-1], cache=caches.mode_edge, **eq14_kwargs)
        elif name == "E":
            if len(sequence) < 2:
                result.times[name] = 0.0
                continue
            edge = ContactEdge(sequence[-2], sequence[-1])
            report = check_eq14(scene, edge, cache=caches.mode_edge, **eq14_kwargs)
        elif name == "KSO":
            report = check_kso(scene, sequence, cache=caches.kso, **kso_kwargs)
        else:  # TO
            raise NotImplementedError(
                "filter TO (Eq. 17) is Milestone 6 and is not implemented. Use a "
                "variant without it -- F1 or F2 -- rather than treating a missing "
                "filter as a pass: F3 and F4 without TO are not the paper's variants."
            )

        result.times[name] = time.perf_counter() - started
        if report.cached:
            cached.append(name)
        if not report.feasible:
            result.feasible = False
            result.rejected_by = name
            result.status = report.result.status
            result.cached = tuple(cached)
            return result
        # "the cost J returned by the FINAL filter in F" -- so each pass overwrites,
        # and whatever the last surviving filter reported is what the node keeps.
        result.cost = float(report.result.cost)
        result.status = report.result.status

    result.cached = tuple(cached)
    return result


def mode_is_cached_infeasible(caches: Caches, mode: ContactMode) -> bool:
    """Was this mode already rejected by filter M? Reporting only, no solve."""
    hit = caches.mode_edge._entries.get(mode)
    return hit is not None and not hit.feasible
