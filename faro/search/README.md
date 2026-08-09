# `faro/search` — Feasibility-Guided Tree Search (Algorithm 1, Section III)

Searches the discrete space of contact-mode **sequences**, using the optimization
filters of Section II as pruning. This is the module that turns Eq. 14 and Eq. 15
from "checks you can run" into "a method that finds plans".

## What it implements

| Paper | Here |
|---|---|
| Alg. 1 (the whole loop) | `search.py::search` |
| Eq. 18, cost-based UCT selection | `uct.py::uct_value`, `best_child`, `select` |
| Eq. 19, progressive widening | `uct.py::widening_limit` |
| `PROPOSESUCCESSOR` | `uct.py::propose_successor` |
| `VERIFY`, filters applied left to right | `verify.py::verify` |
| `K_feas` / `K_infeas` | `verify.py::Caches` (wrapping the existing per-filter caches) |
| Eq. 21's variants F1–F4 | `verify.py::VARIANTS` |
| `GOAL(C+)` | `tree.py::Goal` |
| `INITIALIZETREE`, `ADDCHILD` | `tree.py::make_root`, `search.py` line 12 |

## Run it

```bash
conda activate faro
python scripts/05_tree_search.py --dry-run        # UCT only, no solving, ~5 s
python scripts/05_tree_search.py --budget 300     # a real five-minute search
python scripts/05_tree_search.py --budget 7200    # the paper's budget
```

`--dry-run` replaces VERIFY with "everything passes". It exists because Eqs. 18 and
19 are pure arithmetic and worth seeing on their own, before solver verdicts are
mixed in. It should produce a tree that widens with depth and reaches the depth cap.

## What to look at, and it is not the solution count

**The rejections and the cache.** A search that admits everything is enumerating,
not searching. Table II attributes the method's advantage on the hard task directly
to caching:

> "Since contact modes and transitions are repeatedly revisited during search,
> cached infeasibility results allow previously checked queries to be rejected
> without resolving the corresponding optimization problems. This substantially
> reduces redundant exploration and explains the large performance gap between the
> KSO and M,E,KSO variants." — Section IV-B

The per-iteration log marks which verdicts came from cache. Once the mode cache
fills, iterations should get dramatically cheaper — that transition is the effect.

## Three things worth understanding before reading the output

**The cache is a memo, not a phase.** Alg. 1 line 3 initializes `K_feas, K_infeas`
*inside* the search, empty. There is no separate cache-building stage in the paper,
and the KSO is never run on a sequence that failed M or E — that is the filter
ordering, not the cache.

**The three filters see different things.** M gets the proposed mode `c+` alone; E
gets the transition `c(v) → c+`; KSO gets the whole candidate sequence `C+`. So M
and E are local and reusable across the entire tree, while a KSO key is usually
unique. Table II's hard task shows 1415 KSO attempts against 814.6 nodes — nearly
every KSO call is a new sequence, so its cache almost never hits. **The caching win
is entirely in M and E.**

**Default filters are `[M, E]`, which is not one of the paper's variants.** Every
variant in Eq. 21 contains KSO or TO. Our acados KSO regenerates and recompiles C
per distinct sequence (~580 s), which in a two-hour budget allows roughly twelve
calls against the paper's 1415. Until the KSO is reusable across sequences, `[M, E]`
builds the tree and the caches honestly. TO raises `NotImplementedError` rather than
being skipped: F3 without TO is not F3, and a missing filter must never read as a
pass.

## Ambiguities declared here

The paper does not state these; each is a config parameter with a stated default.
See `docs/ambiguities.md` #30–#33.

- **#30 `GOAL`** — never defined. Expressed as a partial terminal mode
  (`box_bottom → tabletop`), because "the box is on the platform" is a contact
  condition, not a configuration. `released` optionally demands the hands be free.
- **#31 successor set** — never defined. Default `all` (108 modes, matching the
  paper's stated branching factor); `single` restricts to one-interface changes.
- **#32 visit-count updates** — never stated. Incremented along the selection path.
  Without it Eq. 19 never widens and Eq. 18's exploration never decays.
- **#33 `J` is not backed up** — Section III assigns a node "the cost J returned by
  the final filter", so `J` describes the node itself and there is no backup step.
  That is the literal reading and it is unusual for a UCT; a deep, cheap subtree is
  invisible to its ancestors' scores.

## Reproducibility is load-bearing here

Alg. 1 **caches verdicts**, so a filter that flips under thread scheduling does not
produce occasional noise — it produces a permanent, arbitrary answer, because
whichever verdict arrived first is kept and reused for the rest of the run. This is
why `faro/utils/determinism.py` pins the BLAS and why every entry point imports
`faro` before numpy. `seed` controls `PROPOSESUCCESSOR`'s shuffle and nothing else,
matching Section IV-B's "different seeds controlling the random successor selection".

## Tests

`tests/test_search.py` — 36 tests, almost all solver-free via `verify_fn` injection.
They pin Eq. 19's values, Eq. 18's *direction* (cost-based: argmin, minus the bonus
— the opposite of textbook UCT), the branching factor of 108, the depth cap, and
that TO raises rather than passing.
