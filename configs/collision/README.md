# `configs/collision` — swappable Eq. 9 pair policies

Each file here is a `collision:` block that overrides the one in the scene config.
`scripts/03_mode_feasibility.py --scenarios a b c` runs the tour under each and puts
the results side by side.

Two independent knobs, and they are NOT the same thing:

| key | decides | is a |
|---|---|---|
| `include_pairs` | which pair classes exist **at all** | modelling decision |
| `always_active` | which classes ignore `activation_distance` | safety decision |
| `activation_distance` | how far apart a pair may be and still get a row | speed knob |

A class dropped from `include_pairs` is genuinely unwatched — the audit does not see
it either, because the audit measures the pairs in the model. A class in
`always_active` is watched at every pass no matter how far away it is.

Selector vocabulary (`faro/scene/collision.py::select_pairs`):

* a **group** — `robot`, `box`, `floor`, `tabletop` — every pair touching it
* a **class** — `robot-box`, either order — exactly that class
* `all`

Unknown selectors raise, listing what exists. A typo that silently selected nothing
would look identical to a working config while disabling the protection it was added
to provide.

On `box_placement` the classes are: `robot-robot` 579, `box-robot` 36, `floor-robot`
36, `robot-tabletop` 36, `box-floor` 1, `box-tabletop` 1 — 689 total.

## Measured

Ten-target tour, `box_placement`, `base_orientation: 10`, current joint groups.
`pen` is the worst interpenetration audited over the pairs **in that model**;
`lean` is torso pitch, the only readout of pose quality Eq. 14 offers.

| scenario | pairs | always-on | rows | worst pen | `grasp` lean | total |
|---|---|---|---|---|---|---|
| `all_pairs` | 689 | 0 | 689 | **0.00 mm** | 60.3° | 277.2 s |
| `cutoff` | 689 | 0 | 48–80 | **0.00 mm** | **41.0°** | **67.5 s** |
| `task_always_on` | 689 | 74 | 111–141 | 5.44 mm | 49.4° | 105.8 s |
| `task_and_floor` | 689 | 110 | 135–154 | 0.01 mm | 49.4° | 88.3 s |
| `no_self_collision` | 72 | 72 | 72 | 0.07 mm | 49.4° | 111.0 s |
| `task_only_full` | 72 | 0 | 72 | 0.07 mm | 49.4° | 109.4 s |

**`all_pairs` is dominated.** It is 4.1x slower than `cutoff`, audits no cleaner, and
returns a worse pose (60.3 deg of lean against 41.0, and both knees pinned at their
hyperextension limit). The complete row set is not the conservative choice it looks
like — it is the same feasible set reached by a worse path.

Four things in this table are worth more than the numbers.

**`no_self_collision` and `task_only_full` are identical on all ten targets.** They
must be: when `always_active` covers every pair, the cutoff cannot do anything. That
is a semantic check, not a coincidence, and it is pinned by
`test_exempting_every_pair_is_the_same_as_no_cutoff`.

**Adding rows made the audit worse, then better.** `task_always_on` introduces 3.59 mm
on `place` and 5.44 mm on `front-rear` where plain `cutoff` had none; adding the floor
on top removes them again. Rows do not monotonically buy safety — Eq. 14 is nonconvex,
so changing the row set moves which local minimum Ipopt reaches and whether the refresh
loop settles inside its cap. Both dirty entries ran long (7.2 s, 9.4 s), which is the
loop failing to converge rather than the constraint being violated by the solve.

**Dropping 579 self-collision pairs did not make it faster.** `no_self_collision` is
*slower* than `cutoff` (111 s vs 67.5 s). Time here is Ipopt iterating, not row count;
`edge-regrasp` alone is 44–149 s in every policy because it is infeasible and burns
every restart. Row count is a bad proxy for cost.

**One infeasible target sets the budget.** `edge-regrasp` is 54% of `all_pairs`' total
and 66% of `no_self_collision`'s. Whatever the pair policy, the cost of this filter is
dominated by the modes it *rejects* — which is worth knowing before Alg. 1, where
rejection is the common case and the whole point.

### If you want one recommendation

`task_and_floor`. Its always-on set is exactly the 110 non-self-collision pairs, so the
cutoff applies only to `robot-robot` — the one class where its assumption (a body
cannot travel far in one step) actually holds, since robot links are bounded by
kinematics and a movable object's pose is a free decision variable. That is a
principled split rather than a tuned one, and it audits clean at 88 s.

`cutoff` is faster and also audits clean on this tour. The case for paying the extra
31% is that the tour is ten hand-picked targets, and `cutoff`'s clean audit is partly
which local minima it happened to reach — not a property it is structurally guaranteed
to keep across all 108 modes.
