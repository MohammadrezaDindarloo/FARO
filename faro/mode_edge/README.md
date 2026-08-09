# `faro/mode_edge` — contact mode and edge feasibility (Eq. 14)

The cheapest filter in FARO's hierarchy, and the first one that actually solves
anything. It answers one question: **is there any configuration in which this set of
contacts holds at once?**

```
min_q  (q - q_nom)^T W (q - q_nom)
s.t.   contact(q, c)  <= 0     (7a), (7b)
       collision(q)   <= 0     (9)
       limits(q)      <= 0     (13a)                                        (14)
```

with `q = (q^r, q^o1, ..., q^ono)` — the robot configuration **and** every movable
object's pose.

## What is deliberately absent

Eq. 14 cites 7a and 7b but **not** 7c or 7d, and 13a but **not** 13b or 13c. That is
the hierarchy of Fig. 3 working as designed: this is an inverse-kinematics problem,
so there are no forces and no time, and friction, torque and velocity have nothing to
act on. Adding them would make the cheap filter expensive *and* make it reject modes
a full TO could realize — the false negatives Table III measures.

`tests/test_mode_edge.py::test_eq_14_uses_only_7a_7b_and_13a` checks this on the row
labels, so it cannot be satisfied by a comment.

## Files

| file | what it is |
|---|---|
| `problem.py` | builds Eq. 14 from a scene and a mode/edge. `RegularizationWeights` is W. |
| `feasibility.py` | the filters M and E of Section III, plus Alg. 1's `K_feas`/`K_infeas` cache. |

The discrete vocabulary lives one level down in `faro/core/modes.py`, and the
symbolic `q` in `faro/scene/symbolic.py`, so KSO and TO can reuse both without
importing anything from here.

## Run it

```bash
python scripts/03_mode_feasibility.py --list      # the ten-entry tour
python scripts/03_mode_feasibility.py             # solve each, see the pose
python scripts/03_mode_feasibility.py --sweep     # all 108 modes of Table IV
```

Current measurement on the ten-entry tour, **all 689 pairs** (the default):

```
10/10 match their predicted verdict     294 s total, 29.4 s per target
  penetration audit (all 689 pairs):    <= 0.46 mm on every converged answer
```

How the mode's own contact pairs are handled turned out to matter more than anything
else, and both obvious options are wrong:

| contact pairs | tour time | worst penetration | |
|---|---|---|---|
| hard `sd >= 0` | 427 s | 5 mm | exact, but every solve runs to the refresh cap |
| dropped | 238 s | **128 mm** | fastest, and lets the forearm into the box |
| **relaxed to `sd >= -1 mm`** | **294 s** | **0.46 mm** | |

Dropping them is the standard advice and it fails here for a specific reason: a patch
is not a body. The hand patch is a 27×30 mm window on the wrist cut face, while
`left_wrist_yaw_link_0` is the whole forearm hull — drop the pair and nothing
constrains the forearm at all. `lift`, which holds the box in mid-air by both hands,
came back feasible with the arm **12.8 cm inside it**. Keeping them hard is exact but
costs 2.7×, because they sit at exactly `sd = 0` and round-off stops the audit ever
calling the answer clean. 1 mm of slack converges and is nowhere near enough for a
forearm to vanish.

With `activation_distance: 0.10` the tour runs in **70.7 s** with identical verdicts.

`--sweep` is what shows the filter doing its job. Measurement on
`box_placement`, all 108 modes of Table IV:

| | contact + limits only | **full Eq. 14** (with Eq. 9) |
|---|---|---|
| feasible | 101 | **106** |
| `Maximum_Iterations_Exceeded` | 4 | **0** |
| `Infeasible_Problem_Detected` | 3 | **2** |
| time per mode | 0.21 s | **27.1 s** |

Read that table carefully, because the obvious reading of it is wrong. Adding Eq. 9
adds constraints, so it can only ever *shrink* the feasible set — 101 → 106 is not
collision making things possible. It is the **false-negative rate falling**: grounding
`q_nom`, raising `max_iter` to 1000 and adding restarts landed together with collision,
and between them they recovered every one of the four `Maximum_Iterations_Exceeded`
verdicts. Those four were false negatives all along, which is what we suspected of them
at the time and can now say with a measurement instead of a suspicion.

The two survivors are both hands on **perpendicular** box faces (`box_front` with
`box_right`, `box_left` with `box_rear`) — the arms would have to wrap a corner.

## The two statuses are not the same answer

Section II-D defines infeasibility as the solver failing to converge, so the verdict
is the solver's status — and that status carries information the bool throws away:

* **`Infeasible_Problem_Detected`** is the stronger claim: Ipopt converged to a
  nonzero minimum of constraint violation. Eq. 14 is **nonconvex**, so this is a
  *local* certificate and not a proof — we watched one mode go from this verdict to
  `Solve_Succeeded` purely by grounding `q_nom`.
* **`Maximum_Iterations_Exceeded`** is a claim about *this solve* only. Section IV-C
  names it as the source of the paper's own false negatives, and in our sweep all four
  such verdicts turned out to be exactly that.

`FeasibilityReport.explain()` says which one you got and why it matters. Collapsing
them would hide the difference between pruning correctly and losing a branch.

## Three things worth knowing before you trust a verdict

**1. Eq. 7a is not written the way the paper writes it.** The default is
`R[0,2] = 0, R[1,2] = 0, R[2,2] >= 0` rather than `log3(R)_{x,y} = 0`. Same feasible
set, same optimum to 1e-6, but the literal form makes Ipopt return
`Invalid_Number_Detected` on the grasp and place modes of this very scene. The
mechanism is **not diagnosed** — see `normal_alignment_rows` and ambiguity #23 for
what was and was not established. `alignment="log3"` selects the paper's form.

**2. The unit-quaternion rows are ours, not the paper's.** Eq. 6 says `q ∈ SE(3)`,
Eq. 14 lists no constraint pinning the quaternion norm, and without one the solver can
shrink a quaternion to cheat the cost instead of moving the robot. Ambiguity #21.

**3. Verdicts were non-deterministic until we pinned the BLAS.** The same mode
returned `Solve_Succeeded` and `Infeasible_Problem_Detected` on different calls in one
process, with a byte-identical NLP — threaded BLAS reductions inside MUMPS, on a
problem near its feasibility boundary. `faro/utils/determinism.py` fixes it and
explains why it has to run before `import numpy`. This matters more than it sounds:
Alg. 1 *caches* verdicts, so non-determinism does not produce noise, it produces a
permanent arbitrary answer.

## Collision (Eq. 9) — what is in the pair set

Every pair, minus three structural exclusions. 39 bodies give **741** raw pairs;
**689** survive.

| excluded | why it is not a tuning choice |
|---|---|
| two geometries on the **same link** | rigidly connected, so `sd` is a constant — the row can never be influenced by `q` |
| **parent–child** links | they touch by construction; that is what a joint is. All 12 pairs interpenetrating at the nominal pose are of this kind |
| **environment–environment** | both static; `sd` does not depend on `q` at all |

Everything else is in: robot self-collision across the tree, robot↔box,
robot↔floor/platform, box↔floor/platform. All three exclusions are flags, so the
literal 741 set can be built and measured.

**The G1's meshes are not convex** — 24 `BVHModelOBBRSS`, plus 8 spheres and 4
cylinders. Section II-C 2 says *"all bodies … are modeled using convex geometries"*,
and Schulman's frozen-witness model genuinely needs it: on a non-convex mesh the
witness pair jumps between concave features, so the frozen constraint describes a
different pocket of geometry than the one the solver is moving into. They are replaced
by their convex hulls.

**Contact pairs are kept.** With the paper's `margin = 0`, Eq. 7a's `p_z = 0` and
Eq. 9's `0 ≤ sd` are compatible at exactly `sd = 0` — which is what a sole on the
floor measures. The consequence is that *coincident witness points are the ordinary
case*, not a corner case, so `query_witness` falls back (coal's own normal, then
centre-to-centre) instead of raising. It used to raise, and one degenerate pair out of
689 aborted the whole solve.

**Eq. 14 is solved as a sequence, not once.** The frozen witnesses are only valid near
where they were taken, so it is solve → re-query → solve again, bounded by
`refresh_iterations`. Only the final solve decides the verdict.

## Two things that only showed up once the answer was audited

**A converged, feasible answer can be 9.3 cm inside the box.** With a fixed three
refresh passes, `lift` returned `Solve_Succeeded` with every one of its 689 collision
rows satisfied — and 9.3 cm of interpenetration. Each row was linearized at the
*previous* iterate, so the convex model was satisfied and the true geometry was not.
The loop now stops only when the configuration has settled **and** the answer is clean
when re-queried against all pairs; `refresh_iterations` is a cap, and reaching it shows
up in `max_penetration` rather than being hidden. Every demo now audits at 0.00000.

**A positive collision margin makes every mode infeasible.** The usual reason to set
`margin` to a few millimetres is to keep the solver off the constraint boundary. It
cannot be done here: a mode that asserts a contact puts that pair at `sd = 0.000000`
exactly, so `margin − sd ≤ 0` with `margin > 0` is unsatisfiable. At `margin = 0.002`
all ten tour entries flip to infeasible. The paper's `margin = 0` is not just its
choice — it is the only one available unless the mode's own contact pairs are excluded
from Eq. 9, which is a package deal, not an independent knob.

## Where the time goes, and how it got 43× cheaper

The ten-demo tour, same verdicts throughout:

```
82.46 s  ->  19.07 s        8.25  ->  1.91 s per check
```

and before that work it was 333 s. The number that mattered was never the mean — it
was the **infeasible** check, because a pruning filter says *no* far more often than
*yes*, and `edge-regrasp` alone was 79% of the tour at 65 s.

Four measurements, in the order they were made:

**1. It is Ipopt, not construction.** One pass on `grasp` splits 3.23 s of solve
against 0.66 s of symbolic building. Rebuilding the NLP every pass is ~17%, so
caching it would buy almost nothing. (An earlier version of this file claimed the
opposite; it was wrong.)

**2. The refresh loop never terminated on its own.** The rule was "settled AND
clean", and the *settled* half never fired — re-linearizing at a new point perturbs
the rows enough to keep `max|dq|` above `1e-4` forever, so the iterate circles the
constraint boundary and the **cap**, not convergence, ended every loop. `lift` used
20 of 20 passes and was already clean at pass 1.

Dropping that half alone made things **slower** (413 s → 548 s): the loop could then
exit on a pass whose solve had *failed* but whose geometry happened to be clean, and
`check` discards a non-converged result and restarts — buying a fresh twenty passes.
The rule that works is **converged AND clean**.

**3. A failed solve returns finite wreckage.** Entries around `1e308` — finite, so an
`isfinite` guard sails past them. Feeding that back as the next linearization point
made Ipopt reject the point *without running* (`Invalid_Number_Detected`, 0
iterations) and return it unchanged, so the loop spun **19 no-op passes**, twice over.
The same coordinates reaching GJK raised `coincident body origins` and **took the
process down** — inside Alg. 1, a lost two-hour run and its entire cache.
`_is_configuration` bounds the magnitude; `tests/test_mode_edge.py` pins it.

**4. A failing pass is not a reason to stop.** `place` reports
`Infeasible_Problem_Detected` on passes 1, 2 and 3 and converges on pass 4. Stopping
at the first such status — which looked obviously right — would have made it a false
negative.

## Deciding an edge without solving it

An edge is the union `c1 ∪ c2`, so an interface that changes partner appears with
**both**. Eq. 7a then needs one flat patch flush against two patches at once — and
when those two are fixed to the same rigid body, that is decidable from geometry:
normals must be parallel *and* the planes must coincide. Three pairs fail on this
scene:

```
left_hand    box_left  + box_front    perpendicular normals
right_hand   box_right + box_rear     perpendicular normals
box_bottom   floor     + tabletop     parallel, planes 0.40 m apart
```

**6176 of 11556 ordered edges (53.4%)** are settled this way, in microseconds instead
of 35–51 s. This is an accelerator, never a relaxation: it returns the same verdict
Eq. 14 returns, for the same reason. Validated on eight sampled caught edges — Eq. 14
agreed eight times out of eight. `contradictory_contacts()` returns the reason, `""`
when nothing is decidable, and an empty string is **not** a feasibility claim.

Only edges are ever tested: Eq. 1 gives a mode exactly one partner per interface, so
a mode cannot contradict itself this way.

## The collision cutoff, swept

`activation_distance: 0.10` — chosen by measurement, not by taste. Every value from
0.05 to 0.50 gives **verdicts identical** to all-pairs; the cost is a U-curve:

| cutoff | per check | vs 689 |
|---|---|---|
| null (689 pairs) | 33.3 s | 1.0× |
| 0.05 | 12.1 s | 2.8× |
| **0.10** | **7.9 s** | **4.2×** |
| 0.20 | 9.6 s | 3.5× |
| 0.30 | 12.7 s | 2.6× |
| 0.50 | 43.1 s | 0.8× |

Both ends cost, for opposite reasons: too few rows and the refresh loop needs more
passes to settle the active set (`grasp` takes 6 at 0.05 against 2 at 0.10); too many
and every solve pays for pairs that were never near mattering. At the reference
answers the closest non-contact pair is **0.6–9.5 mm**, so 0.10 carries ~10× the
headroom the geometry needs, while 0.30 quadruples the rows to capture pairs at a
median distance of 0.43 m.

The active set does **not** thrash: it converges in two passes and stops. Pass 1 of
`grasp` "succeeds" while 283 mm inside something, because those pairs had no rows;
pass 2 adds them and it comes out clean. That is Schulman's procedure working, and
it is why the audit — not a movement test — has to be the stopping criterion.

**Eq. 15 needs the opposite** and reads `kso_activation_distance` instead. acados
requires a fixed parameter-block size, so its pair set is frozen once at `q_init`;
a frozen cutoff measured `QP_Solver_Failed` against `Solve_Succeeded` for the full
set. `SceneCollisionModel.activation_distance_for` is the one place that resolves it.
