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

## Cost, and the knob you may want

Every pair gets a row by default (`activation_distance: null`), which is the slow and
complete option. Schulman's activation distance is implemented and measured: at
**0.10 m** only 53 of 689 pairs are active at the nominal pose, giving a ~12× cut in
rows, **6–20× faster**, and **identical verdicts** on every entry compared.

Worth knowing before deciding: Table II's own counts bound this filter much harder than
the two-hour budget does. On the hard task the `M,E,KSO` run spends ~850 s on 1415 KSO
attempts and ~5150 s on 79.6 TO attempts, leaving ~1200 s for at least 2830 mode and
edge checks — **under ~0.4 s each**. Whatever the paper is doing, it is not spending
tens of seconds per mode.
