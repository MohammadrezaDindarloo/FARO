# `faro/kso` — Kinematic Sequence Optimization (Eq. 15)

The middle filter of Fig. 3. Where Eq. 14 asks *"does a configuration exist for this
mode?"*, Eq. 15 asks it of a whole **sequence** at once, and adds the one constraint
that only makes sense across time.

```
min_{q_0:K}  sum_{s=0}^{K} (q_s - q_nom)^T W (q_s - q_nom)
s.t.  q_0 = q_init,   q_K in Q_goal,
      for s in 1..K :   contact(q_s, c_{s-1} u c_s) <= 0   (7a), (7b)
                        collision(q_s)              <= 0   (9)
                        limits(q_s)                 <= 0   (13a)
      for s in 0..K-1:  for each interface satisfying (16a) or (16b):
                        contact(q_s, q_{s+1}, c_s)   = 0   (8)              (15)
```

with `c_K := c_{K-1}`, so **K modes give K+1 configurations**.

## Files

| file | what it is |
|---|---|
| `problem.py` | Eq. 15 assembled in CasADi — cost, the Eq. 8 no-slip blocks, `q_init`, `Q_goal` |
| `acados_solver.py` | the multi-phase OCP, code generation, and the `.so` cache |
| `feasibility.py` | the filter: verdict, cache, warm start |

## Run it

```bash
conda activate faro
python scripts/04_kso_demo.py -s reach          # in Meshcat
python scripts/04_kso_demo.py --list
python scripts/04_kso_demo.py -s regrasp        # the intended negative
```

## What Eq. 15 adds over Eq. 14, and what it still does not have

**Adds: consistency across time.** Eq. 8 forbids a contact that persists (16a) or is
released (16b) from sliding between adjacent knots. That is the whole difference
between the `KSO` and `M,E` rows of Table II, and it is why the sequence is one NLP
rather than K independent solves.

**Still absent: forces.** No dynamics, no gravity, no balance — Section IV-A measures
this directly, at 70.0% of the TO's constraint types. So `infeasible` here never
means "the robot could not hold this"; it means no sequence of configurations
satisfies the contacts, the no-slip condition, collision and the joint limits at
once. A sequence of head-first dives is perfectly acceptable to Eq. 15.

**The union is not a detail.** Knot `s` is constrained by `c_{s-1} u c_s` — the
*edge*, not the mode. An earlier version used `contact(q_s, c_s)` and was a strictly
weaker filter: it passed sequences whose transitions cannot happen.

## Two solvers, on purpose

Section II-G assigns them: **Eq. 14 → Ipopt, Eq. 15 → acados SQP**, and this module
honours that. The Ipopt KSO path and its outer witness-refresh loop are deleted.

That loop existed only because an interior-point method cannot change its constraint
functions mid-solve. An SQP re-solves a QP every iteration, so carrying the GJK
witness data as `model.p` parameters puts Schulman's sequential convex procedure
*inside* the iterations, where the paper has it. Re-linearizing becomes a `memcpy`
instead of a `gcc` run. Measured on `reach`: one SQP call, 119 ms, all residuals at
machine precision — against up to 20 full NLP solves before.

## The cost that shapes everything: code generation

acados generates and compiles C for each **distinct problem structure**:

```
first build (cold)   ~580 s      once per distinct sequence
build with cache       10.3 s
solve                   0.17 s
```

The `.so` is cached under `third_party/acados_generated/`, keyed by a fingerprint of
everything that affects the generated C — scene, sequence, pair count, collision
settings, solver options. That key is not optional: keying on too little served a
solver compiled for a *different* pair set and reported its answer as this one's.

**Do not interrupt a cold build.** Once you see `rendered solver templates
successfully`, let it finish — acados wipes the directory before regenerating, so an
interrupted build leaves no `.so` and the next run starts over.

**This is the binding constraint on running Alg. 1 with F2.** Every candidate in a
tree search is a different sequence, so at ~580 s each a two-hour budget allows about
twelve KSO calls against the paper's 1415. Table I reports 0.14–1.93 s per KSO, which
is not compatible with regenerating C per sequence — so the paper must be reusing one
compiled structure across sequences. Making the mode assignment a *runtime* input
(constraint bounds set per solve, one solver per horizon length K) is the open item.

## Rejecting a sequence without generating anything

Eq. 15 constrains knot `s` by the edge `c_{s-1} u c_s`, so a sequence containing a
contradictory transition cannot have a solution — and that is decidable from scene
geometry alone. `check()` tests every edge with `contradictory_contacts()` before any
code is generated.

It is worth much more here than in Eq. 14: a miss costs a **580 s compile**, not a
40 s solve. `regrasp` now returns in **0.43 s**.

## Collision: Eq. 15 wants every pair, and Eq. 14 does not

`kso_activation_distance: null` — all 689 pairs — and this is not a preference.
Measured on `reach`:

```
 53 pairs   QP_Solver_Failed   stat=2.18e+00
689 pairs   Solve_Succeeded    stat=1.46e-07   119 ms
```

The reason the two stages disagree is structural. Eq. 14 **re-selects** its pair set
at every refresh pass, which is what makes a cutoff safe there. acados needs a fixed
parameter-block size, so `fixed_pairs` chooses the set **once at `q_init`** and
freezes it. A pair that becomes relevant en route then never gets a row, while the
frozen ones are driven onto their boundaries — HPIPM reports the contradiction as
`MIN_STEP`. Ipopt tolerated it; an SQP does not.

`relax_contact_pairs` must stay **on**, and the two knobs are not independent:
turning it off helps at 53 pairs and is catastrophic at 689 (inequality residual
0.98), because Eq. 7 wants a patch touching while Eq. 9 wants the same bodies apart.
Something has to give where 7 and 9 meet; the paper does not say what, so this is a
**declared deviation**.

## The acados gotcha that cost the most

**The initial stage is a separate constraint.** `con_h_expr` applies from stage 1
onward; stage 0 needs `con_h_expr_0` / `lh_0` / `uh_0`. With `N_list = [1]*K`, phase
0's only stage *is* the initial stage — so setting `model.con_h_expr` there generated
and compiled the rows and then never applied them. **Eq. 8 across the first
transition silently vanished** while acados reported `eq = 2.2e-16`.

Solver status alone said everything was fine. Only the drift table caught it: the
feet slid **521.656 mm** between `q_0` and `q_1`. If you add constraints here, check
the drift, not the residual.

## Warm starts come from the edges, not the modes

`warm_start()` seeds knot `s` from Eq. 14 on `edge_at(sequence, s)`. Seeding from the
mode `c_s` alone hands the SQP a point satisfying the *wrong* contact set, so every
knot arrives violating its own 7a/7b rows and the solver spends its budget on
feasibility restoration — measured as constraints at `1e-16` while stationarity stuck
at 2.18.

This is also what Alg. 1 does anyway: filter E runs on every edge before a sequence
is assembled, so with a cache these answers are free by the time the KSO sees them.

## Open question: `q_init`

Eq. 15 pins `q_0 = q_init` and Eq. 8 then freezes every persisting contact — so
`q_init` decides where the feet are **for the whole sequence**. In FARO it is an
*input*: the robot's actual configuration at the root of the tree search. We currently
synthesize it by solving Eq. 14 on `c_0`, which optimizes toward the nominal pose
knowing nothing about the box to be grasped later, and `reach` is INFEASIBLE as a
result — the feet land 521.7 mm from a stance that can reach.

Note what that does *not* prove: `QP_Solver_Failed` is a local method giving up, not
a certificate. Filter E proves *a* grasping stance exists; whether one exists with
the feet where `q_init` put them is untested.

Declaring `q_init` in the scene YAML is the most paper-faithful fix, and the tree
search settles it naturally — in Alg. 1 the root's configuration is a task input
shared by every KSO in the run.
