# KSO on acados SQP — design

Section II-G is explicit and both halves matter:

> "Both (15) and (17) are implemented using a direct multiple-shooting transcription.
>  ... The KSO is solved with the SQP solver in acados [26]."

## What the first attempt got wrong

It generated and compiled new C **every refresh pass**, because the collision rows were
baked into `con_h_expr` as constants. That is why it looked expensive and why the
outer refresh loop seemed unavoidable.

## The fix: witness data as acados PARAMETERS

Eq. 9 is `0 <= n . (T_A(x) p_A - T_B(x) p_B)`. The three quantities GJK returns --
`p_A`, `p_B`, `n` -- are **constants of the linearization, not of the problem**. In
acados they belong in `model.p`, not in the expression tree:

```
model.p = vertcat(*[p_A_i, p_B_i, n_i for each active pair])     # 9 numbers per pair
model.con_h_expr = <Eq. 9 written against those symbols>
```

Consequences, and they are the whole point:

* **C is generated ONCE per sequence shape.** Re-linearizing becomes
  `solver.set(k, "p", witness_values)` -- a memcpy, not a `gcc` invocation.
* **The outer refresh loop collapses into the SQP.** Schulman's method IS sequential
  convex optimization: each SQP iteration already re-solves a QP at the current
  iterate. With the witnesses as parameters they can be refreshed between SQP
  iterations rather than between full solves. That removes the 9-20 complete NLP
  solves per KSO that make the Ipopt path ~100x slower than Table I.
* The pair set must be FIXED per knot for the parameter vector to have a fixed size.
  With `activation_distance` the active set changes as the robot moves, so either the
  cutoff is disabled for the acados path or the parameter block is sized to the
  full pair set and inactive pairs are given a trivially-satisfied witness.

## Structure

One phase per knot (`AcadosMultiphaseOcp`, `N_list=[1]*K`), because Eq. 15's knots
carry different constraints: s=0 has none (it is pinned by `q_0 = q_init`) and each
s in 1..K carries `c_{s-1} u c_s`, whose pair count varies along the sequence.

| acados | Eq. 15 |
|---|---|
| phase/stage `k` | knot `s` |
| `x_k` | `q_s` |
| `u_k` | `q_{s+1} - q_s` (encoding only; Eq. 15 has NO control) |
| `x_{k+1} = x_k + u_k` | exact, linear -- a change of variables, not a model |
| `constraints.x0` on phase 0 | `q_0 = q_init` |
| `con_h_expr` at k | 7a, 7b on `c_{s-1} u c_s`; 9; 13a; unit-quat; **Eq. 8** via `x_k + u_k` |
| `con_h_expr_e` | knot K, plus `q_K in Q_goal` |
| `cost LINEAR_LS` | `(q_s - q_nom)^T W (q_s - q_nom)` |

Eq. 8 is why `u` exists at all: an acados path constraint sees `(x_k, u_k)` and never
`x_{k+1}`, so coupling `q_s` to `q_{s+1}` needs the increment in scope.

## The open problem, stated honestly

`u` carries no cost, because Eq. 15 has no cost on it. GAUSS_NEWTON then has a zero
Hessian block in those directions. Measured behaviour of the first attempt:

| setting | result |
|---|---|
| `SQP` + GAUSS_NEWTON | `QP_Solver_Failed` (ACADOS_MINSTEP) at iteration 1 |
| `SQP_WITH_FEASIBLE_QP` | iterates, then diverges: 1237 iters, base at z=5.8 m |
| `SQP` + MERIT_BACKTRACKING / FUNNEL_L1PEN | iterates, hits the cap |

Adding a cost on `u` fixes the conditioning and is NOT allowed -- it changes Eq. 15.

Two legitimate routes that do not touch the formulation:

1. **Warm start.** Section IV-C: *"The few observed false negatives were primarily
   associated with poor initialization or insufficient solver iterations in the KSO."*
   `faro.kso.feasibility.warm_start` already stacks per-mode Eq. 14 answers. Ipopt
   tolerates the cold start from `q_nom`; acados evidently does not. **Try this first
   -- it is one line and the paper points straight at it.**
2. **`EXACT` Hessian instead of GAUSS_NEWTON.** The cost is quadratic and the
   curvature then comes from the constraints too. Note this reopens the `cpin.log3`
   NaN (ambiguity #23), so it requires `yaw="column"` and `alignment="column"`.

## Order of work

1. witness data -> `model.p`
2. warm start from Eq. 14
3. verify `reach` and `pick` match the Ipopt answers to ~1e-6
4. only then delete the Ipopt KSO path

Step 4 is last on purpose: until the acados path solves, deleting the Ipopt one leaves
no working KSO. `faro/mode_edge/` is untouched by all of this -- Eq. 14 stays on Ipopt,
which is what the paper specifies.
