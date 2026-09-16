# Deviation register — every place FARO differs from the paper

The point of this file is that "are we aligned?" should be answerable by reading one
page, not by grepping docstrings. Every row is either **FORCED** (a consequence of a
substitution we had to make) or **CHOSEN** (we decided it), and chosen deviations
should be justified or removed.

Status as of 2026-09-16: Milestones 0–3 done, Milestones 4 and 5 in progress, TO and LLM not started.

## 1. Aligned — verified, not assumed

| Paper | Where | Checked by |
|---|---|---|
| Eq. 1/2 contact modes, `∃!b`, consistency | `faro/core/modes.py` | `tests/test_modes.py` |
| Table IV interfaces, branching factor 108 | `configs/scenes/box_placement.yaml` | `Scene.branching_factor()` |
| Eq. 7a, 7b patch contact | `faro/constraints/contact.py` | `tests/test_milestone2_*` |
| Eq. 8 no-slip, Eq. 16a/16b gating | `faro/constraints/no_slip.py` | `--drift`, 0.000 mm |
| Eq. 9 Schulman signed distance, coal/GJK | `faro/constraints/collision.py` | penetration audit |
| Eq. 13a position limits only in 14/15 | `faro/constraints/limits.py` | row-label test |
| Eq. 14 objective, constraint set | `faro/mode_edge/problem.py` | `test_eq_14_uses_only_7a_7b_and_13a` |
| **Eq. 14 solved with Ipopt** (§II-G) | `configs/solvers/ipopt_mode_edge.yaml` | — |
| Eq. 15: K+1 configs, `c_K := c_{K-1}` | `faro/kso/problem.py` | `n_knots`, demo output |
| Eq. 15: `contact(q_s, c_{s-1} ∪ c_s)` | `edge_at()` | demo output |
| Eq. 15: `q_0 = q_init`, `q_K ∈ Q_goal` | `build_problem` | 43 anchor rows |
| Eq. 15: s∈{1..K} constraints, s∈{0..K-1} Eq. 8 | `build_problem` | demo output |
| Eq. 15 objective `Σ_{s=0}^{K}` | `SequenceWeights` | — |
| Alg. 1 caches `K_feas`/`K_infeas` | `FeasibilityCache`, `KSOCache` | — |
| **Eq. 15 solved with acados SQP** (§II-G) | `faro/kso/acados_solver.py` | `reach`: 119 ms, residuals ~1e-16 |
| Eq. 15 witnesses re-linearized inside the SQP (Schulman, as §II-C 2 cites) | `model.p` | no outer refresh loop |
| Alg. 1 loop, Eq. 18 argmin, Eq. 19, k=1, α=0.5, C=3 | `faro/search/` | `tests/test_search.py` |
| Eq. 21 variants F1–F4, filters applied left to right, stop at first failure | `faro/search/verify.py` | `tests/test_search.py` |
| §IV-B: depth cap 5 switches, 2 h budget, 108 successors | `configs/search/default.yaml` | — |
| Pinocchio + CasADi + coal stack (§II-G) | throughout | — |

## 2. FORCED deviations

### 2a. Resolved by moving the KSO to acados (was F1–F5)

The previous version of this file listed F1–F5 as consequences of solving Eq. 15
with Ipopt. The KSO now runs on acados SQP, so their status is:

| # | Was | Now |
|---|---|---|
| F1 | KSO solved with Ipopt | **gone.** acados SQP, as §II-G says. Ipopt KSO code is deleted. (`configs/solvers/ipopt_kso.yaml` is still in the repo, but nothing loads it.) |
| F2 | Outer collision-refresh loop (up to 20 NLP solves) | **gone for Eq. 15.** Witnesses are `model.p`; `solve(refreshes=1)`. Eq. 14 still has its loop, because it is still on Ipopt. |
| F3 | Column forms of Eq. 7a / Eq. 8 instead of `log3` | **still present**, see F6 |
| F4 | Random restarts | **still present** (`restarts=2`) in both Eq. 14 and Eq. 15 |
| F5 | No multiple-shooting transcription | **gone.** A multi-phase OCP with one phase per knot |

### 2b. Current forced deviations

| # | Deviation | Why it exists |
|---|---|---|
| F6 | **Column forms** of Eq. 7a and the Eq. 8 yaw row, not `log3` | `cpin.log3` gives a NaN in its exact Hessian, and the acados KSO uses `hessian_approx=EXACT` (see F8). `alignment="log3"` / `yaw="log3"` are still available. |
| F7 | **Control `u_k = q_{s+1} − q_s`** carrying no cost | An acados path constraint only sees `(x_k, u_k)`, and Eq. 8 couples adjacent knots. This is an exact linear change of variables. It is not a model. |
| F8 | **EXACT Hessian + `CONVEXIFY` regularization**, not Gauss–Newton | With no cost on `u`, a GN Hessian has a zero block and HPIPM stops at iteration 1. Adding a cost on `u` would change Eq. 15, so we don't. |
| F9 | **Eq. 14 refresh loop** | Ipopt cannot change its constraint functions mid-solve. §II-G does specify Ipopt for Eq. 14, so this is the cost of following the paper. |
| F10 | **TO on Ipopt instead of Hippo** (planned) | Hippo is not public |

## 3. CHOSEN deviations — our decisions, each defensible but ours

| # | Deviation | Default | Note |
|---|---|---|---|
| C1 | `activation_distance` cutoff on Eq. 9 | Eq. 14: 0.10 · Eq. 15: `null` (all 689 pairs) | Schulman's, cited by the paper; threshold not given (#19b) |
| C2 | `relax_contact_pairs`: mode-held pairs get `sd ≥ −1 mm` | on | paper says `0 ≤ sd` flat. Measured: dropping them lets a forearm 12.8 cm into the box; keeping them hard costs 2.7× |
| C3 | `include_pairs` / `always_active` pair-class policy | all / none | paper never defines the pair set (#19) |
| C4 | unit-quaternion rows | on | Eq. 6 says `q ∈ SE(3)`; Eq. 14/15 list no norm constraint (#21) |
| C5 | `W` values, `joint_groups` | base_orientation 10 | paper names W, never gives it (#22) |
| C6 | `q_nom` grounded on the soles | — | paper does not define the nominal pose |
| C7 | `smoothness` term in the KSO cost | **0.0** | NOT in Eq. 15. Inert at 0.0; delete if it is never used |
| C10 | Eq. 15 collision pair set frozen once at `q_init` (`fixed_pairs`) | — | acados needs a fixed parameter size. It is only safe with the full pair set (C1). |
| C11 | Contradictory edges rejected from geometry, before any solve | on | This is a fact that follows from Eq. 7a, not a relaxation, but the paper describes no such shortcut. |
| C12 | Tree-search default filters `[M, E]` | on | **Not an Eq. 21 variant.** Temporary until the KSO stops needing per-sequence code generation. |
| C13 | `GOAL` = partial terminal mode; successor set = all 108; visits incremented along the path; `J` not backed up | — | ambiguities #30–#33 |
| C8 | box size, platform height, μ, μ_r | — | none given by the paper (#6, #13) |
| C9 | single-threaded BLAS pin | on | ours; verdicts were non-deterministic without it |

## 4. Known-unfaithful or broken, to fix

| Deviation | Why it matters |
|---|---|
| **`q_init` is not a task input.** When `q_init=None`, the KSO uses knot 0 of the initial guess (the nominal pose). | The paper treats `q_init` as the task's initial state. With our choice, Eq. 8 fixes the feet where the nominal-pose solve put them, and `reach` becomes infeasible. The user still has to decide this; the options are in `faro/kso/README.md`. |
| **`q_init` is missing from the acados build fingerprint and the `_BUILT` key**, although it is compiled in as `constraints.x0` | A second call on the same sequence with a different `q_init` can silently reuse the first call's solver. This is a correctness bug, not a documentation gap. |
| **Per-sequence code generation**, ~580 s cold | Table I reports 0.14–1.93 s per KSO, so the paper must reuse one compiled structure across sequences. |
| KSO results with restarts: the `.so` is built with the first attempt's `q_init` | Restarts perturb knot 0 of the guess, but the solver keeps the `q_init` from the first attempt |
| `Q_goal` is passed as a partial mode (`goal=`) that adds contact rows at the last knot | The paper names `Q_goal` without defining it. Ours is a reasonable reading, but it is still a guess. |
| G1 processes still show ~177% CPU despite the BLAS pin | C9 may not be covering everything, and Milestone 3 proved threading can flip a verdict |

## 5. Not yet built

- **Eq. 17 TO** (Milestone 6): Eq. 10/11 dynamics, Eq. 12, 13b, 13c and Eq. 7c/7d
  friction exist as constraint blocks (Milestone 2), but nothing assembles them into an
  NLP. The time scaling `T̄_s`, 20 knots per mode (§IV-A) and `w_T` are also missing.
- **Alg. 1 line 17** `TRAJECTORYOPT` on goal-reaching sequences (it depends on the TO).
- **LLM sampling** (§IV-C, Milestone 7).
- **Evaluation harnesses:** Table I (constraint coverage, decision-variable reduction),
  Table II (5 seeds × 2 scenes × 4 variants), Table III (FNR/FPR/speedup), and the
  hard scene of Fig. 4.

## The one-line summary

The **formulations** of Eqs. 14 and 15 match the paper and are tested, and each is
solved with the solver §II-G names. What stops a run like the paper's is **cost, not
correctness**: the KSO regenerates C for every sequence. The open **correctness**
items are how `q_init` is chosen and the gap in the cache key.
