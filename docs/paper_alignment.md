# Deviation register — every place FARO differs from the paper

The point of this file is that "are we aligned?" should be answerable by reading one
page, not by grepping docstrings. Every row is either **FORCED** (a consequence of a
substitution we had to make) or **CHOSEN** (we decided it), and chosen deviations
should be justified or removed.

Status as of Milestone 4.

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
| Pinocchio + CasADi + coal stack (§II-G) | throughout | — |

## 2. FORCED deviations — consequences of the Ipopt substitution

**These are one cluster with one cause.** §II-G specifies acados SQP for the KSO; we
use Ipopt. Everything below follows from that, and moving the KSO to acados is
expected to remove all four at once — which is the real argument for doing it.

| # | Deviation | Why it exists | Removed by acados? |
|---|---|---|---|
| F1 | **KSO solved with Ipopt, not acados SQP** | acados attempt diverges; see `kso_acados_design.md` | — (this IS the deviation) |
| F2 | **Outer collision-refresh loop** (up to 20 full NLP solves) | Eq. 9 freezes GJK witnesses and must be re-linearized. In an SQP that happens per iteration for free; interior-point wants fixed constraint functions. **The paper describes no such loop.** | YES — witnesses become `model.p` |
| F3 | **Column forms for Eq. 7a and Eq. 8** instead of the paper's `log3` | `cpin.log3`'s exact Hessian returns NaN; only matters with `hessian_approximation: exact`. Measured: log3 solves fine under `limited-memory`. | YES if GAUSS_NEWTON |
| F4 | **Random restarts** on failure | escapes the `R_zz = ±1` branch trap that the column form of 7a introduces | YES — follows from F3 |
| F5 | **Direct multiple-shooting transcription** (§II-G) not used for Eq. 15 | our Ipopt path builds one flat NLP | YES — it is acados' native structure |

## 3. CHOSEN deviations — our decisions, each defensible but ours

| # | Deviation | Default | Note |
|---|---|---|---|
| C1 | `activation_distance` cutoff on Eq. 9 | 0.10 | Schulman's, cited by the paper; threshold not given (#19b) |
| C2 | `relax_contact_pairs`: mode-held pairs get `sd ≥ −1 mm` | on | paper says `0 ≤ sd` flat. Measured: dropping them lets a forearm 12.8 cm into the box; keeping them hard costs 2.7× |
| C3 | `include_pairs` / `always_active` pair-class policy | all / none | paper never defines the pair set (#19) |
| C4 | unit-quaternion rows | on | Eq. 6 says `q ∈ SE(3)`; Eq. 14/15 list no norm constraint (#21) |
| C5 | `W` values, `joint_groups` | base_orientation 10 | paper names W, never gives it (#22) |
| C6 | `q_nom` grounded on the soles | — | paper does not define the nominal pose |
| C7 | `smoothness` term in the KSO cost | **0.0** | NOT in Eq. 15. Inert at 0.0; delete if it is never used |
| C8 | box size, platform height, μ, μ_r | — | none given by the paper (#6, #13) |
| C9 | single-threaded BLAS pin | on | ours; verdicts were non-deterministic without it |

## 4. Known-unfaithful, to fix

| Deviation | Why it matters |
|---|---|
| `Q_goal` is inferred as "terminal mode's own contacts" | paper names it and never defines it; ours is a guess and should be a config object |
| `q_init` defaults to `q_nom` | the paper's `q_init` is the task's initial state; these coincide only by accident |
| G1 processes still show ~177% CPU despite the BLAS pin | C9 may not be covering everything, and Milestone 3 proved threading can flip a verdict |

## 5. Not yet built

Eq. 17 (TO, Hippo — not public), Alg. 1 tree search, LLM sampling, Eq. 10/11
dynamics, Eq. 12/13b/13c, Eq. 7c/7d friction (declared in config, unused).

## The one-line summary

The **formulation** of Eqs. 14 and 15 is faithful and tested. The **solver** for Eq. 15
is not, and five of our deviations (F1–F5) are that single substitution wearing
different hats.
