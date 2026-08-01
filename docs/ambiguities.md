# Paper ambiguities and our chosen defaults

The FARO paper omits many implementation details. This file is the running log of
every gap we hit, what we chose instead, and where the knob lives so it can be
tuned later.

Format: one entry per ambiguity, added as we encounter it.

| # | Paper ref | What's missing | Our default | Config knob |
|---|---|---|---|---|
| 1 | whole paper | **The robot is never named.** No model, no DoF count, no simulator anywhere in the text (verified against the full PDF). | **Unitree G1, 29 DoF** (`g1_29dof_rev_1_0.urdf`), box object, MuJoCo. Inferred via ref [6] DynaRetarget — see `paper_notes.md` §1 for the evidence chain. | `configs/robots/` |
| 2 | §II-G | Paper uses acados (KSO) + Hippo (TO); Hippo is closed-source. | Ipopt for all three stages. Mode/edge matches the paper exactly; KSO/TO solve times will differ (interior-point vs SQP) — compare feasibility verdicts, not wall-clock. | `configs/solvers/` |
| 3 | §II-C 1 | Friction coefficients μ (linear) and μ_t (torsional) are never given. | To be set at Milestone 2; note the paper specifies a **pyramidal** friction approximation, so the constraint stays linear in force. | `configs/scenes/` |
| 4 | §II-C 1 | Patch half-extents `h` for hands/feet/box faces are never given. | **Feet: derived from the URDF, not guessed** — the G1's foot collision model is 4 spheres (r=0.005) at x∈{-0.05,+0.12}, y∈{-0.025,+0.03}, z=-0.03, giving a sole rectangle `h=(0.085, 0.03)` centred at (0.035, 0, -0.035). Box faces follow from the box size. | `configs/scenes/` |
| 5 | §II-D (Eq. 14) | The nominal pose that Eq. 14 regularizes toward is never given, yet it determines *which* solution the NLP finds. | A lightly crouched stance with the leg pitch chain summing to ~0 so the soles stay flat. | `configs/robots/g1.yaml` → `nominal_configuration` |
| 6 | §IV-B | Box dimensions and mass are never given. | 0.30 m cube, 2.0 kg, starting on the floor at x=0.45. Width chosen so the ±y faces sit inside the G1's hand span. | `configs/scenes/` → `objects.box` |
| 7 | §II-C 1 | Hand patch geometry — where the contact patch on the arm actually is. | **RESOLVED — the hand is cut off at the wrist and the patch sits on the cut face** (see #7b). Geometry is then *measured, not authored*: `*_hand_palm_joint` bolts the hand on at x = 0.0415 in `*_wrist_yaw_link`, so that plane IS the cut, and the collision mesh's cross-section there gives `h = (0.0269, 0.0301)` centred at (0.0415, ±0.003, 0). Normal along **+x of the wrist-yaw link** (out along the forearm), `rpy_deg [90, 0, 90]`. | `configs/scenes/` → `patches.left_hand_patch` |
| 7b | Fig. 1, 4, 5 | **The paper's end effector is not the G1's hand.** Its figures show the arms ending in a flat stub with the rectangular patch on it. The model is never named in the text. | **RESOLVED — the hand is removed at the wrist and replaced by the patch.** §II-C 1 models every end-effector as a *rectangular patch* ("all end-effectors … are modeled as rectangular patches"), reaffirmed in the Conclusion ("assumes planar contact interfaces"), so the hand mesh was never in the math. Consequence that IS substantive: the patch normal runs **along the forearm**, not sideways out of a palm, so the robot presses the box's ±y faces between its two forearms. Costs 3.4 cm of reach per arm (shoulder→patch 0.3492 → 0.3150 m). The URDF is unchanged — we keep `g1_29dof_rev_1_0` and simply do not use the `*_rubber_hand` link for contact. | `configs/scenes/` → `patches.left_hand_patch` |
| 7c | §II-C 2 (Eq. 9) | **The hand geometry conflicts with a wrist-mounted patch.** The patch sits on the wrist cut face (x = 0.0415), but the `*_rubber_hand` link extends to x = 0.1318 — 0.09 m PAST the contact point, straight through whatever the robot touches. | **RESOLVED — the hand links are removed.** Not hidden: a viewer-only fix would leave 0.170 kg × 2 in Eq. 10's centroidal dynamics. They are **commented out, not deleted**, in a generated URDF we own (`assets://robots/g1_description/urdf/g1_29dof_faro.urdf`, built by `python -m faro.robots.urdf_surgery`), with an in-file note; uncommenting restores the full hand. Both are leaf links on FIXED joints, so nq/nv and every joint index are unchanged — total mass 33.341 → **33.001 kg**. Still open for Milestone 3: adding convex collision geometry for the wrist patch itself. | `configs/robots/g1.yaml` → `urdf` |
| 8 | §II-C 1 | **Patch normal sign convention is never stated.** | **RESOLVED at Milestone 2.** Eq. 7a's `log₃(R)ₓ,ᵧ=0` forces a pure yaw, so the paper implicitly assumes **parallel** z-axes in contact. We keep uniform **outward** normals (the paper's convention is not uniform — Eq. 7b's containment makes a box face `b` for a hand but the box bottom `a` on the floor) and apply one `Rx(π)` flip in `relative_patch_transform`. `flip_b=False` selects the paper's convention. | `faro/constraints/frames.py` |
| 10 | §III | UCT exploration constant, progressive-widening exponent. | **NOT AMBIGUOUS** — the paper states them: `k = 1.0`, `α = 0.5`, `C = 3`, with `M(v) = max{1, kN(v)^α}` (Eq. 19) and Eq. 18 an arg**min** over cost `J`. | `configs/tasks/` (Milestone 6) |
| 11 | §II-C 2 (Eq. 9) | Safety margin for the collision constraint. | Paper writes `0 ≤ sd`, i.e. **margin 0**. We default to 0.0 to match, exposed as a parameter — a few mm keeps Ipopt off the constraint boundary and usually helps convergence. | `collision_avoidance(margin=…)` |
| 12 | §II-C 2 | How often the frozen GJK witness points/normal are refreshed. Not stated. | Witness data is frozen per linearization point (Schulman et al.); refresh cadence becomes a solver-loop parameter at Milestones 3–5. **Now quantified:** `9-drift` measures the cost of *not* refreshing — 0.5 rad of box tilt makes the frozen model report 0.038 m of clearance while the true signed distance is −0.034 m. Under pure translation of a flat face the error is exactly zero, so the cadence must be driven by **rotation**, not by distance travelled. | `faro/constraints/collision.py` |
| 13 | §II-C 1 | Friction coefficients μ (linear) and μ_r (torsional) are never given. | To set at Milestone 3 with the first real solve. Friction is **pyramidal** per the paper, so the constraint stays linear in force. | `configs/scenes/` |
| 14 | §II-C 4 | `v_τ,max` for the Eq. 13c torque-speed envelope is not given, and the G1 URDF has no torque-speed data. | Take `τ_max` from the URDF effort limits; `v_τ,max` from the URDF velocity limits as a first estimate. | `configs/robots/` |
| 9 | §IV-B | Platform ("tabletop") dimensions and height. | 0.60×0.60 m at z=0.40, at x=0.9. | `configs/scenes/` → `patches.tabletop` |
| 15 | §II-C 1 (Eq. 7d) | **Which frame Eq. 7d's moment lives in — the paper contradicts itself.** The prose says p, R, f and κ are all "expressed in frame b", but 7d bounds the moment by `^a ξ_a` — a's half-extents in **a's own** frame, and the superscript is deliberate since 7b writes `^b ξ_a`. The two readings differ as soon as the patches differ by a yaw, which **7a explicitly leaves free**. | **We follow `^a ξ_a`**: the moment is transported into patch a's frame (`κ_a = Rᵀ(κ_b − p × f)`) before the CoP rows are formed. 7d is a centre-of-pressure bound and the CoP must lie inside *patch a*; read literally in frame b it instead confines the CoP to a rectangle of a's **dimensions** but b's **orientation**. Measured on a G1 sole (85×30 mm) at 45°: the frame-b reading rejects a CoP well inside the foot **and** accepts one 3× outside it. Eq. 7c is deliberately **not** transported — a friction *pyramid* is not rotationally symmetric, so it is only defined once a frame is chosen, and the paper's choice is b (the support surface). | `contact_wrench(R_rel=…, p_rel=…)` |
| 17 | §II-C 3 (Eq. 11b) | Object inertia values are never given, and a hand-picked diagonal is easy to make **non-physical**. | Derive the probe body's inertia from its **dimensions** (`m/12·[b²+c², a²+c², a²+b²]`), never type it in. The previous hand-picked `diag(0.10, 0.50, 0.90)` violates the triangle inequality `I_x + I_y ≥ I_z` (0.60 < 0.90) — no rigid body has that inertia, so the demo was evaluating a correct formula on an impossible object. Enforced by a test. | `faro/scenarios/constraint_demos.py` → `_SPIN_SIZE` |
| 16 | §II-C 3 | **Eqs. 10b and 11b use opposite 6-vector orderings**, and both are correct. 10b is written `[linear; angular]` (matching Pinocchio's `Ag`, `Force`, and frame Jacobians); 11b follows Lynch & Park and is `[angular; linear]` (`G = diag(I, m·I₃)`, `ad_V = [[ω] 0; [v] [ω]]`). | **Keep both**, since rewriting either departs from its source. The boundary is explicit instead: a contact wrench crosses it (entering 10b and 11b with opposite signs) and must go through `swap_spatial_ordering`. Blocks swapped by accident give a plausible answer with torques where forces belong. | `faro/constraints/frames.py` |

## Environment gotchas discovered (not paper ambiguities, but worth remembering)

- **CasADi reserves the Function names `jac`, `hess`, `null`.** Naming a
  `ca.Function` `"jac"` fails with a confusing *"Function name is not valid"*.
  Use `dfk`, `d_something`, etc.
- **`MUJOCO_GL` must be set before the first `import mujoco`.** MuJoCo caches its
  GL backend at import; setting the variable afterwards has no effect. On this
  cluster `MUJOCO_GL=egl` works (A40 present).
- **`example-robot-data` 5.x is data-only** — URDFs/meshes under
  `$CONDA_PREFIX/share/example-robot-data/robots/`, no Python module. See SETUP.md §4b.
- **Floating base:** `nq - nv == 1` is the signature of a quaternion-parameterised
  free-flyer root. Useful assertion to catch an accidental fixed-base URDF load.

## Already known, to be resolved in later milestones

These are flagged now because they are visible from the paper text alone, and will
each get a full entry when the corresponding milestone is built.

- **Eq. 7 (patch-to-patch contact)** — patch half-extents, and whether normal
  alignment uses the full `log(R)` or only its tangential components.
- **Eq. 9 (collision)** — the safety margin, and how the GJK signed distance is
  smoothed to stay differentiable near contact.
- **Eq. 14 (mode/edge)** — the nominal-pose regularization weight, and what the
  nominal pose actually is.
- **Eq. 15 (KSO)** — cost weights and the initialization strategy across the K+1
  configurations.
- **Eq. 17 (TO)** — knot count N, per-contact-stage time-scaling bounds,
  multiple-shooting integrator choice, and problem scaling.
- **Alg. 1** — UCT exploration constant, progressive-widening exponent, and the
  cache key definition for feasible/infeasible results.
- **Friction** — friction coefficient values, and the cone vs. pyramid
  approximation choice.
