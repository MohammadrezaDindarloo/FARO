# Paper ambiguities and our chosen defaults

The FARO paper omits many implementation details. This file is the running log of
every gap we hit, what we chose instead, and where the knob lives so it can be
tuned later.

Format: one entry per ambiguity, added as we encounter it.

| # | Paper ref | What's missing | Our default | Config knob |
|---|---|---|---|---|
| 1 | whole paper | **The robot is never named.** No model, no DoF count, no simulator anywhere in the text (verified against the full PDF). | **Unitree G1, 29 DoF** (`g1_29dof_rev_1_0.urdf`), box object, MuJoCo. Inferred via ref [6] DynaRetarget — see `paper_notes.md` §1 for the evidence chain. | `configs/robots/` |
| 2 | §II-G | Paper uses acados (KSO) + Hippo (TO); Hippo is closed-source. | **UPDATED at Milestone 4:** Ipopt for Eq. 14 and acados SQP for Eq. 15, both as the paper does. Ipopt is planned for the TO in place of Hippo, so compare TO feasibility verdicts, not wall-clock. | `configs/solvers/` (Eq. 14), `acados_solver.build(...)` args (Eq. 15) |
| 3 | §II-C 1 | Friction coefficients μ (linear) and μ_r (torsional) are never given. | **Superseded by #13.** To be set at Milestone 2; note the paper specifies a **pyramidal** friction approximation, so the constraint stays linear in force. | `configs/scenes/` |
| 4 | §II-C 1 | Patch half-extents `h` for hands/feet/box faces are never given. | **Feet: derived from the URDF, not guessed** — the G1's foot collision model is 4 spheres (r=0.005) at x∈{-0.05,+0.12}, y∈{-0.025,+0.03}, z=-0.03, giving a sole rectangle `h=(0.085, 0.03)` centred at (0.035, 0, -0.035). Box faces follow from the box size. | `configs/scenes/` |
| 5 | §II-D (Eq. 14) | The nominal pose that Eq. 14 regularizes toward is never given, yet it determines *which* solution the NLP finds. | A lightly crouched stance with the leg pitch chain summing to ~0 so the soles stay flat. | `configs/robots/g1.yaml` → `nominal_configuration` |
| 6 | §IV-B | Box dimensions and mass are never given. | 0.30 m cube, 2.0 kg, starting on the floor at x=0.45. Width chosen so the ±y faces sit inside the G1's hand span. | `configs/scenes/` → `objects.box` |
| 7 | §II-C 1 | Hand patch geometry — where the contact patch on the arm actually is. | **RESOLVED — the hand is cut off at the wrist and the patch sits on the cut face** (see #7b). Geometry is then *measured, not authored*: `*_hand_palm_joint` bolts the hand on at x = 0.0415 in `*_wrist_yaw_link`, so that plane IS the cut, and the collision mesh's cross-section there gives `h = (0.0269, 0.0301)` centred at (0.0415, ±0.003, 0). Normal along **+x of the wrist-yaw link** (out along the forearm), `rpy_deg [90, 0, 90]`. | `configs/scenes/` → `patches.left_hand_patch` |
| 7b | Fig. 1, 4, 5 | **The paper's end effector is not the G1's hand.** Its figures show the arms ending in a flat stub with the rectangular patch on it. The model is never named in the text. | **RESOLVED — the hand is removed at the wrist and replaced by the patch.** §II-C 1 models every end-effector as a *rectangular patch* ("all end-effectors … are modeled as rectangular patches"), reaffirmed in the Conclusion ("assumes planar contact interfaces"), so the hand mesh was never in the math. Consequence that IS substantive: the patch normal runs **along the forearm**, not sideways out of a palm, so the robot presses the box's ±y faces between its two forearms. Costs 3.4 cm of reach per arm (shoulder→patch 0.3492 → 0.3150 m). The URDF is unchanged — we keep `g1_29dof_rev_1_0` and simply do not use the `*_rubber_hand` link for contact. | `configs/scenes/` → `patches.left_hand_patch` |
| 7c | §II-C 2 (Eq. 9) | **The hand geometry conflicts with a wrist-mounted patch.** The patch sits on the wrist cut face (x = 0.0415), but the `*_rubber_hand` link extends to x = 0.1318 — 0.09 m PAST the contact point, straight through whatever the robot touches. | **RESOLVED — the hand links are removed.** Not hidden: a viewer-only fix would leave 0.170 kg × 2 in Eq. 10's centroidal dynamics. They are **commented out, not deleted**, in a generated URDF we own (`assets://robots/g1_description/urdf/g1_29dof_faro.urdf`, built by `python -m faro.robots.urdf_surgery`), with an in-file note; uncommenting restores the full hand. Both are leaf links on FIXED joints, so nq/nv and every joint index are unchanged — total mass 33.341 → **33.001 kg**. Still open for Milestone 3: adding convex collision geometry for the wrist patch itself. | `configs/robots/g1.yaml` → `urdf` |
| 8 | §II-C 1 | **Patch normal sign convention is never stated.** | **RESOLVED at Milestone 2.** Eq. 7a's `log₃(R)ₓ,ᵧ=0` forces a pure yaw, so the paper implicitly assumes **parallel** z-axes in contact. We keep uniform **outward** normals (the paper's convention is not uniform — Eq. 7b's containment makes a box face `b` for a hand but the box bottom `a` on the floor) and apply one `Rx(π)` flip in `relative_patch_transform`. `flip_b=False` selects the paper's convention. | `faro/constraints/frames.py` |
| 10 | §III | UCT exploration constant, progressive-widening exponent. | **NOT AMBIGUOUS** — the paper states them: `k = 1.0`, `α = 0.5`, `C = 3`, with `M(v) = max{1, kN(v)^α}` (Eq. 19) and Eq. 18 an arg**min** over cost `J`. | `configs/search/default.yaml` → `uct:` |
| 11 | §II-C 2 (Eq. 9) | Safety margin for the collision constraint. | **Margin 0, as the paper writes, and it is not merely a preference — it is forced.** The usual argument for a few mm is that it keeps Ipopt off the constraint boundary. Measured here, any positive margin makes **every** mode infeasible: a mode that asserts a contact puts that pair at `sd = 0.000000` exactly, and `margin − sd ≤ 0` with `margin > 0` cannot be satisfied. At `margin = 0.002` all ten tour entries flip to infeasible. So a positive margin is only available together with excluding the mode's own contact pairs from Eq. 9 — a package deal, not an independent knob. | `collision_avoidance(margin=…)` |
| 12 | §II-C 2 | How often the frozen GJK witness points/normal are refreshed. Not stated. | Witness data is frozen per linearization point (Schulman et al.); refresh cadence becomes a solver-loop parameter at Milestones 3–5. **Now quantified:** `9-drift` measures the cost of *not* refreshing — 0.5 rad of box tilt makes the frozen model report 0.038 m of clearance while the true signed distance is −0.034 m. Under pure translation of a flat face the error is exactly zero, so the cadence must be driven by **rotation**, not by distance travelled. | `faro/constraints/collision.py` |
| 14 | §II-C 4 | `v_τ,max` for the Eq. 13c torque-speed envelope is not given, and the G1 URDF has no torque-speed data. | Take `τ_max` from the URDF effort limits; `v_τ,max` from the URDF velocity limits as a first estimate. | `configs/robots/` |
| 9 | §IV-B | Platform ("tabletop") dimensions and height. | 0.60×0.60 m at z=0.40, at x=0.9. | `configs/scenes/` → `patches.tabletop` |
| 15 | §II-C 1 (Eq. 7d) | **Which frame Eq. 7d's moment lives in — the paper contradicts itself.** The prose says p, R, f and κ are all "expressed in frame b", but 7d bounds the moment by `^a ξ_a` — a's half-extents in **a's own** frame, and the superscript is deliberate since 7b writes `^b ξ_a`. The two readings differ as soon as the patches differ by a yaw, which **7a explicitly leaves free**. | **We follow `^a ξ_a`**: the moment is transported into patch a's frame (`κ_a = Rᵀ(κ_b − p × f)`) before the CoP rows are formed. 7d is a centre-of-pressure bound and the CoP must lie inside *patch a*; read literally in frame b it instead confines the CoP to a rectangle of a's **dimensions** but b's **orientation**. Measured on a G1 sole (85×30 mm) at 45°: the frame-b reading rejects a CoP well inside the foot **and** accepts one 3× outside it. Eq. 7c is deliberately **not** transported — a friction *pyramid* is not rotationally symmetric, so it is only defined once a frame is chosen, and the paper's choice is b (the support surface). | `contact_wrench(R_rel=…, p_rel=…)` |
| 19 | §II-C 2 (Eq. 9) | **Which collision pairs.** The paper says only *"all bodies in the robot and scene are modeled using convex geometries"* and *"for each collision pair A and B"* — it never defines the set, never mentions self-collision, and never says what happens to a pair the contact mode requires to be TOUCHING. | **RESOLVED at Milestone 3 — every pair, minus three structural exclusions.** 39 bodies → **741** raw pairs → **689** after excluding: (a) two geometries on the *same* link (their signed distance is a constant, so the row can never be influenced by `q`); (b) *parent–child* links, which touch by construction — that is what a joint is, and all 12 pairs interpenetrating at the nominal pose are of this kind; (c) *environment–environment*, both static. Everything else is in: robot self-collision across the tree, robot↔box, robot↔floor/platform, box↔floor/platform. All three are flags, so the literal 741 set can be built and measured. **The G1's meshes are not convex** (24 `BVHModelOBBRSS`, 8 spheres, 4 cylinders) — replaced by their convex hulls, which is what the paper's own sentence asks for and what Schulman's frozen-witness model requires. **Contact pairs ARE excluded** (`exclude_contact_pairs: true`), mode-dependently. The usual argument — that `0 ≤ sd` contradicts Eq. 7a's `p_z = 0` — is *false* at the paper's `margin = 0`: they meet at exactly `sd = 0`, which is what a resting sole measures, and both hold to machine precision. The real reason is performance, and not the obvious one. Those ~11 body pairs out of 689 barely change the row count (we predicted the exclusion would save nothing, and were wrong); they sit at exactly `sd = 0`, so round-off reads as a hair of penetration, the audit-driven refresh loop never declares the answer clean, and every solve runs to its cap. Measured over four targets: **136.2 s and 3/10/10/10 passes with them in, 50.0 s and 2/4/4/5 with them out** — 2.7×, identical verdicts, clean audit either way. It is also what makes a positive margin possible at all. Cost: a patch is not a body (the hand patch is a 27×30 mm window on the whole forearm hull), so `max_penetration` is deliberately audited over **every** pair including the excluded ones. That does force a robust degenerate-normal fallback in `query_witness` — coincident witness points are the *normal* state for every asserted contact, not a corner case. | `faro/scene/collision.py` |
| 12b | §II-C 2 | **The refresh loop for Eq. 14 is never described.** Eq. 9 freezes GJK witness points, so it is a first-order model valid only near where they were taken — but the paper states Eq. 14 as a single optimization. | Eq. 14 is solved as a **sequence**: solve → re-query witnesses at the answer → solve again. Only the **final** solve decides the verdict. **Stopping on "the configuration stopped moving" is not enough**, and that is measured: with a fixed 3 passes, `lift` came back converged, feasible, and **9.3 cm inside the box** with every one of its 689 collision rows satisfied — each row having been linearized at the *previous* iterate. The loop now stops only when the configuration has settled AND the answer is collision-free re-queried against **all** pairs; `refresh_iterations: 10` is a cap, and reaching it is reported through `max_penetration` rather than hidden. | `configs/scenes/` → `collision:` |
| 19b | §II-C 2 (Eq. 9) | **Schulman's activation distance is never given.** The paper adopts "the signed-distance approximation of Schulman et al." wholesale; that method only generates rows for pairs close enough to become active within one convex subproblem, and the threshold is not stated. | **Current defaults (2026-09-16): `activation_distance: 0.10` for Eq. 14 and `kso_activation_distance: null` (all 689 pairs) for Eq. 15.** The reasons are in `faro/mode_edge/README.md` and `faro/kso/README.md`. The text that follows is the original Milestone 3 reasoning for `null`: **Default `null` — every pair gets a row.** Chosen deliberately: Section IV-B gives the search a two-hour budget, so a slow-but-complete filter is in the spirit of the method and nothing is guessing which pairs matter. The knob is implemented and measured: at **0.10 m**, 53 of 689 pairs are active at the nominal pose and 52–64 at solved configurations — a ~12× cut in rows, **6–20× faster**, and **identical verdicts** on every entry compared. Every row generated is exactly Eq. 9 either way; the threshold decides which rows can be *active*, not what they say. **Counter-argument on record:** Table II's own counts bound the filter far harder than two hours does — on the hard task, ~850 s goes to 1415 KSO attempts and ~5150 s to 79.6 TO attempts, leaving ~1200 s for ≥2830 mode/edge checks, i.e. **under ~0.4 s each**. Whatever the paper does here, it is not spending tens of seconds per mode. | `configs/scenes/` → `collision.activation_distance` |
| 13 | §II-C 1 | Friction coefficients μ (linear) and μ_r (torsional) are never given. | **RESOLVED at Milestone 3 as config, not code.** `mu: 0.6` (rubber on hard floor; the pyramid is *inscribed* in the cone so a diagonal push is limited to μ/√2 = 0.42 — conservative, which is the right direction for a feasibility check). `mu_torsional: 0.02` — note this is a **length**, not dimensionless, since `\|κ_z\| ≤ μ_r f_z` relates a moment to a force; ≈ μ × a characteristic patch radius, and the G1 sole is 0.17 × 0.06 m. **Neither enters Eq. 14** — that problem cites only 7a, 7b, 9, 13a and has no forces for friction to act on. They are declared now so KSO and TO read them from config rather than finding them typed into a call. | `configs/scenes/` → `friction:` |
| 25 | §II-D | **`q_nom` is never specified, and the obvious choice is broken.** `RobotModel.q_nominal` sets joint angles on top of `pin.neutral`, leaving the floating base at the world origin. | **Ground it on the soles.** Invisible until Eq. 9 existed; once collision is measured, the raw nominal has **seven robot links inside the floor slab**, the deepest by 6.7 cm, so Eq. 14 started from a point violating the collision constraint before contacts were considered. `Scene.nominal_configuration()` drops the base so the lowest sole rests at z = 0. Consequence worth noting: fixing this flipped one mode from `Infeasible_Problem_Detected` to `Solve_Succeeded` — a reminder that for a **nonconvex** NLP, Ipopt's infeasibility detection is a *local* certificate, never a proof. | `faro/scene/scene.py` |
| 26 | §II-C 1 (Eq. 7a) | **Our column form of Eq. 7a has two disconnected branches, and the nominal pose starts inside the wrong one.** `R[0,2] = R[1,2] = 0` is satisfied by both `R[2,2] = +1` (facing) and `−1` (back-to-back); the inequality `R[2,2] ≥ 0` picks the first but cannot *reach* it — the two components are disconnected on the surface the equalities define. | **Seeded restarts** (`restarts=2`). At q_nom the left hand sits at `R[2,2] = −0.19` w.r.t. `box_left` — already in the forbidden branch — and grasp stalled with `7a:R_zz≥0` violated by 1.000071. No single nominal posture fixes it: `box_left` (+y) and `box_front` (+x) are perpendicular, so aligning to one anti-aligns the other. This is a genuine **cost of deviating from the paper's log₃ form**, which has only one branch, and is recorded as such rather than as a free win. See #23. | `faro/mode_edge/feasibility.py` |
| 18 | §II-C 3 (Eq. 11) | **`W^o_env` is never defined.** The paper writes `W_ext := W_env + W_grav + Σ_a W_a` and then never says what `W_env` contains — drag, a spring, a conveyor, a magnet? None of its tasks obviously need one. | Carried as an **optional** argument to `external_wrench`, defaulting to absent. Not hard-coded to zero: that would bake an assumption into the algebra where a reader cannot see it. The other two terms are built from `gravity_wrench` and `transform_wrench_to_body`. | `external_wrench(W_env=…)` |
| 17 | §II-C 3 (Eq. 11b) | Object inertia values are never given, and a hand-picked diagonal is easy to make **non-physical**. | Derive the probe body's inertia from its **dimensions** (`m/12·[b²+c², a²+c², a²+b²]`), never type it in. The previous hand-picked `diag(0.10, 0.50, 0.90)` violates the triangle inequality `I_x + I_y ≥ I_z` (0.60 < 0.90) — no rigid body has that inertia, so the demo was evaluating a correct formula on an impossible object. Enforced by a test. | `faro/scenarios/constraint_demos.py` → `_SPIN_SIZE` |
| 16 | §II-C 3 | **Eqs. 10b and 11b use opposite 6-vector orderings**, and both are correct. 10b is written `[linear; angular]` (matching Pinocchio's `Ag`, `Force`, and frame Jacobians); 11b follows Lynch & Park and is `[angular; linear]` (`G = diag(I, m·I₃)`, `ad_V = [[ω] 0; [v] [ω]]`). | **Keep both**, since rewriting either departs from its source. The boundary is explicit instead: a contact wrench crosses it (entering 10b and 11b with opposite signs) and must go through `swap_spatial_ordering`. Blocks swapped by accident give a plausible answer with torques where forces belong. | `faro/constraints/frames.py` |

| 20 | §II-A (Eq. 1) vs Table IV | **Eq. 1's consistency rule contradicts the paper's own scene.** Eq. 1 assigns each interface `a ∈ I` **exactly one** partner (`∃! b`) and demands `(a,b) ∈ c ⟺ (b,a) ∈ c`. But Table IV lets `left foot`, `right foot` **and** `box bottom` all name `floor`. Read literally, `floor` would need three partners at once. | **Environment patches host any number of partners; robot and object patches host at most one.** `floor` and `tabletop` are surfaces, not end-effectors, and reading Eq. 1 strictly would forbid standing on two feet. Our `ContactMode` stores the assignment **one-directionally** over the declared interfaces and materializes the reverse pair only where the constraints need it (`active_pairs`), so the contradiction never has to be resolved in the data. | `faro/core/modes.py` |
| 21 | §II-B (Eq. 6) vs Eq. 14 | **The unit-quaternion constraint is never written.** Eq. 6 puts `qʳ ∈ SE(3)×Rⁿ` and `qᵒ ∈ SE(3)`, but an NLP optimizes over Rᵐ, and both Pinocchio and this code parameterize SE(3) with a **unit** quaternion — 7 numbers for 6 DoF. Eq. 14 lists no constraint pinning the norm. | **One equality `‖quat‖² − 1 = 0` per pose.** Not a modelling choice: a non-unit quaternion's rotation matrix is scaled by `‖q‖²`, so without it the solver can cheat the cost by shrinking the robot's effective rotation instead of moving the robot. Squared norm, not the norm — `sqrt` has an infinite derivative at 0. The alternative (optimizing in the 6-D tangent space, as a Lie-group solver would) changes what the variables *are* and is deferred. | `faro/scene/symbolic.py` → `unit_quaternion_rows` |
| 22 | §II-D (Eq. 14) | **`W` and `q_nom` are named and never given.** | `W = I` by default — the only value the paper's notation supports without invention — exposed as per-group knobs (base position, base orientation, joints, object position, object orientation) since a single number is rarely what you want once tuning starts. `q_nom` is the robot's nominal posture plus each object's initial pose, and doubles as the initial guess. **Not an ambiguity:** the plain subtraction `(q − q_nom)` on the quaternion looks like a manifold error and is not — for unit quaternions `‖q − q_nom‖² = 2(1 − cos(θ/2))`, strictly increasing in the geodesic angle, so it ranks rotations identically. That holds *only* because #21 pins the norm. | `faro/mode_edge/problem.py` → `RegularizationWeights` |
| 23 | §II-C 1 (Eq. 7a) | **`log₃(R)ₓ,ᵧ = 0` is fragile inside the Eq. 14 NLP — cause not diagnosed.** With the paper's literal expression Ipopt returns `Invalid_Number_Detected` (NaN in the Lagrangian Hessian) on the grasp and place modes and on the grasp→front edge of the box-placement scene. | **Default to an equivalent form: `R[0,2] = 0`, `R[1,2] = 0`, `R[2,2] ≥ 0`.** Alignment is a statement about R's third column ("patch normals are aligned with their local z-axes"); the two equalities force the a-frame z-axis to have no x/y component, orthonormality then gives `R[2,2] = ±1`, and the inequality picks aligned over back-to-back. Same feasible set (checked on 400 random rotations), same optimum to 1e-6 wherever both converge, and every row is **linear in the entries of R** so nothing in it can produce a NaN. Both forms are kept; `alignment="log3"` is the paper-literal one and Milestone 2's scenarios still use it. **Open:** the mechanism. `log₃` probed alone at θ=0, θ=π, and on the non-orthogonal matrices a non-unit quaternion yields was finite in value, Jacobian *and* Hessian every time; random probing of the assembled rows and of the full Lagrangian Hessian did not reproduce it. An attempt to blame specific contact pairs did not replicate under a different pair ordering, so no per-pair attribution is claimed. What *does* replicate: the whole-form failure, in every row ordering tried. | `faro/constraints/frames.py` → `normal_alignment_rows` |
| 24 | §II-D | **"a number of maximum iterations" is never given** — yet it *is* the feasibility verdict, so it directly sets the false-negative rate the paper measures in Table III. | **300** for Eq. 14, with measured headroom rather than a guess: every mode that converges in the box-placement scene does so in under 50 iterations (~6× margin). Section IV-C blames the paper's own false negatives on "insufficient solver iterations", so this is a knob to revisit if modes start failing that should not. | `configs/solvers/ipopt_mode_edge.yaml` |

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

## Milestone 4 (KSO, Eq. 15)

**#23 RECLASSIFIED -- this was never a paper ambiguity. It is an artifact of our
solver substitution.** Measured on the `reach` sequence:

| | `yaw=log3` (paper-literal) | `yaw=column` (ours) |
|---|---|---|
| `hessian_approximation: exact` | **Invalid_Number_Detected** | Solve_Succeeded |
| `hessian_approximation: limited-memory` | **Solve_Succeeded** | Solve_Succeeded |

The failure needs the EXACT Hessian. Section II-G solves the KSO with acados SQP,
which uses Gauss-Newton -- constraint Jacobians only, never exact second derivatives
of constraint expressions. So the paper can write `log3` in Eq. 7a and Eq. 8 and
never encounter this, and our column forms are a deviation forced by choosing Ipopt,
not a gap in the paper.

Open decision *(written when the KSO was on Ipopt)*: keep `column` + exact Hessian,
or move to `log3` + limited-memory. **Now that the KSO is on acados** with
`hessian_approx=EXACT` (Gauss–Newton stalls, see `kso_acados_design.md`), the
column forms are still needed there too. The claim above that "acados uses
Gauss-Newton, so the paper never hits this" assumes a configuration we could not use.

**#23 background -- the mechanism.**
Recorded in Milestone 3 as reproducible but unexplained. Milestone 4 localized it:
`cpin.log3`'s **second** derivative returns NaN for some inputs. Value and Jacobian
are clean, which is why Milestone 2's finite-difference checks and Eq. 14's
`alignment="column"` default both missed it, and why it only surfaces with
`hessian_approximation: exact`. Ipopt reports `nlp_hess_l failed: NaN detected`
followed by `Invalid_Number_Detected`.

Not established: the precise condition. It does not reproduce as a clean function of
rotation angle or axis -- 0/1600 random rotations produced it, while several
structured probes did. A plausible but UNDEMONSTRATED account is that `cpin.log3`
branches on the angle and second-order AD evaluates both sides, letting a `0/0` in
the inactive branch propagate as `NaN * 0 = NaN`.

Consequence: `log3` is avoided in constraint expressions. Eq. 7a already used the
column form; Eq. 8's yaw row now does too (`yaw="column"`), with the paper-literal
form kept as an option.

**#27 -- how many configurations a sequence has.** Eq. 2 writes
`C = (c_0, ..., c_{K-1})` while the KSO is described over `K+1` configurations. We
build **one configuration per mode given**, which matches Eq. 16's own indexing
("partner at step s and step s+1") and makes the question moot in code.

**#28 -- Eq. 15's cost.** Named, never written. Default is the direct generalization
of Eq. 14: sum the same weighted nominal-distance over every step. A `smoothness`
term penalizing `q_{s+1} - q_s` is implemented and defaults to **0.0**. Config:
`kso:` in the scene file. See `SequenceWeights`.

**#29 -- RESOLVED, see `docs/eq15_kso.md`: Eq. 15 does write `q_0 = q_init`, and the
KSO now enforces it. What is still open is where `q_init` comes from (see
`paper_alignment.md` §4). Original entry, kept for history:** nothing pins the initial state. Eq. 15 as printed makes every object pose a
free variable at every step, including the first, so the box may start somewhere other
than where it is. Measured on `stand -> grasp`: it slides from x = 0.45 to x = 0.047 in
step 0, before the robot touches it, because that is cheaper than reaching. Inside
Alg. 1 this cannot be the intent -- a node has a known state. `anchor_initial=True`
adds the constraint; it is **off** by default because the paper does not write it.

## Milestone 5 (tree search, Alg. 1 and Section III)

**#30 -- `GOAL(C+)` is never defined.** Alg. 1 line 13 calls it and the paper says
only that "the objective is to place the white box on the platform" (Fig. 4). We
express it as a **partial terminal mode** the last mode must match, because "the box
is on the platform" is a contact condition and not a configuration. Default for
`box_placement`: `box_bottom -> tabletop`; interfaces not named stay unconstrained,
so it does not dictate where the hands are. `goal.released` optionally also requires
interfaces to be free ("placed AND let go"), **off** by default since the paper's
sentence does not ask for it. Config: `goal:` in `configs/search/*.yaml`.

**#31 -- the successor set of `PROPOSESUCCESSOR` is never defined.** One number
constrains it: "the allowed contact interfaces are listed in Table IV, yielding a
maximum branching factor of 108" (Section IV-B). 108 is exactly the Cartesian product
over Table IV, i.e. **all modes** -- not the 8 reachable by changing one interface.
So `successor_policy: all` is the faithful default, and a transition may change
several contacts at once; filter E is what rejects the ones that cannot happen at an
instant. `single` (one interface per transition) is available and measurable.

**#32 -- which nodes have `N(.)` incremented is never stated.** Eq. 18 and Eq. 19
both read visit counts, and nothing in Alg. 1 says when they change. We increment
along the **selection path**, the UCT convention. It is not cosmetic: without it
Eq. 19 never widens past one child and Eq. 18's exploration term never decays, so
the search expands each node once and degenerates into a random walk.

**#33 -- `J` is not backed up.** Section III: a candidate "is added to the tree and
assigned the cost J returned by the final filter in F". Read literally, `J(u)` in
Eq. 18 is the node's **own** cost, fixed at admission, with no backup step -- which
is unusual for a UCT, because a cheap subtree is then invisible in its ancestors'
scores. We implement the literal reading. Flagged rather than silently "fixed",
since a backup would change which branches the search prefers.

**Not an ambiguity, but recorded here:** `J` is a raw Eq. 14/15 objective (order
10-15 on this scene) while Eq. 18's exploration term is at most `C*sqrt(log(N+1))`,
a few units. The two are comparable without normalization, so none is applied. On a
scene where `J` is much larger, exploration would go inert -- that needs handling
explicitly, not by silently rescaling the paper's coefficients.

## Still to be resolved in later milestones

Resolved items from this list were removed on 2026-09-16: UCT constants (stated in
§III, #10), pyramid friction (stated, #13), the collision margin (#11), Eq. 7a form
(#23), `q_nom`/W (#22, #25), and the Eq. 15 structure (#27–#29).

- **Eq. 15 (KSO)** — where `q_init` comes from; what `Q_goal` is (a partial mode, for now).
- **Eq. 17 (TO)** — Δt, `T̄_min`/`T̄_max`, `w_T`, the running and terminal costs φ/φ_N,
  and problem scaling. The knot count is stated as 20 per mode (§IV-A), so it is not an
  ambiguity. The integrator is stated as backward Euler (Eqs. 10a, 11a).
- **Eq. 13c** — `v_τ,max` (#14).
- **Alg. 1** — the cache key for `K_feas`/`K_infeas` beyond mode/edge/sequence identity.
  Also whether a KSO verdict can be reused across different `q_init` values (today it
  silently is, see `paper_alignment.md` §4).
- **§IV-B hard scene** — the dimensions of the large platform in Fig. 4 (not in our configs yet).
- **§IV-C** — the LLM prompt, the scene description format, and the plan output format.
