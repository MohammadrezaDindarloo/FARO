# FARO paper — extracted implementation facts

Source: **FARO: Feasibility-Aware Robot Motion Optimization**, Michal Ciebielski,
Shafeef Omar, Aaron Johnson, Majid Khadiv. arXiv:2607.18362v1 [cs.RO], 20 Jul 2026.
TUM MIRMI. CC BY 4.0. Video: https://youtu.be/R6qCHoCormQ

**Re-checked 2026-09-16 against the arXiv v1 PDF.** Reference numbers and notation
below follow that PDF. Earlier versions of this file used numbers from a different
rendering; for example, DynaRetarget is **[19]**, not [6].

Everything below is quoted or directly derived from the paper text. Anything
*inferred* is labelled as such.

---

## 1. The robot — NOT STATED IN THE PAPER

**The paper never names its robot, its DoF count, or its simulator.** Verified by
extracting the full PDF text and searching: only the phrase "humanoid robot"
appears. This is a real reproducibility gap, acknowledged implicitly by the paper's
own limitations section.

**Our conclusion: Unitree G1 (29 DoF), manipulating a box, simulated in MuJoCo.**
Confidence: high, but *inferred*. The evidence chain:

1. §IV-A: sequences marked `*` "are additionally executed on the real robot using
   an RL-based trajectory-tracking controller **[19]**".
2. **[19]** = Dhedin, Taouil, **Omar**, Yu, Tao, Dai, Khadiv (2026), *DynaRetarget:
   Dynamically-feasible retargeting using sampling-based trajectory optimization*,
   arXiv:2602.06827 — same lab, and shares author Shafeef Omar with FARO.
3. DynaRetarget states verbatim: *"The dataset contains hundreds of motions of a
   **G1 humanoid robot** interacting with a **box**, including pick-and-place,
   kicking, and pushing or dragging motions."* Simulator: **MuJoCo** (RL trained in
   Isaac Lab).
4. FARO's Table IV interfaces (two hands ↔ box faces, two feet ↔ floor, box bottom
   ↔ floor/tabletop) exactly match a humanoid + single box.
5. FARO's task names — Pick Place, Toss to Table, Double/Triple Catch, Juggle,
   Climb — are G1-scale loco-manipulation with a box.

### Which G1 variant to use

`example-robot-data` ships two, both loading cleanly with a free-flyer root:

| URDF | nq | nv | actuated |
|---|---|---|---|
| `g1_29dof_rev_1_0.urdf` | 36 | 35 | 29 |
| `g1_29dof_with_hand_rev_1_0.urdf` | 50 | 49 | 43 |

**Use the 29-DoF variant (no articulated hands).** FARO models every end-effector
as a *rectangular contact patch* (§II-C 1), never as an articulated gripper — the
"hand" is just a patch frame. The 14 extra finger DoF in the `with_hand` variant
would add pure NLP cost with zero modelling benefit.

Moreover the paper **removes the hand entirely at the wrist** and puts the patch on
the resulting cut face (visible in Figs. 1, 4, 5 — the arms end in a flat stub). So
our patch is anchored to `*_wrist_yaw_link` at the hand's own mounting plane
(x = 0.0415), with its normal pointing **out along the forearm**. The robot presses
the box's ±y faces between the ends of its two forearms rather than pinching with
palms. See `ambiguities.md` #7 / #7b.

---

## 2. Solvers used by the paper (§II-G)

> "The KSO is solved with the SQP solver in **acados**, whereas the TO is solved
> with the SQP solver **Hippo**. The mode/edge optimizations are solved with **Ipopt**."

| Stage | Paper's solver | Ours |
|---|---|---|
| Mode/edge (Eq. 14) | Ipopt | **Ipopt — exact match** |
| KSO (Eq. 15) | acados SQP | **acados SQP — match** (since Milestone 4) |
| TO (Eq. 17) | Hippo SQP | Ipopt, planned (Hippo is not open source) |

Also from §II-G: **both (15) and (17) use direct multiple shooting**. Our KSO is a
multi-phase acados OCP with one phase per knot.

Expect our TO **solve times to differ** from Table I, since Ipopt is interior-point
and Hippo is SQP.
Compare *feasibility verdicts* and *constraint coverage*, not wall-clock, when
validating against the paper.

## 3. Their library stack — identical to ours

Refs [25] **CasADi**, [24] **Pinocchio**, [22] **Coal**, [21] **GJK** (Gilbert-Johnson-Keerthi),
[20] Schulman et al. (the signed-distance approximation behind Eq. 9),
[23] Lynch & Park *Modern Robotics* (object twist–wrench dynamics, Eq. 11b),
[26] **acados**, [27] **Hippo**, [28] **Ipopt**, [30] smoothed distance functions
(future work in §V), [31] MotionDisco (higher-level search, future work).

Our stack choice therefore matches the paper's exactly, apart from Hippo.

---

## 4. Constraint details (§II-C) — for Milestone 2

**All contact interfaces are rectangular planar patches** with patch-to-patch
unilateral contact. Notation (PDF): `p`, `R`, `f`, `κ` = relative position,
rotation, force, moment of patch *a* w.r.t. patch *b*, expressed in frame *b*.
**Patch normals are aligned with their local z-axes.** `ξ` = half-extents, and
`ᵇξₐ` = the half-extents of *a* expressed in frame *b*. The wrench is `λ_e = (f_e, κ_e)`.

- **Eq. 7a** — aligns patch normals via `log(R)` and enforces zero normal
  separation, **leaving relative in-plane position and yaw free**.
- **Eq. 7a** is `log₃(R)_{x,y} = 0, p_z = 0`.
- **Eq. 7b** — `|p_{x,y}| ≤ (ᵇξ_b)_{x,y} − (ᵇξ_a)_{x,y}`: patch *a* lies inside patch *b*.
- **Eq. 7c** — unilateral contact + a **pyramidal** (not smooth-cone) approximation
  of Coulomb friction, coefficient μ.
- **Eq. 7d** — torsional friction `|κ_z| ≤ μ_r f_z` + centre-of-pressure bounds
  `|(κ_x, κ_y)| ≤ f_z ((ᵃξₐ)_y, (ᵃξₐ)_x)`. Note the x↔y swap and the frame-*a* half-extents.
- **Eq. 8** — no-slip for a sticking contact active across adjacent timesteps,
  `(p^{s+1} − p^s)_{x,y} = 0`, `log₃((R^s)ᵀR^{s+1})_z = 0`.
- **Eq. 9** — `0 ≤ sd_AB(x) ≈ n̂ · (T_A(x) p_A − T_B(x) p_B)`, with witnesses from GJK.
- **Eq. 10** — backward-Euler integration over `T̄Δt` of q (on the manifold), v and
  the centroidal momentum h. `ḣ` is gravity plus contact wrenches, and `h = A(q) v`.
- **Eq. 11** — object backward Euler, `W_ext = G V̇ − [ad_V]ᵀ G V`, with
  `W_ext = W_env + W_grav + Σ_a W_a`.
- **Eq. 12** — `τ_j = S(M v̇ + b − Σ J_eᵀ λ_e)`. **Eq. 13a/b** position/velocity
  limits. **Eq. 13c** `|τ_j| + (τ_max / v_τ,max)|v_j| ≤ τ_max`.
- **Which constraints each problem uses:** Eq. 14 uses 7a, 7b, 9, 13a. Eq. 15 uses
  7a, 7b, 8, 9, 13a. Eq. 17 uses all of 7, 8, 9, 10, 11, 13.

Note for Milestone 2: the friction cone is **pyramidal**, which keeps the
constraint linear in the force variables. Do not substitute a quadratic cone.

Limitation stated in §V: "The current formulation assumes **planar contact
interfaces**" — extending to curved geometry would need differentiable SDFs [7].

## 5. Evaluation setup — for Milestones 5–7

- **TO discretization: 20 optimization knots per mode.** Eq. 17 adds one time
  scale `T̄_s ∈ [T̄_min, T̄_max]` per mode, penalized by `w_T Σ(T̄_s − 1)`. None of
  `T̄_min`, `T̄_max`, `w_T`, Δt, φ or φ_N is given.
- Eight human-defined contact sequences: Climb 1/2, Pick Place 1/2, Toss to Table,
  Double Catch, Triple Catch, Juggle. Modes per sequence: 4–11 (avg 8.4).
- **Table I results:** TO 9.73–117.62 s (avg 64.70); KSO 0.14–1.93 s (avg 0.60).
  KSO ≈ **2 orders of magnitude faster**, retains **70.0%** of TO constraint types,
  **74.8×** reduction in decision variables. Omitted constraints are the dynamic ones.
- **Tree search (§IV-B):** two box-placement task variants, 5 seeds, **2-hour budget**,
  **max depth 5 switches**, **max branching factor 108**. Hard task: TO-only baseline
  expands 12.8 nodes / 0 solutions; with FARO filters 814.6 nodes / 26.4 solutions.
  Filter variants (Eq. 21): F1=(KSO), F2=(M,E,KSO), F3=(M,E,KSO,TO), F4=(TO).
  **Table II** (mean over 5 seeds):

  | | Easy KSO | Easy M,E,KSO | Easy M,E,KSO,TO | Easy TO | Hard KSO | Hard M,E,KSO | Hard M,E,KSO,TO | Hard TO |
  |---|---|---|---|---|---|---|---|---|
  | solutions | 73.6 | 83.8 | 8.0 | 20.0 | 0.4 | 26.4 | 1.2 | 0.0 |
  | first sol. [s] | 192 | 194 | 1455 | 951 | 2671 | 2763 | 5399 | n/a |
  | KSO attempts | 410.6 | 410.0 | 230.8 | 0 | 1660.8 | 1415.0 | 437.4 | 0 |
  | TO attempts | 182.0 | 176.2 | 223.0 | 208.4 | 5.4 | 79.6 | 164.0 | 216.2 |
  | tree nodes | 393.8 | 380.8 | 90.6 | 87.8 | 218.2 | 814.6 | 52.8 | 12.8 |
- **LLM (§IV-C):** **GPT-5.5**, prompted for **100 diverse contact plans** per scene,
  repeated **5×**. KSO used as a classifier of TO feasibility; false-negative rate
  near zero. Figure 5 shows four panels, 1-a, 1-b, 2-a and 2-b: two scenes, each
  with an easier `-a` and a harder `-b` variant. **Table III:** FNR 0.0 / 0.0 / 1.3 /
  0.0 %, FPR 24.2 / 4.2 / 41.4 / 19.5 %, speedup 3.8 / 15.5 / 2.4 / 7.4×. The few
  false negatives are blamed on "poor initialization or insufficient solver
  iterations in the KSO".

### Table IV — allowed contact interactions (defines the search space)

| Interface | Allowed contacts |
|---|---|
| Left hand | box left, box front, free |
| Right hand | box right, box rear, free |
| Left foot | floor, free |
| Right foot | floor, free |
| Box bottom | floor, tabletop, free |

## 6. Formulation structure (§II-A, §II-B)

- A **contact mode** assigns, to every interface, a pair marking it either in
  unilateral contact with another interface or free (Eq. 1), subject to a
  consistency condition. A sequence indexes these over time (Eq. 2).
- **State** splits into robot and object parts (Eq. 5): robot = configuration `q`,
  generalized velocity `v`, **centroidal momentum**; each object = pose + body velocity.
- **Control**: robot = generalized acceleration + **end-effector wrenches**;
  each object = body acceleration + external environment wrench.

Note the robot control includes wrenches, not joint torques — torques enter only
through the actuation limits (Eqs. 12–13), where a selection matrix picks the
actuated components.
