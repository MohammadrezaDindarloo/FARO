# FARO paper — extracted implementation facts

Source: **FARO: Feasibility-Aware Robot Motion Optimization**, Michal Ciebielski,
Shafeef Omar, Aaron Johnson, Majid Khadiv. arXiv:2607.18362v1 [cs.RO], 20 Jul 2026.
TUM MIRMI. CC BY 4.0. Video: https://youtu.be/R6qCHoCormQ

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
   an RL-based trajectory-tracking controller **[6]**".
2. **[6]** = Dhedin, Taouil, **Omar**, Yu, Tao, Dai, Khadiv (2026), *DynaRetarget:
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
| KSO (Eq. 15) | acados SQP | Ipopt (acados optional later) |
| TO (Eq. 17) | Hippo SQP | Ipopt (Hippo is not open source) |

Expect our KSO/TO **solve times to differ** from Table I: interior-point (Ipopt)
vs. SQP (acados/Hippo) is an algorithmic difference, not an implementation bug.
Compare *feasibility verdicts* and *constraint coverage*, not wall-clock, when
validating against the paper.

## 3. Their library stack — identical to ours

Refs [2] **CasADi**, [3] **Pinocchio**, [16] **Coal**, [8] **GJK** (Gilbert-Johnson-Keerthi),
[18] Schulman et al. (SQP collision handling, basis for the Eq. 9 formulation),
[12] Lynch & Park *Modern Robotics* (cited for §II-C 3, the dynamics).

Our stack choice therefore matches the paper's exactly, apart from Hippo.

---

## 4. Constraint details (§II-C) — for Milestone 2

**All contact interfaces are rectangular planar patches** with patch-to-patch
unilateral contact. Notation: `p_ij`, `R_ij`, `f_ij`, `m_ij` = relative position,
rotation, force, moment of patch *i* w.r.t. patch *j*, expressed in frame *j*.
**Patch normals are aligned with their local z-axes.** `h` = half-extents.

- **Eq. 7a** — aligns patch normals via `log(R)` and enforces zero normal
  separation, **leaving relative in-plane position and yaw free**.
- **Eq. 7b** — keeps patches within the admissible region defined by half-extents.
- **Eq. 7c** — unilateral contact + a **pyramidal** (not smooth-cone) approximation
  of Coulomb friction, coefficient μ.
- **Eq. 7d** — torsional friction (coefficient μ_t) + centre-of-pressure bounds
  induced by the finite patch size.
- **Eq. 8** — no-slip for a sticking contact active across adjacent timesteps,
  constraining **both in-plane position and in-plane rotation** across timesteps.

Note for Milestone 2: the friction cone is **pyramidal**, which keeps the
constraint linear in the force variables. Do not substitute a quadratic cone.

Limitation stated in §V: "The current formulation assumes **planar contact
interfaces**" — extending to curved geometry would need differentiable SDFs [7].

## 5. Evaluation setup — for Milestones 5–7

- **TO discretization: 20 optimization knots per mode.**
- Eight human-defined contact sequences: Climb 1/2, Pick Place 1/2, Toss to Table,
  Double Catch, Triple Catch, Juggle. Modes per sequence: 4–11 (avg 8.4).
- **Table I results:** TO 9.73–117.62 s (avg 64.70); KSO 0.14–1.93 s (avg 0.60).
  KSO ≈ **2 orders of magnitude faster**, retains **70.0%** of TO constraint types,
  **74.8×** reduction in decision variables. Omitted constraints are the dynamic ones.
- **Tree search (§IV-B):** two box-placement task variants, 5 seeds, **2-hour budget**,
  **max depth 5 switches**, **max branching factor 108**. Hard task: TO-only baseline
  expands 12.8 nodes / 0 solutions; with FARO filters 814.6 nodes / 26.4 solutions.
- **LLM (§IV-C):** **GPT-5.5**, prompted for **100 diverse contact plans** per scene,
  repeated **5×**. KSO used as a classifier of TO feasibility; false-negative rate
  near zero. Four scenes, each with an easier `-a` and harder `-b` variant.

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
