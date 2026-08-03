# `faro.scenarios` — seeing the constraints hold and break

Twenty-eight physical stories, one per behaviour of Eqs. 7–13. Each sweeps **one**
physical parameter and shows the constraint responding — in numbers, and in Meshcat.

```bash
python scripts/02_constraint_playground.py --list          # what's available
python scripts/02_constraint_playground.py --kill-stale    # all, animated
python scripts/02_constraint_playground.py -s 7d-pitch     # just one
python scripts/02_constraint_playground.py --no-viz        # headless table
python scripts/02_constraint_playground.py --auto          # no prompts
```

## Navigating

The paper's own statement of each equation is printed above its animation, so the
algebra and the motion are on screen together. After each scenario, a **single
keypress** (no ENTER needed) decides what happens next:

| Key | Action |
|---|---|
| **ENTER** | next scenario |
| **SPACE** | replay this one — watch it again against the printed equation |
| **p** | previous scenario |
| **q** | quit |

Replay is the useful one: sweeps run in a few seconds, and the moment a patch turns
red or the force arrow crosses the pyramid wall is easy to miss the first time.

Piped and non-interactive runs play straight through, so `--no-viz` still works in
scripts and CI without hanging on the prompt.

## Why this exists

Milestones 3–5 assemble these constraints into NLPs with thousands of rows, where a
swapped axis or a sign error surfaces only as `Infeasible_Problem_Detected`. There
is no way to bisect that. So every constraint is pinned down **alone** first.

The mechanism that makes this more than a smoke test: each scenario declares a
**`predicted_crossing`** — the parameter value at which the paper's equation says
the constraint *must* start failing, derived by hand from the equation and the
scene's real numbers, **independently of the implementation**. The tests then compare
prediction against measurement.

All 28 currently agree, most to machine precision:

| Scenario | Eq. | Predicted | Measured |
|---|---|---|---|
| `7a-lift` | 7a | 0 m | 0.0000 |
| `7a-tilt` | 7a | 0 rad | 0.0000 |
| `7a-yaw` | 7a | **never** | never |
| `7b-slide` | 7b | 0.15 m | 0.1500 |
| `7b-hand-x` | 7b | 0.1231 m | 0.1231 |
| `7b-hand-y` | 7b | 0.1199 m | 0.1199 |
| `7b-yaw` | 7b | 0.2278 rad | 0.2278 |
| `7c-slip` | 7c | 140 N | 140.00 |
| `7c-pull` | 7c | 0 N | 0.0000 |
| `7d-roll` | 7d | 6.0 Nm | 6.0000 |
| `7d-pitch` | 7d | 17.0 Nm | 17.000 |
| `7d-torsion` | 7d | 10.0 Nm | 10.000 |
| `8-slip` | 8 | 0 m | 0.0000 |
| `8-spin` | 8 | 0 rad | 0.0000 |
| `8-lift` | 8 | **never** | never |
| `8-carry` | 8 | **never** | never |
| `9-penetrate` | 9 | 0 m | 0.0000 |
| `9-drift` | 9 | **never** (frozen model) | never |
| `10-weight` | 10 | 323.74 N | 323.74 |
| `10-moment` | 10 | 0 m | 0.0000 |
| `10-swing` | 10 | 0 rad/s | 0.0000 |
| `10a-rollout` | 10a | **never** | never |
| `10a-forward` | 10a | 0 s | 0.0000 |
| `11-hold` | 11 | 19.62 N | 19.620 |
| `11-spin` | 11 | 0 rad/s | 0.0000 |
| `13a-knee` | 13a | 2.8798 rad | 2.8798 |
| `13b-speed` | 13b | 20.0 rad/s | 20.000 |
| `13c-envelope` | 13c | 10.0 rad/s | 10.000 |

## What each one teaches

**`7a-lift` / `7a-tilt` / `7a-yaw` — what "contact" actually constrains.**
Lifting breaks `p_z = 0`; tilting breaks `log₃(R)ₓ,ᵧ = 0`; **yawing breaks nothing
at all**, through a full half-turn. That last one is the point of the pair: Eq. 7a
pins three DoF and deliberately leaves in-plane position and yaw free — the freedom
the robot needs to turn while walking.

**`7b-slide` / `7b-yaw` — containment, not overlap.**
The box fails at exactly `0.30 − 0.15 = 0.15 m`, the *difference* of half-extents.
Then `7b-yaw` rotates a box that **never moves** and the constraint still fails at
13.05°, because rotating enlarges its footprint in the platform's frame. That is
what the `ᵇξ_a` term is for; without it this failure is invisible.

**`7b-hand-x` / `7b-hand-y` — Eq. 7b applies to end-effectors, not just objects.**
Section II-C 1: *"This includes contact between **end-effector patches**, patches on
movable objects, and static environment patches."* Table IV's *"Left hand → box
left"* is exactly such a pair. These two matter for reasons the box sweeps can't
cover:

- they run through the robot's **forward kinematics** — patch `a` hangs off a robot
  link rather than an object pose, a different code path entirely;
- they use the **wrist cut-face geometry** — the paper removes the G1's hand at the
  wrist and puts the patch on the cut surface (ambiguity #7b), so the contact normal
  runs *along the forearm*, not sideways out of a palm;
- that cut is **0.0269 × 0.0301**, so the bound is `0.1231` along x but `0.1199`
  along y. A single scalar "patch radius" cannot produce that, and the
  box-on-platform sweeps can never reveal it because those patches are square.

Note which way round that goes: the cut is **taller than it is wide**, so sliding the
box *vertically* runs out of room **sooner** (0.1199) than sliding it sideways
(0.1231). The old palm patch was wider than tall and gave the opposite ordering —
which is exactly why the test asserts the direction-free invariant (*the larger
half-extent leaves less room*) rather than a hard-coded `y > x`.

The robot **stands still with its forearm out and the box slides across the cut
face**. The box is the movable object, so it is the thing that moves; posing the
robot to meet a fixed box would need full-body IK and reads as the robot flying onto
the box.

Only the patch's *orientation* has to be exact, and that is solved in closed form.
The G1's wrist is roll(X)·pitch(Y)·yaw(Z) with no fixed rotation between the joints,
so the patch's world rotation is `R0 · Rx(a)Ry(b)Rz(c) · R_patch`; requiring it to
equal the orientation that seats an **upright, axis-aligned** box gives the three
angles directly from an XYZ Euler extraction. The shoulder and elbow are four
hand-picked constants, all inside their Eq. 13a limits. The box pose then comes from
*inverting* the Eq. 7a relation rather than approximating it, so 7a holds to machine
precision at every step and 7b is the only thing the sweep can break.

**`7c-slip` / `7c-pull` — the friction pyramid, and why contacts can't pull.**
**The foot does not move, and that is correct.** Eq. 7c constrains the contact
*wrench*, not the configuration; the sweep varies the applied force with `q` held
fixed, and Milestone 2 has no integrator to turn a friction violation into motion —
that needs Eq. 17. "Slip" here is the force arrow leaving the pyramid, not the sole
sliding. Slip at `μf_z = 140 N`. Unilaterality bites the instant `f_z` goes negative — the
floor is not glue, and this single row is what forces the robot to *lift* a foot
rather than hang from it. In the viewer, watch the force arrow leave the blue
pyramid. Note it's a **pyramid**: the corners stick out further than the edges,
which is why `fₓ = f_y = μf_z` is admissible.

**`7d-torsion` — the row with no other visual consequence.**
A moment about the contact normal leaves the force unchanged (arrow and pyramid
static) and does not appear in `x_cop = κ_y/f_z`, `y_cop = −κ_x/f_z` (dot stays put).
So the scene is *genuinely frozen* through this whole sweep — hence its own indicator:
a **blue ring** of radius `μ_r f_z` (the capacity) and a **yellow arc** of radius
`|κ_z|` (the applied twist), swept in the direction of its sign. Read it exactly like
the arrow against the pyramid: the arc grows through the ring at 10 Nm. Torsional
friction is far weaker than linear friction, which is why pivoting on one foot is easy.

**`7d-roll` vs `7d-pitch` — the most valuable pair here.**
Same foot, same load. Roll fails at **6.0 Nm**, pitch at **17.0 Nm** — a ratio of
2.83, exactly `0.085 / 0.030`, the sole's length-to-width ratio. A foot is long
front-to-back, so it resists pitching far better than rolling. **This is what the
x↔y swap in Eq. 7d encodes.** Swap the subscripts and every unit test still passes
while the robot silently tips over sideways in simulation. In the viewer the pink
CoP dot slides to the toe (far) or the edge (soon).

**`8-slip` / `8-spin` / `8-lift` / `8-carry` — all three rows, and both freedoms.**
Eq. 7a was happy for the foot to sit *anywhere* on the floor. Eq. 8 says it may not
*move* once established. `8-slip` slides it and the residual equals the slide
distance exactly.

But Eq. 8 has **three rows**, and sliding only reaches two of them. The other three
scenarios exist because `8-slip` alone cannot distinguish a correct implementation
from several broken ones:

- **`8-spin`** pivots the foot on its own contact point, so `dp_x`/`dp_y` stay at
  machine zero and `8:dyaw` is the only row that can move. A sign error in the yaw
  row was invisible before this — `8-slip` does not even *watch* that row.
- **`8-lift`** raises the robot 20 cm and **nothing happens**. Eq. 8 is an *in-plane*
  lock; the normal direction is Eq. 7a's `p_z = 0`. Were it a full pose lock, a foot
  could never leave the floor and the robot could never take a step.
- **`8-carry`** moves a hand *and* the box it holds together. Both patches sweep a
  quarter metre and the residual stays at exactly zero, because Eq. 8 fixes the pose
  of `a` **relative to** `b`. Written against world poses it would forbid moving a
  grasped object at all — making the paper's own box-placement task infeasible. The
  floor never moves, so `8-slip` and `8-spin` cannot tell the two readings apart.

**`9-penetrate` vs `9-drift` — what the `≈` in Eq. 9 actually costs.**
Eq. 9 is exact for the *material points* GJK froze, and drifts once those stop being
the closest pair. `9-penetrate` slides a flat face straight in: the closest pair never
changes, so the frozen model reproduces the true signed distance **exactly**, through
contact and into penetration. `9-drift` holds a 2 cm clearance and **tilts** the box
instead: the contact migrates to a different corner, the frozen pair does not follow,
and Eq. 9 reports clearance *growing* from 0.020 → 0.038 m while the box is actually
**0.034 m inside the platform** — a 7 cm error, in the dangerous direction. That gap is
the linearization error, and it is the whole argument for refreshing witness data
**between** solver iterations. GJK is combinatorial, so it cannot go inside the graph
in any case.

> A previous version of this section claimed the opposite — that a refreshed query
> can never report penetration because `coal.distance()` is unsigned. It is not:
> `DistanceRequest.enable_signed_distance` defaults to `True`. That claim was propped
> up by a sign error in `query_witness`, which built its normal from `w_A − w_B` and so
> **reversed it under overlap**, turning 6 cm of interpenetration into `sd = +0.06`.
> Both are fixed; see `tests/test_conventions.py`.

**Every failure is visible.** Eqs. 9, 11 and 13 constrain an *object* or a *joint*,
not a contact patch, so the usual green→red patch had nothing to colour and eight
scenarios failed silently — the box drove clean through the platform and stayed its
normal colour. Those now tint the body (`highlight_objects`) or drop a green/red bead
at the joint (`status_at`), and `test_every_violating_scenario_shows_the_violation_on_screen`
requires one of the two from any scenario whose sweep can fail.

**`10a-rollout` / `10a-forward` — validating an integrator without a solver.**
The robot **walks**: legs in antiphase, each foot lifting ~9 cm in turn, arms
counter-swinging, torso advancing 0.38 m while swaying ±3.6°, every joint inside its
Eq. 13a limits. That legibility is the point — an earlier version used random
accelerations, which satisfied Eq. 10a just as exactly and was useless, because a robot
tumbling through the air cannot be looked at and judged.
Eqs. 10a and 11a are the only constraints here that span more than one state, so the
tempting place to exercise them is inside the TO. That is the wrong place to exercise
them *first*: if an integrator's debut is inside a solver and the solve fails,
`Infeasible_Problem_Detected` cannot separate a bad transcription from a bad initial
guess, bad scaling, or a genuinely infeasible problem — there is nothing to bisect.

So [`trajectories.py`](trajectories.py) writes the trajectory by hand. `10a-rollout`
flies one built to satisfy Eq. 10a exactly and the defect stays at **machine zero
(~1e-18) at every knot and every dt**. `10a-forward` transcribes the *same motion* with
`v_i` instead of `v_{i+1}` — forward Euler — and the defect is non-zero at any dt > 0.

Two independent checks, and they test different things.

**1. The defect.** Given `(q_i, q_{i+1}, v_{i+1})`, does the constraint row evaluate to
zero? That is exactly what the NLP sees. Checked over four step sizes in
`tests/test_trajectories.py`:

| dt | backward-Euler defect | forward-Euler defect | ratio |
|---|---|---|---|
| 0.040 | 6.9e-18 | 3.018e-03 | |
| 0.020 | 1.7e-18 | 7.545e-04 | 4.00× |
| 0.010 | 2.2e-19 | 1.886e-04 | 4.00× |
| 0.005 | 5.4e-20 | 4.715e-05 | 4.00× |

Halving dt quarters the defect — exactly O(dt²), the local truncation error of a
first-order method. A defect that merely *looks* small proves nothing; one that scales
at the predicted rate is a measurement.

**2. The convergence.** The defect check has a limit worth naming: the trajectory is
*built* from the recurrence, so the answer is zero by construction. It proves the
implementation matches the recurrence — not that the recurrence reproduces the true
motion. So a second test marches forward with Eq. 10a from `q_0` alone and compares
against a **closed-form** ground truth:

| dt | base-rotation error | joint error | ratio |
|---|---|---|---|
| 0.0400 | 6.679e-02 | 4.341e-02 | |
| 0.0200 | 3.307e-02 | 2.150e-02 | 2.02× |
| 0.0100 | 1.645e-02 | 1.070e-02 | 2.01× |
| 0.0050 | 8.207e-03 | 5.335e-03 | 2.00× |

Halving dt **halves** the error — O(dt), first-order *global* convergence, which is
what backward Euler must give. The ground truth is exact, not a fine-dt reference: the
base twist is angular-only about a **fixed body axis**, so the rotations commute and the
time-ordered exponential collapses to `R(t) = R₀·exp₃(z·∫ω)`. Comparing against our own
integrator at small dt would have been circular.

Two details make or break this test, and both bit during development:

- the horizon is a **partial** period. Over a full one the rectangle sum of a cosine
  cancels exactly, the error collapses to ~1e-16 at every dt, and the test passes while
  measuring nothing;
- dt must **divide** the horizon exactly. Using `round(horizon/dt)` ends each run at a
  different time and the ratios come out as noise (1.12, 1.76, 3.22).

A control test pins the difference between *wrong* and merely *coarse*: replacing the
exponential map with plain addition on the quaternion makes the error **grow** as dt
shrinks (0.119 → 0.245, ratio 0.62) rather than halving. Divergence, not inaccuracy. The base is given a real **angular** velocity
on purpose: revolute joints integrate correctly under plain addition, so a bug that
only breaks the exponential map would hide in a purely translating trajectory.

**Scope, stated honestly:** Eq. 10a is a *kinematic* identity. Making 10a and 10b hold
together — `h = A(q)v` at every knot *and* `hdot` the net external wrench — has no
closed-form rollout. That coupling **is** the trajectory optimization, and it belongs
to Milestone 5.

**`10-weight` / `10-moment` — the centroidal dynamics, made static.**
Eqs. 10 and 11 need velocities and accelerations, which a swept configuration does
not have. Both scenarios instead pin the system in *static equilibrium* and sweep the
one quantity that must then balance — the contact wrench — which is exactly Eq. 10b's
first line at rest. `10-weight` asks how hard the floor must push and answers
**323.74 N**, the robot's own weight from the URDF. `10-moment` then keeps that force
but slides it sideways: the linear row stays at zero while the angular row grows
immediately, because the moment arm `(p − c)` is measured **from the centre of mass**.
That is "why you fall over", and why Eq. 7d's CoP has to stay under the body.

**`11-hold` / `11-spin` — the object has its own dynamics.**
Eq. 11 is Newton-Euler for the box alone, separate from the robot's Eq. 10. At rest
the external wrench must vanish, so the support equals `2.0 × 9.81 = 19.62 N`. FARO
optimizes over robot **and** object states — that is what lets it plan a placement.

`11-spin` covers the part `11-hold` cannot reach: the gyroscopic term `−[ad_V]ᵀ G V`.
Spin a body at constant rate with nothing pushing it and Eq. 11b becomes
`0 = −[ad_V]ᵀ G V`, which the term says is impossible — an anisotropic body needs a
moment just to keep spinning steadily. The residual **is** the textbook Euler term
`ω × (Iω)`, and the test checks it against a plain numpy cross product, sharing no
code with the adjoint formulation.

Two things worth knowing about this pair:

- **The scene's box is a cube**, so `I = k·Id` and `ω × (Iω) = 0` for *every* ω. No
  scenario using it could ever exercise the gyroscopic term — it would pass for the
  wrong reason. `11-spin` therefore uses an elongated probe body, and a test asserts
  the cube really is isotropic so this stays true if the box changes.
- Wrenches here are **angular-first**: indices 0–2 are the moment, 3–5 the force.
  `11-hold` originally wrote its support force at index 2 — the moment-about-z slot —
  and still passed, because with `V = V̇ = 0` the residual was just the number the
  scenario had computed by hand. The equation did no work. It now builds the weight
  with `gravity_wrench`, so a wrong slot no longer cancels.

**`13a-knee` / `13b-speed` / `13c-envelope` — three limits, three different stages.**
`13a` binds exactly at the URDF's `q_max` and is the **only** one mode/edge (Eq. 14)
and KSO (Eq. 15) can enforce. `13b` needs a velocity and `13c` needs a torque, so only
the TO (Eq. 17) sees them — which *is* the constraint-coverage gap Table I reports
between KSO and TO.

`13c-envelope` is the one worth watching: it is **not a box** on (τ, v). Torque and
speed trade off linearly, so a joint holding half its peak torque reaches only
**half** its no-load speed — 10.0 of 20.0 rad/s. A box constraint would have allowed
full torque at full speed, which no real actuator can do. It is built from four linear
rows per joint rather than `fabs`, which is not differentiable exactly at `v = 0`
where joints rest.

## Layering

Deliberately split so a graphics problem can never be mistaken for a physics problem:

| Layer | Module | Depends on Meshcat? |
|---|---|---|
| Physics | `faro/scenarios/constraint_demos.py` | **no** |
| Drawing | `faro/viz/constraint_viz.py` | yes |
| Runner | `scripts/02_constraint_playground.py` | optional (`--no-viz`) |
| Tests | `tests/test_milestone2_scenarios.py` | **no** |

## What to watch in the viewer

- **Patches turn green → red** at the crossing
- **Yellow arrow** = contact force; **blue pyramid** = the Eq. 7c friction limit.
  Slip is the arrow crossing the pyramid wall — the *foot itself never moves* in the
  7c/7d sweeps, because those constrain the wrench, not the configuration.
  Both are drawn in the **contact frame** (`viz.constraint_viz.contact_frame`): our
  patch normals are uniformly outward, so a sole's own +z points *down*, and drawing
  there once buried this entire overlay under the floor
- **Pink rectangle** = the patch outline; **pink dot** = the centre of pressure.
  Eq. 7d fails when the dot reaches the rectangle's edge
- **Blue ring / yellow arc** (`7d-torsion` only) = torsional capacity `μ_r f_z` vs the
  applied `κ_z`. Nothing else moves in that sweep — the twist has no other visual effect
- **Yellow up-arrows vs a red down-arrow** (Eqs. 10, 11) = the two sides of a balance.
  `10-weight` grows the contact forces until they match the weight; `10-moment` keeps
  them equal but slides the support out from under the **white CoM dot**, opening up a
  purple moment arrow; `11-hold` does the same for the box; `11-spin` shows the spin
  (yellow) against the gyroscopic moment it demands (purple), which grows as `ω²`.
  Arrows carry a per-vector scale, since the robot weighs 324 N and the box 19.6 N
- The overlay draws **on top of the robot** (`depthTest=False`): the CoP rectangle lies
  in the sole plane and would otherwise render inside the G1's shoe mesh, invisible
- The terminal prints each watched row and marks `<-- CROSSES HERE`

## A real bug this caught

The G1's nominal standing pose had a leg pitch chain summing to `−0.01` rather than
`0`, leaving a 0.01 rad sole tilt. Since `q_nominal` is both the Eq. 14
regularization target *and* the initial guess for every stage, every solve would
have started from a point that violated Eq. 7a for no reason. Invisible in the
unit tests; obvious the moment `7a-yaw` refused to report "never violated".
