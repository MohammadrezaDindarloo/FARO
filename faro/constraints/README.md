# `faro.constraints` — shared constraint blocks (paper Eqs. 7–13)

Section II-C of the paper. These are the constraints that **every** stage reuses:

> "These optimization problems … share a common set of contact, collision, dynamics,
> and limit constraints, while differing in the variables they optimize and the
> level of feasibility they enforce." — Section II

Each file implements one paper equation as reusable **CasADi expressions**. No file
here builds or solves an NLP — that is Milestones 3–5.

## Contents

| File | Equation | What it constrains |
|---|---|---|
| `contact.py` | **7a–7d** | patch-to-patch contact: normal alignment, containment, friction, CoP |
| `no_slip.py` | **8**, **16** | sticking contacts don't slide; when the condition applies |
| `collision.py` | **9** | GJK signed-distance approximation |
| `dynamics_robot.py` | **10a, 10b** | centroidal robot dynamics |
| `dynamics_object.py` | **11a, 11b** | rigid-body object dynamics |
| `limits.py` | **12, 13a–c** | torque definition; position, velocity, torque-speed limits |
| `frames.py` | — | SE(3) helpers and the patch-frame convention |
| `block.py` | — | uniform `ConstraintBlock` container |

## Which stage uses which

This table is the thing to keep straight — it's what makes the hierarchy a
*hierarchy* rather than three copies of the same problem:

| Stage | Contact | No-slip | Collision | Dynamics | Limits |
|---|---|---|---|---|---|
| **Mode/edge** (Eq. 14) | 7a, 7b | — | 9 | — | 13a |
| **KSO** (Eq. 15) | 7a, 7b | 8 | 9 | — | 13a |
| **TO** (Eq. 17) | 7 (all) | 8 | 9 | 10, 11 | 13 (all) |

Mode/edge and KSO are kinematic: 7c/7d, 13b and 13c are not merely skipped for
speed, they are **undefined** there — those problems have no force or velocity
variables. This is exactly the paper's finding that KSO retains "70.0% of the
constraint types present in the TO", omitting the dynamic ones.

## Conventions

**Sign.** `eq == 0`, `ineq <= 0` throughout, matching the paper's own
`contact(q,c) <= 0` style, so blocks drop straight into an NLP.

**Patch normals — the one real reconciliation.** FARO stores every patch's `+z` as
its **outward** normal (uniform, physical, visible in Meshcat). The paper's Eq. 7a,
`log₃(R)ₓ,ᵧ = 0`, forces R to be a *pure yaw*, i.e. it assumes mating patches have
**parallel** z-axes. Outward normals are **anti-parallel** in contact, so
`relative_patch_transform` applies one fixed `Rx(π)` flip to patch b. That is the
only place the choice is encoded, and `flip_b=False` selects the paper's convention
directly.

Why not just adopt the paper's convention? Because it isn't uniform. Eq. 7b's
containment makes the pair asymmetric (`a` ⊆ `b`), so a box face is `b` when a hand
presses it but the box *bottom* is `a` when it rests on the floor — the same object
would need opposite conventions on different faces.

## Three things that are easy to get subtly wrong

1. **Eq. 7b is containment, not overlap.** `|pₓ,ᵧ| ≤ (ᵇξ_b) − (ᵇξ_a)`, where `ᵇξ_a`
   is a's rectangle *rotated into b's frame*. A patch larger than its support is
   infeasible even perfectly centred, and a yawed patch needs more axis-aligned
   room than an aligned one.
2. **Eq. 7d swaps x and y**: `|κₓ| ≤ f_z(ξₐ)ᵧ`, `|κᵧ| ≤ f_z(ξₐ)ₓ`. Correct physics —
   a moment about x comes from a lever arm along y. "Tidying" the subscripts to
   match is a plausible-looking bug that mis-sizes the support polygon.
3. **Eqs. 10a/11a are backward Euler**, with the right-hand side at `i+1`. And `⊕`
   is manifold integration — `q + v·dt` would knock the base quaternion off the
   unit sphere.

## Verify

```bash
pytest tests/test_milestone2_constraints.py -v   # algebra, numeric
pytest tests/test_milestone2_symbolic.py -v      # symbolic, on the real G1
```

The numeric suite checks each constraint is satisfied at a physically correct
configuration **and violated, in the right row, by a specific physical error**. The
symbolic suite then checks the same expressions build against symbolic `q`, are
differentiable, and agree with finite differences and with Pinocchio's numeric
`ccrba` / `centerOfMass` — i.e. that Ipopt will actually have something to work with
in Milestone 3.
