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
| 7 | §II-C 1 | Palm patch geometry. Unlike the feet, the G1's `*_rubber_hand` links carry **no collision geometry at all** — only a visual mesh — so there is nothing to read a palm patch off. | Authored estimate: `h=(0.035, 0.02)` (~0.07×0.04 m palm), palms facing inward. | `configs/scenes/` → `patches.left_palm` |
| 8 | §II-C 1 | **Patch normal sign convention is never stated.** | Local `+z` = outward normal (away from the owning body). Contact is then face-to-face with **anti-parallel** z-axes, so Eq. 7a's alignment residual must include that flip. Revisit when Eq. 7a is written. | `faro/core/patches.py` |
| 9 | §IV-B | Platform ("tabletop") dimensions and height. | 0.60×0.60 m at z=0.40, at x=0.9. | `configs/scenes/` → `patches.tabletop` |

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
