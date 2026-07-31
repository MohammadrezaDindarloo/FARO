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
| 4 | §II-C 1 | Patch half-extents `h` for hands/feet/box faces are never given. | Derive from the G1 URDF foot geometry and the box dimensions at Milestone 2. | `configs/scenes/` |

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
