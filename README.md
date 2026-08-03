# FARO

A from-scratch reimplementation of **"FARO: Feasibility-Aware Robot Motion
Optimization"** (arXiv 2607.18362). The authors did not release the method code,
so every component here is rebuilt from the paper.

This is a **learning project**: each module is independently runnable, documented
against the specific paper equations it implements, and validated in simulation
with visualization before the next one is built.

## What FARO is

A hierarchy of increasingly expensive feasibility checks over discrete contact-mode
sequences. Cheap checks prune infeasible contact plans before the expensive ones
run.

| Stage | Paper | What it answers | Cost |
|---|---|---|---|
| **Mode / edge feasibility** | Eq. 14 | Is there *one* configuration `q` satisfying this contact mode? | cheap |
| **KSO** (kinematic sequence opt.) | Eq. 15 | Is there a *geometrically* consistent sequence of K+1 configurations? | medium |
| **TO** (trajectory opt.) | Eq. 17 | Is there a full *dynamically* feasible trajectory? | expensive |
| **Tree search** | Alg. 1 | Which contact-mode sequence should we try? (UCT + progressive widening) | — |
| **LLM sampling** | — | Propose plausible contact plans to filter | — |

Shared constraint definitions (Section II-C, Eqs. 7–13) live in one place,
`faro/constraints/`, and are reused by every stage.

## Scope

- **In scope:** simulation-only validation of the full hierarchy, Milestones 0–7.
- **Out of scope:** the RL controller and real-hardware execution.
- **No Hippo.** The paper's TO solver is not open source. Everything uses
  **Ipopt via CasADi**; the solver interface is structured so acados can be added
  later as an optional backend.

## Setup

See **[SETUP.md](SETUP.md)** for the full cluster walkthrough (env creation,
Meshcat over VS Code port forwarding, MuJoCo headless rendering).

```bash
cd ~/repos/FARO
conda env update -f environment.yml --prune   # or: conda env create -f environment.yml
conda activate faro
pip install -e . --no-deps
python scripts/00_check_install.py
```

## Repository layout

```
FARO/
├── environment.yml            # the `faro` conda env (all conda-forge binaries)
├── pyproject.toml             # makes `faro` importable via `pip install -e .`
├── SETUP.md                   # cluster setup, Meshcat forwarding, GL backends
├── configs/                   # YAML: swap robot / scene / solver without code changes
│   ├── robots/  scenes/  solvers/  tasks/
├── assets/                    # local URDFs + meshes
├── faro/                      # the importable package
│   ├── core/                  # shared vocabulary: ContactMode, ContactPatch, Interface
│   ├── robots/                # URDF loading, Pinocchio wrappers (numeric + symbolic)
│   ├── scene/                 # scene definition, allowed contact interfaces
│   ├── constraints/           # shared CasADi constraint blocks, Eqs. 7-13
│   ├── solvers/               # backend-agnostic NLP interface; Ipopt now, acados later
│   ├── mode_edge/             # mode + edge feasibility, Eq. 14
│   ├── kso/                   # kinematic sequence optimization, Eq. 15
│   ├── to/                    # trajectory optimization, Eq. 17
│   ├── search/                # feasibility-guided UCT tree search, Alg. 1
│   ├── llm/                   # LLM contact-plan sampling
│   ├── viz/                   # Meshcat + MuJoCo helpers
│   └── utils/                 # config loading, paths, logging
├── scripts/                   # numbered, runnable, standalone demos
├── tests/                     # unit tests per module
├── notebooks/                 # optional library exploration
└── docs/                      # paper notes, equation mapping, design decisions
```

Each `faro/` subpackage gets its own `README.md` explaining what it does, which
equations it implements, how to run it, and what correct output looks like — added
as that milestone is built.

## Milestone status

| # | Milestone | Script | Status |
|---|---|---|---|
| 0 | Environment & scaffolding | `00_check_install.py` | **done** — 17/17 checks pass |
| 1 | Robot loading & visualization | `01_load_and_visualize_robot.py` | **done** — 48/48 tests pass, branching factor 108 reproduced |
| 2 | Shared constraints (Eqs. 7–13) | `02_constraint_playground.py` | **done** — 441 tests; 28/28 constraints fail exactly where predicted |
| 3 | Mode/edge feasibility (Eq. 14) | `02_mode_edge_demo.py` | not started |
| 4 | KSO (Eq. 15) | `03_kso_demo.py` | not started |
| 5 | TO (Eq. 17) | `04_to_demo.py` | not started |
| 6 | Tree search (Alg. 1) | `05_tree_search_demo.py` | not started |
| 7 | LLM contact-plan sampling | `06_llm_sampling_demo.py` | not started |

## Stack

CasADi (symbolic + autodiff) · Pinocchio (kinematics/dynamics, centroidal momentum) ·
coal (GJK witness points + normals) · Ipopt (NLP) · MuJoCo (physics + headless render) ·
Meshcat (browser 3D viz) · acados (optional, later)

## Robot

**Unitree G1, 29 DoF** (floating base: nq=36, nv=35), manipulating a box. The paper
never names its robot — see [docs/paper_notes.md](docs/paper_notes.md) §1 for how
this was established.

The hands are **cut off at the wrist**, as the paper does: its figures show the arms
ending in a flat stub with the contact patch on the cut face. We load a generated
URDF in which the hand links are *commented out rather than deleted*, with a note
explaining why — uncomment them to get the full hand back.

```bash
python -m faro.robots.urdf_surgery     # regenerate assets/ from example-robot-data
```

Both hand links are leaves on fixed joints, so `nq`/`nv` and every joint index are
unchanged; only 2 × 0.170 kg of mass and two visual meshes go away (33.341 →
33.001 kg). See [docs/ambiguities.md](docs/ambiguities.md) #7b / #7c.

## Milestone 1 — what to look for in the viewer

```bash
conda activate faro
python scripts/01_load_and_visualize_robot.py --mode sweep
```

Forward port **7000** in VS Code, open `http://127.0.0.1:7000/static/`.

Check these five things — each one is a convention that Milestone 2 will build on:

1. **Feet on the floor, not hovering or sunk.** The green sole patches sit exactly on
   `z = 0`. The G1's sole is 0.035 m below its ankle frame, so placing the robot by the
   ankle would leave it floating.
2. **Every red normal stub points outward.** Soles point *down*, the hand patches point
   *out along each forearm* (the paper cuts the hand off at the wrist and puts the patch
   on the cut face), box faces point *away* from the box, floor and platform point *up*.
   A stub pointing into its own body is the bug this visualization exists to catch.
3. **Green sole patches match the real foot outline** — longer forward (+x) than back,
   because the sole spans x ∈ [-0.05, +0.12] in the URDF.
4. **The orange box patches move with the box**, the green robot patches move with the
   joints during the sweep, and the blue floor/platform never move.
5. **The box is out of arm's reach** from the standing pose. That is intentional (see
   below), not a misconfigured scene.

Console output should end with `Raw branching factor: 108`.

## Milestone 2 — seeing the constraints hold and break

```bash
python scripts/02_constraint_playground.py --list       # 28 scenarios
python scripts/02_constraint_playground.py              # animated in Meshcat
python scripts/02_constraint_playground.py --no-viz     # headless table
```

Each scenario sweeps one physical parameter and declares, **independently of the
implementation**, where the paper's equation says the constraint must fail. All 28
agree, most to machine precision. Highlights:

- **`7d-roll` vs `7d-pitch`** — same foot, same load: roll fails at 6.0 Nm, pitch at
  17.0 Nm. Ratio 2.83 = the sole's length/width. That's what the x↔y swap in Eq. 7d
  encodes; swap it and the robot silently tips sideways while every unit test passes.
- **`7b-hand-x` vs `7b-hand-y`** — Eq. 7b governs **end-effectors**, not just objects
  (§II-C 1: *"includes contact between end-effector patches …"*). The robot holds a
  forearm out and the box slides across the wrist cut face: 0.1231 m sliding sideways
  but only 0.1199 m sliding vertically, because the cut is 0.0269 × 0.0301 — taller
  than wide. Runs through forward kinematics, unlike the box sweeps.
- **`7b-yaw`** — a box that **never moves** violates Eq. 7b at 13.05° of rotation,
  because yaw enlarges its footprint in the platform's frame.
- **`9-penetrate` vs `9-drift`** — sliding a flat face straight in, the frozen witness
  model is **exact** and tracks the signed distance through contact. Tilt the box
  instead and it reports **growing clearance while the box is 3.4 cm inside the
  platform** — the closest-point pair moved and the frozen one did not. That drift is
  the linearization error, and the whole case for refreshing between solver iterations.
- **`7a-yaw`** — nothing happens through a full half-turn. Eq. 7a leaves yaw free by
  design; that's the freedom needed to turn while walking.

See [faro/scenarios/README.md](faro/scenarios/README.md) for the full table and what
to watch in the viewer.

## Open paper ambiguities

The paper omits many implementation details (cost weights, tolerances,
initialization strategies, frame conventions). Every such gap is called out
explicitly in the relevant module README and exposed as a tunable parameter in
`configs/` rather than silently hard-coded. Running list: `docs/ambiguities.md`.
