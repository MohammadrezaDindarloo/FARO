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
| 1 | Robot loading & visualization | `01_load_and_visualize_robot.py` | not started |
| 2 | Shared constraints (Eqs. 7–13) | — (tests only) | not started |
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

**Unitree G1, 29 DoF** (`g1_29dof_rev_1_0.urdf`, floating base: nq=36, nv=35),
manipulating a box. The paper never names its robot — see
[docs/paper_notes.md](docs/paper_notes.md) §1 for how this was established.

## Open paper ambiguities

The paper omits many implementation details (cost weights, tolerances,
initialization strategies, frame conventions). Every such gap is called out
explicitly in the relevant module README and exposed as a tunable parameter in
`configs/` rather than silently hard-coded. Running list: `docs/ambiguities.md`.
