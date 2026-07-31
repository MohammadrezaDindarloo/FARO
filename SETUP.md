# SETUP

Cluster setup for FARO. Target: Linux x86_64 (verified on Ubuntu 24.04), conda at
`/opt/conda`, environments in `~/.conda/envs`.

---

## 1. Create the `faro` environment

From the repo root:

```bash
cd ~/repos/FARO
conda env create -f environment.yml
```

An empty `faro` env already exists on this machine (Python 3.11, no packages).
`conda env create` will refuse to overwrite it, so pick one:

```bash
# Option A -- update the existing env in place (fastest)
conda env update -f environment.yml --prune

# Option B -- start clean (use this if anything looks wrong later)
conda env remove -n faro -y
conda env create -f environment.yml
```

Then activate and install the FARO package itself in editable mode:

```bash
conda activate faro
pip install -e . --no-deps
```

`--no-deps` matters: every dependency is a conda-forge binary, and pip must not be
allowed to "helpfully" replace the conda-installed `pinocchio` / `casadi` / `numpy`
with incompatible PyPI wheels.

### If the solve is slow

The classic conda solver can take a long time on an environment this size. If it
stalls, use libmamba:

```bash
conda install -n base conda-libmamba-solver -y
conda env update -f environment.yml --prune --solver=libmamba
```

---

## 2. Verify the environment

```bash
conda activate faro
python scripts/00_check_install.py
```

Expected: every check `PASS`, with at most warnings on the *optional* ones
(`example-robot-data`, `mujoco headless render`, `openai`). Anything marked
`FAIL` must be fixed before Milestone 1.

For full tracebacks on a failure:

```bash
FARO_CHECK_TRACEBACK=1 python scripts/00_check_install.py
```

And the pytest equivalent:

```bash
pytest tests/ -v
```

### The one check that really matters: `pinocchio.casadi`

FARO writes all kinematics as **symbolic** CasADi expressions and hands their
derivatives to Ipopt. That requires Pinocchio to have been compiled *with* CasADi
support. The conda-forge `pinocchio` 4.1 build depends on `casadi >= 3.7.2`, so
this should just work — but if `pinocchio.casadi` fails to import, stop and tell
me, because it invalidates the modelling approach rather than one module.

---

## 3. Why conda-forge and not source builds

Every library in the stack (CasADi w/ Ipopt, Pinocchio 4.1 w/ CasADi bindings,
coal, MuJoCo, Meshcat) has a maintained conda-forge binary. Building Pinocchio or
coal from source pulls in Boost, Eigen, urdfdom, and hpp-fcl version-matching pain
that has nothing to do with understanding the paper. We use binaries.

**The one exception is `acados`** (optional faster SQP backend, Milestone 5+). It
has no usable conda package and does need a CMake source build plus `t_renderer`.
It is deliberately *not* in `environment.yml`: the solver interface is designed so
acados can be added later as an alternative backend without touching any other
module. Do not build it now.

### Docker fallback

If a library misbehaves on the cluster in a way that is not a config problem,
the fallback is a container rather than fighting the host:

```bash
# ~/repos/FARO/docker/Dockerfile  (not written yet -- only if we need it)
FROM condaforge/miniforge3:latest
COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml && conda clean -afy
```

Run it with `--network=host` so Meshcat port forwarding still works. We will only
write this if we actually hit a wall.

---

## 4. Viewing Meshcat from the cluster (via VS Code)

Meshcat runs a small web server on the cluster node and you view it in a browser
on your laptop. VS Code Remote makes this nearly automatic.

**Start the viewer** (any FARO script that opens Meshcat, e.g. Milestone 1's
`scripts/01_load_and_visualize_robot.py`). It prints something like:

```
You can open the visualizer by visiting the following URL:
http://127.0.0.1:7000/static/
```

**Then:**

1. In VS Code (connected to the cluster over Remote-SSH), open the **PORTS** panel
   — next to TERMINAL, or `Ctrl+Shift+P` → *"Ports: Focus on Ports View"*.
2. VS Code usually auto-detects port `7000` and forwards it. If not, click
   **Forward a Port** and enter `7000`.
3. Click the 🌐 globe icon next to the forwarded port, or open
   `http://127.0.0.1:7000/static/` in your **local** browser.

You should get a dark 3D viewport with a grid and axis triad.

### Notes and gotchas

- **Always use `http://127.0.0.1:7000/static/` locally**, not the cluster's
  hostname. The forwarding tunnel maps your laptop's `localhost:7000` to the
  cluster's `localhost:7000`.
- **The trailing `/static/` is required.** `http://127.0.0.1:7000/` alone shows
  nothing useful.
- **Keep the Python process alive.** The Meshcat server dies with the script, and
  the browser goes blank. Our demo scripts hold the process open at the end for
  this reason.
- **Port already in use** (a leftover viewer from an earlier run): pass a
  different port, or clean up with `pkill -f meshcat`.
- **Compute nodes vs login node.** If you `srun` onto a compute node, VS Code
  forwards ports from the node your Remote-SSH session is attached to. Simplest
  path for Milestones 1–4: run the interactive/visual scripts on whichever node
  your VS Code session is on. These are small problems and do not need a GPU.

### Fallback: manual SSH tunnel

If VS Code's forwarding misbehaves, from your **laptop**:

```bash
ssh -N -L 7000:localhost:7000 <user>@<cluster-host>
```

then open `http://127.0.0.1:7000/static/`.

---

## 4b. Robot assets (`example-robot-data`)

**The conda-forge `example-robot-data` 5.x package is data-only.** It installs 42
robot directories of URDFs + meshes but ships **no Python module** — there is no
`import example_robot_data`, and no separate `example-robot-data-python` package on
conda-forge. Assets live at:

```
$CONDA_PREFIX/share/example-robot-data/robots/
```

We load these URDFs directly through Pinocchio. That is better for FARO anyway:
asset paths become explicit entries in `configs/robots/` rather than being hidden
inside a loader helper, which is what makes the robot swappable.

**Always load with a free-flyer root**, since FARO targets floating-base legged
robots — a default fixed-base load silently drops the 6 base DoF:

```python
model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
```

Legged/humanoid models available: `g1_description` (Unitree G1, 29 DoF humanoid),
`talos_data` (Talos humanoid), `simple_humanoid_description`, `bolt` (small biped),
`solo` / `solo12`, `anymal`, `go1`, `go2`, `a1`, `icub`, `cassie`, `hector`.

---

## 5. MuJoCo headless rendering (only needed from Milestone 5)

Cluster nodes have no display. MuJoCo needs an explicit GL backend:

```bash
export MUJOCO_GL=egl      # GPU offscreen rendering -- preferred (A40 present)
# export MUJOCO_GL=osmesa # CPU software fallback: conda install -c conda-forge mesalib
```

`scripts/00_check_install.py` reports this as a **warning**, not a failure —
nothing before Milestone 5 needs rendered pixels, and Meshcat covers all
visualization until then.

To make it permanent for the env:

```bash
conda env config vars set MUJOCO_GL=egl -n faro
conda activate faro   # re-activate for it to take effect
```

---

## 6. LLM API key (Milestone 7 only)

Not needed until the final milestone. When we get there:

```bash
conda env config vars set OPENAI_API_KEY=sk-... -n faro
```

Do not commit keys. `.env` is already gitignored.
