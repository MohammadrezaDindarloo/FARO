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

**The one exception is `acados`**, which is **required from Milestone 4 on**. §II-G
solves the KSO (Eq. 15) with acados SQP, and `faro/kso/` has no other backend. It has
no usable conda package and needs a CMake source build plus `t_renderer`, so it is
not in `environment.yml`. It lives in `third_party/acados`, and
`faro/utils/acados_env.py` expects the built libraries under `third_party/acados/lib/`
(`libacados.so`, `libblasfeo.so`, `libhpipm.so`, `libqpOASES_e.so`).

**Reproducibility gap, still open:** `third_party/acados` is committed as a submodule
pointer (commit `a26c7d0`) with **no `.gitmodules`**, so a fresh clone gets an empty
directory. The error message in `acados_env.py` also points to
`scripts/setup_acados.sh`, **which does not exist**. The build steps still need to be
written down here.

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

**You do not need to work any of this out yourself.** Every FARO script that opens a
viewer detects where it is running and prints instructions with the real hostname,
username, Slurm node and port already filled in — ready to copy and paste. Nothing
is hard-coded, because the Slurm node changes on every allocation.

Just run the script and read the box it prints.

### One-time setup on a cluster (recommended)

A compute node cannot know how *your laptop* reaches the cluster: the address that
works from outside is usually not one visible from inside (gateways, VPNs, and
`~/.ssh/config` aliases). Tell FARO your alias once:

```bash
conda env config vars set FARO_SSH_GATEWAY=mlp1 -n faro
conda activate faro   # re-activate for it to take effect
```

Use whatever `Host` name you already `ssh` to. With that set, the printed command is
exactly right, e.g.:

```
ssh -N -L 7000:crannog04:7000 mlp1
```

Without it, FARO falls back to reverse-resolving the SSH entry point, which is
usually right but occasionally names a gateway you do not normally use.

### What the script prints

**Running locally** — no tunnel, no forwarding:

```
==============================================================================
  OPEN THIS:  http://127.0.0.1:7000/static/
==============================================================================
  Running locally -- no tunnel or port forwarding needed.
```

Add `--open` to launch the browser automatically (ignored on a remote session).

**Running on a cluster** — the options it can prove will work, best first:

```
  [1] SSH TUNNEL -- run this on YOUR LAPTOP and leave it open:
        ssh -N -L 7000:crannog04:7000 mlp1
     then open:  http://127.0.0.1:7000/static/

  [2] Direct access will NOT work: crannog04.inf.ed.ac.uk resolves only to a
        private address, so no VPN makes it routable from your laptop.

  [3] VS Code TUNNELS is active on this host. ...
```

Note option 2: FARO resolves the node's address and, when it is private
(RFC 1918, e.g. `192.168.x.x`), says so instead of offering a URL that cannot work.

### The three situations, and why they differ

| Situation | How the browser reaches the viewer |
|---|---|
| **Local machine** | Directly — `http://127.0.0.1:<port>/static/`. |
| **Remote-SSH** | VS Code forwards to your laptop's `127.0.0.1`. PORTS panel → forward the port. |
| **VS Code Tunnels** (`code tunnel`) | Ports come out on a `*.devtunnels.ms` URL, **not** `127.0.0.1`. Click the 🌐 globe in the PORTS panel and **append `/static/`**. If blank, right-click → **Port Visibility → Public**, since Meshcat needs a WebSocket. |

An SSH tunnel (option 1) works in all remote cases and has no auth or WebSocket
restrictions, which is why it is listed first.

You should get a dark 3D viewport with a grid and axis triad.

### Notes and gotchas

- **The trailing `/static/` is mandatory — this is the most common trap.**
  Meshcat routes `/` to a *WebSocket* handler, so a normal browser request to the
  bare origin returns:

  ```
  HTTP 400   Can "Upgrade" only to "WebSocket".
  ```

  That error means **your tunnel is working perfectly** — you reached the server and
  merely asked for the wrong path. Add `/static/`:

  | URL | Result |
  |---|---|
  | `http://127.0.0.1:7000/` | ❌ `Can "Upgrade" only to "WebSocket".` |
  | `http://127.0.0.1:7000/static/` | ✅ the viewer |

  This bites especially with VS Code's 🌐 globe icon, which opens the bare origin.
- **Keep the Python process alive.** The Meshcat server dies with the script, and
  the browser goes blank. Our demo scripts hold the process open at the end for
  this reason.
- **Port already in use — the most common cause of "I see nothing".** Meshcat scans
  upward from 7000 and offers **no way to request a port**, so a leftover server
  from a killed script silently pushes the new one onto 7001. If you then forward
  7000, you are looking at the *old, empty* viewer. Every script prints the port it
  actually got; trust that banner, not the default.

  Clean up orphaned viewers (safe — it only kills servers whose parent script has
  already exited, never one another terminal is using):

  ```bash
  python scripts/01_load_and_visualize_robot.py --kill-stale
  ```

  To see what is holding the ports:

  ```bash
  python -c "from faro.viz.meshcat_ports import find_meshcat_servers as f; [print(s) for s in f()]"
  ```

  Note each meshcat server binds **two** ports: ZeroMQ on 6000+ and HTTP on 7000+.
  You forward the HTTP one.
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
