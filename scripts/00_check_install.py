#!/usr/bin/env python
"""Milestone 0 -- environment validation.

Verifies that the `faro` conda environment is correct. This does more than import
each library: every check runs a *minimal real computation*, because the failure
modes that actually bite on a cluster (pinocchio built without CasADi support,
coal not returning witness points, MuJoCo unable to find a GL backend) all import
fine and only break when used.

Run:
    conda activate faro
    python scripts/00_check_install.py

Every check is independent -- one failure never hides the others.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback

# MuJoCo picks its GL backend at *first import* and caches it, so this must be set
# before anything imports mujoco -- setting it later in the render check is too late.
# EGL = GPU offscreen rendering, the right choice on a headless cluster node.
os.environ.setdefault("MUJOCO_GL", "egl")

# --- tiny console helpers (no third-party deps: this script must run even in a
# --- half-built environment) -------------------------------------------------
_TTY = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


GREEN, RED, YELLOW, BOLD, DIM = "32", "31", "33", "1", "2"

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_RESULTS: list[tuple[str, str, str]] = []


def check(name: str, optional: bool = False):
    """Decorator: run a check fn, capture its outcome, never let it kill the script.

    The fn returns a short detail string (usually a version). Raising means the
    check failed -- or, if `optional`, only warns.
    """

    def deco(fn):
        label = f"{name:<28}"
        try:
            detail = fn() or ""
            status = PASS
            colour = GREEN
        except Exception as exc:  # noqa: BLE001 -- we deliberately catch everything
            detail = f"{type(exc).__name__}: {exc}"
            status = WARN if optional else FAIL
            colour = YELLOW if optional else RED
            if os.environ.get("FARO_CHECK_TRACEBACK"):
                traceback.print_exc()
        print(f"  [{_c(status, colour)}] {label} {_c(detail, DIM)}")
        _RESULTS.append((name, status, detail))
        return fn

    return deco


def section(title: str) -> None:
    print(f"\n{_c(title, BOLD)}")


# =============================================================================
print(_c("\nFARO environment check", BOLD))
print(_c(f"python  {sys.executable}", DIM))
print(_c(f"platform {platform.platform()}", DIM))

# -----------------------------------------------------------------------------
section("Core scientific stack")


@check("python")
def _python():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"need >=3.10, found {platform.python_version()}")
    return platform.python_version()


@check("numpy")
def _numpy():
    import numpy as np

    a = np.linalg.solve(np.eye(3) * 2.0, np.ones(3))
    assert np.allclose(a, 0.5)
    return np.__version__


@check("scipy")
def _scipy():
    import scipy
    from scipy.spatial.transform import Rotation

    # Rotation is used later for frame conventions / log(R) sanity checks.
    Rotation.from_euler("z", 0.1).as_matrix()
    return scipy.__version__


@check("yaml")
def _yaml():
    import yaml

    assert yaml.safe_load("a: 1")["a"] == 1
    return yaml.__version__


@check("matplotlib")
def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")  # cluster: never assume an interactive display
    return matplotlib.__version__


# -----------------------------------------------------------------------------
section("Optimization backbone  (CasADi + Ipopt)")


@check("casadi")
def _casadi():
    import casadi as ca

    x = ca.SX.sym("x")
    f = ca.Function("f", [x], [ca.sin(x) ** 2])
    assert abs(float(f(0.5)) - 0.2298488) < 1e-5
    return ca.__version__


@check("casadi -> ipopt NLP")
def _casadi_ipopt():
    """Solve min (x-3)^2 s.t. x >= 0. Confirms Ipopt is linked and callable.

    This is the exact machinery every FARO stage uses (Eq. 14 / 15 / 17), just
    at the smallest possible scale.
    """
    import casadi as ca

    x = ca.SX.sym("x")
    nlp = {"x": x, "f": (x - 3.0) ** 2}
    solver = ca.nlpsol("s", "ipopt", nlp, {"ipopt.print_level": 0, "print_time": False})
    sol = solver(x0=0.0, lbx=0.0, ubx=10.0)
    x_opt = float(sol["x"])
    if abs(x_opt - 3.0) > 1e-6:
        raise RuntimeError(f"expected x*=3, got {x_opt}")
    stats = solver.stats()
    if not stats.get("success", False):
        raise RuntimeError(f"ipopt did not converge: {stats.get('return_status')}")
    return f"x*={x_opt:.6f}  status={stats['return_status']}"


# -----------------------------------------------------------------------------
section("Robot kinematics / dynamics  (Pinocchio + coal)")


@check("pinocchio")
def _pinocchio():
    import numpy as np
    import pinocchio as pin

    # Build a 1-DoF revolute arm from scratch -- no URDF needed, so this check
    # is independent of any asset being present.
    model = pin.Model()
    model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "j1")
    data = model.createData()
    pin.forwardKinematics(model, data, np.zeros(model.nq))
    return f"{pin.__version__}  (nq={model.nq})"


@check("pinocchio.casadi")
def _pinocchio_casadi():
    """THE critical check.

    `pinocchio.casadi` only exists if the conda-forge pinocchio was compiled with
    CasADi support. Without it, none of FARO can be written symbolically and the
    whole project needs a different plan. Verified here by actually building a
    symbolic forward-kinematics function.
    """
    import casadi as ca
    import numpy as np
    import pinocchio as pin
    import pinocchio.casadi as cpin

    model = pin.Model()
    model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "j1")
    model.appendBodyToJoint(1, pin.Inertia.Identity(), pin.SE3.Identity())

    cmodel = cpin.Model(model)          # symbolic mirror of the numeric model
    cdata = cmodel.createData()
    q = ca.SX.sym("q", model.nq)
    cpin.forwardKinematics(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)

    # Symbolic joint placement -> a differentiable CasADi Function.
    fk = ca.Function("fk", [q], [cdata.oMi[1].translation])
    assert np.allclose(np.array(fk(np.zeros(model.nq))).ravel(), np.zeros(3), atol=1e-9)

    # And it must be differentiable -- that is the entire point.
    # NOTE: CasADi reserves the function names 'jac', 'hess' and 'null'; naming this
    # Function "jac" raises a confusing "Function name is not valid" error.
    dfk = ca.Function("dfk", [q], [ca.jacobian(cdata.oMi[1].translation, q)])
    dfk(np.zeros(model.nq))
    return "symbolic FK + jacobian OK"


@check("coal (GJK collision)")
def _coal():
    """Signed distance + witness points + normal between two boxes (Eq. 9 input).

    FARO's collision constraint needs the *witness points and separating normal*,
    not just a boolean overlap flag, so we assert those are populated.
    """
    import numpy as np

    try:
        import coal
    except ImportError:  # older stacks still expose the pre-rename module
        import hppfcl as coal

    box_a = coal.Box(1.0, 1.0, 1.0)
    box_b = coal.Box(1.0, 1.0, 1.0)
    tf_a = coal.Transform3s()
    tf_b = coal.Transform3s()
    tf_b.setTranslation(np.array([3.0, 0.0, 0.0]))  # 2.0 m gap between faces

    req, res = coal.DistanceRequest(), coal.DistanceResult()
    dist = coal.distance(box_a, tf_a, box_b, tf_b, req, res)
    if abs(dist - 2.0) > 1e-6:
        raise RuntimeError(f"expected distance 2.0, got {dist}")

    p1, p2 = res.getNearestPoint1(), res.getNearestPoint2()
    if np.linalg.norm(np.asarray(p2) - np.asarray(p1)) < 1e-9:
        raise RuntimeError("witness points not populated")
    ver = getattr(coal, "__version__", "?")
    return f"{ver}  d={dist:.3f}  witness OK"


def _erd_root() -> str:
    """Locate the example-robot-data asset root inside the active conda env.

    Defined before the check that uses it: @check runs the decorated function
    immediately, so any helper it calls must already exist.
    """
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    root = os.path.join(prefix, "share", "example-robot-data", "robots")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no robot assets at {root}")
    return root


@check("example-robot-data URDFs")
def _erd():
    """Robot assets for Milestone 1 onwards.

    NOTE: conda-forge `example-robot-data` 5.x is a DATA-ONLY package -- URDFs and
    meshes under $CONDA_PREFIX/share/example-robot-data/robots/, with no Python
    module (there is no `example_robot_data` import, and no separate python package
    on conda-forge). We load the URDFs directly through Pinocchio, which is better
    for us anyway: asset paths become explicit config, not a hidden helper.

    Loaded here with a FREE-FLYER root, because FARO targets floating-base legged
    robots -- a fixed-base load would silently drop the 6 base DoF.
    """
    import pinocchio as pin

    root = _erd_root()
    urdf = os.path.join(root, "talos_data", "robots", "talos_reduced.urdf")
    if not os.path.isfile(urdf):
        raise FileNotFoundError(urdf)

    model = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
    n_robots = len(os.listdir(root))
    # Floating base contributes 7 to nq (3 translation + 4 quaternion) and 6 to nv.
    if model.nq - model.nv != 1:
        raise RuntimeError(f"expected quaternion floating base, nq={model.nq} nv={model.nv}")
    return f"{n_robots} robot dirs; talos_reduced nq={model.nq} nv={model.nv} (floating base)"


# -----------------------------------------------------------------------------
section("Simulation + visualization")


@check("mujoco")
def _mujoco():
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <body name="b"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    if data.time <= 0:
        raise RuntimeError("simulation did not advance")
    return f"{mujoco.__version__}  stepped to t={data.time:.3f}s"


@check("mujoco headless render", optional=True)
def _mujoco_render():
    """Offscreen rendering needs a GL backend (EGL on a headless cluster node).

    WARN-only: nothing before Milestone 5 needs it, and Meshcat covers all our
    visualization until then. If this warns, see SETUP.md (`MUJOCO_GL=egl`).
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="2 2 .1"/>
        <body pos="0 0 .5"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=64, width=64) as r:
        r.update_scene(data)
        img = r.render()
    return f"rendered {img.shape} via MUJOCO_GL={os.environ['MUJOCO_GL']}"


@check("meshcat")
def _meshcat():
    """Import only -- we do NOT start a server here (that would bind a port).

    Milestone 1 opens the actual viewer; see SETUP.md for VS Code port forwarding.
    """
    import meshcat
    import meshcat.geometry as g

    g.Box([1, 1, 1])  # geometry construction works without a live server
    return getattr(meshcat, "__version__", "installed")


@check("pinocchio MeshcatVisualizer")
def _pin_meshcat():
    from pinocchio.visualize import MeshcatVisualizer  # noqa: F401

    return "available"


# -----------------------------------------------------------------------------
section("FARO package")


@check("import faro")
def _faro():
    """Confirms `pip install -e .` was run inside the faro env."""
    import faro

    return f"{faro.__version__} from {os.path.dirname(faro.__file__)}"


@check("openai (Milestone 7)", optional=True)
def _openai():
    import openai

    return f"{openai.__version__} (API key not required until Milestone 7)"


# =============================================================================
n_fail = sum(1 for _, s, _ in _RESULTS if s == FAIL)
n_warn = sum(1 for _, s, _ in _RESULTS if s == WARN)
n_pass = sum(1 for _, s, _ in _RESULTS if s == PASS)

print(f"\n{_c('Summary', BOLD)}: {n_pass} passed, {n_warn} warnings, {n_fail} failed")

if n_fail:
    print(_c("\nFAILED checks (must fix before Milestone 1):", RED))
    for name, status, detail in _RESULTS:
        if status == FAIL:
            print(f"  - {name}: {detail}")
    print(_c("\nRe-run with FARO_CHECK_TRACEBACK=1 for full tracebacks.", DIM))
    sys.exit(1)

if n_warn:
    print(_c("\nWarnings (optional components, safe to proceed):", YELLOW))
    for name, status, detail in _RESULTS:
        if status == WARN:
            print(f"  - {name}: {detail}")

print(_c("\nEnvironment OK -- ready for Milestone 1.\n", GREEN))
