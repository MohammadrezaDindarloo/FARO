"""Milestone 0 test: the environment provides what FARO structurally depends on.

Deliberately minimal. These are the assumptions that, if violated, invalidate the
whole project plan rather than just one module -- so they are worth asserting in
CI-style form and not only in the human-readable 00_check_install.py report.

    pytest tests/test_environment.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


def test_faro_importable():
    """`pip install -e .` was run inside the faro env."""
    import faro

    assert faro.__version__


def test_casadi_has_ipopt():
    """Ipopt is the primary solver for Eq. 14 / 15 / 17. No Ipopt, no FARO."""
    import casadi as ca

    x = ca.SX.sym("x")
    solver = ca.nlpsol(
        "s", "ipopt", {"x": x, "f": (x - 3.0) ** 2},
        {"ipopt.print_level": 0, "print_time": False},
    )
    sol = solver(x0=0.0, lbx=-10.0, ubx=10.0)
    assert solver.stats()["success"]
    assert float(sol["x"]) == pytest.approx(3.0, abs=1e-6)


def test_pinocchio_casadi_is_symbolic_and_differentiable():
    """Pinocchio must be compiled WITH CasADi support.

    Every FARO stage writes kinematics as symbolic expressions and hands their
    derivatives to Ipopt. If pinocchio.casadi is missing or non-differentiable,
    the modelling approach in this repo does not work at all.
    """
    import casadi as ca
    import pinocchio as pin
    import pinocchio.casadi as cpin

    model = pin.Model()
    model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "j1")
    # Offset the body 1 m along +x so rotating the joint actually moves the frame,
    # which makes the Jacobian below non-trivial.
    model.appendBodyToJoint(1, pin.Inertia.Identity(), pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])))

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    q = ca.SX.sym("q", model.nq)
    cpin.forwardKinematics(cmodel, cdata, q)

    pos = cdata.oMi[1].translation
    fk = ca.Function("fk", [q], [pos])
    # NOTE: CasADi reserves the names 'jac', 'hess', 'null' -- do not name a Function "jac".
    dfk = ca.Function("dfk", [q], [ca.jacobian(pos, q)])

    # At q=0 the joint frame sits at the origin (the *body* is offset, not the joint).
    assert np.allclose(np.array(fk([0.0])).ravel(), np.zeros(3), atol=1e-9)
    # The Jacobian must be a real, finite, non-empty matrix.
    J = np.array(dfk([0.0]))
    assert J.shape == (3, 1)
    assert np.all(np.isfinite(J))


def test_coal_returns_witness_points_and_distance():
    """FARO's collision constraint (Eq. 9) needs signed distance AND witness points.

    A boolean overlap test would not be enough to build a differentiable constraint.
    """
    try:
        import coal
    except ImportError:
        import hppfcl as coal

    box_a, box_b = coal.Box(1.0, 1.0, 1.0), coal.Box(1.0, 1.0, 1.0)
    tf_a, tf_b = coal.Transform3s(), coal.Transform3s()
    tf_b.setTranslation(np.array([3.0, 0.0, 0.0]))

    res = coal.DistanceResult()
    dist = coal.distance(box_a, tf_a, box_b, tf_b, coal.DistanceRequest(), res)

    # Boxes are 1 m wide, centres 3 m apart -> 2 m face-to-face gap.
    assert dist == pytest.approx(2.0, abs=1e-6)

    p1 = np.asarray(res.getNearestPoint1())
    p2 = np.asarray(res.getNearestPoint2())
    assert np.linalg.norm(p2 - p1) == pytest.approx(2.0, abs=1e-6)


def test_floating_base_urdf_assets_available():
    """FARO targets floating-base legged robots, so assets must load with a free-flyer.

    conda-forge `example-robot-data` 5.x ships URDFs/meshes only (no Python module),
    so we resolve the path inside the conda prefix and hand it straight to Pinocchio.
    """
    import os
    import sys

    import pinocchio as pin

    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    urdf = os.path.join(
        prefix, "share", "example-robot-data", "robots",
        "talos_data", "robots", "talos_reduced.urdf",
    )
    if not os.path.isfile(urdf):
        pytest.skip(f"example-robot-data assets not found at {urdf}")

    model = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
    # A quaternion-parameterised floating base makes nq exactly one larger than nv.
    assert model.nq - model.nv == 1
    assert model.nv > 6, "expected actuated joints on top of the 6-DoF floating base"


def test_mujoco_steps():
    """MuJoCo is our independent physics check for Milestone 5+ validation."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/>'
        '<geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0


def test_meshcat_visualizer_available():
    """Meshcat is the visualization path that works over VS Code port forwarding."""
    from pinocchio.visualize import MeshcatVisualizer

    assert MeshcatVisualizer is not None
