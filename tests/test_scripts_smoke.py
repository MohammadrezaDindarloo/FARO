"""The demo scripts must at least run.

Added after a real escape: cutting the hand off at the wrist removed the
`*_rubber_hand` frame, but `01_load_and_visualize_robot.py` still had that frame name
hard-coded in a print loop. The whole test suite passed, because nothing here
executed a script -- the failure only surfaced when a human ran it.

These tests import each script as a module and call its non-visual entry points, so a
stale frame name, a renamed patch or a changed config key fails in CI rather than in
front of someone.

Meshcat is deliberately never started: the visual layer is exercised by hand, and a
server would make these slow and flaky.

    pytest tests/test_scripts_smoke.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from faro.scene.scene import Scene

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str) -> ModuleType:
    """Import a numbered script by path (they are not an importable package)."""
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_config("box_placement")


def test_every_script_imports():
    """A syntax error or a bad top-level import should never reach the user."""
    for path in sorted(SCRIPTS.glob("*.py")):
        _load(path.name)


def test_milestone1_report_runs_against_the_current_scene(scene, capsys):
    """The exact call that broke: it touches real frames by name.

    Any patch moved to a different parent link, or any renamed frame, shows up here.
    """
    module = _load("01_load_and_visualize_robot.py")
    module.report_scene(scene, scene.robot.q_nominal)

    out = capsys.readouterr().out
    assert "Raw branching factor: 108" in out
    assert "left_hand_patch" in out
    assert "rubber_hand" not in out, "the hand is cut off at the wrist; this frame is gone"


def test_milestone1_reports_every_patch_in_the_scene(scene, capsys):
    """The report must not silently skip a patch that was added to the config."""
    module = _load("01_load_and_visualize_robot.py")
    module.report_scene(scene, scene.robot.q_nominal)

    out = capsys.readouterr().out
    for name in scene.patches:
        assert name in out, f"patch {name} is missing from the Milestone 1 report"


def test_milestone2_playground_runs_headless(scene):
    """Every scenario builds and sweeps without a viewer."""
    from faro.scenarios.constraint_demos import ALL_SCENARIOS, resolve_predictions, run_sweep

    resolve_predictions(scene)
    for scenario in ALL_SCENARIOS:
        result = run_sweep(scenario, scene)
        assert result.steps, f"{scenario.key} produced no steps"


def test_every_robot_frame_named_in_a_config_exists(scene):
    """Config-level guard: a patch pointing at a removed link must fail loudly.

    This is the general form of the bug -- `urdf_surgery` can remove any link, and a
    patch left pointing at it would otherwise fail deep inside a solve.
    """
    from faro.core.patches import Attachment

    for patch in scene.patches.values():
        if patch.attachment is Attachment.ROBOT:
            assert scene.robot.model.existFrame(patch.parent), (
                f"patch {patch.name!r} is attached to {patch.parent!r}, which the robot "
                f"model does not have -- was that link removed from the URDF?"
            )
