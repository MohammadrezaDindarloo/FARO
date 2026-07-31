"""Filesystem path resolution.

One place that knows where things live, so no other module hard-codes a path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# faro/utils/paths.py -> faro/utils -> faro -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
ASSET_DIR = REPO_ROOT / "assets"


def conda_prefix() -> Path:
    """Active environment prefix.

    Uses $CONDA_PREFIX when the env is activated, and falls back to sys.prefix so
    that calling the env's python directly (e.g. `.../envs/faro/bin/python foo.py`)
    still works.
    """
    return Path(os.environ.get("CONDA_PREFIX") or sys.prefix)


def example_robot_data_share() -> Path:
    """Directory to pass to Pinocchio as the `package://` search root.

    conda-forge `example-robot-data` 5.x is a DATA-ONLY package: URDFs and meshes,
    no Python module. Its URDFs reference meshes as

        package://example-robot-data/robots/g1_description/meshes/foo.STL

    so the search root must be the directory *containing* `example-robot-data`,
    i.e. `$CONDA_PREFIX/share` -- not the `robots/` directory itself.
    """
    share = conda_prefix() / "share"
    if not (share / "example-robot-data" / "robots").is_dir():
        raise FileNotFoundError(
            f"example-robot-data assets not found under {share}. "
            "Is the `faro` conda env active? See SETUP.md section 4b."
        )
    return share


def robot_asset(*parts: str) -> Path:
    """Absolute path to a file inside example-robot-data's `robots/` tree.

    >>> robot_asset("g1_description", "urdf", "g1_29dof_rev_1_0.urdf")
    """
    path = example_robot_data_share() / "example-robot-data" / "robots" / Path(*parts)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_asset(spec: str) -> Path:
    """Resolve an asset reference from a config file.

    Supported forms, in order:
      - "erd://g1_description/urdf/g1.urdf"  -> example-robot-data
      - "assets://box.urdf"                  -> the repo's own assets/ dir
      - any absolute path
      - a path relative to the repo root
    """
    if spec.startswith("erd://"):
        return robot_asset(*spec[len("erd://"):].split("/"))
    if spec.startswith("assets://"):
        return ASSET_DIR / spec[len("assets://"):]
    path = Path(spec)
    return path if path.is_absolute() else REPO_ROOT / path
