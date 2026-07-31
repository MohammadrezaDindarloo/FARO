"""YAML config loading.

FARO's hard requirement is that robot, scene and solver are swappable *as data*.
Everything a module needs to be reconfigured therefore arrives through here rather
than as a constant in code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from faro.utils.paths import CONFIG_DIR


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def load_config(kind: str, name: str) -> dict[str, Any]:
    """Load `configs/<kind>/<name>.yaml`.

    >>> load_config("robots", "g1")
    >>> load_config("scenes", "box_placement")

    `name` may also be a path to a file elsewhere, so a config can live outside the
    repo without any code change.
    """
    candidate = Path(name)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return load_yaml(candidate)
    return load_yaml(CONFIG_DIR / kind / f"{name}.yaml")


def require(cfg: dict[str, Any], key: str, context: str = "config") -> Any:
    """Fetch a required key, failing loudly rather than defaulting silently.

    Used for values where a wrong silent default would produce a plausible-looking
    but incorrect optimization problem -- which is far worse than a crash.
    """
    if key not in cfg:
        raise KeyError(f"{context}: missing required key '{key}' (have: {sorted(cfg)})")
    return cfg[key]
