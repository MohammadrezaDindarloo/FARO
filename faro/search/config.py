"""Turn `configs/search/*.yaml` into Alg. 1's arguments.

Kept apart from `search.py` for the same reason every other stage separates its
config from its mathematics: the search must be runnable from Python with explicit
arguments and no YAML anywhere, so that tests can drive it without a config file
and so that a sweep over C or alpha is a loop rather than ten near-identical files.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from faro.core.modes import ContactMode
from faro.scene.scene import Scene
from faro.search.tree import Goal
from faro.search.uct import DEFAULT_ALPHA, DEFAULT_C, DEFAULT_K
from faro.utils.config import load_config


def _full_mode(scene: Scene, partial: dict) -> ContactMode:
    """A partial `{interface: partner}` completed with `free` everywhere else.

    Eq. 1 requires an assignment for EVERY interface, and `ContactMode.validate`
    enforces it. Writing the root and the goal as partial assignments in YAML is
    much less error-prone than listing five interfaces every time, but the mode that
    reaches the solver has to be complete -- an omitted interface would silently
    drop its contact constraints rather than mean "free".
    """
    unknown = set(partial) - set(scene.interfaces)
    if unknown:
        raise ValueError(
            f"config names interfaces the scene does not declare: {sorted(unknown)}. "
            f"Known: {sorted(scene.interfaces)}"
        )
    return ContactMode.from_dict({name: partial.get(name) for name in scene.interfaces})


@dataclass
class SearchConfig:
    """Everything Alg. 1 needs, resolved against a scene."""

    root: ContactMode
    goal: Goal
    budget: float = 7200.0
    max_depth: int = 5
    successor_policy: str = "all"
    filters: tuple = ("M", "E")
    C: float = DEFAULT_C
    k: float = DEFAULT_K
    alpha: float = DEFAULT_ALPHA
    seed: int = 0
    name: str = "default"

    @classmethod
    def from_config(cls, scene: Scene, name_or_cfg: str | dict = "default") -> "SearchConfig":
        cfg = (load_config("search", name_or_cfg)
               if isinstance(name_or_cfg, str) else dict(name_or_cfg))

        known = {"name", "uct", "budget", "max_depth", "successor_policy",
                 "filters", "goal", "root", "seed"}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(
                f"unknown keys in search config: {sorted(unknown)}. Known: {sorted(known)}"
            )

        uct = dict(cfg.get("uct") or {})
        goal_cfg = dict(cfg.get("goal") or {})
        if "require" not in goal_cfg:
            raise KeyError(
                "search config: `goal.require` is missing. Alg. 1 line 13 needs a "
                "GOAL test, and defaulting it would silently search for nothing."
            )

        filters = cfg.get("filters", ["M", "E"])
        return cls(
            root=_full_mode(scene, dict(cfg.get("root") or {})),
            goal=Goal(require=dict(goal_cfg["require"]),
                      released=tuple(goal_cfg.get("released") or ())),
            budget=float(cfg.get("budget", 7200.0)),
            max_depth=int(cfg.get("max_depth", 5)),
            successor_policy=str(cfg.get("successor_policy", "all")),
            filters=filters if isinstance(filters, str) else tuple(filters),
            C=float(uct.get("C", DEFAULT_C)),
            k=float(uct.get("k", DEFAULT_K)),
            alpha=float(uct.get("alpha", DEFAULT_ALPHA)),
            seed=int(cfg.get("seed", 0)),
            name=str(cfg.get("name", "default")),
        )

    def as_kwargs(self) -> dict:
        """The keyword arguments `search.search` takes, minus scene/root/goal."""
        return {
            "budget": self.budget, "filters": self.filters, "C": self.C, "k": self.k,
            "alpha": self.alpha, "max_depth": self.max_depth,
            "successor_policy": self.successor_policy, "seed": self.seed,
        }
