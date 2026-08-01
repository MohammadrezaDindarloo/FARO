"""Physical demonstrations of the constraint blocks (paper Eqs. 7-13).

Each scenario tells one physical story -- lifting a foot, sliding a box off a
platform, pushing a contact force outside its friction pyramid -- and sweeps a
single parameter from "satisfied" through to "violated".

Why this exists: Milestones 3-5 assemble these constraints into NLPs with thousands
of rows, where a wrong sign or a swapped axis shows up only as "infeasible". The
scenarios pin down, one constraint at a time and *before* that happens, both

  * that the constraint means what the paper says it means, and
  * WHERE exactly it breaks -- each scenario predicts its own crossing point
    analytically from the paper's equation, and the tests check the measured
    crossing against that prediction.

Layered so the physics is testable without graphics: this package is headless, and
`faro.viz.constraint_viz` renders the same scenarios in Meshcat.
"""

from faro.scenarios.constraint_demos import (
    ALL_SCENARIOS,
    Scenario,
    ScenarioStep,
    SweepResult,
    get_scenario,
    run_sweep,
)

__all__ = [
    "ALL_SCENARIOS",
    "Scenario",
    "ScenarioStep",
    "SweepResult",
    "get_scenario",
    "run_sweep",
]
