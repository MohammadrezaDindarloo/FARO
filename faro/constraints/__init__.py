"""Shared constraint blocks -- paper Section II-C, Eqs. 7-13.

Reusable CasADi expressions for every constraint the FARO hierarchy shares. See
README.md in this directory for the equation-to-file map and, importantly, the
table of which stage enforces which subset.

Nothing here builds or solves an NLP; that is Milestones 3-5.
"""

from faro.constraints.block import ConstraintBlock, merge
from faro.constraints.collision import (
    WitnessData,
    collision_avoidance,
    query_witness,
    signed_distance_expr,
)
from faro.constraints.contact import (
    contact_full,
    contact_kinematic,
    contact_wrench,
    patch_containment_is_possible,
)
from faro.constraints.dynamics_object import (
    gravity_wrench,
    object_integration,
    object_newton_euler,
    spatial_inertia,
    transform_wrench_to_body,
)
from faro.constraints.dynamics_robot import (
    center_of_mass,
    centroidal_consistency,
    centroidal_momentum_rate,
    integrate_configuration,
    robot_dynamics,
    total_mass,
)
from faro.constraints.frames import (
    RX_PI,
    log3,
    relative_patch_transform,
    relative_transform,
    rotated_half_extents,
    skew,
)
from faro.constraints.limits import (
    actuated_slice,
    actuated_torque,
    default_velocity_limits,
    joint_position_limits,
    joint_velocity_limits,
    torque_speed_limits,
)
from faro.constraints.no_slip import applies_between, no_slip

__all__ = [
    # plumbing
    "ConstraintBlock", "merge",
    # Eq. 7
    "contact_kinematic", "contact_wrench", "contact_full", "patch_containment_is_possible",
    # Eq. 8 / 16
    "no_slip", "applies_between",
    # Eq. 9
    "WitnessData", "collision_avoidance", "query_witness", "signed_distance_expr",
    # Eq. 10
    "robot_dynamics", "centroidal_consistency", "centroidal_momentum_rate",
    "integrate_configuration", "center_of_mass", "total_mass",
    # Eq. 11
    "object_newton_euler", "object_integration", "spatial_inertia",
    "gravity_wrench", "transform_wrench_to_body",
    # Eqs. 12-13
    "actuated_slice", "actuated_torque", "joint_position_limits",
    "joint_velocity_limits", "torque_speed_limits", "default_velocity_limits",
    # frames / conventions
    "RX_PI", "log3", "relative_transform", "relative_patch_transform",
    "rotated_half_extents", "skew",
]
