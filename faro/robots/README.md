# `faro.robots` — robot loading and Pinocchio wrapping

The only module that knows how a robot gets from a URDF into memory. Everything
downstream receives a `RobotModel` and never touches a file path again — that is
what makes the robot swappable by editing `configs/robots/*.yaml`.

## Paper grounding

No single equation. Provides the substrate for:
- **Eq. 12** joint position/velocity limits (`joint_limits()`)
- **Eq. 13** torque-speed limits (actuation selection via `n_actuated`)
- **Eq. 10** centroidal robot dynamics (Milestone 5, via Pinocchio's CMM)
- Symbolic kinematics for **Eqs. 14 / 15 / 17** (`casadi_model()`)

## Floating base is mandatory

FARO models humanoid loco-manipulation, so the root joint is a **free-flyer**:

```python
model = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
```

A fixed-base load silently drops the 6 base DoF and makes the formulation *wrong*,
not merely inaccurate — the underactuation of those 6 DoF is precisely why contact
forces are needed at all. So it is asserted, never assumed: `nq - nv == 1` is the
signature of a quaternion-parameterised free-flyer, and `_check_floating_base`
raises if it does not hold.

For the G1: **nq=36, nv=35, 29 actuated joints**.

## Design notes

- `frame_id()` raises on an unknown frame. Pinocchio's `getFrameId` instead returns
  `model.nframes`, which turns a typo into silent nonsense far downstream.
- `build_configuration()` takes **named** joint angles, so postures in configs stay
  readable and survive a robot swap.
- `place_on_ground()` is purely kinematic — it positions the robot, it does **not**
  check static balance. Balance is a dynamics question and only arises at TO (Eq. 17).
- `q_nominal` is not cosmetic: it is the regularization target of the mode/edge NLP
  (Eq. 14) and the initial guess everywhere, so it affects *which* solutions are
  found and how fast. It is a tuning knob — see ambiguity #5.

## Run

```python
from faro.robots.robot_model import RobotModel
robot = RobotModel.from_config("g1")
print(robot)   # RobotModel(name='g1_29dof', nq=36, nv=35, actuated=29)
```

## Verify

```bash
pytest tests/test_milestone1_scene.py -k robot -v
```
