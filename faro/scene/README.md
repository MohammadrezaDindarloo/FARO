# `faro.scene` — scene assembly

> "The scene defines the robot, movable objects, environment geometry, and available
> contact interfaces from which candidate contact-mode sequences are generated."
> — paper Fig. 2 caption

A `Scene` is the single object later stages consume. It holds no optimization state
and does no solving.

## Paper grounding

- **Eq. 5b** — objects carry their own pose and body velocity, separate from the robot
- **Eq. 11** — rigid-body object dynamics (mass + inertia carried here, used at Milestone 5)
- **Table IV** — the allowed-contact graph, reproduced exactly in `configs/scenes/box_placement.yaml`
- **Fig. 4** — the box-placement task ("place the white box on the platform")

## Objects are not part of the robot model

The box is deliberately **not** added as an extra floating joint on the Pinocchio
model. The paper gives each object its own state (Eq. 5b) and its own dynamics
(Eq. 11), separate from the robot's centroidal dynamics (Eq. 10). Fusing them would
merge two things the paper keeps apart and would make Eq. 11 impossible to write
cleanly.

## Face naming

World-frame at the object's nominal orientation:

```
front = +x    rear = -x    left = +y    right = -y    top = +z    bottom = -z
```

This is what makes Table IV consistent: *"Left hand → box left, box front"* are the
`+y` and `+x` faces, on the robot's left. Face patches are **generated** from the box
geometry rather than hand-written in YAML — six SE3s with correct outward normals and
correctly-paired half-extents are easy to get subtly wrong by hand, and a flipped
normal would grip the box by the wrong face while still solving.

## Validation

`Scene.validate()` runs at load and checks that every robot patch names a real frame,
every object patch names a real object, and every `allowed` entry names a real patch.
A typo would otherwise surface as a mysteriously infeasible NLP several milestones
later.

## The branching factor is a paper reproduction check

```
3 (left hand) × 3 (right hand) × 2 (left foot) × 2 (right foot) × 3 (box bottom) = 108
```

Section IV-B: *"yielding a maximum branching factor of 108"*. `scene.branching_factor()`
returns exactly 108, which is strong evidence our reading of Table IV is correct.

Caveat: this is the raw combinatorial count. It ignores the consistency condition of
Section II-A (if A touches B then B touches A). Treat a future mismatch as a question
about consistency handling, not a bug here.

## Run

```python
from faro.scene.scene import Scene
scene = Scene.from_config("box_placement")
print(scene.summary())
```

## Verify

```bash
pytest tests/test_milestone1_scene.py -k "scene or box or foot or palm" -v
```
