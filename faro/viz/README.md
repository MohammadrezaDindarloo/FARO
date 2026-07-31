# `faro.viz` — visualization

Meshcat runs a small web server on the cluster node and renders in a browser on your
laptop through VS Code port forwarding. That is why it — not an OpenGL window — is
the visualization path for Milestones 1–4. See [SETUP.md](../../SETUP.md) §4.

## Why patches are drawn

Patch placements and normal directions are the easiest thing in the whole project to
get silently wrong: a flipped normal or a mis-signed half-extent produces constraints
that **solve happily and mean the wrong thing**. Drawing them makes the convention
visible and checkable by eye *before* any solver depends on it.

## Colour key

| Colour | Meaning |
|---|---|
| 🟢 green | robot patch — moves with `q` |
| 🟠 orange | object patch — moves with the object pose |
| 🔵 blue | environment patch — fixed in the world |
| 🔴 red stub | the patch's outward `+z` normal |

Every red stub must point **away** from the body it belongs to. Soles point down,
palms point inward at each other, floor and platform point up.

## Gotcha worth knowing

meshcat/three.js cylinders run along their local **Y** axis, so pointing one along
`+Z` needs an extra `Rx(90°)`. Forgetting it is a classic silent 90-degree bug; it is
handled once by `_Y_TO_Z` in `meshcat_viz.py`.

## Run

```python
from faro.scene.scene import Scene
from faro.viz.meshcat_viz import SceneVisualizer

scene = Scene.from_config("box_placement")
vis = SceneVisualizer(scene)
vis.display(scene.robot.q_nominal)
print(vis.url)
```

Robot patches are re-placed on every `display(q)`; objects move via
`update_object_pose(name, pose)`, since object pose is separate state (Eq. 5b).

## Verify

Run `scripts/01_load_and_visualize_robot.py` and open the viewer — see the Milestone 1
section of the top-level [README](../../README.md) for exactly what to look for.
