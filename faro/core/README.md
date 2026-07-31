# `faro.core` — shared vocabulary

The common language every other module speaks. Contains **no optimization code**.

This package exists so that `mode_edge/`, `kso/`, `to/` and `search/` never import
from each other. All four need to agree on what a "contact patch" is; if that
definition lived in `mode_edge/`, then KSO would depend on mode/edge purely by
accident of build order, and the modules would stop being swappable.

## Paper grounding

Section II-A (contact-mode definition) and Section II-C 1 (contact):

> "all end-effectors and environment contact interfaces are modeled as rectangular
> patches with patch-to-patch unilateral contact"
> "The patch normals are aligned with their local z-axes."
> "The vector `h` denotes the half-extents of a rectangular patch"

## Contents

| Object | Meaning |
|---|---|
| `ContactPatch` | A rectangular planar patch: parent frame + placement + half-extents `h`. |
| `Attachment` | `ROBOT` / `OBJECT` / `ENVIRONMENT` — determines which decision variables a patch's placement depends on. |
| `Interface` | A named contact site plus its allowed partners (paper Table IV). |

`Attachment` is the load-bearing distinction: it decides whether a patch's world
placement is a function of the robot configuration `q`, of an object's pose, or of
nothing at all. Milestone 2's constraint code branches on exactly this.

## The normal sign convention

**Ours, not the paper's — the paper never states a sign.**

> A patch's local **+z points outward**: away from the body it belongs to, into the
> free space it can touch.

So the floor's normal is `+z` (up), and a foot sole's normal points *down*, out of
the foot. Two patches in contact are therefore face-to-face, with **anti-parallel**
z-axes.

Consequence for Milestone 2: Eq. 7a's normal-alignment residual must account for
that flip. This is deliberately isolated in one place and revisited when Eq. 7a is
actually written; see [docs/ambiguities.md](../../docs/ambiguities.md).

## Verify

```bash
pytest tests/test_milestone1_scene.py -k "patch or box_face" -v
```

You should see the six box faces each produce an outward normal along the expected
axis, proper rotations (det = +1), and half-extents equal to the box's *other* two
dimensions.
