"""Commenting links out of a URDF, reversibly.

The FARO paper cuts the robot's hand off at the wrist and puts the rectangular
contact patch on the resulting cut face (docs/ambiguities.md #7b). Keeping the G1's
`*_rubber_hand` link would leave a 13 cm five-fingered mesh sticking out 0.09 m past
the contact patch -- straight through whatever the robot is touching.

Hiding the mesh in the viewer would fix the picture but not the model: the link's
0.170 kg would still sit in Eq. 10's centroidal dynamics.

So the hand is removed from a URDF that lives in **our** repo. Crucially it is
**commented out, not deleted**: the original markup stays in the file inside an XML
comment, next to a note explaining why. Putting the full hand back is an edit, not a
rewrite -- delete the comment markers and reload.

    python -m faro.robots.urdf_surgery      # regenerate assets/ from example-robot-data

The installed `example-robot-data` URDF is never modified.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# example-robot-data source and the derived copy we actually load.
SOURCE_SPEC = "erd://g1_description/urdf/g1_29dof_rev_1_0.urdf"
OUTPUT_SPEC = "assets://robots/g1_description/urdf/g1_29dof_faro.urdf"

HAND_LINKS = ["left_rubber_hand", "right_rubber_hand"]

HAND_NOTE = """
================================================================================
FARO: HAND REMOVED AT THE WRIST - commented out, not deleted.

WHY (docs/ambiguities.md #7b):
  The paper cuts the robot's hand off at the wrist and puts the rectangular
  contact patch on the resulting cut face. Its figures (1, 4, 5) show the arms
  ending in a flat stub.

  Our contact patch therefore sits at x = 0.0415 in `*_wrist_yaw_link` - exactly
  the plane where `*_hand_palm_joint` below bolts the hand on. With the hand still
  attached, its 13 cm five-fingered mesh would extend 0.09 m PAST the contact
  patch and pass straight through whatever the robot is touching.

  Hiding the mesh in the viewer was rejected: it would fix only the picture. The
  link's 0.170 kg would still enter Eq. 10's centroidal dynamics, and any collision
  geometry added later (ambiguity #7c) would still intersect the box.

WHAT THIS COSTS:
  Both are LEAF links on FIXED joints, so nq / nv and every joint index are
  unchanged. Only 0.170 kg each (2 x 0.5 % of the 33.34 kg total), one frame each,
  and one visual mesh each go away. No collision geometry is lost - these links
  never had any.

TO PUT THE FULL HAND BACK:
  Delete the comment markers around the joint + link blocks below, on both sides,
  and reload. Then move the contact patch back onto the palm in
  `configs/scenes/box_placement.yaml` - otherwise the patch stays at the wrist and
  the hand hangs uselessly past it, which is the bug this note exists to prevent.

This file is GENERATED from example-robot-data by
`python -m faro.robots.urdf_surgery`. If you edit it by hand, that regeneration
will overwrite your edits.
================================================================================
"""


class LinkRemovalError(ValueError):
    """A link named for removal cannot be removed safely."""


def _check_removable(root: ET.Element, name: str) -> list[ET.Element]:
    """Return the joints attaching `name`, or raise if removing it is unsafe.

    Only *leaf* links attached by a **fixed** joint may go. Both restrictions are
    deliberate:

      * removing a link with children would silently orphan the subtree;
      * removing a link on a movable joint would change `nq`/`nv` and renumber every
        configuration index, quietly invalidating `q_nominal`, the joint limits of
        Eq. 13a, and every cached contact-mode result.

    A fixed leaf link is the one case where removal only subtracts mass, a frame and
    some geometry -- which is exactly what "the paper cut the hand off" means.
    """
    links = {link.get("name") for link in root.findall("link")}
    if name not in links:
        raise LinkRemovalError(
            f"cannot remove link {name!r}: no such link. Known links: {sorted(links)}"
        )

    joints = root.findall("joint")
    parents = [j for j in joints if (j.find("child") is not None
                                     and j.find("child").get("link") == name)]
    children = [j for j in joints if (j.find("parent") is not None
                                      and j.find("parent").get("link") == name)]
    if children:
        kids = sorted(j.find("child").get("link") or "?" for j in children)
        raise LinkRemovalError(
            f"cannot remove link {name!r}: it still has child links {kids}. "
            f"Remove those first, or it would orphan the subtree."
        )
    for joint in parents:
        if joint.get("type") != "fixed":
            raise LinkRemovalError(
                f"cannot remove link {name!r}: attached by {joint.get('name')!r} of type "
                f"{joint.get('type')!r}, not 'fixed'. Removing it would change nq/nv "
                f"and renumber every joint index."
            )
    return parents


def _ensure_comment_safe(text: str, what: str) -> None:
    """XML comments may not contain `--`. Raise rather than emit a corrupt file.

    Caught this the hard way: the explanatory note below originally used `--` as a
    dash, which produced a URDF that no parser would read.
    """
    if "--" in text:
        raise LinkRemovalError(
            f"cannot comment out: {what} contains '--', which is illegal inside an "
            f"XML comment. Use a single dash."
        )


def comment_out_links(urdf_xml: str, link_names: list[str], note: str = "") -> str:
    """Return `urdf_xml` with each named link (and its fixed joint) commented out.

    The original markup is preserved verbatim inside an XML comment, so the change
    is reversible by hand. `note` is emitted once, before the first commented block.
    """
    if not link_names:
        return urdf_xml

    _ensure_comment_safe(note, "the explanatory note")
    root = ET.fromstring(urdf_xml)
    first = True

    for name in link_names:
        joints = _check_removable(root, name)
        link = next(el for el in root.findall("link") if el.get("name") == name)

        # Serialize before detaching, so the comment holds the original markup.
        blocks = [ET.tostring(el, encoding="unicode").strip() for el in (*joints, link)]
        body = "\n".join(blocks)
        # "--" is illegal inside an XML comment; the URDF has none, but a mesh path
        # or a future edit could, and a silently corrupt file is worse than a raise.
        _ensure_comment_safe(body, f"the markup of {name!r}")

        index = list(root).index(joints[0] if joints else link)
        for el in (*joints, link):
            root.remove(el)

        if first and note:
            root.insert(index, ET.Comment(note))
            index += 1
            first = False
        root.insert(index, ET.Comment(f"\n{body}\n"))

    return ET.tostring(root, encoding="unicode")


def link_report(urdf_xml: str, link_names: list[str]) -> list[str]:
    """Human-readable summary of what would be commented out, for logging."""
    root = ET.fromstring(urdf_xml)
    report = []
    for link in root.findall("link"):
        name = link.get("name")
        if name not in link_names:
            continue
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value")) if inertial is not None else 0.0
        report.append(
            f"{name}: {mass:.3f} kg, {len(link.findall('visual'))} visual, "
            f"{len(link.findall('collision'))} collision geometries"
        )
    return report


def build_faro_urdf(overwrite: bool = True) -> Path:
    """Generate `assets://.../g1_29dof_faro.urdf` from example-robot-data.

    Mesh references stay as `package://example-robot-data/...`, which still resolve
    because the loader passes example-robot-data's share directory as `package_dirs`.
    So the copy carries no meshes of its own.
    """
    from faro.utils.paths import resolve_asset

    source = resolve_asset(SOURCE_SPEC)
    output = resolve_asset(OUTPUT_SPEC)
    if output.exists() and not overwrite:
        return output

    xml = comment_out_links(source.read_text(), HAND_LINKS, HAND_NOTE)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "<?xml version=\"1.0\" ?>\n"
        f"<!-- GENERATED by `python -m faro.robots.urdf_surgery` from {source.name}. -->\n"
        + xml
        + "\n"
    )
    return output


def main() -> None:
    from faro.utils.paths import resolve_asset

    source = resolve_asset(SOURCE_SPEC)
    for line in link_report(source.read_text(), HAND_LINKS):
        print(f"  commenting out  {line}")
    output = build_faro_urdf()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
