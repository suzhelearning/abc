"""Render validated :class:`UrdfModel` data as deterministic MuJoCo XML."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .urdf_model import UrdfJoint, UrdfLink, UrdfModel


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(item) for item in rpy)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=float,
    )


def rpy_to_mujoco_quat(rpy: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert URDF fixed-axis RPY to MuJoCo's ``w x y z`` quaternion."""
    matrix = _rpy_matrix(rpy)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * root
        x = (matrix[2, 1] - matrix[1, 2]) / root
        y = (matrix[0, 2] - matrix[2, 0]) / root
        z = (matrix[1, 0] - matrix[0, 1]) / root
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            root = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            x = 0.25 * root
            y = (matrix[0, 1] + matrix[1, 0]) / root
            z = (matrix[0, 2] + matrix[2, 0]) / root
            w = (matrix[2, 1] - matrix[1, 2]) / root
        elif index == 1:
            root = math.sqrt(max(0.0, 1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2])) * 2.0
            x = (matrix[0, 1] + matrix[1, 0]) / root
            y = 0.25 * root
            z = (matrix[1, 2] + matrix[2, 1]) / root
            w = (matrix[0, 2] - matrix[2, 0]) / root
        else:
            root = math.sqrt(max(0.0, 1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2])) * 2.0
            x = (matrix[0, 2] + matrix[2, 0]) / root
            y = (matrix[1, 2] + matrix[2, 1]) / root
            z = 0.25 * root
            w = (matrix[1, 0] - matrix[0, 1]) / root
    quat = np.asarray((w, x, y, z), dtype=float)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat *= -1.0
    return tuple(float(item) for item in quat)


def _joint_map(model: UrdfModel) -> dict[str, UrdfJoint]:
    return {joint.child: joint for joint in model.joints}


def arm_projection_links(model: UrdfModel) -> frozenset[str]:
    """Return base, arm and fixed wrist-frame links for the 14-DoF view."""
    by_child = _joint_map(model)
    keep = {model.root, "l_wrist", "r_wrist"}
    for wrist in ("l_wrist", "r_wrist"):
        cursor = wrist
        while cursor != model.root:
            joint = by_child.get(cursor)
            if joint is None:
                raise ValueError(f"missing wrist chain for {wrist!r}")
            keep.add(cursor)
            keep.add(joint.parent)
            cursor = joint.parent
    # Fixed arm-base children and their seven revolute links per side.
    changed = True
    while changed:
        changed = False
        for joint in model.joints:
            if joint.parent in keep and joint.child not in keep and (
                joint.type == "fixed" or joint.name.startswith("Joint")
            ):
                keep.add(joint.child)
                changed = True
    return frozenset(keep)


def _add_inertial(parent: ET.Element, link: UrdfLink) -> None:
    if link.inertial is None:
        return
    inertia = np.asarray(link.inertial.inertia, dtype=float)
    values = (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])
    ET.SubElement(
        parent,
        "inertial",
        pos=_fmt(link.inertial.com),
        mass=f"{link.inertial.mass:.17g}",
        fullinertia=_fmt(values),
    )


def _add_geometry(
    body: ET.Element,
    link: UrdfLink,
    mesh_assets: Mapping[str, tuple[str, str, tuple[float, float, float]]],
    collision_assets: Mapping[tuple[str, int], Sequence[str]],
) -> None:
    for geometry in link.visuals:
        asset_name, _, _ = mesh_assets[str(geometry.path.resolve())]
        ET.SubElement(
            body,
            "geom",
            type="mesh",
            mesh=asset_name,
            pos=_fmt(geometry.origin),
            quat=_fmt(rpy_to_mujoco_quat(geometry.rpy)),
            contype="0",
            conaffinity="0",
            group="1",
            density="0",
            rgba="0.75 0.75 0.75 1",
        )
    for index, geometry in enumerate(link.collisions):
        pieces = collision_assets.get((link.name, index))
        if not pieces:
            raise ValueError(f"missing validated collision artifact for {link.name}[{index}]")
        pose = {"pos": _fmt(geometry.origin), "quat": _fmt(rpy_to_mujoco_quat(geometry.rpy))}
        for piece in pieces:
            ET.SubElement(
                body,
                "geom",
                type="mesh",
                mesh=piece,
                contype="1",
                conaffinity="1",
                group="0",
                density="0",
                **pose,
            )


def _children(model: UrdfModel) -> dict[str, tuple[UrdfJoint, ...]]:
    result: dict[str, list[UrdfJoint]] = {}
    for joint in model.joints:
        result.setdefault(joint.parent, []).append(joint)
    return {key: tuple(value) for key, value in result.items()}


def render_mjcf(
    model: UrdfModel,
    output_path: str | Path,
    *,
    mesh_assets: Mapping[str, tuple[str, str, tuple[float, float, float]]],
    collision_assets: Mapping[tuple[str, int], Sequence[str]],
    mode: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Write one full or arm-projection MJCF and return joint/exclude maps."""
    if mode not in {"full", "arm"}:
        raise ValueError("mode must be 'full' or 'arm'")
    allowed = frozenset(link.name for link in model.links) if mode == "full" else arm_projection_links(model)
    by_name = {link.name: link for link in model.links}
    child_map = _children(model)
    root = ET.Element("mujoco", model="tianji_wuji2_unified" if mode == "full" else "tianji_wuji2_arm")
    ET.SubElement(root, "compiler", angle="radian", meshdir=".", inertiafromgeom="false")
    ET.SubElement(root, "option", timestep=f"{1.0 / 480.0:.17g}", integrator="implicitfast")
    ET.SubElement(root, "size", nuser_jnt="1")
    asset_element = ET.SubElement(root, "asset")
    for source, (asset_name, asset_path, scale) in sorted(mesh_assets.items()):
        del source
        attributes = {"name": asset_name, "file": asset_path}
        if tuple(scale) != (1.0, 1.0, 1.0):
            attributes["scale"] = _fmt(scale)
        ET.SubElement(asset_element, "mesh", inertia="shell", **attributes)
    source_mesh_names = {asset_name for asset_name, _, _ in mesh_assets.values()}
    collision_names = sorted(
        {piece for pieces in collision_assets.values() for piece in pieces} - source_mesh_names
    )
    for piece_name in collision_names:
        ET.SubElement(asset_element, "mesh", name=piece_name, file=f"collision/{piece_name}.stl", inertia="shell")

    worldbody = ET.SubElement(root, "worldbody")
    joint_order: list[str] = []
    excludes: list[tuple[str, str]] = []

    def visit(parent_element: ET.Element, link_name: str, parent_joint: UrdfJoint | None) -> None:
        if link_name not in allowed:
            return
        link = by_name[link_name]
        attributes = {"name": link_name}
        if parent_joint is not None:
            attributes.update(pos=_fmt(parent_joint.origin), quat=_fmt(rpy_to_mujoco_quat(parent_joint.rpy)))
        body = ET.SubElement(parent_element, "body", **attributes)
        _add_inertial(body, link)
        if parent_joint is not None and parent_joint.type == "revolute":
            if parent_joint.limit is None:
                raise ValueError(f"revolute joint {parent_joint.name!r} has no limit")
            ET.SubElement(
                body,
                "joint",
                name=parent_joint.name,
                type="hinge",
                axis=_fmt(parent_joint.axis),
                range=_fmt(parent_joint.limit),
                limited="true",
                damping="0.1",
            )
            joint_order.append(parent_joint.name)
        _add_geometry(body, link, mesh_assets, collision_assets)
        if link_name in {"l_wrist", "r_wrist"}:
            ET.SubElement(body, "site", name=f"{link_name}_target", pos="0 0 0", size="0.008", rgba="1 0.2 0.2 1")
            camera_name = "left_wrist" if link_name == "l_wrist" else "right_wrist"
            camera_x = "-0.035" if link_name == "l_wrist" else "0.035"
            ET.SubElement(
                body,
                "camera",
                name=camera_name,
                pos=f"{camera_x} 0 0.005",
                quat="1 0 0 0",
                fovy="70",
            )
        for child_joint in child_map.get(link_name, ()):
            if child_joint.child not in allowed:
                continue
            excludes.append((link_name, child_joint.child))
            visit(body, child_joint.child, child_joint)

    visit(worldbody, model.root, None)
    if mode == "full":
        ET.SubElement(worldbody, "geom", name="ground", type="plane", size="5 5 0.1", contype="1", conaffinity="1")
        ET.SubElement(worldbody, "light", name="key_light", pos="1 -1 3", dir="-1 1 -3", directional="true")
        ET.SubElement(
            worldbody,
            "camera",
            name="top",
            pos="0.5 0 1.8",
            xyaxes="0 1 0 -0.9977851579 0 0.0665190105",
            fovy="70",
        )

    actuator = ET.SubElement(root, "actuator")
    joint_by_name = {joint.name: joint for joint in model.joints}
    for name in joint_order:
        joint = joint_by_name[name]
        is_arm = name.startswith("Joint")
        effort = abs(float(joint.effort)) if joint.effort is not None and abs(float(joint.effort)) > 0 else 1.0
        attributes = {
            "name": f"{name}_position",
            "joint": name,
            "kp": "500" if is_arm else "1",
            "forcerange": f"{-effort:.17g} {effort:.17g}",
            "forcelimited": "true",
        }
        if is_arm:
            attributes["dampratio"] = "1"
        if joint.limit is not None:
            attributes.update(ctrlrange=_fmt(joint.limit), ctrllimited="true")
        ET.SubElement(actuator, "position", **attributes)

    contact = ET.SubElement(root, "contact")
    for parent, child in excludes:
        ET.SubElement(contact, "exclude", body1=parent, body2=child)
    ET.indent(root, space="  ")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True, short_empty_elements=True)
    return tuple(joint_order), tuple(excludes)


__all__ = ["arm_projection_links", "render_mjcf", "rpy_to_mujoco_quat"]
