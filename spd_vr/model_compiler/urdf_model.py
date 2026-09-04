"""Strict, dependency-light parser for the authoritative Tianji/Wuji2 URDF."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import xml.etree.ElementTree as ET

import numpy as np


Vec3 = tuple[float, float, float]
Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class Inertial:
    mass: float
    com: Vec3
    inertia: Mat3

    @property
    def origin(self) -> Vec3:
        return self.com

    @property
    def tensor(self) -> Mat3:
        return self.inertia


@dataclass(frozen=True, slots=True)
class MeshGeometry:
    filename: str
    path: Path
    origin: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)

    @property
    def resolved_path(self) -> Path:
        return self.path


@dataclass(frozen=True, slots=True)
class UrdfLink:
    name: str
    inertial: Inertial | None
    visuals: tuple[MeshGeometry, ...] = ()
    collisions: tuple[MeshGeometry, ...] = ()
    source_index: int = -1
    has_geometry: bool = False

    @property
    def meshes(self) -> tuple[MeshGeometry, ...]:
        return self.visuals + self.collisions


@dataclass(frozen=True, slots=True)
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)
    axis: Vec3 = (0.0, 0.0, 0.0)
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None
    source_index: int = -1

    @property
    def xyz(self) -> Vec3:
        return self.origin

    @property
    def limit(self) -> tuple[float, float] | None:
        if self.lower is None or self.upper is None:
            return None
        return self.lower, self.upper

    @property
    def source_xml_index(self) -> int:
        return self.source_index


@dataclass(frozen=True, slots=True)
class UrdfModel:
    links: tuple[UrdfLink, ...]
    joints: tuple[UrdfJoint, ...]
    source_path: Path
    _root: str
    manifest: Mapping[str, Mapping[str, int]]

    @property
    def root(self) -> str:
        return self._root

    @property
    def revolute_joints(self) -> tuple[UrdfJoint, ...]:
        return tuple(joint for joint in self.joints if joint.type == "revolute")

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return tuple(
            joint.name
            for joint in self.revolute_joints
            if joint.name.startswith("Joint") and joint.name[5:6].isdigit()
        )

    def hand_joint_names(self, side: str) -> tuple[str, ...]:
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        prefix = "l_" if side == "left" else "r_"
        return tuple(joint.name for joint in self.revolute_joints if joint.name.startswith(prefix))

    def link(self, name: str) -> UrdfLink:
        for link in self.links:
            if link.name == name:
                return link
        raise KeyError(name)

    def fixed_transform(self, parent: str, child: str) -> tuple[Vec3, Mat3]:
        """Return the parent-to-child translation and rotation for a fixed path."""
        if parent == child:
            return (0.0, 0.0, 0.0), _identity()
        by_child = {joint.child: joint for joint in self.joints}
        path: list[UrdfJoint] = []
        cursor = child
        while cursor != parent:
            joint = by_child.get(cursor)
            if joint is None:
                raise ValueError(f"{parent!r} is not an ancestor of {child!r}")
            if joint.type != "fixed":
                raise ValueError(f"path {parent!r}->{child!r} contains non-fixed joint {joint.name!r}")
            path.append(joint)
            cursor = joint.parent
        translation: Vec3 = (0.0, 0.0, 0.0)
        rotation: Mat3 = _identity()
        for joint in reversed(path):
            translation, rotation = _compose(
                (translation, rotation), (joint.origin, _rpy_matrix(joint.rpy))
            )
        return translation, rotation


def load_urdf(path: str | Path, *, source_bytes: bytes | None = None) -> UrdfModel:
    source = Path(path).resolve()
    try:
        robot = ET.fromstring(source_bytes) if source_bytes is not None else ET.parse(source).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"unable to parse URDF {source}: {exc}") from exc
    if robot.tag != "robot":
        raise ValueError("URDF root must be <robot>")

    children = list(robot)
    link_elements: list[tuple[int, ET.Element]] = [
        (index, element) for index, element in enumerate(children) if element.tag == "link"
    ]
    joint_elements: list[tuple[int, ET.Element]] = [
        (index, element) for index, element in enumerate(children) if element.tag == "joint"
    ]
    links_by_name: dict[str, UrdfLink] = {}
    for source_index, element in link_elements:
        name = _required(element, "name", "link")
        if name in links_by_name:
            raise ValueError(f"duplicate link {name!r}")
        visuals, visual_geometry = _parse_geometry(element, "visual", source.parent)
        collisions, collision_geometry = _parse_geometry(element, "collision", source.parent)
        has_geometry = visual_geometry or collision_geometry
        inertial_element = element.find("inertial")
        if inertial_element is None:
            if name in {"TCP_Link_L", "TCP_Link_R"}:
                inertial = Inertial(0.05, (0.0, 0.0, 0.0), _zeros())
            elif has_geometry:
                raise ValueError(f"link {name!r} is missing inertial")
            else:
                inertial = None
        else:
            inertial = _parse_inertial(inertial_element, name, allow_point_mass=name in {"TCP_Link_L", "TCP_Link_R"})
        links_by_name[name] = UrdfLink(name, inertial, visuals, collisions, source_index, has_geometry)

    if not links_by_name:
        raise ValueError("URDF has no links")
    joints: list[UrdfJoint] = []
    joint_names: set[str] = set()
    incoming: dict[str, UrdfJoint] = {}
    outgoing: dict[str, list[UrdfJoint]] = {name: [] for name in links_by_name}
    for source_index, element in joint_elements:
        name = _required(element, "name", "joint")
        if name in joint_names:
            raise ValueError(f"duplicate joint {name!r}")
        joint_names.add(name)
        joint_type = _required(element, "type", f"joint {name!r}")
        if joint_type not in {"fixed", "revolute", "continuous", "prismatic", "floating", "planar"}:
            raise ValueError(f"unsupported joint type {joint_type!r}")
        parent_element = element.find("parent")
        child_element = element.find("child")
        if parent_element is None or child_element is None:
            raise ValueError(f"joint {name!r} must have parent and child")
        parent = _required(parent_element, "link", f"joint {name!r} parent")
        child = _required(child_element, "link", f"joint {name!r} child")
        if parent not in links_by_name or child not in links_by_name:
            raise ValueError(f"joint {name!r} references unknown link")
        if child in incoming:
            raise ValueError(f"link {child!r} has multiple parent joints")
        origin, rpy = _parse_pose(element.find("origin"), f"joint {name!r} origin")
        axis = _parse_vector(element.find("axis").get("xyz") if element.find("axis") is not None else None, 3, f"joint {name!r} axis", default=(0.0, 0.0, 0.0))
        if joint_type in {"revolute", "continuous", "prismatic"} and np.linalg.norm(axis) <= 0.0:
            raise ValueError(f"joint {name!r} has zero axis")
        lower = upper = effort = velocity = None
        limit = element.find("limit")
        if joint_type == "revolute":
            if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                raise ValueError(f"revolute joint {name!r} is missing limits")
            lower = _finite_float(limit.attrib["lower"], f"joint {name!r} lower")
            upper = _finite_float(limit.attrib["upper"], f"joint {name!r} upper")
            if lower > upper:
                raise ValueError(f"joint {name!r} has inverted limits")
            if "effort" in limit.attrib:
                effort = _finite_float(limit.attrib["effort"], f"joint {name!r} effort")
            if "velocity" in limit.attrib:
                velocity = _finite_float(limit.attrib["velocity"], f"joint {name!r} velocity")
        elif limit is not None:
            for key in ("lower", "upper", "effort", "velocity"):
                if key in limit.attrib:
                    value = _finite_float(limit.attrib[key], f"joint {name!r} {key}")
                    if key == "lower": lower = value
                    elif key == "upper": upper = value
                    elif key == "effort": effort = value
                    else: velocity = value
        joint = UrdfJoint(name, joint_type, parent, child, origin, rpy, axis, lower, upper, effort, velocity, source_index)
        joints.append(joint)
        incoming[child] = joint
        outgoing[parent].append(joint)

    roots = [name for name in links_by_name if name not in incoming]
    if len(roots) != 1:
        raise ValueError(f"URDF must have one root, found {roots}")
    root = roots[0]

    state: dict[str, int] = {}
    ordered_links: list[UrdfLink] = []
    ordered_joints: list[UrdfJoint] = []

    def visit(link_name: str) -> None:
        marker = state.get(link_name, 0)
        if marker == 1:
            raise ValueError(f"cycle involving link {link_name!r}")
        if marker == 2:
            return
        state[link_name] = 1
        ordered_links.append(links_by_name[link_name])
        for joint in sorted(outgoing[link_name], key=lambda item: item.source_index):
            ordered_joints.append(joint)
            visit(joint.child)
        state[link_name] = 2

    visit(root)
    if len(ordered_links) != len(links_by_name):
        disconnected = sorted(set(links_by_name) - {link.name for link in ordered_links})
        raise ValueError(f"URDF is disconnected: {disconnected}")
    for link in ordered_links:
        if link.inertial is None:
            joint = incoming.get(link.name)
            if joint is None or joint.type != "fixed" or link.name not in {
                "marker_wuji2_r", "marker_tianji_r", "r_thumb_tip", "r_index_finger_tip",
                "r_middle_finger_tip", "r_ring_finger_tip", "r_pinky_tip", "marker_wuji2_l",
                "marker_tianji_l", "l_thumb_tip", "l_index_finger_tip", "l_middle_finger_tip",
                "l_ring_finger_tip", "l_pinky_tip",
            }:
                raise ValueError(f"link {link.name!r} has invalid missing inertial")
    manifest = MappingProxyType({
        "links": MappingProxyType({link.name: link.source_index for link in ordered_links}),
        "joints": MappingProxyType({joint.name: joint.source_index for joint in ordered_joints}),
    })
    return UrdfModel(tuple(ordered_links), tuple(ordered_joints), source, root, manifest)


def aggregate_fixed_point_masses(model: UrdfModel) -> UrdfModel:
    """Fold the two explicitly-defined TCP point masses into their Link7 parents."""
    links = {link.name: link for link in model.links}
    for side in ("L", "R"):
        tcp_name = f"TCP_Link_{side}"
        parent_name = f"Link7_{side}"
        tcp = links.get(tcp_name)
        parent = links.get(parent_name)
        if tcp is None or parent is None or tcp.inertial is None:
            continue
        parent_inertial = parent.inertial
        if parent_inertial is None:
            raise ValueError(f"aggregation parent {parent_name!r} has no inertial")
        translation, rotation = model.fixed_transform(parent_name, tcp_name)
        mp = parent_inertial.mass
        mc = tcp.inertial.mass
        parent_com = np.asarray(parent_inertial.com, dtype=float)
        child_com = np.asarray(translation) + np.asarray(rotation) @ np.asarray(tcp.inertial.com)
        total_mass = mp + mc
        total_com = (mp * parent_com + mc * child_com) / total_mass
        parent_tensor = np.asarray(parent_inertial.inertia, dtype=float)
        child_tensor = np.asarray(rotation) @ np.asarray(tcp.inertial.inertia, dtype=float) @ np.asarray(rotation).T
        total_tensor = _shift_tensor(parent_tensor, mp, parent_com, total_com) + _shift_tensor(child_tensor, mc, child_com, total_com)
        links[parent_name] = replace(
            parent,
            inertial=Inertial(total_mass, _vec3(total_com), _mat3(total_tensor)),
        )
        links[tcp_name] = replace(tcp, inertial=None)
    return replace(model, links=tuple(links[link.name] for link in model.links))


def _parse_inertial(element: ET.Element, name: str, *, allow_point_mass: bool) -> Inertial:
    mass_element = element.find("mass")
    inertia_element = element.find("inertia")
    if mass_element is None or "value" not in mass_element.attrib or inertia_element is None:
        raise ValueError(f"link {name!r} has incomplete inertial")
    mass = _finite_float(mass_element.attrib["value"], f"link {name!r} mass")
    if mass <= 0:
        raise ValueError(f"link {name!r} mass must be positive")
    com, inertial_rpy = _parse_pose(element.find("origin"), f"link {name!r} inertial origin")
    inertial_rotation = np.asarray(_rpy_matrix(inertial_rpy), dtype=float)
    values = {
        key: _finite_float(inertia_element.attrib.get(key, "nan"), f"link {name!r} inertia {key}")
        for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    }
    matrix = np.array([[values["ixx"], values["ixy"], values["ixz"]], [values["ixy"], values["iyy"], values["iyz"]], [values["ixz"], values["iyz"], values["izz"]]], dtype=float)
    link_matrix = inertial_rotation @ matrix @ inertial_rotation.T
    if allow_point_mass:
        if not math.isclose(mass, 0.05, rel_tol=0.0, abs_tol=1e-12) or not np.allclose(matrix, 0.0, atol=0.0):
            raise ValueError(f"link {name!r} TCP point mass must be 0.05kg with zero inertia")
        return Inertial(mass, com, _zeros())
    eigenvalues = np.linalg.eigvalsh(link_matrix)
    if np.any(eigenvalues <= 0.0):
        raise ValueError(f"link {name!r} inertia is not positive definite")
    principal = np.sort(eigenvalues)
    if any(principal[i] + principal[j] < principal[k] for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0))):
        raise ValueError(f"link {name!r} inertia violates triangle inequalities")
    return Inertial(mass, com, _mat3(link_matrix))


def _parse_geometry(
    link: ET.Element, kind: str, base: Path
) -> tuple[tuple[MeshGeometry, ...], bool]:
    geometries: list[MeshGeometry] = []
    geometry_present = False
    for element in link.findall(kind):
        geometry = element.find("geometry")
        if geometry is None:
            raise ValueError(f"link {link.get('name')!r} {kind} is missing geometry")
        mesh = geometry.find("mesh")
        origin, rpy = _parse_pose(element.find("origin"), f"{kind} origin")
        _parse_vector(element.get("scale"), 3, f"{kind} scale", default=(1.0, 1.0, 1.0))
        if mesh is None:
            primitive = next(
                (geometry.find(tag) for tag in ("box", "cylinder", "sphere") if geometry.find(tag) is not None),
                None,
            )
            if primitive is None:
                raise ValueError(f"link {link.get('name')!r} {kind} has unsupported geometry")
            if primitive.tag == "box":
                size = _parse_vector(primitive.get("size"), 3, f"{kind} box size", required=True)
                if any(value <= 0.0 for value in size):
                    raise ValueError(f"{kind} box size must be positive")
            elif primitive.tag == "cylinder":
                _positive_float(primitive.get("length"), f"{kind} cylinder length")
                _positive_float(primitive.get("radius"), f"{kind} cylinder radius")
            else:
                _positive_float(primitive.get("radius"), f"{kind} sphere radius")
            # Primitive geometry is intentionally not flattened into MeshGeometry,
            # but it still counts as geometry and its pose is fully validated.
            geometry_present = True
            continue
        filename = mesh.get("filename") or mesh.get("file")
        if not filename:
            raise ValueError(f"link {link.get('name')!r} {kind} mesh is missing filename")
        if filename.startswith("package://") or filename.startswith("file://"):
            raise ValueError(f"unsupported mesh URI {filename!r}")
        resolved = (base / filename).resolve()
        try:
            resolved.relative_to(base.resolve())
        except ValueError as exc:
            raise ValueError(f"mesh path escapes URDF directory: {filename!r}") from exc
        if not resolved.is_file():
            raise ValueError(f"missing mesh {filename!r}")
        scale = _parse_vector(mesh.get("scale"), 3, f"{kind} mesh scale", default=(1.0, 1.0, 1.0))
        geometries.append(MeshGeometry(filename, resolved, origin, rpy, scale))
        geometry_present = True
    return tuple(geometries), geometry_present


def _parse_pose(element: ET.Element | None, name: str) -> tuple[Vec3, Vec3]:
    if element is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return (
        _parse_vector(element.get("xyz"), 3, f"{name} xyz"),
        _parse_vector(element.get("rpy"), 3, f"{name} rpy"),
    )


def _parse_vector(
    value: str | None,
    length: int,
    name: str,
    *,
    default: tuple[float, ...] | None = None,
    required: bool = False,
) -> tuple[float, ...]:
    if value is None:
        if required:
            raise ValueError(f"{name} is missing")
        if default is None:
            return tuple(0.0 for _ in range(length))
        return default
    try:
        result = tuple(float(item) for item in value.split())
    except ValueError as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if len(result) != length or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def _positive_float(value: str | None, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value: str | None, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _required(element: ET.Element, attribute: str, name: str) -> str:
    value = element.get(attribute)
    if not value:
        raise ValueError(f"{name} is missing {attribute}")
    return value


def _identity() -> Mat3:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _zeros() -> Mat3:
    return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _rpy_matrix(rpy: Vec3) -> Mat3:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), (-sp, cp * sr, cp * cr))


def _compose(first: tuple[Vec3, Mat3], second: tuple[Vec3, Mat3]) -> tuple[Vec3, Mat3]:
    p1, r1 = np.asarray(first[0]), np.asarray(first[1])
    p2, r2 = np.asarray(second[0]), np.asarray(second[1])
    return _vec3(p1 + r1 @ p2), _mat3(r1 @ r2)


def _shift_tensor(tensor: np.ndarray, mass: float, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    displacement = source - target
    return tensor + mass * ((displacement @ displacement) * np.eye(3) - np.outer(displacement, displacement))


def _vec3(value: np.ndarray) -> Vec3:
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _mat3(value: np.ndarray) -> Mat3:
    return tuple(tuple(float(item) for item in row) for row in value)  # type: ignore[return-value]
