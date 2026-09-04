"""Deterministic, contact-enabled procedural SPD scene assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np

TABLE_Z = 0.90
WORKSPACE_CENTER = (0.45, 0.0, TABLE_Z)
WORKSPACE_X = (0.10, 0.80)
WORKSPACE_Y = (-0.55, 0.55)
MAX_RESET_CANDIDATES = 32

JENGA_BLOCK_SIZE = (0.075, 0.025, 0.015)
LETTER_BLOCK_SIZE = (0.040, 0.040, 0.040)
PLATE_RADIUS = 0.100
PLATE_THICKNESS = 0.008
CUP_OUTER_RADIUS = 0.040
CUP_HEIGHT = 0.090
CUP_WALL = 0.004
BOTTLE_RADIUS = 0.035
BOTTLE_HEIGHT = 0.180
BIN_INNER_SIZE = (0.350, 0.250, 0.150)

CLASS_IDS = {
    "jenga_block": 1,
    "letter_block": 2,
    "plate": 3,
    "cup": 4,
    "mug": 5,
    "bottle": 6,
    "rack": 7,
    "mug_tree": 8,
    "bin": 9,
    "domino": 10,
}
BASE_MASSES = {
    "jenga_block": 0.045,
    "letter_block": 0.040,
    "plate": 0.120,
    "cup": 0.055,
    "mug": 0.070,
    "bottle": 0.110,
    "rack": 0.500,
    "mug_tree": 0.400,
    "bin": 0.800,
    "domino": 0.030,
}


class SceneResetError(RuntimeError):
    """Raised when deterministic placement cannot pass the contact gate."""


@dataclass(frozen=True)
class ObjectSpec:
    instance_id: int
    class_id: int
    class_name: str
    name: str
    position: tuple[float, float, float]
    yaw_rad: float
    size: tuple[float, ...]
    mass_kg: float
    friction: float
    contact_group: str
    assembled: bool
    color_rgb: tuple[float, float, float]
    geoms: tuple[dict[str, Any], ...]

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneBuildResult:
    scene: str
    task: str
    seed: int
    candidate: int
    objects: tuple[ObjectSpec, ...]
    sampled_values: dict[str, Any]
    worldbody: ET.Element

    def manifest(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "task": self.task,
            "seed": self.seed,
            "candidate": self.candidate,
            "table": {
                "top_z_m": TABLE_Z,
                "workspace_center": list(WORKSPACE_CENTER),
                "x_range_m": list(WORKSPACE_X),
                "y_range_m": list(WORKSPACE_Y),
            },
            "sampled_values": self.sampled_values,
            "objects": [item.manifest() for item in self.objects],
        }

    def xml_string(self) -> str:
        return ET.tostring(self.worldbody, encoding="unicode")


def _quat_z(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def _box_geom(name: str, size: tuple[float, float, float], *, pos=(0.0, 0.0, 0.0), rgba=(0.5, 0.5, 0.5, 1.0)) -> dict[str, Any]:
    return {"name": name, "type": "box", "size": tuple(value * 0.5 for value in size), "pos": tuple(pos), "rgba": tuple(rgba)}


def _cylinder_geom(name: str, radius: float, height: float, *, pos=(0.0, 0.0, 0.0), rgba=(0.5, 0.5, 0.5, 1.0)) -> dict[str, Any]:
    return {"name": name, "type": "cylinder", "size": (radius, height * 0.5), "pos": tuple(pos), "rgba": tuple(rgba)}


def _capsule_geom(name: str, radius: float, fromto: tuple[float, ...], *, rgba=(0.5, 0.5, 0.5, 1.0)) -> dict[str, Any]:
    return {"name": name, "type": "capsule", "size": (radius,), "fromto": tuple(fromto), "rgba": tuple(rgba)}


def _geoms_for(class_name: str, size: tuple[float, ...], color: tuple[float, float, float], instance_id: int) -> tuple[dict[str, Any], ...]:
    rgba = (*color, 1.0)
    prefix = f"obj{instance_id}"
    if class_name in {"jenga_block", "letter_block", "domino", "rack", "mug_tree"}:
        if class_name == "domino":
            size = (size[0], size[1], size[2])
        return (_box_geom(f"{prefix}_geom", size, rgba=rgba),)
    if class_name == "plate":
        return (_cylinder_geom(f"{prefix}_geom", size[0], size[1], rgba=rgba),)
    if class_name in {"cup", "mug"}:
        radius, height, wall = size[:3]
        bottom = _cylinder_geom(f"{prefix}_bottom", max(0.001, radius - wall), wall, pos=(0.0, 0.0, wall * 0.5), rgba=rgba)
        side = radius - wall * 0.5
        wall_box = wall * 0.5
        walls = (
            _box_geom(f"{prefix}_wall_xp", (wall, 2.0 * radius, height), pos=(side, 0.0, height * 0.5), rgba=rgba),
            _box_geom(f"{prefix}_wall_xn", (wall, 2.0 * radius, height), pos=(-side, 0.0, height * 0.5), rgba=rgba),
            _box_geom(f"{prefix}_wall_yp", (2.0 * radius - 2.0 * wall_box, wall, height), pos=(0.0, side, height * 0.5), rgba=rgba),
            _box_geom(f"{prefix}_wall_yn", (2.0 * radius - 2.0 * wall_box, wall, height), pos=(0.0, -side, height * 0.5), rgba=rgba),
        )
        if class_name == "mug":
            handle = _capsule_geom(
                f"{prefix}_handle", wall,
                (radius, 0.0, height * 0.72, radius + 0.040, 0.0, height * 0.72),
                rgba=rgba,
            )
            return (bottom, *walls, handle)
        return (bottom, *walls)
    if class_name == "bottle":
        radius, height = size[:2]
        body = _cylinder_geom(f"{prefix}_body", radius, height * 0.78, pos=(0.0, 0.0, height * 0.39), rgba=rgba)
        neck = _capsule_geom(
            f"{prefix}_neck", radius * 0.65,
            (0.0, 0.0, height * 0.72, 0.0, 0.0, height * 0.98), rgba=rgba,
        )
        return (body, neck)
    if class_name == "bin":
        width, depth, height = size[:3]
        wall = 0.008
        bottom = _box_geom(f"{prefix}_bottom", (width, depth, wall), pos=(0.0, 0.0, wall * 0.5), rgba=rgba)
        walls = (
            _box_geom(f"{prefix}_x1", (wall, depth, height), pos=((width - wall) * 0.5, 0.0, height * 0.5), rgba=rgba),
            _box_geom(f"{prefix}_x2", (wall, depth, height), pos=(-(width - wall) * 0.5, 0.0, height * 0.5), rgba=rgba),
            _box_geom(f"{prefix}_y1", (width - 2.0 * wall, wall, height), pos=(0.0, (depth - wall) * 0.5, height * 0.5), rgba=rgba),
            _box_geom(f"{prefix}_y2", (width - 2.0 * wall, wall, height), pos=(0.0, -(depth - wall) * 0.5, height * 0.5), rgba=rgba),
        )
        return (bottom, *walls)
    raise ValueError(f"unknown procedural class: {class_name}")


def _bounding_radius(obj: ObjectSpec) -> float:
    return math.sqrt(sum(value * value for value in obj.size)) * 0.5


def _object_interpenetrates(a: ObjectSpec, b: ObjectSpec) -> bool:
    if a.assembled and b.assembled and a.class_name == b.class_name == "cup":
        return False
    def half_xy(obj: ObjectSpec) -> tuple[float, float]:
        if obj.class_name in {"plate", "cup", "mug", "bottle"}:
            radius = obj.size[0]
            return radius, radius
        half_x, half_y = obj.size[0] * 0.5, obj.size[1] * 0.5
        c, s = abs(math.cos(obj.yaw_rad)), abs(math.sin(obj.yaw_rad))
        return c * half_x + s * half_y, s * half_x + c * half_y
    ax, ay = half_xy(a)
    bx, by = half_xy(b)
    if abs(a.position[0] - b.position[0]) >= ax + bx:
        return False
    if abs(a.position[1] - b.position[1]) >= ay + by:
        return False
    def z_bounds(obj: ObjectSpec) -> tuple[float, float]:
        if obj.class_name in {"cup", "mug", "bottle", "rack", "mug_tree", "bin"}:
            return obj.position[2], obj.position[2] + obj.size[-1]
        half_z = obj.size[-1] * 0.5 if obj.class_name != "plate" else obj.size[1] * 0.5
        return obj.position[2] - half_z, obj.position[2] + half_z
    a_low, a_high = z_bounds(a)
    b_low, b_high = z_bounds(b)
    return min(a_high, b_high) > max(a_low, b_low)


class ProceduralSceneBuilder:
    """Build one scene/task with all random values in its manifest."""

    def __init__(self, scene: str, task: str, seed: int) -> None:
        self.scene = scene
        self.task = task
        self.seed = int(seed)

    def _layout(self) -> list[tuple[str, tuple[float, float, float], bool]]:
        scene, task = self.scene, self.task
        if scene == "jenga":
            if task == "hollow_tower":
                return [("jenga_block", (0.45 + (i % 3 - 1) * 0.078, -0.03 + (i // 3) * 0.027, TABLE_Z + JENGA_BLOCK_SIZE[2] * 0.5 + (i // 3) * 0.032), True) for i in range(9)]
            if task == "tower":
                return [("jenga_block", (0.45 + (i % 3 - 1) * 0.078, 0.0, TABLE_Z + JENGA_BLOCK_SIZE[2] * 0.5 + (i // 3) * 0.032), True) for i in range(9)]
            if task == "dominos":
                return [("domino", (0.16 + (i % 5) * 0.12, -0.16 + (i // 5) * 0.12, TABLE_Z + 0.035), False) for i in range(9)]
            if task == "criss_cross":
                return [("jenga_block", (0.45 + (i % 3 - 1) * 0.078, 0.0, TABLE_Z + JENGA_BLOCK_SIZE[2] * 0.5 + (i // 3) * 0.032), True) for i in range(9)]
            if task in {"handover_lr", "handover_rl"}:
                return [("jenga_block", (0.45, 0.0, TABLE_Z + JENGA_BLOCK_SIZE[2] * 0.5), False)]
        if scene == "spelling_blocks":
            if task == "spelling":
                return [("letter_block", (0.22 + (i % 4) * 0.10, -0.20 + (i // 4) * 0.10, TABLE_Z + 0.02), False) for i in range(8)]
            if task == "sort_and_unload":
                return [("letter_block", (0.22 + (i % 4) * 0.10, -0.18 + (i // 4) * 0.10, TABLE_Z + 0.02), False) for i in range(8)]
            if task == "pyramid":
                return [("letter_block", (0.40 + (i % 3 - 1) * 0.045, 0.05 + (i // 3) * 0.045, TABLE_Z + 0.02 + (i // 3) * 0.043), True) for i in range(6)]
            if task == "vowel_consonant_sort":
                return [("letter_block", (0.22 + (i % 5) * 0.10, -0.18 + (i // 5) * 0.10, TABLE_Z + 0.02), False) for i in range(10)]
        if scene == "mugs" and task == "hang_mug":
            return [("mug", (0.32, 0.0, TABLE_Z), False), ("mug_tree", (0.58, 0.0, TABLE_Z + 0.15), True)]
        if scene == "dishes":
            if task == "rack_dishes":
                return [("plate", (0.25 + (i % 2) * 0.30, -0.25 + (i // 2) * 0.30, TABLE_Z + PLATE_THICKNESS * 0.5), False) for i in range(4)] + [("rack", (0.70, 0.25, TABLE_Z + 0.06), True)]
            if task == "plate_dishes":
                return [("plate", (0.25 + (i % 2) * 0.30, -0.25 + (i // 2) * 0.30, TABLE_Z + PLATE_THICKNESS * 0.5), False) for i in range(4)]
        if scene == "cups":
            if task == "pyramid":
                return [("cup", (0.42 + (i % 3 - 1) * 0.07, 0.04 + (i // 3) * 0.07, TABLE_Z), True) for i in range(6)]
            if task == "stack_two_threes":
                return [("cup", (0.31 + (i % 3) * 0.07, -0.10 + (i // 3) * 0.07, TABLE_Z), True) for i in range(6)]
            if task == "unstack":
                return [("cup", (0.45, 0.0, TABLE_Z + i * 0.04), True) for i in range(3)]
        if scene == "bottles" and task == "toss_in_bin":
            return [("bottle", (0.20 + i * 0.10, -0.16 + (i % 2) * 0.12, TABLE_Z), False) for i in range(4)] + [("bin", (0.75, 0.0, TABLE_Z + 0.075), True)]
        raise KeyError(f"unknown SPD task: {scene}/{task}")

    @staticmethod
    def _size(class_name: str) -> tuple[float, ...]:
        if class_name == "jenga_block":
            return JENGA_BLOCK_SIZE
        if class_name == "domino":
            return (0.060, 0.012, 0.070)
        if class_name == "letter_block":
            return LETTER_BLOCK_SIZE
        if class_name == "plate":
            return (PLATE_RADIUS, PLATE_THICKNESS)
        if class_name in {"cup", "mug"}:
            return (CUP_OUTER_RADIUS, CUP_HEIGHT, CUP_WALL)
        if class_name == "bottle":
            return (BOTTLE_RADIUS, BOTTLE_HEIGHT)
        if class_name == "rack":
            return (0.25, 0.16, 0.12)
        if class_name == "mug_tree":
            return (0.12, 0.12, 0.30)
        if class_name == "bin":
            return BIN_INNER_SIZE
        raise KeyError(class_name)

    def _sample_candidate(self, rng: np.random.Generator, candidate: int) -> tuple[ObjectSpec, ...]:
        objects: list[ObjectSpec] = []
        for instance_id, (class_name, base_position, assembled) in enumerate(self._layout(), start=1):
            size = self._size(class_name)
            jitter = np.zeros(2, dtype=np.float64) if assembled else rng.uniform(-0.04, 0.04, size=2)
            yaw = 0.0 if assembled else float(rng.uniform(-math.radians(15.0), math.radians(15.0)))
            position = (float(base_position[0] + jitter[0]), float(base_position[1] + jitter[1]), float(base_position[2]))
            if not (WORKSPACE_X[0] <= position[0] <= WORKSPACE_X[1] and WORKSPACE_Y[0] <= position[1] <= WORKSPACE_Y[1]):
                raise SceneResetError(f"object {instance_id} leaves workspace")
            mass = BASE_MASSES[class_name] * float(rng.uniform(0.8, 1.2))
            friction = float(rng.uniform(0.6, 1.2))
            color = tuple(float(value) for value in (0.15 + 0.75 * rng.random(3)))
            object_spec = ObjectSpec(
                instance_id=instance_id,
                class_id=CLASS_IDS[class_name],
                class_name=class_name,
                name=f"{self.scene}_{self.task}_object_{instance_id:03d}",
                position=position,
                yaw_rad=yaw,
                size=tuple(float(value) for value in size),
                mass_kg=mass,
                friction=friction,
                contact_group="hand_object",
                assembled=assembled,
                color_rgb=color,
                geoms=tuple(_geoms_for(class_name, size, color, instance_id)),
            )
            if any(_object_interpenetrates(object_spec, previous) for previous in objects):
                raise SceneResetError(f"initial object interpenetration at candidate {candidate}")
            objects.append(object_spec)
        return tuple(objects)

    @staticmethod
    def _worldbody(objects: Iterable[ObjectSpec]) -> ET.Element:
        worldbody = ET.Element("worldbody")
        ET.SubElement(worldbody, "geom", name="scene_table", type="box", pos="0.45 0 0.875", size="0.40 0.55 0.025", contype="1", conaffinity="1", rgba="0.30 0.26 0.22 1")
        for obj in objects:
            body = ET.SubElement(
                worldbody,
                "body",
                name=obj.name,
                pos=" ".join(f"{value:.12g}" for value in obj.position),
                quat=" ".join(f"{value:.12g}" for value in _quat_z(obj.yaw_rad)),
            )
            inertia = max(1.0e-6, obj.mass_kg * max(obj.size) ** 2 / 12.0)
            ET.SubElement(
                body,
                "inertial",
                pos="0 0 0",
                mass=f"{obj.mass_kg:.12g}",
                diaginertia=f"{inertia:.12g} {inertia:.12g} {inertia:.12g}",
            )
            ET.SubElement(body, "joint", name=f"{obj.name}_free", type="free", damping="0.02")
            for geom in obj.geoms:
                attributes = {
                    "name": geom["name"],
                    "type": geom["type"],
                    "contype": "1",
                    "conaffinity": "1",
                    "group": str(obj.class_id),
                    "user": f"{obj.instance_id} {obj.class_id}",
                    "friction": f"{obj.friction:.12g} 0.005 0.0001",
                    "rgba": " ".join(f"{value:.12g}" for value in geom["rgba"]),
                }
                if geom["type"] == "capsule":
                    attributes["size"] = f"{geom['size'][0]:.12g}"
                    attributes["fromto"] = " ".join(f"{value:.12g}" for value in geom["fromto"])
                elif geom["type"] == "cylinder":
                    attributes["size"] = " ".join(f"{value:.12g}" for value in geom["size"])
                    attributes["pos"] = " ".join(f"{value:.12g}" for value in geom["pos"])
                else:
                    attributes["size"] = " ".join(f"{value:.12g}" for value in geom["size"])
                    attributes["pos"] = " ".join(f"{value:.12g}" for value in geom["pos"])
                ET.SubElement(body, "geom", **attributes)
        return worldbody

    def build(self) -> SceneBuildResult:
        rng = np.random.default_rng(self.seed)
        last_error: Exception | None = None
        for candidate in range(MAX_RESET_CANDIDATES):
            try:
                objects = self._sample_candidate(rng, candidate)
                values = {
                    "mass_multiplier_range": [0.8, 1.2],
                    "friction_range": [0.6, 1.2],
                    "xy_jitter_m": 0.04,
                    "yaw_jitter_deg": 15.0,
                    "candidate": candidate,
                    "object_sizes_m": {str(item.instance_id): list(item.size) for item in objects},
                    "object_masses_kg": {str(item.instance_id): item.mass_kg for item in objects},
                    "object_friction": {str(item.instance_id): item.friction for item in objects},
                    "object_colors": {str(item.instance_id): list(item.color_rgb) for item in objects},
                }
                return SceneBuildResult(self.scene, self.task, self.seed, candidate, objects, values, self._worldbody(objects))
            except SceneResetError as exc:
                last_error = exc
        raise SceneResetError(
            f"scene reset failed after {MAX_RESET_CANDIDATES} candidates for {self.scene}/{self.task} seed={self.seed}: {last_error}"
        )


def contact_gate(
    model: Any,
    data: Any,
    object_body_names: set[str],
    allowed_interpenetration_pairs: set[frozenset[str]] | None = None,
) -> None:
    """Reject unexpected object-object initial penetration after ``mj_forward``."""
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mujoco is required for contact gating") from exc
    allowed_interpenetration_pairs = allowed_interpenetration_pairs or set()
    mujoco.mj_forward(model, data)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if contact.dist >= -1e-7:
            continue
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom1]))
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[contact.geom2]))
        if first in object_body_names and second in object_body_names:
            if frozenset((first, second)) in allowed_interpenetration_pairs:
                continue
            raise SceneResetError(f"initial interpenetration: {first} vs {second}")


__all__ = [
    "BASE_MASSES",
    "BIN_INNER_SIZE",
    "CLASS_IDS",
    "CUP_HEIGHT",
    "CUP_OUTER_RADIUS",
    "CUP_WALL",
    "JENGA_BLOCK_SIZE",
    "LETTER_BLOCK_SIZE",
    "MAX_RESET_CANDIDATES",
    "ObjectSpec",
    "PLATE_RADIUS",
    "PLATE_THICKNESS",
    "ProceduralSceneBuilder",
    "SceneBuildResult",
    "SceneResetError",
    "TABLE_Z",
    "WORKSPACE_CENTER",
    "WORKSPACE_X",
    "WORKSPACE_Y",
    "contact_gate",
]
