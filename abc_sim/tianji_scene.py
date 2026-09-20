"""Compose local Tianji MJCF assets with a physical hammer/table scene.

No robot collision masks, inertias, gains, or meshes are replaced. Robot
collision-only geometry is hidden from cameras by its visualization group.
The scanned hammer uses four longitudinal convex collision regions, an explicit
fixture mass/inertia, and its original visual mesh. These are simulation
assumptions, not a calibrated real-hammer contact model.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numbers(values):
    return " ".join(format(float(value), ".12g") for value in np.asarray(values).ravel())


@dataclass
class SceneConfig:
    robot_xml: Path
    hammer_mesh: Path
    output: Path
    table_height: float = 0.90
    hammer_xy: tuple[float, float] = (0.45, 0.08)
    hammer_mass: float = 0.30
    initial_qpos_path: Path | None = None
    fit_wrist_cameras: bool = False


def _fit_wrist_cameras(root, robot_xml, pose_path, table_height):
    """Choose fixed wrist-local mounts from a nominal initial workspace view.

    This is not hardware calibration. The resulting cameras remain rigidly
    attached to the wrist and never track privileged object state at runtime.
    """
    import mujoco

    pose = json.loads(Path(pose_path).expanduser().read_text())
    if not isinstance(pose, dict) or len(pose.get("joint_names", [])) != 54:
        raise ValueError("camera fitting needs initial_qpos_path with 54 joint_names and qpos")
    qpos = np.asarray(pose.get("qpos"), dtype=np.float64)
    if qpos.shape != (54,) or not np.isfinite(qpos).all() or len(set(pose["joint_names"])) != 54:
        raise ValueError("invalid camera-fitting initial pose")
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    data = mujoco.MjData(model)
    for name, value in zip(pose["joint_names"], qpos):
        joint = model.joint(name).id
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_range[joint, 0] <= value <= model.jnt_range[joint, 1]:
            raise ValueError(f"camera-fitting joint outside limits: {name}")
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    report = []
    target = np.array([0.45, 0.0, table_height + 0.02])
    for name, side in (("left_wrist", 1), ("right_wrist", -1)):
        camera = model.camera(name).id
        body = int(model.cam_bodyid[camera])
        rotation = data.xmat[body].reshape(3, 3)
        eye = data.xpos[body] + np.array([0.0, side * 0.06, 0.12])
        local_eye = rotation.T @ (eye - data.xpos[body])
        z_axis = eye - target
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
        if np.linalg.norm(x_axis) < 1e-8:
            raise ValueError("nominal wrist camera view is vertical")
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        axes = np.concatenate((rotation.T @ x_axis, rotation.T @ y_axis))
        element = next(element for element in root.iter("camera") if element.get("name") == name)
        for attribute in ("quat", "euler", "axisangle", "zaxis", "xyaxes", "target"):
            element.attrib.pop(attribute, None)
        element.set("mode", "fixed")
        element.set("pos", _numbers(local_eye))
        element.set("xyaxes", _numbers(axes))
        report.append({"camera": name, "wrist_body": model.body(body).name,
                       "local_position": local_eye.tolist(), "local_xyaxes": axes.tolist(),
                       "initial_world_position": eye.tolist(),
                       "calibration": "nominal workspace-facing simulation mount, not hardware calibrated"})
    return report


def build_tianji_scene(config: SceneConfig) -> dict:
    robot_xml = config.robot_xml.expanduser().resolve()
    hammer_mesh = config.hammer_mesh.expanduser().resolve()
    output = config.output.expanduser().resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"scene output already exists: {output}")
    if not np.isfinite([config.table_height, *config.hammer_xy, config.hammer_mass]).all() or config.table_height <= 0 or config.hammer_mass <= 0:
        raise ValueError("table height and hammer mass must be finite and positive")
    root = ET.parse(robot_xml).getroot()
    if root.tag != "mujoco" or root.findall("include"):
        raise ValueError("robot_xml must be a standalone MuJoCo XML model")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    mesh_dir = (robot_xml.parent / compiler.get("meshdir", ".")).resolve()
    texture_dir = (robot_xml.parent / compiler.get("texturedir", ".")).resolve()
    asset = root.find("asset")
    world = root.find("worldbody")
    if asset is None or world is None:
        raise ValueError("robot MJCF requires asset and worldbody sections")
    for element in asset:
        if "file" in element.attrib:
            base = mesh_dir if element.tag == "mesh" else texture_dir if element.tag == "texture" else robot_xml.parent
            target = (base / element.attrib["file"]).resolve()
            if not target.is_file():
                raise FileNotFoundError(target)
            element.set("file", str(target))
    compiler.set("meshdir", ".")
    compiler.set("texturedir", ".")
    for geom in world.iter("geom"):
        if geom.get("name") == "ground":
            geom.set("group", "2")
        elif geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0":
            # MuJoCo geom groups affect visualization only, not collisions.
            geom.set("group", "3")
    if any(body.get("name") == "hammer" for body in world.iter("body")):
        raise ValueError("robot scene already contains a hammer body")
    if hammer_mesh.suffix.lower() != ".obj":
        raise ValueError("hammer_mesh must be an OBJ in metre units")
    vertices = []
    with hammer_mesh.open() as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError("malformed OBJ vertex")
                vertices.append([float(value) for value in fields[1:4]])
    vertices = np.unique(np.asarray(vertices, dtype=np.float64), axis=0)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("hammer OBJ must contain finite 3D vertices")
    lower, upper = vertices.min(0), vertices.max(0)
    extents = upper - lower
    if np.any(extents <= 0) or not 0.03 <= float(extents.max()) <= 1.0:
        raise ValueError("hammer bounds are degenerate or not plausible metre units")
    center = (lower + upper) / 2
    ET.SubElement(asset, "mesh", name="tianji_hammer_visual", file=str(hammer_mesh), inertia="shell")
    ET.SubElement(asset, "material", name="tianji_table_material", rgba="0.55 0.40 0.27 1", specular="0.1", shininess="0.1")
    ET.SubElement(world, "geom", name="tianji_table", type="box", size="0.35 0.55 0.025",
                  pos=_numbers((0.45, 0, config.table_height - 0.025)),
                  material="tianji_table_material", group="2", friction="0.8 0.005 0.0001")
    position = (*config.hammer_xy, config.table_height - lower[2] + 0.002)
    body = ET.SubElement(world, "body", name="hammer", pos=_numbers(position))
    ET.SubElement(body, "freejoint", name="hammer_free")
    inertia = config.hammer_mass / 12 * np.array([
        extents[1] ** 2 + extents[2] ** 2,
        extents[0] ** 2 + extents[2] ** 2,
        extents[0] ** 2 + extents[1] ** 2,
    ])
    ET.SubElement(body, "inertial", pos=_numbers(center), mass=str(config.hammer_mass), diaginertia=_numbers(inertia))
    ET.SubElement(body, "geom", name="hammer_visual", type="mesh", mesh="tianji_hammer_visual",
                  contype="0", conaffinity="0", group="1", rgba="0.20 0.23 0.26 1")
    axis = int(np.argmax(extents))
    cuts = lower[axis] + extents[axis] * np.array([0, 0.30, 0.60, 0.85, 1])
    regions = []
    for index, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
        selected = vertices[(vertices[:, axis] >= lo) & (vertices[:, axis] <= hi)]
        if len(selected) < 4 or np.linalg.matrix_rank(selected - selected.mean(0)) < 3:
            raise ValueError("hammer collision region is not three-dimensional")
        name = f"hammer_collision_{index}"
        # With vertices and no faces, MuJoCo computes a convex hull.
        ET.SubElement(asset, "mesh", name=name, vertex=_numbers(selected))
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name,
                      group="3", friction="0.8 0.005 0.0001", condim="4")
        regions.append({"axis": axis, "interval": [float(lo), float(hi)], "vertices": len(selected)})
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1280")
    global_visual.set("offheight", "720")
    camera_mounts = []
    if config.fit_wrist_cameras:
        if config.initial_qpos_path is None:
            raise ValueError("--fit-wrist-cameras requires --initial-qpos-path")
        camera_mounts = _fit_wrist_cameras(root, robot_xml, config.initial_qpos_path, config.table_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    with output.open("xb") as stream:
        ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)
    manifest = {
        "robot_xml": str(robot_xml), "robot_sha256": _sha256(robot_xml),
        "hammer_mesh": str(hammer_mesh), "hammer_sha256": _sha256(hammer_mesh),
        "scene_xml": str(output), "scene_sha256": _sha256(output),
        "table_height_m": config.table_height, "hammer_position_m": list(position),
        "hammer_mass_kg": config.hammer_mass, "hammer_bounds_m": [lower.tolist(), upper.tolist()],
        "collision_regions": regions,
        "collision_model": "four longitudinal convex regions of scan vertices; not contact-qualified",
        "robot_physics": "source collision masks, gains, inertias and meshes preserved",
        "camera_mounts": camera_mounts,
    }
    with output.with_suffix(".json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest
