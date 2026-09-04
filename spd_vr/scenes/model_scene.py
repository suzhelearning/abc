"""Merge one procedural scene into the verified unified plant."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from .scene_builder import SceneBuildResult, SceneResetError, contact_gate


def write_scene_model(base_model: str | Path, result: SceneBuildResult, output_model: str | Path) -> Path:
    base_model, output_model = Path(base_model), Path(output_model)
    root = ET.parse(base_model).getroot()
    size = root.find("size")
    if size is None:
        size = ET.Element("size")
        root.insert(0, size)
    size.set("nuser_geom", "2")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise SceneResetError("base model has no worldbody")

    for mesh in root.iter("mesh"):
        file_name = mesh.attrib.get("file")
        if not file_name:
            continue
        source = (base_model.parent / file_name).resolve()
        if not source.is_file():
            raise SceneResetError(f"base model mesh not found: {source}")
        mesh.set("file", os.path.relpath(source, output_model.parent))
    # ``Element.append`` moves a node out of its current parent.  Copy each
    # generated body/geom so the immutable-looking ``SceneBuildResult`` can be
    # reused to produce a second byte-identical scene model (and to compute a
    # fresh manifest) without silently losing the first scene's objects.
    for child in result.worldbody:
        child = copy.deepcopy(child)
        if child.tag == "body" and any(existing.attrib.get("name") == child.attrib.get("name") for existing in worldbody.findall("body")):
            raise SceneResetError(f"duplicate scene body: {child.attrib.get('name')}")
        if child.tag == "geom" and any(existing.attrib.get("name") == child.attrib.get("name") for existing in worldbody.findall("geom")):
            raise SceneResetError(f"duplicate scene geom: {child.attrib.get('name')}")
        worldbody.append(child)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_model, encoding="utf-8", xml_declaration=True)
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(output_model))
        data = mujoco.MjData(model)
        object_names = {item.name for item in result.objects}
        allowed_pairs = {
            frozenset((left.name, right.name))
            for index, left in enumerate(result.objects)
            for right in result.objects[index + 1:]
            if left.assembled and right.assembled and left.class_name == right.class_name == "cup"
        }
        contact_gate(model, data, object_names, allowed_pairs)
    except ImportError:
        pass
    except Exception:
        output_model.unlink(missing_ok=True)
        raise
    return output_model


__all__ = ["write_scene_model"]
