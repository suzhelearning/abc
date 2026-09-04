"""Validation for deterministic ``spd-scene`` JSON manifests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping
from numbers import Integral, Real

from .registry import get_task


class SceneManifestError(ValueError):
    """Raised when scene provenance is incomplete or internally inconsistent."""


def _finite_real(value: object, label: str, *, positive: bool = False) -> float:
    """Validate a JSON number without accepting booleans or non-finite values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SceneManifestError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise SceneManifestError(f"{label} must be {qualifier}")
    return result


def _vector(
    value: object,
    length: int,
    label: str,
    *,
    positive: bool = False,
    unit_interval: bool = False,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SceneManifestError(f"{label} must contain {length} values")
    values = [_finite_real(item, label, positive=positive) for item in value]
    if unit_interval and any(item < 0.0 or item > 1.0 for item in values):
        raise SceneManifestError(f"{label} values must be in [0,1]")
    return values


def _sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise SceneManifestError(f"{label} must be a SHA-256 hex string")


def _validate_object(object_value: Mapping[str, Any], index: int, geom_names: set[str]) -> str:
    label = f"reset.objects[{index}]"
    name = object_value.get("name")
    if not isinstance(name, str) or not name:
        raise SceneManifestError(f"{label}.name must be a non-empty string")
    for field in ("position", "yaw_rad", "mass_kg", "friction", "geoms"):
        if field not in object_value:
            raise SceneManifestError(f"{label}.{field} is missing")
    _vector(object_value["position"], 3, f"{label}.position")
    _finite_real(object_value["yaw_rad"], f"{label}.yaw_rad")
    _finite_real(object_value["mass_kg"], f"{label}.mass_kg", positive=True)
    _finite_real(object_value["friction"], f"{label}.friction", positive=True)

    if "instance_id" in object_value:
        instance_id = object_value["instance_id"]
        if isinstance(instance_id, bool) or not isinstance(instance_id, Integral) or int(instance_id) <= 0:
            raise SceneManifestError(f"{label}.instance_id must be a positive integer")
    if "class_id" in object_value:
        class_id = object_value["class_id"]
        if isinstance(class_id, bool) or not isinstance(class_id, Integral) or int(class_id) <= 0:
            raise SceneManifestError(f"{label}.class_id must be a positive integer")
    for field in ("class_name", "contact_group"):
        if field in object_value and (not isinstance(object_value[field], str) or not object_value[field]):
            raise SceneManifestError(f"{label}.{field} must be a non-empty string")
    if "assembled" in object_value and type(object_value["assembled"]) is not bool:
        raise SceneManifestError(f"{label}.assembled must be a boolean")
    if "size" in object_value:
        raw_size = object_value["size"]
        if not isinstance(raw_size, (list, tuple)) or not raw_size:
            raise SceneManifestError(f"{label}.size must be a non-empty vector")
        _vector(raw_size, len(raw_size), f"{label}.size", positive=True)
    if "color_rgb" in object_value:
        _vector(object_value["color_rgb"], 3, f"{label}.color_rgb", unit_interval=True)

    geoms = object_value["geoms"]
    if not isinstance(geoms, list) or not geoms:
        raise SceneManifestError(f"{label}.geoms must be a non-empty list")
    for geom_index, geom in enumerate(geoms):
        geom_label = f"{label}.geoms[{geom_index}]"
        if not isinstance(geom, Mapping):
            raise SceneManifestError(f"{geom_label} must be a mapping")
        geom_name = geom.get("name")
        if not isinstance(geom_name, str) or not geom_name:
            raise SceneManifestError(f"{geom_label}.name must be a non-empty string")
        if geom_name in geom_names:
            raise SceneManifestError(f"duplicate scene geom name: {geom_name}")
        geom_names.add(geom_name)
        geom_type = geom.get("type")
        if geom_type not in {"box", "cylinder", "capsule"}:
            raise SceneManifestError(f"{geom_label}.type is unsupported")
        if geom_type == "box":
            _vector(geom.get("size"), 3, f"{geom_label}.size", positive=True)
            if "pos" in geom:
                _vector(geom["pos"], 3, f"{geom_label}.pos")
        elif geom_type == "cylinder":
            _vector(geom.get("size"), 2, f"{geom_label}.size", positive=True)
            if "pos" in geom:
                _vector(geom["pos"], 3, f"{geom_label}.pos")
        else:
            _vector(geom.get("size"), 1, f"{geom_label}.size", positive=True)
            _vector(geom.get("fromto"), 6, f"{geom_label}.fromto")
        if "rgba" not in geom:
            raise SceneManifestError(f"{geom_label}.rgba is missing")
        _vector(geom["rgba"], 4, f"{geom_label}.rgba", unit_interval=True)
    return name


def _validate_reset(reset: Mapping[str, Any], task: Any) -> None:
    if reset.get("scene") != task.scene or reset.get("task") != task.name:
        raise SceneManifestError("scene reset does not match the registered task")
    seed = reset.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise SceneManifestError("scene reset seed must be an integer")
    candidate = reset.get("candidate")
    if isinstance(candidate, bool) or not isinstance(candidate, Integral) or int(candidate) < 0:
        raise SceneManifestError("scene reset candidate must be a non-negative integer")
    objects = reset.get("objects")
    if not isinstance(objects, list) or not objects:
        raise SceneManifestError("scene reset must contain a non-empty objects list")
    names: list[str] = []
    geom_names: set[str] = set()
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise SceneManifestError(f"reset.objects[{index}] must be a mapping")
        names.append(_validate_object(item, index, geom_names))
    if len(set(names)) != len(names):
        raise SceneManifestError("reset object names must be unique")


def load_scene_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneManifestError(f"cannot read scene manifest: {manifest_path}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "spd-vr-scene-v1":
        raise SceneManifestError("unsupported scene manifest schema")
    qualified = document.get("task")
    if not isinstance(qualified, str) or "/" not in qualified:
        raise SceneManifestError("scene manifest task must be scene/task")
    try:
        task = get_task(qualified)
    except KeyError as exc:
        raise SceneManifestError(f"unknown scene task: {qualified}") from exc
    if document.get("prompt") not in {None, task.prompt}:
        raise SceneManifestError("scene manifest prompt differs from registry")
    duration = document.get("target_duration_s")
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, Real)
        or not math.isfinite(float(duration))
        or not math.isclose(float(duration), task.target_duration_s, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise SceneManifestError("scene manifest duration differs from Table-2 registry")
    table2 = document.get("table2")
    if table2 is not None and (
        not isinstance(table2, Mapping)
        or isinstance(table2.get("episodes"), bool)
        or not isinstance(table2.get("episodes"), Integral)
        or isinstance(table2.get("minutes"), bool)
        or not isinstance(table2.get("minutes"), Integral)
        or table2.get("episodes") != task.table2_episodes
        or table2.get("minutes") != task.table2_minutes
    ):
        raise SceneManifestError("scene manifest Table-2 statistics differ from registry")
    reset = document.get("reset")
    if not isinstance(reset, Mapping):
        raise SceneManifestError("scene manifest must contain a reset mapping")
    _validate_reset(reset, task)
    objects = reset.get("objects")
    names = document.get("object_bodies")
    if not isinstance(objects, list) or not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise SceneManifestError("scene manifest must contain reset.objects and object_bodies")
    if any(not name for name in names):
        raise SceneManifestError("object_bodies must contain non-empty names")
    reset_names = [item["name"] for item in objects]
    if names != reset_names or len(set(names)) != len(names):
        raise SceneManifestError("object_bodies does not match reset object names")
    model = document.get("model")
    if model is not None:
        if not isinstance(model, Mapping) or not isinstance(model.get("path"), str) or not model["path"]:
            raise SceneManifestError("scene model path must be a non-empty string")
        _sha256(model.get("sha256"), "scene model sha256")
    builder_hashes = document.get("builder_source_sha256")
    if builder_hashes is not None:
        if not isinstance(builder_hashes, Mapping) or not builder_hashes:
            raise SceneManifestError("builder_source_sha256 must be a non-empty mapping")
        for name, digest in builder_hashes.items():
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".py"):
                raise SceneManifestError("builder source names must be simple Python filenames")
            _sha256(digest, f"builder_source_sha256[{name}]")
    return document


__all__ = ["SceneManifestError", "load_scene_manifest"]
