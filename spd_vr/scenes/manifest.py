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
    if not isinstance(reset, Mapping) or reset.get("scene") != task.scene or reset.get("task") != task.name:
        raise SceneManifestError("scene reset does not match the registered task")
    objects = reset.get("objects")
    names = document.get("object_bodies")
    if not isinstance(objects, list) or not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise SceneManifestError("scene manifest must contain reset.objects and object_bodies")
    if any(not name for name in names):
        raise SceneManifestError("object_bodies must contain non-empty names")
    if any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("name"), str)
        or not item["name"]
        for item in objects
    ):
        raise SceneManifestError("reset.objects must contain named object mappings")
    reset_names = [item["name"] for item in objects]
    if names != reset_names or len(set(names)) != len(names):
        raise SceneManifestError("object_bodies does not match reset object names")
    return document


__all__ = ["SceneManifestError", "load_scene_manifest"]
