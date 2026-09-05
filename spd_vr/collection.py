"""Aggregate auditable SPD-VR episode collections.

This module is deliberately separate from single-episode validation.  An
episode can be structurally valid while a collection is still unusable: it
may have no contact-qualified window, no source-time provenance, incomplete
task coverage, or far less than the planned amount of data.  The command here
reports those dimensions explicitly and only enforces the aggregate target
when the caller opts into the formal gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from .data import count_training_windows, scan_episodes, validate_episode
from .scenes.registry import TASK_REGISTRY


COLLECTION_METADATA_KEYS = ("run_id", "operator_id", "pico_serial")
DEFAULT_TARGET_HOURS = 75.0
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def collection_metadata(
    *,
    run_id: str | None = None,
    operator_id: str | None = None,
    pico_serial: str | None = None,
) -> dict[str, str]:
    """Return validated collection identity fields for an episode manifest.

    The three fields are optional for simulation smoke.  Once one is supplied,
    all three are required so a formal collection cannot silently mix an
    anonymous operator, device, or run into its audit ledger.
    """
    values = {
        "run_id": run_id,
        "operator_id": operator_id,
        "pico_serial": pico_serial,
    }
    present = [name for name, value in values.items() if value is not None]
    if not present:
        return {}
    if len(present) != len(values):
        missing = ", ".join(name for name, value in values.items() if value is None)
        raise ValueError(f"collection metadata requires all fields; missing {missing}")
    result: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"collection metadata {name} must be a non-empty string")
        result[name] = value.strip()
    return result


def _duration_ns(values: np.ndarray) -> int:
    values = np.asarray(values)
    if values.ndim != 1 or values.size < 2:
        return 0
    return int(max(int(values[-1]) - int(values[0]), 0))


def _qualified_duration_ns(
    timestamps: np.ndarray,
    index: np.ndarray,
    segments: np.ndarray,
) -> int:
    total = 0
    for start, end in np.asarray(segments, dtype=np.int64):
        if end - start < 2:
            continue
        start_raw = int(index[int(start)])
        end_raw = int(index[int(end) - 1])
        total += max(int(timestamps[end_raw]) - int(timestamps[start_raw]), 0)
    return int(total)


def _task_from_manifest(manifest: Mapping[str, Any]) -> str | None:
    scene_manifest = manifest.get("scene_manifest")
    if isinstance(scene_manifest, Mapping):
        task = scene_manifest.get("task")
        if isinstance(task, str) and task:
            return task
    task = manifest.get("task")
    scene = manifest.get("scene")
    if isinstance(task, str) and task:
        if "/" in task:
            return task
        if isinstance(scene, str) and scene:
            return f"{scene}/{task}"
    return None


def _metadata_errors(manifest: Mapping[str, Any]) -> list[str]:
    value = manifest.get("collection")
    if not isinstance(value, Mapping):
        return ["manifest collection metadata is missing"]
    errors: list[str] = []
    for key in COLLECTION_METADATA_KEYS:
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            errors.append(f"collection.{key} is missing or empty")
    return errors


def _scene_seed(manifest: Mapping[str, Any]) -> int | None:
    scene_manifest = manifest.get("scene_manifest")
    if isinstance(scene_manifest, Mapping):
        reset = scene_manifest.get("reset")
        if isinstance(reset, Mapping) and isinstance(reset.get("seed"), int) and not isinstance(
            reset.get("seed"), bool
        ):
            return int(reset["seed"])
    value = manifest.get("seed")
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _artifact_hash(manifest: Mapping[str, Any], key: str) -> str | None:
    value = manifest.get(key)
    return value if isinstance(value, str) and _SHA256_RE.fullmatch(value) else None


@dataclass(frozen=True, slots=True)
class EpisodeAudit:
    """Machine-readable metrics and gate status for one episode."""

    path: str
    ok: bool
    errors: tuple[str, ...] = ()
    task: str | None = None
    scene: str | None = None
    seed: int | None = None
    model_sha256: str | None = None
    urdf_sha256: str | None = None
    collision_manifest_sha256: str | None = None
    raw_frames: int = 0
    training_frames: int = 0
    raw_duration_s: float = 0.0
    qualified_duration_s: float = 0.0
    source_duration_s: float = 0.0
    source_timestamps_valid: bool = False
    valid_both_sides_frames: int = 0
    valid_both_sides_fraction: float = 0.0
    contact_eligible_frames: int = 0
    usable_training_windows: int = 0
    checksum_verified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_episode(
    path: str | Path,
    *,
    verify_checksums: bool = True,
    require_metadata: bool = False,
    require_usable_training: bool = False,
) -> EpisodeAudit:
    """Validate and summarize one episode without changing it."""
    episode_path = Path(path).resolve()
    errors: list[str] = []
    manifest: Mapping[str, Any] = {}
    try:
        manifest = validate_episode(episode_path, verify_checksums=verify_checksums)
    except Exception as exc:
        return EpisodeAudit(
            path=str(episode_path),
            ok=False,
            errors=(str(exc),),
            checksum_verified=verify_checksums,
        )

    task = _task_from_manifest(manifest)
    if task is None:
        errors.append("episode manifest does not identify an SPD scene/task")
    elif task not in TASK_REGISTRY:
        errors.append(f"episode task is not in the SPD registry: {task}")
    if require_metadata:
        errors.extend(_metadata_errors(manifest))
    seed = _scene_seed(manifest)
    if require_metadata and seed is None:
        errors.append("scene/task seed is missing from the episode manifest")
    artifact_hashes = {
        key: _artifact_hash(manifest, key)
        for key in ("model_sha256", "urdf_sha256", "collision_manifest_sha256")
    }
    if require_metadata:
        for key, value in artifact_hashes.items():
            if value is None:
                errors.append(f"manifest {key} is missing or not a SHA-256 hex string")

    try:
        with h5py.File(episode_path, "r") as handle:
            timestamps = np.asarray(handle["raw/timestamp_ns"][:], dtype=np.int64)
            source_timestamps = np.asarray(
                handle["raw/pico/source_timestamp_ns"][:], dtype=np.int64
            )
            index = np.asarray(handle["training/index_30hz"][:], dtype=np.int64)
            segments = np.asarray(handle["training/segments_30hz"][:], dtype=np.int64)
            validity = np.asarray(handle["raw/validity/sides"][:], dtype=np.bool_)
            eligible = np.asarray(
                handle["training/contact_eligible"][:], dtype=np.bool_
            )
    except Exception as exc:
        errors.append(f"cannot read collection metrics: {exc}")
        return EpisodeAudit(
            path=str(episode_path),
            ok=False,
            errors=tuple(errors),
            task=task,
            checksum_verified=verify_checksums,
        )

    source_valid = bool(
        source_timestamps.size
        and np.all(source_timestamps > 0)
        and np.all(np.diff(source_timestamps) > 0)
    )
    if require_metadata and not source_valid:
        errors.append("PICO source timestamps are not strictly positive and increasing")
    usable_windows = count_training_windows(segments)
    if require_usable_training and usable_windows <= 0:
        errors.append("episode has no complete contact-eligible 258-row SPD window")
    both_valid = np.all(validity, axis=1) if validity.ndim == 2 else np.zeros(0, dtype=np.bool_)
    task_scene = task.split("/", 1)[0] if task and "/" in task else None
    return EpisodeAudit(
        path=str(episode_path),
        ok=not errors,
        errors=tuple(errors),
        task=task,
        scene=task_scene,
        seed=seed,
        model_sha256=artifact_hashes["model_sha256"],
        urdf_sha256=artifact_hashes["urdf_sha256"],
        collision_manifest_sha256=artifact_hashes["collision_manifest_sha256"],
        raw_frames=int(timestamps.size),
        training_frames=int(index.size),
        raw_duration_s=_duration_ns(timestamps) / 1e9,
        qualified_duration_s=_qualified_duration_ns(timestamps, index, segments) / 1e9,
        source_duration_s=_duration_ns(source_timestamps) / 1e9 if source_valid else 0.0,
        source_timestamps_valid=source_valid,
        valid_both_sides_frames=int(np.count_nonzero(both_valid)),
        valid_both_sides_fraction=(float(np.mean(both_valid)) if both_valid.size else 0.0),
        contact_eligible_frames=int(np.count_nonzero(eligible)),
        usable_training_windows=int(usable_windows),
        checksum_verified=verify_checksums,
    )


@dataclass(frozen=True, slots=True)
class CollectionAudit:
    """Aggregate report for a directory of episodes."""

    ok: bool
    episodes: tuple[EpisodeAudit, ...]
    raw_hours: float
    qualified_hours: float
    source_hours: float
    tasks: tuple[str, ...]
    scenes: tuple[str, ...]
    missing_tasks: tuple[str, ...]
    target_hours: float
    target_met: bool
    required_all_tasks: bool
    require_target: bool
    artifact_hashes_consistent: bool

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["episodes"] = [episode.as_dict() for episode in self.episodes]
        return result


def audit_collection(
    root: str | Path,
    *,
    target_hours: float = DEFAULT_TARGET_HOURS,
    require_target: bool = False,
    require_all_tasks: bool = False,
    require_metadata: bool = False,
    require_usable_training: bool = False,
    verify_checksums: bool = True,
) -> CollectionAudit:
    """Aggregate episode gates and optionally enforce the formal target."""
    if not np.isfinite(float(target_hours)) or float(target_hours) <= 0.0:
        raise ValueError("target_hours must be positive and finite")
    paths = scan_episodes(root)
    audits = tuple(
        audit_episode(
            path,
            verify_checksums=verify_checksums,
            require_metadata=require_metadata,
            require_usable_training=require_usable_training,
        )
        for path in paths
    )
    tasks = tuple(sorted({item.task for item in audits if item.task is not None}))
    scenes = tuple(sorted({item.scene for item in audits if item.scene is not None}))
    required_tasks = set(TASK_REGISTRY) if require_all_tasks else set()
    missing_tasks = tuple(sorted(required_tasks.difference(tasks)))
    raw_hours = sum(item.raw_duration_s for item in audits) / 3600.0
    qualified_hours = sum(item.qualified_duration_s for item in audits) / 3600.0
    source_hours = sum(item.source_duration_s for item in audits) / 3600.0
    artifact_tuples = {
        (item.model_sha256, item.urdf_sha256, item.collision_manifest_sha256)
        for item in audits
        if item.model_sha256 is not None
        or item.urdf_sha256 is not None
        or item.collision_manifest_sha256 is not None
    }
    artifact_hashes_consistent = len(artifact_tuples) <= 1
    target_met = qualified_hours >= float(target_hours)
    ok = bool(audits) and all(item.ok for item in audits)
    if require_all_tasks and missing_tasks:
        ok = False
    if require_target and not target_met:
        ok = False
    if require_metadata and (not artifact_tuples or not artifact_hashes_consistent):
        ok = False
    return CollectionAudit(
        ok=ok,
        episodes=audits,
        raw_hours=float(raw_hours),
        qualified_hours=float(qualified_hours),
        source_hours=float(source_hours),
        tasks=tasks,
        scenes=scenes,
        missing_tasks=missing_tasks,
        target_hours=float(target_hours),
        target_met=target_met,
        required_all_tasks=require_all_tasks,
        require_target=require_target,
        artifact_hashes_consistent=artifact_hashes_consistent,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, help="HDF5 file or directory")
    parser.add_argument("--target-hours", type=float, default=DEFAULT_TARGET_HOURS)
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="fail unless contact-qualified duration reaches --target-hours",
    )
    parser.add_argument(
        "--require-all-tasks",
        action="store_true",
        help="fail unless all 17 registered scene/tasks are represented",
    )
    parser.add_argument(
        "--require-metadata",
        action="store_true",
        help="require run_id/operator_id/pico_serial and valid source timestamps",
    )
    parser.add_argument(
        "--require-usable-training",
        action="store_true",
        help="fail an episode with no complete contact-eligible 258-row window",
    )
    parser.add_argument("--no-checksums", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = audit_collection(
            args.episodes,
            target_hours=args.target_hours,
            require_target=args.require_target,
            require_all_tasks=args.require_all_tasks,
            require_metadata=args.require_metadata,
            require_usable_training=args.require_usable_training,
            verify_checksums=not args.no_checksums,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report.ok else 1


__all__ = [
    "COLLECTION_METADATA_KEYS",
    "CollectionAudit",
    "DEFAULT_TARGET_HOURS",
    "EpisodeAudit",
    "audit_collection",
    "audit_episode",
    "collection_metadata",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
