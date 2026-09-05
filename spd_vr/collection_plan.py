"""Create and validate the approved SPD-VR collection schedule.

The scene registry already records the episode counts and minutes from SPD
Table 2.  This module turns that source of truth into a deterministic,
operator-facing plan before any human recording starts.  It is a plan, not a
collection ledger: no episode is considered collected or qualified here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import tempfile
from numbers import Integral, Real
from typing import Any

from .scenes.registry import TASKS, TASK_REGISTRY


COLLECTION_PLAN_SCHEMA = "spd-vr-collection-plan-v1"
_TABLE2_TOTAL_MINUTES = sum(spec.table2_minutes for spec in TASKS)
_TABLE2_TOTAL_EPISODES = sum(spec.table2_episodes for spec in TASKS)


def _positive_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _seed(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _identity(
    *,
    run_id: str | None,
    operator_id: str | None,
    pico_serial: str | None,
) -> dict[str, str] | None:
    values = {
        "run_id": run_id,
        "operator_id": operator_id,
        "pico_serial": pico_serial,
    }
    present = [name for name, value in values.items() if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        missing = ", ".join(name for name, value in values.items() if value is None)
        raise ValueError(f"collection plan identity requires all fields; missing {missing}")
    result: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"collection plan identity {name} must be a non-empty string")
        result[name] = value.strip()
    return result


def _episode_id(task: str, task_index: int) -> str:
    return f"{task.replace('/', '--')}-{task_index:04d}"


def build_collection_plan(
    *,
    seed_start: int = 0,
    target_hours: float = 75.0,
    run_id: str | None = None,
    operator_id: str | None = None,
    pico_serial: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic schedule for all registered SPD tasks.

    The registry quotas are intentionally not scaled to fit a requested target:
    changing the Table-2 episode distribution would make the experiment
    incomparable.  A target below the 75.25-hour registry total is therefore
    allowed and leaves an explicit qualified reserve; a larger target fails
    until the operator supplies a new reviewed schedule.
    """

    seed_start = _seed(seed_start, "seed_start")
    target_hours = _positive_finite(target_hours, "target_hours")
    qualified_target_hours = _TABLE2_TOTAL_MINUTES / 60.0
    if target_hours > qualified_target_hours:
        raise ValueError(
            "target_hours exceeds the registered Table-2 schedule; "
            "review a larger schedule instead of silently scaling quotas"
        )
    identity = _identity(
        run_id=run_id,
        operator_id=operator_id,
        pico_serial=pico_serial,
    )

    quotas: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    ordinal = 0
    for spec in TASKS:
        quota = {
            "scene": spec.scene,
            "task": spec.name,
            "qualified_task": spec.qualified_name,
            "prompt": spec.prompt,
            "table2_episodes": int(spec.table2_episodes),
            "table2_minutes": int(spec.table2_minutes),
            "target_duration_s": float(spec.target_duration_s),
            "seed_start": seed_start + ordinal,
            "seed_end": seed_start + ordinal + spec.table2_episodes - 1,
        }
        quotas.append(quota)
        for task_index in range(1, spec.table2_episodes + 1):
            episodes.append(
                {
                    "ordinal": ordinal + 1,
                    "episode_id": _episode_id(spec.qualified_name, task_index),
                    "scene": spec.scene,
                    "task": spec.qualified_name,
                    "task_episode_index": task_index,
                    "task_episode_count": int(spec.table2_episodes),
                    "seed": seed_start + ordinal,
                    "target_duration_s": float(spec.target_duration_s),
                }
            )
            ordinal += 1

    plan: dict[str, Any] = {
        "schema_version": COLLECTION_PLAN_SCHEMA,
        "status": "planned",
        "data_collected": False,
        "plan_id": "spd-table2-v1",
        "source": {
            "registry": "spd_vr.scenes.registry.TASKS",
            "table2_total_minutes": _TABLE2_TOTAL_MINUTES,
            "table2_total_episodes": _TABLE2_TOTAL_EPISODES,
        },
        "target_hours": float(target_hours),
        "qualified_target_hours": float(qualified_target_hours),
        "task_count": len(TASKS),
        "episode_count": len(episodes),
        "seed_start": seed_start,
        "seed_policy": "seed = seed_start + zero_based_global_episode_ordinal",
        "task_quotas": quotas,
        "episodes": episodes,
    }
    if identity is not None:
        plan["collection_identity"] = identity
    return validate_collection_plan(plan)


def validate_collection_plan(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a plan against the current 17-task registry and seed policy."""

    if not isinstance(document, Mapping):
        raise ValueError("collection plan must be a JSON object")
    if document.get("schema_version") != COLLECTION_PLAN_SCHEMA:
        raise ValueError(f"schema_version must be {COLLECTION_PLAN_SCHEMA}")
    if document.get("status") != "planned" or document.get("data_collected") is not False:
        raise ValueError("collection plan must remain a planned, non-collected artifact")
    if document.get("plan_id") != "spd-table2-v1":
        raise ValueError("unsupported collection plan id")
    target_hours = _positive_finite(document.get("target_hours"), "target_hours")
    qualified_hours = _positive_finite(
        document.get("qualified_target_hours"), "qualified_target_hours"
    )
    if target_hours > qualified_hours:
        raise ValueError("target_hours exceeds qualified_target_hours")
    seed_start = _seed(document.get("seed_start"), "seed_start")
    if document.get("seed_policy") != "seed = seed_start + zero_based_global_episode_ordinal":
        raise ValueError("unsupported seed policy")
    if document.get("task_count") != len(TASKS):
        raise ValueError("task_count does not match the current registry")
    if document.get("episode_count") != _TABLE2_TOTAL_EPISODES:
        raise ValueError("episode_count does not match Table-2 quotas")
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("source metadata is missing")
    if source.get("registry") != "spd_vr.scenes.registry.TASKS":
        raise ValueError("collection plan source registry is not recognized")
    if source.get("table2_total_minutes") != _TABLE2_TOTAL_MINUTES:
        raise ValueError("Table-2 total minutes do not match the registry")
    if source.get("table2_total_episodes") != _TABLE2_TOTAL_EPISODES:
        raise ValueError("Table-2 total episodes do not match the registry")

    identity = document.get("collection_identity")
    if identity is not None:
        if not isinstance(identity, Mapping):
            raise ValueError("collection_identity must be an object")
        if set(identity) != {"run_id", "operator_id", "pico_serial"}:
            raise ValueError("collection_identity must contain run_id/operator_id/pico_serial")
        for key, value in identity.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"collection_identity.{key} must be non-empty")

    raw_quotas = document.get("task_quotas")
    if not isinstance(raw_quotas, list) or len(raw_quotas) != len(TASKS):
        raise ValueError("task_quotas must list every registered task exactly once")
    seen_tasks: set[str] = set()
    expected_ordinal = 0
    for raw_quota, spec in zip(raw_quotas, TASKS, strict=True):
        if not isinstance(raw_quota, Mapping):
            raise ValueError("task quota must be an object")
        if raw_quota.get("qualified_task") != spec.qualified_name:
            raise ValueError("task quota order or task name does not match the registry")
        if spec.qualified_name in seen_tasks:
            raise ValueError("duplicate task quota")
        seen_tasks.add(spec.qualified_name)
        if raw_quota.get("scene") != spec.scene or raw_quota.get("task") != spec.name:
            raise ValueError("task quota scene/task does not match the registry")
        if raw_quota.get("prompt") != spec.prompt:
            raise ValueError(f"task quota prompt differs for {spec.qualified_name}")
        if raw_quota.get("table2_episodes") != spec.table2_episodes:
            raise ValueError(f"task quota episode count differs for {spec.qualified_name}")
        if raw_quota.get("table2_minutes") != spec.table2_minutes:
            raise ValueError(f"task quota minutes differ for {spec.qualified_name}")
        duration = raw_quota.get("target_duration_s")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, Real)
            or not math.isfinite(float(duration))
            or not math.isclose(float(duration), spec.target_duration_s, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"task quota duration differs for {spec.qualified_name}")
        if raw_quota.get("seed_start") != seed_start + expected_ordinal:
            raise ValueError(f"task quota seed_start differs for {spec.qualified_name}")
        expected_ordinal += spec.table2_episodes
        if raw_quota.get("seed_end") != seed_start + expected_ordinal - 1:
            raise ValueError(f"task quota seed_end differs for {spec.qualified_name}")

    raw_episodes = document.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != _TABLE2_TOTAL_EPISODES:
        raise ValueError("episodes do not match the registered Table-2 quota")
    seen_ids: set[str] = set()
    ordinal = 0
    for spec in TASKS:
        for task_index in range(1, spec.table2_episodes + 1):
            item = raw_episodes[ordinal]
            if not isinstance(item, Mapping):
                raise ValueError("collection plan episode must be an object")
            expected = {
                "ordinal": ordinal + 1,
                "episode_id": _episode_id(spec.qualified_name, task_index),
                "scene": spec.scene,
                "task": spec.qualified_name,
                "task_episode_index": task_index,
                "task_episode_count": spec.table2_episodes,
                "seed": seed_start + ordinal,
            }
            for key, value in expected.items():
                if item.get(key) != value:
                    raise ValueError(f"episode {ordinal + 1} field {key} differs from plan")
            duration = item.get("target_duration_s")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, Real)
                or not math.isfinite(float(duration))
                or not math.isclose(float(duration), spec.target_duration_s, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(f"episode {ordinal + 1} target duration differs from registry")
            episode_id = item["episode_id"]
            if episode_id in seen_ids:
                raise ValueError(f"duplicate episode_id: {episode_id}")
            seen_ids.add(episode_id)
            ordinal += 1
    return dict(document)


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON plan path")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--target-hours", type=float, default=75.0)
    parser.add_argument("--run-id")
    parser.add_argument("--operator-id")
    parser.add_argument("--pico-serial")
    args = parser.parse_args(argv)
    try:
        plan = build_collection_plan(
            seed_start=args.seed_start,
            target_hours=args.target_hours,
            run_id=args.run_id,
            operator_id=args.operator_id,
            pico_serial=args.pico_serial,
        )
        _write_json_atomic(args.output, plan)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1
    report = dict(plan)
    report["ok"] = True
    report["output"] = str(args.output.expanduser().resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "COLLECTION_PLAN_SCHEMA",
    "build_collection_plan",
    "main",
    "validate_collection_plan",
]


if __name__ == "__main__":
    raise SystemExit(main())

