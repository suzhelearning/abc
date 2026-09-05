"""Build and validate task-level SPD-VR evaluation/ablation evidence.

The evaluator consumes binary task outcomes from an already reviewed split. It
does not run a policy or manufacture outcomes.  Its purpose is to make the
required comparison dimensions, provenance, and confidence intervals
machine-checkable before a release audit accepts the report.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any

from .scenes.registry import TASK_REGISTRY


EVALUATION_INPUT_SCHEMA = "spd-vr-evaluation-input-v1"
EVALUATION_SCHEMA = "spd-vr-evaluation-v1"
PLANNED_ABLATIONS = (
    "visual_input",
    "history",
    "contact_filtering",
    "actual_qpos_labels",
    "streaming_inference",
)
_VARIANTS = ("full", *PLANNED_ABLATIONS)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_WILSON_Z = 1.959963984540054


def _metadata(document: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    if document.get("schema_version") != schema:
        raise ValueError(f"schema_version must be {schema}")
    commit = document.get("git_commit")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise ValueError("git_commit must be a 40-character SHA-1")
    values: dict[str, Any] = {"git_commit": commit}
    for key in ("dataset_split_sha256", "model_config_sha256", "dino_checkpoint_sha256"):
        value = document.get(key)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{key} must be a SHA-256 hex string")
        values[key] = value
    seed = document.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    values["seed"] = int(seed)
    return values


def _outcomes(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must contain at least one binary outcome")
    successes = 0
    for index, item in enumerate(value):
        if isinstance(item, bool):
            success = item
        elif isinstance(item, Integral) and not isinstance(item, bool) and int(item) in (0, 1):
            success = bool(item)
        else:
            raise ValueError(f"{label}[{index}] must be binary (0/1 or boolean)")
        successes += int(success)
    return successes, len(value)


def _wilson(successes: int, count: int) -> tuple[float, float]:
    rate = successes / count
    z2 = _WILSON_Z * _WILSON_Z
    denominator = 1.0 + z2 / count
    center = (rate + z2 / (2.0 * count)) / denominator
    half = (
        _WILSON_Z
        * math.sqrt(rate * (1.0 - rate) / count + z2 / (4.0 * count * count))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _validate_input(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise ValueError("evaluation input must be a JSON object")
    metadata = _metadata(document, schema=EVALUATION_INPUT_SCHEMA)
    tasks = document.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("tasks must be an object keyed by registered scene/task")
    expected_tasks = set(TASK_REGISTRY)
    actual_tasks = set(tasks)
    missing = sorted(expected_tasks - actual_tasks)
    extra = sorted(actual_tasks - expected_tasks)
    if missing:
        raise ValueError(f"missing registered tasks: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unknown evaluation tasks: {', '.join(extra)}")
    return metadata


def build_evaluation_report(document: Mapping[str, Any]) -> dict[str, Any]:
    """Convert binary task outcomes into a release-auditable report."""

    metadata = _validate_input(document)
    tasks = document["tasks"]
    results: dict[str, dict[str, dict[str, Any]]] = {}
    confidence_intervals: dict[str, dict[str, list[float]]] = {}
    total_episodes = 0
    for task in TASK_REGISTRY:
        task_values = tasks[task]
        if not isinstance(task_values, Mapping):
            raise ValueError(f"evaluation outcomes for {task} must be an object")
        if set(task_values) != set(_VARIANTS):
            missing = sorted(set(_VARIANTS) - set(task_values))
            extra = sorted(set(task_values) - set(_VARIANTS))
            detail = []
            if missing:
                detail.append(f"missing {', '.join(missing)}")
            if extra:
                detail.append(f"unknown {', '.join(extra)}")
            raise ValueError(f"evaluation variants for {task}: {'; '.join(detail)}")
        task_results: dict[str, dict[str, Any]] = {}
        task_intervals: dict[str, list[float]] = {}
        for variant in _VARIANTS:
            successes, count = _outcomes(task_values[variant], f"{task}.{variant}")
            interval = _wilson(successes, count)
            task_results[variant] = {
                "successes": successes,
                "n": count,
                "success_rate": successes / count,
                "confidence_interval": [interval[0], interval[1]],
            }
            task_intervals[variant] = [interval[0], interval[1]]
            total_episodes += count
        results[task] = task_results
        confidence_intervals[task] = task_intervals

    report: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA,
        "ok": True,
        "baseline": "full",
        "ablations": list(PLANNED_ABLATIONS),
        "method": "wilson-95",
        "task_count": len(TASK_REGISTRY),
        "evaluated_episodes": total_episodes,
        **metadata,
        "results": results,
        "confidence_intervals": confidence_intervals,
    }
    return validate_evaluation_report(report)


def _finite_probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite probability in [0,1]")
    return result


def validate_evaluation_report(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a generated report without trusting its summary fields."""

    if not isinstance(document, Mapping):
        raise ValueError("evaluation report must be a JSON object")
    metadata = _metadata(document, schema=EVALUATION_SCHEMA)
    if document.get("ok") is not True:
        raise ValueError("evaluation report ok must be true")
    if document.get("baseline") != "full":
        raise ValueError("evaluation baseline must be full")
    if document.get("ablations") != list(PLANNED_ABLATIONS):
        raise ValueError("evaluation ablations do not match the planned comparison")
    if document.get("method") != "wilson-95":
        raise ValueError("evaluation confidence interval method is not recognized")
    if document.get("task_count") != len(TASK_REGISTRY):
        raise ValueError("evaluation task_count does not cover the registered tasks")
    total = document.get("evaluated_episodes")
    if isinstance(total, bool) or not isinstance(total, Integral) or int(total) <= 0:
        raise ValueError("evaluated_episodes must be a positive integer")

    results = document.get("results")
    intervals = document.get("confidence_intervals")
    if not isinstance(results, Mapping) or set(results) != set(TASK_REGISTRY):
        raise ValueError("evaluation results must cover every registered task")
    if not isinstance(intervals, Mapping) or set(intervals) != set(TASK_REGISTRY):
        raise ValueError("confidence_intervals must cover every registered task")
    counted = 0
    for task in TASK_REGISTRY:
        task_results = results[task]
        task_intervals = intervals[task]
        if not isinstance(task_results, Mapping) or set(task_results) != set(_VARIANTS):
            raise ValueError(f"evaluation results for {task} must cover full and five ablations")
        if not isinstance(task_intervals, Mapping) or set(task_intervals) != set(_VARIANTS):
            raise ValueError(f"confidence intervals for {task} are incomplete")
        for variant in _VARIANTS:
            metric = task_results[variant]
            if not isinstance(metric, Mapping):
                raise ValueError(f"evaluation metric {task}.{variant} must be an object")
            successes = metric.get("successes")
            count = metric.get("n")
            if (
                isinstance(successes, bool)
                or not isinstance(successes, Integral)
                or isinstance(count, bool)
                or not isinstance(count, Integral)
                or int(count) <= 0
                or int(successes) < 0
                or int(successes) > int(count)
            ):
                raise ValueError(f"evaluation counts are invalid for {task}.{variant}")
            count = int(count)
            successes = int(successes)
            rate = _finite_probability(metric.get("success_rate"), f"{task}.{variant}.success_rate")
            expected_rate = successes / count
            if not math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"success_rate does not match counts for {task}.{variant}")
            raw_interval = metric.get("confidence_interval")
            if not isinstance(raw_interval, (list, tuple)) or len(raw_interval) != 2:
                raise ValueError(f"confidence interval is malformed for {task}.{variant}")
            low = _finite_probability(raw_interval[0], f"{task}.{variant}.confidence_interval[0]")
            high = _finite_probability(raw_interval[1], f"{task}.{variant}.confidence_interval[1]")
            if low > high or not low <= rate <= high:
                raise ValueError(f"confidence interval is inconsistent for {task}.{variant}")
            summary_interval = task_intervals[variant]
            if (
                not isinstance(summary_interval, (list, tuple))
                or len(summary_interval) != 2
                or not math.isclose(float(summary_interval[0]), low, rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(float(summary_interval[1]), high, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(f"confidence interval summary differs for {task}.{variant}")
            counted += count
    if int(total) != counted:
        raise ValueError("evaluated_episodes does not match task metrics")
    return dict(document)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evaluation input: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("evaluation input must be a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="binary task outcome JSON")
    parser.add_argument("--output", type=Path, help="optional report JSON path")
    args = parser.parse_args(argv)
    try:
        report = build_evaluation_report(_read_json(args.input).copy())
        if args.output is not None:
            args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
            args.output.expanduser().write_text(
                json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 1
    printed = dict(report)
    if args.output is not None:
        printed["output"] = str(args.output.expanduser().resolve())
    print(json.dumps(printed, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "EVALUATION_INPUT_SCHEMA",
    "EVALUATION_SCHEMA",
    "PLANNED_ABLATIONS",
    "build_evaluation_report",
    "main",
    "validate_evaluation_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
