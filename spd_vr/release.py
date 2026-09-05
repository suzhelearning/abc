"""Fail-closed aggregate audit for a publishable SPD-VR release.

The repository can validate the shape of each evidence artifact, but it cannot
invent vendor permission, a target-GPU run, human demonstrations, or a safety
review.  This command therefore consumes a small manifest of externally
archived JSON records and refuses to call a release ready until every gate is
explicitly green.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .collection_plan import validate_collection_plan
from .evaluation import validate_evaluation_report
from .policy_benchmark import validate_checkpoint_provenance


SCHEMA_VERSION = 1
EVIDENCE_KEYS = (
    "vendor_terms",
    "dino_provenance",
    "policy_benchmark",
    "training_resume",
    "collection_audit",
    "collection_plan",
    "evaluation",
    "safety_review",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReleaseAudit:
    ok: bool
    manifest: str
    checks: tuple[ReleaseCheck, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "manifest": self.manifest,
            "checks": [item.as_dict() for item in self.checks],
        }


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} evidence not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name} evidence JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} evidence must be a JSON object: {path}")
    return value


def _resolve_entry(manifest_path: Path, value: Any, name: str) -> Path:
    if isinstance(value, (str, Path)):
        raw = str(value)
    elif isinstance(value, Mapping) and isinstance(value.get("path"), str):
        raw = value["path"]
    else:
        raise ValueError(f"release manifest entry {name} must contain a path")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path)


def _check_commit(document: Mapping[str, Any], expected_commit: str | None) -> ReleaseCheck:
    actual = document.get("git_commit")
    if not isinstance(actual, str) or not _COMMIT_RE.fullmatch(actual):
        return ReleaseCheck("git_commit", False, "manifest git_commit must be a 40-character SHA-1")
    if expected_commit is not None:
        if not _COMMIT_RE.fullmatch(expected_commit):
            return ReleaseCheck("git_commit", False, "expected_commit must be a 40-character SHA-1")
        if actual.lower() != expected_commit.lower():
            return ReleaseCheck(
                "git_commit",
                False,
                f"manifest commit {actual} does not match expected {expected_commit}",
            )
    return ReleaseCheck("git_commit", True, actual)


def _check_vendor_terms(document: Mapping[str, Any]) -> ReleaseCheck:
    if document.get("redistribution_permitted") is not True:
        return ReleaseCheck("vendor_terms", False, "redistribution_permitted must be true")
    required = ("rights_holder", "written_confirmation", "scope")
    missing = [key for key in required if not isinstance(document.get(key), str) or not document[key].strip()]
    if missing:
        return ReleaseCheck("vendor_terms", False, f"missing written terms fields: {', '.join(missing)}")
    hashes = document.get("asset_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        return ReleaseCheck("vendor_terms", False, "asset_sha256 must be a non-empty mapping")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or not _SHA256_RE.fullmatch(value)
        for name, value in hashes.items()
    ):
        return ReleaseCheck("vendor_terms", False, "asset_sha256 contains an invalid entry")
    return ReleaseCheck("vendor_terms", True, f"terms cover {len(hashes)} hashed assets")


def _check_policy_benchmark(
    document: Mapping[str, Any], provenance: Mapping[str, Any]
) -> ReleaseCheck:
    if document.get("compile") is not True:
        return ReleaseCheck("policy_benchmark", False, "release benchmark must be the reviewed compile run")
    if document.get("chunk_deadline_p95_ok") is not True:
        return ReleaseCheck("policy_benchmark", False, "compiled benchmark did not pass the 30 Hz p95 deadline")
    if document.get("dino_checkpoint_sha256") != provenance.get("sha256"):
        return ReleaseCheck("policy_benchmark", False, "benchmark and DINO provenance hashes differ")
    memory = document.get("peak_memory")
    if (
        not isinstance(memory, Mapping)
        or isinstance(memory.get("peak_allocated_bytes"), bool)
        or not isinstance(memory.get("peak_allocated_bytes"), int)
        or memory.get("peak_allocated_bytes") < 0
        or isinstance(memory.get("peak_reserved_bytes"), bool)
        or not isinstance(memory.get("peak_reserved_bytes"), int)
        or memory.get("peak_reserved_bytes") < 0
    ):
        return ReleaseCheck("policy_benchmark", False, "compiled benchmark is missing CUDA peak-memory evidence")
    return ReleaseCheck("policy_benchmark", True, "compiled p95, checkpoint hash, and peak memory are recorded")


def _check_training(document: Mapping[str, Any]) -> ReleaseCheck:
    if document.get("resume_verified") is not True:
        return ReleaseCheck("training_resume", False, "resume_verified must be true")
    world_size = document.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 8:
        return ReleaseCheck("training_resume", False, "world_size must be at least 8")
    steps = document.get("formal_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 170_000:
        return ReleaseCheck("training_resume", False, "formal_steps must be at least 170000")
    if document.get("numerically_continuous") is not True:
        return ReleaseCheck("training_resume", False, "numerically_continuous must be true")
    return ReleaseCheck("training_resume", True, f"{world_size}-GPU resume and {steps}-step run are recorded")


def _check_collection(document: Mapping[str, Any]) -> ReleaseCheck:
    if document.get("ok") is not True:
        return ReleaseCheck("collection_audit", False, "collection audit did not pass")
    for key in ("target_met", "required_all_tasks", "artifact_hashes_consistent"):
        if document.get(key) is not True:
            return ReleaseCheck("collection_audit", False, f"collection audit field {key} is not true")
    qualified = document.get("qualified_hours")
    target = document.get("target_hours")
    if (
        isinstance(qualified, bool)
        or not isinstance(qualified, (int, float))
        or not math.isfinite(float(qualified))
    ):
        return ReleaseCheck("collection_audit", False, "qualified_hours is missing")
    if (
        isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not math.isfinite(float(target))
        or target <= 0
        or qualified < target
    ):
        return ReleaseCheck("collection_audit", False, "qualified_hours does not reach target_hours")
    return ReleaseCheck("collection_audit", True, f"{qualified:g} qualified hours and all registered tasks are present")


def _check_collection_plan(document: Mapping[str, Any]) -> ReleaseCheck:
    try:
        plan = validate_collection_plan(document)
    except Exception as exc:
        return ReleaseCheck("collection_plan", False, str(exc))
    return ReleaseCheck(
        "collection_plan",
        True,
        f"{plan['task_count']} tasks and {plan['episode_count']} planned episodes are bound",
    )


def _check_evaluation(
    document: Mapping[str, Any],
    *,
    manifest_commit: str | None = None,
    dino_sha256: str | None = None,
) -> ReleaseCheck:
    try:
        report = validate_evaluation_report(document)
    except Exception as exc:
        return ReleaseCheck("evaluation", False, str(exc))
    if manifest_commit is not None and report["git_commit"].lower() != manifest_commit.lower():
        return ReleaseCheck("evaluation", False, "evaluation git_commit differs from release manifest")
    if dino_sha256 is not None and report["dino_checkpoint_sha256"] != dino_sha256:
        return ReleaseCheck("evaluation", False, "evaluation and DINO provenance hashes differ")
    return ReleaseCheck(
        "evaluation",
        True,
        f"{report['task_count']} tasks and {len(report['ablations'])} planned ablations have validated intervals",
    )


def _check_safety(document: Mapping[str, Any]) -> ReleaseCheck:
    if document.get("approved") is not True:
        return ReleaseCheck("safety_review", False, "safety review is not approved")
    scope = document.get("scope")
    if not isinstance(scope, str) or not scope.strip():
        return ReleaseCheck("safety_review", False, "safety review scope is missing")
    return ReleaseCheck("safety_review", True, "approved scope is recorded")


def audit_release(
    manifest_path: str | Path,
    *,
    dino_checkpoint: str | Path | None = None,
    expected_commit: str | None = None,
) -> ReleaseAudit:
    """Validate all external evidence referenced by one release manifest."""

    path = Path(manifest_path).expanduser().resolve()
    checks: list[ReleaseCheck] = []
    try:
        manifest = _load_json(path, "release manifest")
    except Exception as exc:
        return ReleaseAudit(False, str(path), (ReleaseCheck("manifest", False, str(exc)),))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        checks.append(ReleaseCheck("manifest", False, f"schema_version must be {SCHEMA_VERSION}"))
    else:
        checks.append(ReleaseCheck("manifest", True, "schema version accepted"))
    checks.append(_check_commit(manifest, expected_commit))

    evidence: dict[str, Mapping[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name in EVIDENCE_KEYS:
        try:
            evidence_path = _resolve_entry(path, manifest.get(name), name)
            paths[name] = evidence_path
            evidence[name] = _load_json(evidence_path, name)
        except Exception as exc:
            checks.append(ReleaseCheck(name, False, str(exc)))

    if "vendor_terms" in evidence:
        checks.append(_check_vendor_terms(evidence["vendor_terms"]))
    if "dino_provenance" in evidence:
        checkpoint_value = dino_checkpoint or manifest.get("dino_checkpoint")
        try:
            checkpoint_path = _resolve_entry(path, checkpoint_value, "dino_checkpoint")
            provenance = validate_checkpoint_provenance(paths["dino_provenance"], checkpoint_path)
            checks.append(ReleaseCheck("dino_provenance", True, f"validated {provenance['sha256']}"))
        except Exception as exc:
            checks.append(ReleaseCheck("dino_provenance", False, str(exc)))
            provenance = None
    else:
        provenance = None
    if "policy_benchmark" in evidence:
        if provenance is None:
            checks.append(ReleaseCheck("policy_benchmark", False, "DINO provenance must pass first"))
        else:
            checks.append(_check_policy_benchmark(evidence["policy_benchmark"], provenance))
    if "training_resume" in evidence:
        checks.append(_check_training(evidence["training_resume"]))
    if "collection_audit" in evidence:
        checks.append(_check_collection(evidence["collection_audit"]))
    if "collection_plan" in evidence:
        checks.append(_check_collection_plan(evidence["collection_plan"]))
    if "evaluation" in evidence:
        checks.append(
            _check_evaluation(
                evidence["evaluation"],
                manifest_commit=manifest.get("git_commit") if isinstance(manifest.get("git_commit"), str) else None,
                dino_sha256=provenance.get("sha256") if provenance is not None else None,
            )
        )
    if "safety_review" in evidence:
        checks.append(_check_safety(evidence["safety_review"]))
    return ReleaseAudit(bool(checks) and all(item.ok for item in checks), str(path), tuple(checks))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="release evidence manifest JSON")
    parser.add_argument("--dino-checkpoint", type=Path, default=None)
    parser.add_argument("--expected-commit", default=None)
    args = parser.parse_args(argv)
    report = audit_release(
        args.manifest,
        dino_checkpoint=args.dino_checkpoint,
        expected_commit=args.expected_commit,
    )
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report.ok else 1


__all__ = [
    "EVIDENCE_KEYS",
    "ReleaseAudit",
    "ReleaseCheck",
    "audit_release",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
