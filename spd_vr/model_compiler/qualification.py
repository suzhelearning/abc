"""Cached contact-qualification receipts for fast live-run startup.

A receipt is written only after the expensive authoritative artifact and
contact checks succeed.  Runtime verification re-hashes the small set of
authoritative files and checks every collision proxy's recorded size and
nanosecond mtime, avoiding MuJoCo model construction and its QHull work.
Release/acceptance checks continue to use the full cryptographic verifier.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from typing import Any, Mapping

import yaml

from .artifacts import verify_artifacts, verify_contact_qualified


QUALIFICATION_RECEIPT = "contact_qualification.json"
QUALIFICATION_KIND = "spd-vr-contact-qualification"
QUALIFICATION_SCHEMA_VERSION = 1
_OUTPUT_FILES = (
    "model_manifest.yaml",
    "collision_manifest.yaml",
    "unified_plant.xml",
    "arm_ik.xml",
)


class QualificationError(ValueError):
    """Raised when a cached qualification is absent, malformed, or stale."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _output_path(output: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise QualificationError("qualification receipt contains an invalid relative path")
    path = (output / relative).resolve()
    try:
        path.relative_to(output)
    except ValueError as exc:
        raise QualificationError("qualification receipt path escapes the artifact directory") from exc
    return path


def verify_contact_qualification_receipt(
    output_dir: str | Path,
    urdf_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Quickly prove that a previously qualified local artifact bundle is unchanged."""

    output = Path(output_dir).resolve()
    urdf = Path(urdf_path).resolve()
    receipt_file = (
        output / QUALIFICATION_RECEIPT
        if receipt_path is None
        else Path(receipt_path).resolve()
    )
    try:
        document = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(
            f"contact qualification receipt is unavailable: {receipt_file}; "
            "run spd-model --verify --verify-contact --write-receipt"
        ) from exc
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != QUALIFICATION_SCHEMA_VERSION
        or document.get("kind") != QUALIFICATION_KIND
    ):
        raise QualificationError("contact qualification receipt has an invalid schema")

    qualified_files = document.get("qualified_files")
    expected_keys = {"urdf", *_OUTPUT_FILES}
    if not isinstance(qualified_files, Mapping) or set(qualified_files) != expected_keys:
        raise QualificationError("contact qualification receipt file set is incomplete")
    for name in ("urdf", *_OUTPUT_FILES):
        record = qualified_files[name]
        if not isinstance(record, Mapping) or not _is_sha256(record.get("sha256")):
            raise QualificationError(f"qualification file record is malformed: {name}")
        if name == "urdf":
            if record.get("path") != str(urdf):
                raise QualificationError("qualification receipt belongs to a different URDF")
            path = urdf
        else:
            if record.get("path") != name:
                raise QualificationError(f"qualification file path is invalid: {name}")
            path = _output_path(output, name)
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise QualificationError(f"qualified file changed since qualification: {path}")

    full_verification = document.get("full_verification")
    if (
        not isinstance(full_verification, Mapping)
        or isinstance(full_verification.get("contact_records"), bool)
        or not isinstance(full_verification.get("contact_records"), int)
        or full_verification["contact_records"] <= 0
    ):
        raise QualificationError("receipt does not prove a successful full contact verification")

    pieces = document.get("collision_pieces")
    if not isinstance(pieces, list) or not pieces:
        raise QualificationError("qualification receipt has no collision proxy inventory")
    seen: set[str] = set()
    for record in pieces:
        if not isinstance(record, Mapping):
            raise QualificationError("collision proxy receipt record is malformed")
        relative = record.get("path")
        size = record.get("size")
        mtime_ns = record.get("mtime_ns")
        if (
            not isinstance(relative, str)
            or relative in seen
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or isinstance(mtime_ns, bool)
            or not isinstance(mtime_ns, int)
            or mtime_ns <= 0
        ):
            raise QualificationError("collision proxy receipt record is malformed")
        seen.add(relative)
        proxy = _output_path(output, relative)
        try:
            stat = proxy.stat()
        except OSError as exc:
            raise QualificationError(
                f"collision proxy changed since qualification: {proxy}"
            ) from exc
        if stat.st_size != size or stat.st_mtime_ns != mtime_ns:
            raise QualificationError(f"collision proxy changed since qualification: {proxy}")

    return {
        "verification": "cached_contact_qualification",
        "receipt": str(receipt_file),
        "collision_pieces": len(pieces),
        "contact_records": full_verification["contact_records"],
    }


def write_contact_qualification_receipt(
    output_dir: str | Path,
    urdf_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> Path:
    """Run the authoritative checks and atomically publish a runtime receipt."""

    output = Path(output_dir).resolve()
    urdf = Path(urdf_path).resolve()
    verified = verify_artifacts(output / "model_manifest.yaml", urdf)
    contact_report = verify_contact_qualified(
        output / "collision_manifest.yaml", urdf_path=urdf
    )
    if verified.output_dir.resolve() != output:
        raise QualificationError("verified artifacts belong to a different output directory")

    collision_manifest = output / "collision_manifest.yaml"
    try:
        collision_document = yaml.safe_load(collision_manifest.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise QualificationError(f"cannot inventory qualified collision pieces: {exc}") from exc
    records = collision_document.get("records") if isinstance(collision_document, Mapping) else None
    if not isinstance(records, list) or not records:
        raise QualificationError("qualified collision manifest has no records")

    piece_paths: set[Path] = set()
    for record in records:
        pieces = record.get("pieces") if isinstance(record, Mapping) else None
        if not isinstance(pieces, list) or not pieces:
            raise QualificationError("qualified collision manifest has malformed pieces")
        for piece in pieces:
            filename = piece.get("file") if isinstance(piece, Mapping) else None
            piece_paths.add(_output_path(output, filename))

    inventory: list[dict[str, Any]] = []
    for piece in sorted(piece_paths):
        try:
            stat = piece.stat()
        except OSError as exc:
            raise QualificationError(f"qualified collision proxy is unavailable: {piece}") from exc
        inventory.append(
            {
                "path": str(piece.relative_to(output)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    qualified_files: dict[str, dict[str, str]] = {
        "urdf": {"path": str(urdf), "sha256": _sha256(urdf)}
    }
    for name in _OUTPUT_FILES:
        qualified_files[name] = {"path": name, "sha256": _sha256(output / name)}
    document = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "kind": QUALIFICATION_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qualified_files": qualified_files,
        "collision_pieces": inventory,
        "full_verification": {
            "contact_records": contact_report["records"],
            "surface_gate": contact_report["surface_gate"],
        },
    }
    destination = (
        output / QUALIFICATION_RECEIPT
        if receipt_path is None
        else Path(receipt_path).resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


__all__ = [
    "QUALIFICATION_RECEIPT",
    "QualificationError",
    "verify_contact_qualification_receipt",
    "write_contact_qualification_receipt",
]
