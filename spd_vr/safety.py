"""Fail-closed validation for the SPD-VR safety-review evidence record."""

from __future__ import annotations

from datetime import date
from collections.abc import Mapping
from typing import Any


SAFETY_REVIEW_SCHEMA = "spd-vr-safety-review-v1"


def _non_empty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def validate_safety_review(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an explicitly simulation-bounded safety review."""

    if not isinstance(document, Mapping):
        raise ValueError("safety review must be a JSON object")
    if document.get("schema_version") != SAFETY_REVIEW_SCHEMA:
        raise ValueError(f"schema_version must be {SAFETY_REVIEW_SCHEMA}")
    if document.get("approved") is not True:
        raise ValueError("safety review approved must be true")
    for key in ("review_id", "reviewer", "scope"):
        _non_empty(document.get(key), f"safety review {key}")
    reviewed_at = _non_empty(document.get("reviewed_at"), "safety review reviewed_at")
    try:
        date.fromisoformat(reviewed_at)
    except ValueError as exc:
        raise ValueError("safety review reviewed_at must be an ISO date") from exc
    if document.get("simulation_only") is not True:
        raise ValueError("safety review must explicitly set simulation_only=true")
    if document.get("physical_actuation_authorized") is not False:
        raise ValueError("physical_actuation_authorized must be false")
    if document.get("follower_commands_emitted") is not False:
        raise ValueError("follower_commands_emitted must be false")
    hazards = document.get("known_hazards")
    if not isinstance(hazards, list) or not hazards:
        raise ValueError("known_hazards must be a non-empty list")
    hazard_names: list[str] = []
    for index, hazard in enumerate(hazards):
        name = _non_empty(hazard, f"known_hazards[{index}]")
        if name in hazard_names:
            raise ValueError(f"duplicate safety hazard: {name}")
        hazard_names.append(name)
    mitigations = document.get("mitigations")
    if not isinstance(mitigations, Mapping):
        raise ValueError("mitigations must be an object keyed by hazard")
    for hazard in hazard_names:
        _non_empty(mitigations.get(hazard), f"mitigations.{hazard}")
    evidence = document.get("evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        raise ValueError("safety review evidence must be a non-empty object")
    return dict(document)


__all__ = ["SAFETY_REVIEW_SCHEMA", "validate_safety_review"]

