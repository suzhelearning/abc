import pytest

from spd_vr.safety import SAFETY_REVIEW_SCHEMA, validate_safety_review


def _review():
    return {
        "schema_version": SAFETY_REVIEW_SCHEMA,
        "review_id": "review-001",
        "reviewer": "safety-reviewer",
        "reviewed_at": "2026-09-05",
        "scope": "simulation-only SPD-VR evidence",
        "approved": True,
        "simulation_only": True,
        "physical_actuation_authorized": False,
        "follower_commands_emitted": False,
        "known_hazards": ["unqualified_contact", "stale_tracking"],
        "mitigations": {
            "unqualified_contact": "contact manifest gate",
            "stale_tracking": "50 ms side-local HOLD",
        },
        "evidence": {"acceptance": "acceptance.json", "runbook": "external-validation.md"},
    }


def test_safety_review_requires_explicit_simulation_boundary():
    assert validate_safety_review(_review())["simulation_only"] is True

    physical = _review()
    physical["physical_actuation_authorized"] = True
    with pytest.raises(ValueError, match="physical_actuation_authorized"):
        validate_safety_review(physical)


def test_safety_review_rejects_missing_mitigation_or_bad_date():
    missing = _review()
    missing["mitigations"].pop("stale_tracking")
    with pytest.raises(ValueError, match="mitigations.stale_tracking"):
        validate_safety_review(missing)

    bad_date = _review()
    bad_date["reviewed_at"] = "not-a-date"
    with pytest.raises(ValueError, match="ISO date"):
        validate_safety_review(bad_date)

