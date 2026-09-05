import json

import pytest

from spd_vr.collection_plan import (
    COLLECTION_PLAN_SCHEMA,
    build_collection_plan,
    main,
    validate_collection_plan,
)


def test_collection_plan_matches_all_table2_quotas_and_75_hour_target():
    plan = build_collection_plan(seed_start=1000)

    assert plan["schema_version"] == COLLECTION_PLAN_SCHEMA
    assert plan["status"] == "planned"
    assert plan["task_count"] == 17
    assert plan["episode_count"] == 1916
    assert plan["qualified_target_hours"] == pytest.approx(75.25)
    assert plan["target_hours"] == pytest.approx(75.0)
    assert len(plan["episodes"]) == 1916
    assert len({item["episode_id"] for item in plan["episodes"]}) == 1916
    assert plan["episodes"][0]["seed"] == 1000
    assert plan["episodes"][-1]["seed"] == 2915
    assert validate_collection_plan(plan) == plan


def test_collection_plan_is_deterministic_and_covers_each_task():
    first = build_collection_plan(seed_start=7)
    second = build_collection_plan(seed_start=7)
    assert first == second
    counts = {}
    for episode in first["episodes"]:
        counts[episode["task"]] = counts.get(episode["task"], 0) + 1
    assert len(counts) == 17
    assert counts["jenga/hollow_tower"] == 92
    assert counts["bottles/toss_in_bin"] == 350


def test_collection_plan_rejects_tampering_and_invalid_target():
    plan = build_collection_plan()
    tampered = json.loads(json.dumps(plan))
    tampered["episodes"][0]["task"] = "unknown/task"
    with pytest.raises(ValueError, match="episode 1|unknown"):
        validate_collection_plan(tampered)

    with pytest.raises(ValueError, match="target_hours"):
        build_collection_plan(target_hours=76.0)


def test_collection_plan_cli_writes_audit_ready_json(tmp_path, capsys):
    output = tmp_path / "collection-plan.json"
    assert main(["--output", str(output), "--seed-start", "12"]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["seed_start"] == 12
    assert document["status"] == "planned"
    printed = json.loads(capsys.readouterr().out)
    assert printed["episode_count"] == 1916
