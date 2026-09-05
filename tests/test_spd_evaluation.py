import json

import pytest

from spd_vr.evaluation import (
    EVALUATION_INPUT_SCHEMA,
    EVALUATION_SCHEMA,
    PLANNED_ABLATIONS,
    build_evaluation_report,
    main,
    validate_evaluation_report,
)
from spd_vr.scenes.registry import TASK_REGISTRY


def _input():
    variants = ("full", *PLANNED_ABLATIONS)
    return {
        "schema_version": EVALUATION_INPUT_SCHEMA,
        "git_commit": "a" * 40,
        "dataset_split_sha256": "b" * 64,
        "model_config_sha256": "c" * 64,
        "dino_checkpoint_sha256": "d" * 64,
        "seed": 17,
        "tasks": {
            task: {variant: [True, False, True] for variant in variants}
            for task in TASK_REGISTRY
        },
    }


def test_evaluation_report_covers_all_tasks_and_planned_ablations():
    report = build_evaluation_report(_input())

    assert report["schema_version"] == EVALUATION_SCHEMA
    assert report["ok"] is True
    assert report["task_count"] == 17
    assert report["ablations"] == list(PLANNED_ABLATIONS)
    assert report["results"]["jenga/hollow_tower"]["full"]["success_rate"] == pytest.approx(2 / 3)
    assert report["results"]["jenga/hollow_tower"]["full"]["confidence_interval"][0] < 2 / 3
    assert validate_evaluation_report(report) == report


def test_evaluation_report_rejects_missing_task_or_bad_outcome():
    document = _input()
    document["tasks"].pop("bottles/toss_in_bin")
    with pytest.raises(ValueError, match="missing registered tasks"):
        build_evaluation_report(document)

    document = _input()
    document["tasks"]["jenga/hollow_tower"]["full"] = [2]
    with pytest.raises(ValueError, match="binary"):
        build_evaluation_report(document)


def test_evaluation_report_validation_is_fail_closed_for_tampered_interval():
    report = build_evaluation_report(_input())
    report["results"]["jenga/hollow_tower"]["full"]["confidence_interval"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="confidence interval"):
        validate_evaluation_report(report)


def test_evaluation_cli_writes_report(tmp_path, capsys):
    source = tmp_path / "input.json"
    output = tmp_path / "evaluation.json"
    source.write_text(json.dumps(_input()), encoding="utf-8")
    assert main(["--input", str(source), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == EVALUATION_SCHEMA
    assert json.loads(capsys.readouterr().out)["ok"] is True
