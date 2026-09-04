import json

import pytest

from spd_vr import policy_benchmark


def test_policy_benchmark_requires_a_real_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not download"):
        policy_benchmark.benchmark_policy(tmp_path / "missing.pth")


def test_policy_benchmark_rejects_invalid_measurement_options(tmp_path):
    checkpoint = tmp_path / "dino.pth"
    checkpoint.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="measure_ticks"):
        policy_benchmark.benchmark_policy(checkpoint, measure_ticks=0)


def test_policy_benchmark_cli_returns_structured_failure_for_missing_checkpoint(tmp_path, capsys):
    assert policy_benchmark.main(["--dino-checkpoint", str(tmp_path / "missing.pth")]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "DINOv3 checkpoint" in report["error"]
