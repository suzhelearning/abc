import json
import hashlib

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


def test_checkpoint_provenance_binds_terms_source_and_hash(tmp_path):
    checkpoint = tmp_path / "dinov3_vitb16_pretrain_lvd1689m.pth"
    checkpoint.write_bytes(b"checkpoint-bytes")
    provenance = tmp_path / "checkpoint.json"
    provenance.write_text(
        json.dumps(
            {
                "model_id": policy_benchmark.EXPECTED_DINO_MODEL_ID,
                "checkpoint_filename": checkpoint.name,
                "source_url": "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m",
                "license": "dinov3-license",
                "license_url": "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
                "terms_accepted": True,
                "access_date": "2026-09-05",
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    record = policy_benchmark.validate_checkpoint_provenance(provenance, checkpoint)
    assert record["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert record["terms_accepted"] is True


def test_checkpoint_provenance_rejects_hash_mismatch(tmp_path):
    checkpoint = tmp_path / "dinov3_vitb16_pretrain_lvd1689m.pth"
    checkpoint.write_bytes(b"checkpoint-bytes")
    provenance = tmp_path / "checkpoint.json"
    provenance.write_text(
        json.dumps(
            {
                "model_id": policy_benchmark.EXPECTED_DINO_MODEL_ID,
                "checkpoint_filename": checkpoint.name,
                "source_url": "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m",
                "license": "dinov3-license",
                "license_url": "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
                "terms_accepted": True,
                "access_date": "2026-09-05",
                "sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        policy_benchmark.validate_checkpoint_provenance(provenance, checkpoint)


def test_policy_benchmark_requires_provenance_for_formal_deadline(tmp_path, capsys):
    checkpoint = tmp_path / "dinov3_vitb16_pretrain_lvd1689m.pth"
    checkpoint.write_bytes(b"placeholder")
    assert policy_benchmark.main(
        ["--dino-checkpoint", str(checkpoint), "--enforce-deadline"]
    ) == 1
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "error": "--enforce-deadline requires --checkpoint-provenance",
        "ok": False,
    }


def test_peak_memory_report_is_explicit_on_cpu():
    import torch

    report = policy_benchmark._peak_memory_report(torch, torch.device("cpu"))
    assert report == {
        "device_type": "cpu",
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
