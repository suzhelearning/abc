import hashlib
import json

from spd_vr.policy_benchmark import EXPECTED_DINO_MODEL_ID
from spd_vr.release import audit_release


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _evidence(tmp_path):
    checkpoint = tmp_path / "dinov3_vitb16_pretrain_lvd1689m.pth"
    checkpoint.write_bytes(b"official-checkpoint-bytes")
    _write_json(
        tmp_path / "dino.json",
        {
            "model_id": EXPECTED_DINO_MODEL_ID,
            "checkpoint_filename": checkpoint.name,
            "source_url": "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m",
            "license": "dinov3-license",
            "license_url": "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
            "terms_accepted": True,
            "access_date": "2026-09-05",
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    )
    _write_json(
        tmp_path / "vendor.json",
        {
            "rights_holder": "Tianji/Wuji2 vendor",
            "written_confirmation": "permission-record-001",
            "scope": "research publication and generated contact artifacts",
            "redistribution_permitted": True,
            "asset_sha256": {"tianji_wuji2.urdf": "a" * 64},
        },
    )
    _write_json(
        tmp_path / "benchmark.json",
        {
            "compile": True,
            "chunk_deadline_p95_ok": True,
            "dino_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "peak_memory": {"peak_allocated_bytes": 1, "peak_reserved_bytes": 2},
        },
    )
    _write_json(
        tmp_path / "training.json",
        {
            "resume_verified": True,
            "world_size": 8,
            "formal_steps": 170000,
            "numerically_continuous": True,
        },
    )
    _write_json(
        tmp_path / "collection.json",
        {
            "ok": True,
            "target_met": True,
            "required_all_tasks": True,
            "artifact_hashes_consistent": True,
            "qualified_hours": 75,
            "target_hours": 75,
        },
    )
    _write_json(
        tmp_path / "evaluation.json",
        {
            "ok": True,
            "ablations": ["visual", "history", "contact", "actual_qpos", "streaming"],
            "confidence_intervals": {"task": [0.1, 0.2]},
        },
    )
    _write_json(
        tmp_path / "safety.json",
        {"approved": True, "scope": "simulation release only"},
    )
    manifest = {
        "schema_version": 1,
        "git_commit": "1" * 40,
        "dino_checkpoint": checkpoint.name,
        "vendor_terms": "vendor.json",
        "dino_provenance": "dino.json",
        "policy_benchmark": "benchmark.json",
        "training_resume": "training.json",
        "collection_audit": "collection.json",
        "evaluation": "evaluation.json",
        "safety_review": "safety.json",
    }
    manifest_path = tmp_path / "release.json"
    _write_json(manifest_path, manifest)
    return manifest_path, checkpoint


def test_release_audit_is_fail_closed_for_missing_evidence(tmp_path):
    manifest = tmp_path / "release.json"
    _write_json(manifest, {"schema_version": 1, "git_commit": "1" * 40})
    report = audit_release(manifest)
    assert report.ok is False
    assert any(not item.ok for item in report.checks)


def test_release_audit_accepts_complete_archived_evidence(tmp_path):
    manifest, checkpoint = _evidence(tmp_path)
    report = audit_release(manifest, dino_checkpoint=checkpoint, expected_commit="1" * 40)
    assert report.ok is True
    assert {item.name for item in report.checks} >= {
        "manifest",
        "git_commit",
        "vendor_terms",
        "dino_provenance",
        "policy_benchmark",
        "training_resume",
        "collection_audit",
        "evaluation",
        "safety_review",
    }


def test_release_audit_rejects_non_finite_collection_or_negative_memory(tmp_path):
    manifest, checkpoint = _evidence(tmp_path)
    benchmark = tmp_path / "benchmark.json"
    benchmark_document = json.loads(benchmark.read_text())
    benchmark_document["peak_memory"]["peak_allocated_bytes"] = -1
    _write_json(benchmark, benchmark_document)
    report = audit_release(manifest, dino_checkpoint=checkpoint)
    assert report.ok is False
    assert any(item.name == "policy_benchmark" and not item.ok for item in report.checks)

    manifest, checkpoint = _evidence(tmp_path)
    collection = tmp_path / "collection.json"
    collection_document = json.loads(collection.read_text())
    collection_document["qualified_hours"] = float("nan")
    _write_json(collection, collection_document)
    report = audit_release(manifest, dino_checkpoint=checkpoint)
    assert report.ok is False
    assert any(item.name == "collection_audit" and not item.ok for item in report.checks)
