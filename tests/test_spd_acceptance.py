import hashlib
import json
from pathlib import Path

from spd_vr import acceptance
from spd_vr.scenes.registry import TASKS


def test_acceptance_is_fail_closed_for_missing_requested_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(
        acceptance,
        "validate_tasks",
        lambda seed_start, seed_count: {
            "scenes": 6,
            "tasks": 17,
            "resets": 17 * seed_count,
            "seed_start": seed_start,
            "seed_count": seed_count,
        },
    )
    monkeypatch.setattr(
        acceptance,
        "verify_artifacts",
        lambda manifest, urdf: (_ for _ in ()).throw(ValueError("missing artifacts")),
    )
    results = acceptance.run_acceptance(
        repo_root=tmp_path,
        episodes_path=tmp_path / "episodes",
        seed_count=1,
    )
    assert next(item for item in results if item.name == "scenes").ok
    assert not next(item for item in results if item.name == "artifacts").ok
    episodes = next(item for item in results if item.name == "episodes")
    assert episodes.ok is False
    assert "no HDF5" in episodes.detail


def test_acceptance_reports_unrequested_data_without_claiming_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        acceptance,
        "validate_tasks",
        lambda seed_start, seed_count: {
            "scenes": 6,
            "tasks": 17,
            "resets": 17,
            "seed_start": seed_start,
            "seed_count": seed_count,
        },
    )
    monkeypatch.setattr(
        acceptance,
        "verify_artifacts",
        lambda manifest, urdf: (_ for _ in ()).throw(ValueError("not built")),
    )
    results = acceptance.run_acceptance(repo_root=tmp_path)
    episodes = next(item for item in results if item.name == "episodes")
    assert episodes.ok is True
    assert "not requested" in episodes.detail


def test_model_builder_uses_repository_root_and_single_compiler(monkeypatch, tmp_path):
    from spd_vr import model_builder

    calls = {}

    class Result:
        full_model = tmp_path / "unified_plant.xml"
        path = tmp_path / "model_manifest.yaml"
        actuator_calibration = tmp_path / "actuator_calibration.yaml"

    def fake_compile(urdf, output, cache, *, raw_collisions):
        calls.update(urdf=urdf, output=output, cache=cache, raw=raw_collisions)
        return Result()

    monkeypatch.setattr(model_builder, "compile_models", fake_compile)
    returned = model_builder.build_model(tmp_path / "generated", raw_collisions=True)
    assert returned == (Result.full_model, Result.path, Result.actuator_calibration)
    assert calls["urdf"] == model_builder.default_urdf()
    assert calls["output"] == tmp_path / "generated"
    assert calls["cache"] == tmp_path / "collision_cache"
    assert calls["raw"] is True


def test_scene_manifest_acceptance_checks_model_and_builder_provenance(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = TASKS[0]
    result = spec.reset(3)
    model = tmp_path / "scene.xml"
    model.write_bytes(b"scene-model-v1")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = tmp_path / "scene.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "spd-vr-scene-v1",
                "task": spec.qualified_name,
                "reset": result.manifest(),
                "object_bodies": [item.name for item in result.objects],
                "model": {"path": str(model), "sha256": digest(model)},
                "builder_source_sha256": {
                    name: digest(root / "spd_vr" / "scenes" / name)
                    for name in ("registry.py", "scene_builder.py", "model_scene.py")
                },
            }
        ),
        encoding="utf-8",
    )

    passed = acceptance._check_scene_manifest(manifest, root=root, model=model)
    assert passed.ok
    assert acceptance._declared_scene_model(manifest) == model

    model.write_bytes(b"scene-model-tampered")
    failed = acceptance._check_scene_manifest(manifest, root=root, model=model)
    assert failed.ok is False
    assert "model hash" in failed.detail
