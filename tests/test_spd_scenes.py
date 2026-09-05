import numpy as np
import json
import pytest

from spd_vr.augment import SymmetrySpec, augment_trajectory
from spd_vr.scenes.registry import SCENES, TASKS
from spd_vr.scenes.manifest import load_scene_manifest
from spd_vr.scenes.model_scene import write_scene_model
from spd_vr.visual import randomize_instance_colors


def test_registry_has_six_scenes_and_seventeen_deterministic_tasks():
    assert len(SCENES) == 6
    assert len(TASKS) == 17
    for spec in TASKS:
        first = spec.reset(19).manifest()
        second = spec.reset(19).manifest()
        assert first == second
        assert spec.target_duration_s == 60.0 * spec.table2_minutes / spec.table2_episodes


def test_scene_manifest_validates_task_and_object_provenance(tmp_path):
    spec = TASKS[0]
    result = spec.reset(4)
    path = tmp_path / "scene.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "spd-vr-scene-v1",
                "task": spec.qualified_name,
                "reset": result.manifest(),
                "object_bodies": [item.name for item in result.objects],
            }
        ),
        encoding="utf-8",
    )
    assert load_scene_manifest(path)["task"] == spec.qualified_name


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mass_kg", "not-a-number", "mass_kg"),
        ("position", [0.0, 0.0], "position"),
        ("geoms", [], "geoms"),
    ],
)
def test_scene_manifest_rejects_malformed_object_reset_fields(tmp_path, field, value, message):
    spec = TASKS[0]
    result = spec.reset(4)
    document = {
        "schema_version": "spd-vr-scene-v1",
        "task": spec.qualified_name,
        "reset": result.manifest(),
        "object_bodies": [item.name for item in result.objects],
    }
    document["reset"]["objects"][0][field] = value
    path = tmp_path / f"malformed-{field}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_scene_manifest(path)


def test_scene_manifest_rejects_duplicate_geom_names(tmp_path):
    spec = TASKS[0]
    result = spec.reset(4)
    document = {
        "schema_version": "spd-vr-scene-v1",
        "task": spec.qualified_name,
        "reset": result.manifest(),
        "object_bodies": [item.name for item in result.objects],
    }
    objects = document["reset"]["objects"]
    objects[1]["geoms"][0]["name"] = objects[0]["geoms"][0]["name"]
    path = tmp_path / "duplicate-geom.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate scene geom"):
        load_scene_manifest(path)


def test_symmetry_transform_is_consistent_for_labels_and_wrist_views():
    permutation = tuple(reversed(range(54)))
    signs = tuple(1.0 if index % 2 else -1.0 for index in range(54))
    spec = SymmetrySpec(permutation, signs)
    qpos = np.arange(54, dtype=np.float32)
    previous = qpos[None, :]
    future = qpos.reshape(1, 1, 54)
    images = {
        "top": np.arange(2 * 3 * 1, dtype=np.uint8).reshape(2, 3, 1),
        "left_wrist": np.full((2, 3, 1), 7, dtype=np.uint8),
        "right_wrist": np.full((2, 3, 1), 9, dtype=np.uint8),
    }
    transformed = augment_trajectory(qpos, previous, future, images, spec)
    assert np.array_equal(transformed[0], qpos[list(permutation)] * np.asarray(signs))
    assert transformed[3]["left_wrist"][0, 0, 0] == 9
    assert transformed[3]["right_wrist"][0, -1, 0] == 7


def test_segmentation_randomization_changes_instances_not_background():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [10, 20, 30]
    seg = np.zeros((2, 3, 2), dtype=np.int32)
    seg[0, 0, 1] = 4
    result = randomize_instance_colors(rgb, seg, np.random.default_rng(0), strength=1.0)
    assert np.array_equal(result[1, 1], rgb[1, 1])
    assert not np.array_equal(result[0, 0], rgb[0, 0])


def test_scene_model_writer_does_not_consume_generated_worldbody(tmp_path, vendor_urdf):
    # A generated result is deliberately reusable: writing a second model
    # must not mutate the first result's XML children.
    from spd_vr.model_compiler.artifacts import compile_models

    compiled = compile_models(
        vendor_urdf,
        tmp_path / "generated",
        tmp_path / "collision_cache",
        raw_collisions=True,
    )
    result = TASKS[0].reset(2)
    first = write_scene_model(compiled.full_model, result, tmp_path / "first.xml")
    second = write_scene_model(compiled.full_model, result, tmp_path / "second.xml")
    assert first.read_bytes() == second.read_bytes()
    assert len(result.objects) > 0 and len(tuple(result.worldbody)) > 0
