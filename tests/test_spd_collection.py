import numpy as np
import pytest

from spd_vr.collection import (
    audit_collection,
    audit_episode,
    collection_metadata,
)
from spd_vr.contracts import CAMERA_NAMES
from spd_vr.data import CameraFrame, EpisodeWriter, RawFrame


def _frame(index: int) -> RawFrame:
    cameras = {
        name: CameraFrame(
            np.zeros((168, 224, 3), dtype=np.uint8),
            np.zeros((168, 224, 2), dtype=np.int32),
        )
        for name in CAMERA_NAMES
    }
    return RawFrame(
        timestamp_ns=1_000_000_000 + index * 16_666_667,
        qpos=np.full(54, index, dtype=np.float32),
        qvel=np.zeros(54, dtype=np.float32),
        qpos_target=np.full(54, index + 100, dtype=np.float32),
        mujoco_full_state=np.zeros(109, dtype=np.float64),
        cameras=cameras,
        pico_hands=np.zeros((2, 26, 7), dtype=np.float32),
        pico_timestamp_ns=2_000_000_000 + index,
        pico_sequence_id=index,
        tracking_epoch=1,
        pico_scale=np.ones(2, dtype=np.float32),
        validity=np.ones(2, dtype=np.bool_),
        objects={},
        contacts=[{"hand_object": True}],
    )


def _write_episode(path, *, task="jenga/hollow_tower", metadata=True):
    manifest = {
        "scene_manifest": {"task": task, "reset": {"seed": 7}},
        "model_sha256": "a" * 64,
        "urdf_sha256": "b" * 64,
        "collision_manifest_sha256": "c" * 64,
    }
    if metadata:
        manifest["collection"] = collection_metadata(
            run_id="run-001", operator_id="operator-01", pico_serial="pico-01"
        )
    with EpisodeWriter(path, manifest, require_usable_training=True) as writer:
        for index in range(515):
            writer.append(_frame(index))


def test_collection_metadata_requires_a_complete_identity():
    assert collection_metadata() == {}
    assert collection_metadata(
        run_id="run", operator_id="operator", pico_serial="pico"
    ) == {"run_id": "run", "operator_id": "operator", "pico_serial": "pico"}
    with pytest.raises(ValueError, match="requires all fields"):
        collection_metadata(run_id="run")
    with pytest.raises(ValueError, match="non-empty"):
        collection_metadata(run_id="run", operator_id=" ", pico_serial="pico")


def test_episode_audit_reports_qualified_duration_and_provenance(tmp_path):
    path = tmp_path / "episode.hdf5"
    _write_episode(path)
    report = audit_episode(
        path,
        verify_checksums=True,
        require_metadata=True,
        require_usable_training=True,
    )
    assert report.ok
    assert report.task == "jenga/hollow_tower"
    assert report.scene == "jenga"
    assert report.seed == 7
    assert report.model_sha256 == "a" * 64
    assert report.collision_manifest_sha256 == "c" * 64
    assert report.raw_frames == 515
    assert report.training_frames == 258
    assert report.usable_training_windows == 1
    assert report.source_timestamps_valid
    assert report.valid_both_sides_fraction == 1.0
    assert report.qualified_duration_s > 0.0


def test_collection_audit_is_report_only_until_formal_flags_are_requested(tmp_path):
    _write_episode(tmp_path / "episode.hdf5")
    report = audit_collection(tmp_path, require_metadata=True, require_usable_training=True)
    assert report.ok
    assert report.tasks == ("jenga/hollow_tower",)
    assert report.scenes == ("jenga",)
    assert report.target_met is False
    assert report.missing_tasks == ()
    assert report.artifact_hashes_consistent

    formal = audit_collection(
        tmp_path,
        require_metadata=True,
        require_usable_training=True,
        require_target=True,
        require_all_tasks=True,
    )
    assert formal.ok is False
    assert formal.target_met is False
    assert "jenga/hollow_tower" not in formal.missing_tasks
    assert len(formal.missing_tasks) == 16


def test_collection_audit_rejects_missing_formal_metadata_and_unknown_task(tmp_path):
    _write_episode(tmp_path / "anonymous.hdf5", metadata=False)
    anonymous = audit_collection(
        tmp_path / "anonymous.hdf5", require_metadata=True, require_usable_training=True
    )
    assert anonymous.ok is False
    assert "manifest collection metadata is missing" in anonymous.episodes[0].errors

    _write_episode(tmp_path / "unknown.hdf5", task="unknown/task")
    unknown = audit_collection(tmp_path / "unknown.hdf5")
    assert unknown.ok is False
    assert any("not in the SPD registry" in error for error in unknown.episodes[0].errors)
