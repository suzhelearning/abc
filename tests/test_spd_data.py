import json

import h5py
import numpy as np
import pytest

import spd_vr.data as data_module

from spd_vr.contracts import CAMERA_NAMES
from spd_vr.data import (
    CameraFrame,
    EpisodeWriter,
    RawFrame,
    build_training_index,
    build_contact_segments,
    filter_contact_mask,
    sequence_indices,
    validate_normalization,
    validate_episode,
)
from spd_vr.replay import replay_episode
from spd_vr.filter_contacts import filter_episode


def test_contact_filter_removes_only_runs_longer_than_ten_seconds_and_splits_grid():
    timestamps = np.arange(0, 703, dtype=np.int64) * 16_666_667
    contacts = np.ones(timestamps.shape, dtype=np.bool_)
    contacts[10:13] = False  # short idle gap stays eligible
    contacts[100:702] = False  # >10 seconds at 60 Hz
    keep, audit = filter_contact_mask(timestamps, contacts)
    assert bool(np.all(keep[10:13]))
    assert not bool(np.any(keep[100:702]))
    assert audit[0]["samples"] == 602
    grid = np.arange(8, dtype=np.int64)
    eligible = np.array([True, True, False, True, True, True, False, True])
    assert np.array_equal(
        build_contact_segments(grid, eligible),
        np.array([[0, 2], [3, 6], [7, 8]], dtype=np.int64),
    )
    assert np.array_equal(
        build_contact_segments(np.array([0, 2, 3]), np.ones(3, dtype=np.bool_)),
        np.array([[0, 1], [1, 3]], dtype=np.int64),
    )


def test_normalization_requires_finite_positive_54d_vectors():
    valid = {
        "qpos_mean": np.zeros(54, dtype=np.float64),
        "qpos_std": np.ones(54, dtype=np.float64),
        "action_mean": np.zeros(54, dtype=np.float64),
        "action_std": np.ones(54, dtype=np.float64),
    }
    normalized = validate_normalization(valid)
    assert set(normalized) == set(valid)
    assert len(normalized["qpos_mean"]) == 54
    assert validate_normalization(None) == {}

    with pytest.raises(ValueError, match="keys must be exactly"):
        validate_normalization({"qpos_mean": [0.0] * 54})
    invalid_shape = dict(valid, action_mean=[0.0] * 53)
    with pytest.raises(ValueError, match=r"shape \(54,"):
        validate_normalization(invalid_shape)
    invalid_std = dict(valid, qpos_std=[0.0] * 54)
    with pytest.raises(ValueError, match="strictly positive"):
        validate_normalization(invalid_std)
    underflow_std = dict(valid, action_std=[1e-50] * 54)
    with pytest.raises(ValueError, match="positive in float32"):
        validate_normalization(underflow_std)
    invalid_values = dict(valid, action_std=[float("nan")] * 54)
    with pytest.raises(ValueError, match="finite"):
        validate_normalization(invalid_values)


def _cameras():
    return {
        camera: CameraFrame(
            np.zeros((168, 224, 3), dtype=np.uint8),
            np.zeros((168, 224, 2), dtype=np.int32),
        )
        for camera in CAMERA_NAMES
    }


def _frame(index):
    return RawFrame(
        timestamp_ns=1_000_000_000 + index * 16_666_667,
        qpos=np.full(54, index, dtype=np.float32),
        qvel=np.zeros(54, dtype=np.float32),
        qpos_target=np.full(54, index + 100, dtype=np.float32),
        mujoco_full_state=np.zeros(109, dtype=np.float64),
        cameras=_cameras(),
        pico_hands=np.zeros((2, 26, 7), dtype=np.float32),
        pico_timestamp_ns=2_000_000_000 + index,
        pico_sequence_id=index,
        tracking_epoch=1,
        pico_scale=np.ones(2, dtype=np.float32),
        validity=np.ones(2, dtype=np.bool_),
        objects={"object": index},
        contacts=[],
    )


def test_episode_is_atomic_valid_and_keeps_target_separate(tmp_path):
    output = tmp_path / "episode.hdf5"
    with EpisodeWriter(output, {"seed": 7}) as writer:
        for index in range(3):
            writer.append(_frame(index))

    manifest = validate_episode(output, verify_checksums=True)
    assert manifest["seed"] == 7
    assert manifest["policy_action_source"].startswith("raw/action/qpos")
    assert not list(tmp_path.glob("*.staging"))
    with h5py.File(output, "r") as handle:
        assert handle["raw/observation/qpos"].shape == (3, 54)
        assert handle["raw/action/qpos"].shape == (3, 54)
        assert np.array_equal(handle["raw/action/qpos"][:], handle["raw/observation/qpos"][:])
        assert handle["raw/cameras/top/segmentation"].shape == (3, 168, 224, 2)
        assert handle["raw/contacts/hand_object"].shape == (3,)
        assert handle["training/grid_step"].shape == handle["training/index_30hz"].shape
        assert not np.array_equal(
            handle["raw/observation/qpos"][:], handle["raw/action/qpos_target"][:]
        )


def test_30hz_index_and_258_row_training_window_have_no_leakage():
    raw_timestamps = 1_000_000_000 + np.arange(515, dtype=np.int64) * 16_666_667
    index = build_training_index(raw_timestamps)
    assert index.shape == (258,)
    rows = sequence_indices(index, 1)
    assert rows["history"].shape == (256,)
    assert rows["previous"].shape == (256,)
    assert rows["images"].shape == (32,)
    assert rows["future"].shape == (32, 8)
    assert rows["previous"][0] == index[0]
    assert rows["history"][0] == index[1]
    assert rows["future"][0, 0] == index[2]
    assert rows["future"][-1, -1] == index[-1]


def test_empty_episode_is_aborted_instead_of_published(tmp_path):
    output = tmp_path / "empty.hdf5"
    writer = EpisodeWriter(output, {})
    with pytest.raises(ValueError, match="empty episode"):
        writer.finish()
    assert not output.exists()
    assert not list(tmp_path.glob("*.staging"))


def test_episode_writer_can_require_usable_training_before_publish(tmp_path, monkeypatch):
    output = tmp_path / "idle.hdf5"
    writer = EpisodeWriter(output, {}, require_usable_training=True)
    for index in range(3):
        writer.append(_frame(index))

    def no_eligible_rows(timestamps, contacts, *, threshold_ns=0):
        return np.zeros(len(timestamps), dtype=np.bool_), []

    monkeypatch.setattr(data_module, "filter_contact_mask", no_eligible_rows)
    with pytest.raises(ValueError, match="usable contact-eligible training window"):
        writer.finish()
    assert not output.exists()
    assert not list(tmp_path.glob("*.staging"))


def test_episode_writer_rejects_a_segment_shorter_than_one_spd_window(tmp_path, monkeypatch):
    output = tmp_path / "short.hdf5"
    writer = EpisodeWriter(output, {}, require_usable_training=True)
    for index in range(3):
        writer.append(_frame(index))

    monkeypatch.setattr(
        data_module,
        "filter_contact_mask",
        lambda timestamps, contacts, *, threshold_ns=0: (
            np.ones(len(timestamps), dtype=np.bool_),
            [],
        ),
    )
    monkeypatch.setattr(
        data_module,
        "build_contact_segments",
        lambda grid_step, eligible: np.asarray([[0, 1]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="usable contact-eligible training window"):
        writer.finish()
    assert not output.exists()
    assert not list(tmp_path.glob("*.staging"))


def test_replay_structural_mode_validates_full_state_stream(tmp_path):
    output = tmp_path / "replay.hdf5"
    with EpisodeWriter(output, {"model_sha256": "not-checked-without-model"}) as writer:
        for index in range(3):
            writer.append(_frame(index))
    report = replay_episode(output)
    assert report["valid"] is True
    assert report["replayed"] is False
    assert report["raw_frames"] == 3


def test_contact_filter_writes_atomic_provenance_copy(tmp_path):
    source = tmp_path / "source.h5"
    output = tmp_path / "filtered.h5"
    with EpisodeWriter(source, {"seed": 3}) as writer:
        for index in range(3):
            writer.append(_frame(index))
    report = filter_episode(source, output)
    assert report["raw_removed_frames"] == 0
    manifest = validate_episode(output, verify_checksums=True)
    assert manifest["contact_filter"]["derived_from"] == str(source.resolve())
    assert not list(tmp_path.glob("*.staging-*"))


def test_action_stream_cannot_diverge_from_actual_qpos(tmp_path):
    output = tmp_path / "mismatch.h5"
    with EpisodeWriter(output, {}) as writer:
        for index in range(3):
            writer.append(_frame(index))
    with h5py.File(output, "r+") as handle:
        handle["raw/action/qpos"][0, 0] += 1.0
    with pytest.raises(ValueError, match="action/qpos"):
        validate_episode(output)
