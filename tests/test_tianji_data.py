"""Small real HDF5/JPEG fixtures defending the Tianji SPD data contract."""

from io import BytesIO
import json

import h5py
import numpy as np
from PIL import Image
import pytest
import torch

from abc_minimal.config import TrainConfig
from abc_minimal.dataloader import DataParallelScope
from abc_minimal.preprocess import unnormalize
from abc_minimal.tianji_data import (
    CAMERA_NAMES,
    TianjiSequenceDataset,
    compute_tianji_normalization,
    load_tianji_config,
    prepare_spd_data,
    validate_spd_norm_stats,
)


POLICY_COLUMNS = np.r_[0:7, 14:34, 7:14, 34:54]


def _grid(rows):
    steps = np.arange(rows, dtype=np.int64)
    return steps // 30 * 1_000_000_000 + steps % 30 * 1_000_000_000 // 30


def _make_root(tmp_path, *, cameras=("top", "left_wrist"), episodes=2, rows=260):
    config = {
        "schema_version": 1,
        "joint_unit": "rad",
        "policy_rate_hz": 30,
        "image_encoding": "jpeg",
        "decoded_color_order": "RGB",
        "image_width": 1280,
        "image_height": 720,
        "jpeg_quality": 50,
        "robot_config": "wuji2",
        "joint_names": [f"joint_{index}" for index in range(54)],
        "camera_names": list(cameras),
    }
    (tmp_path / "dataset_config.json").write_text(json.dumps(config))
    encoded = BytesIO()
    Image.new("RGB", (1280, 720), (20, 40, 80)).save(encoded, format="JPEG", quality=50)
    jpeg = np.frombuffer(encoded.getvalue(), dtype=np.uint8)
    timestamps = _grid(rows)
    image_rows = np.unique(np.r_[np.arange(0, rows, 30), rows - 1])
    paths = []
    for episode in range(episodes):
        path = tmp_path / f"episode_{episode}.h5"
        paths.append(path)
        with h5py.File(path, "w") as handle:
            handle.attrs.update(schema_version=1, success=True, robot_config="wuji2", task="pick")
            source = (
                np.arange(rows, dtype=np.float32)[:, None]
                + np.arange(54, dtype=np.float32)[None, :] * 100
                + episode * 10000
            )
            for stream, values in (("arms", source[:, :14]), ("hands", source[:, 14:])):
                group = handle.create_group(f"observations/{stream}")
                group.create_dataset("timestamp_ns", data=timestamps)
                group.create_dataset("qpos", data=values)
            for camera in cameras:
                group = handle.create_group(f"images/{camera}")
                group.create_dataset("timestamp_ns", data=timestamps[image_rows])
                payloads = group.create_dataset("jpeg", (len(image_rows),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                for index in range(len(image_rows)):
                    payloads[index] = jpeg
    return paths


def _raw(tensor, dataset, kind):
    stats = {key: torch.tensor(value) for key, value in dataset.norm_stats[kind].items()}
    return unnormalize(tensor, stats).numpy()


def _config(root, *, batch_size=1, train_steps=4):
    config = TrainConfig(seed=7, batch_size=batch_size, num_workers=0, train_steps=train_steps)
    config.spd_data.root = str(root)
    return config


@pytest.mark.parametrize("cameras", [("top", "left_wrist"), CAMERA_NAMES])
def test_real_camera_contract_and_exact_causal_labels(tmp_path, cameras):
    _make_root(tmp_path, cameras=cameras)
    for split in ("train", "val"):
        dataset = TianjiSequenceDataset(tmp_path, split=split)
        sample = dataset[0]
        episode = int(dataset.episodes[0].stem.split("_")[-1])
        columns = POLICY_COLUMNS * 100 + episode * 10000
        np.testing.assert_allclose(_raw(sample["state"], dataset, "state"), np.arange(1, 257)[:, None] + columns, atol=0.002)
        np.testing.assert_allclose(_raw(sample["previous_actions"], dataset, "actions"), np.arange(256)[:, None] + columns, atol=0.002)
        anchors = 1 + np.arange(32) * 8
        future = anchors[:, None] + np.arange(1, 9)[None, :]
        np.testing.assert_allclose(_raw(sample["actions"], dataset, "actions"), future[:, :, None] + columns, atol=0.002)
        assert set(sample["images"]) == set(cameras)
        for images in sample["images"].values():
            assert images.shape == (32, 3, 224, 224)
            assert torch.isfinite(images).all()
        assert sample["camera_validity"].dtype == torch.bool
        assert sample["camera_validity"].tolist() == [[name in cameras for name in CAMERA_NAMES]] * 32
        assert len(dataset) == 3


def test_last_duplicate_wins_without_looking_into_future(tmp_path):
    paths = _make_root(tmp_path)
    for path in paths:
        with h5py.File(path, "r+") as handle:
            group = handle["observations/arms"]
            timestamps, qpos = group["timestamp_ns"][:], group["qpos"][:]
            timestamps = np.insert(timestamps, 11, timestamps[10])
            qpos = np.insert(qpos, 11, qpos[10] + 50000, axis=0)
            del group["timestamp_ns"], group["qpos"]
            group.create_dataset("timestamp_ns", data=timestamps)
            group.create_dataset("qpos", data=qpos)
    dataset = TianjiSequenceDataset(tmp_path)
    actual = _raw(dataset[0]["state"], dataset, "state")[:, 0]
    baseline = int(dataset.episodes[0].stem.split("_")[-1]) * 10000
    assert actual[8] == pytest.approx(baseline + 9)
    assert actual[9] == pytest.approx(baseline + 10 + 50000)
    assert actual[10] == pytest.approx(baseline + 11)


def test_stale_gaps_cannot_be_bridged_by_windows(tmp_path):
    paths = _make_root(tmp_path, rows=560)
    for path in paths:
        with h5py.File(path, "r+") as handle:
            for stream in ("arms", "hands"):
                group = handle[f"observations/{stream}"]
                timestamps, qpos = group["timestamp_ns"][:], group["qpos"][:]
                keep = np.r_[0:260, 300:560]
                del group["timestamp_ns"], group["qpos"]
                group.create_dataset("timestamp_ns", data=timestamps[keep])
                group.create_dataset("qpos", data=qpos[keep])
    dataset = TianjiSequenceDataset(tmp_path, joint_max_age_ms=1)
    # Two usable 260-row segments, not one compacted 520-row segment.
    assert len(dataset) == 6
    for index in (2, 3):
        state = _raw(dataset[index]["state"], dataset, "state")[:, 0]
        assert np.diff(state).max() == pytest.approx(1, abs=0.002)
    # Increasing the explicit limit allows causal holding across this same gap.
    permissive = TianjiSequenceDataset(tmp_path, joint_max_age_ms=2000)
    assert len(permissive) == 560 - 258 + 1


def test_train_normalization_excludes_validation_and_short_segments(tmp_path):
    paths = _make_root(tmp_path, episodes=5, rows=560)
    for path in paths:
        with h5py.File(path, "r+") as handle:
            for stream in ("arms", "hands"):
                group = handle[f"observations/{stream}"]
                timestamps, qpos = group["timestamp_ns"][:], group["qpos"][:]
                keep = np.r_[0:260, 500:560]
                # A valid but too-short segment must not skew fitted moments.
                qpos[500:] += 1_000_000
                del group["timestamp_ns"], group["qpos"]
                group.create_dataset("timestamp_ns", data=timestamps[keep])
                group.create_dataset("qpos", data=qpos[keep])
    dataset = TianjiSequenceDataset(tmp_path, seed=7, joint_max_age_ms=1)
    stats = compute_tianji_normalization(tmp_path, seed=7, joint_max_age_ms=1)
    offsets = [int(path.stem.split("_")[-1]) * 10000 for path in dataset.episodes]
    expected = np.concatenate([np.arange(260) + offset for offset in offsets])
    np.testing.assert_allclose(stats["state"]["mean"], expected.mean() + POLICY_COLUMNS * 100)
    np.testing.assert_allclose(stats["actions"]["std"], expected.std())
    heldout = set(paths) - set(dataset.episodes)
    for path in heldout:
        with h5py.File(path, "r+") as handle:
            for stream in ("arms", "hands"):
                handle[f"observations/{stream}/qpos"][:] += 100_000_000
    assert compute_tianji_normalization(tmp_path, seed=7, joint_max_age_ms=1) == stats


def test_resume_stream_and_saved_normalization(tmp_path):
    _make_root(tmp_path)
    config = _config(tmp_path)
    scope = DataParallelScope(0, 1, "replicated", True)
    stats = {kind: {"mean": [42.0] * 54, "std": [2.0] * 54} for kind in ("state", "actions")}
    full = prepare_spd_data(config, scope, 0, stats)
    resumed = prepare_spd_data(config, scope, 2, stats)
    full_batches = list(full.train_loader)
    resumed_batches = list(resumed.train_loader)
    for uninterrupted, restarted in zip(full_batches[2:], resumed_batches, strict=True):
        for key in ("state", "previous_actions", "actions", "camera_validity"):
            torch.testing.assert_close(uninterrupted[key], restarted[key], rtol=0, atol=0)
        for camera in uninterrupted["images"]:
            torch.testing.assert_close(uninterrupted["images"][camera], restarted["images"][camera], rtol=0, atol=0)
    assert full.norm_stats == stats
    raw = _raw(full_batches[0]["state"], full.train_dataset, "state")
    offset = int(full.train_dataset.episodes[0].stem.split("_")[-1]) * 10000
    assert np.isclose(raw[0, 0, 0], offset + np.arange(1, 4), atol=0.002).any()
    assert full.contract == resumed.contract
    json.dumps(full.contract, allow_nan=False)
    # Iterator creation must not disturb the restored training-noise RNG.
    before = torch.get_rng_state().clone()
    next(iter(resumed.train_loader))
    assert torch.equal(torch.get_rng_state(), before)


def test_validation_ranks_partition_all_windows_without_dropping_tail(tmp_path):
    _make_root(tmp_path)
    config = _config(tmp_path, batch_size=2)
    observed = []
    for rank in range(2):
        bundle = prepare_spd_data(config, DataParallelScope(rank, 2, "replicated", rank == 0), 0)
        for batch in bundle.val_loaders["tianji"]:
            states = _raw(batch["state"], bundle.val_dataset, "state")
            observed.extend(states[:, 0, 0].round().astype(int).tolist())
    assert len(observed) == 3
    assert len(set(observed)) == 3
    assert sorted(observed) == list(range(min(observed), min(observed) + 3))


def test_contract_detects_source_identity_change(tmp_path):
    paths = _make_root(tmp_path)
    config = _config(tmp_path)
    scope = DataParallelScope(0, 1, "replicated", True)
    before = prepare_spd_data(config, scope, 0)
    with h5py.File(paths[0], "r+") as handle:
        handle.attrs["extra_recording_metadata"] = "changed source"
    after = prepare_spd_data(config, scope, 0, before.norm_stats)
    assert before.contract != after.contract


@pytest.mark.parametrize("quality", [True, 50.0, -1, 101])
def test_invalid_jpeg_quality_rejected(tmp_path, quality):
    _make_root(tmp_path)
    path = tmp_path / "dataset_config.json"
    config = json.loads(path.read_text())
    config["jpeg_quality"] = quality
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="jpeg_quality"):
        load_tianji_config(tmp_path)


def test_missing_declared_stream_is_not_masked(tmp_path):
    paths = _make_root(tmp_path, cameras=CAMERA_NAMES)
    with h5py.File(paths[0], "r+") as handle:
        del handle["images/right_wrist"]
    with pytest.raises(ValueError, match="camera groups"):
        TianjiSequenceDataset(tmp_path)


def test_corrupt_declared_jpeg_is_not_masked(tmp_path):
    _make_root(tmp_path)
    dataset = TianjiSequenceDataset(tmp_path)
    with h5py.File(dataset.episodes[0], "r+") as handle:
        handle["images/top/jpeg"][0] = np.array([1, 2, 3], dtype=np.uint8)
    with pytest.raises(ValueError, match="invalid JPEG"):
        dataset[0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_resume_scales_rejected(bad):
    stats = {kind: {"mean": [0.0] * 54, "std": [1.0] * 54} for kind in ("state", "actions")}
    stats["actions"]["std"][0] = bad
    with pytest.raises(ValueError):
        validate_spd_norm_stats(stats)


def test_image_age_limit_prevents_windows_across_stale_views(tmp_path):
    _make_root(tmp_path)
    # Images are recorded at 1 Hz. A 100ms ceiling leaves no contiguous window.
    with pytest.raises(ValueError, match="no complete 258-row"):
        TianjiSequenceDataset(tmp_path, image_max_age_ms=100)


def test_failed_finalized_recording_rejected(tmp_path):
    paths = _make_root(tmp_path)
    with h5py.File(paths[0], "r+") as handle:
        handle.attrs["success"] = False
    with pytest.raises(ValueError, match="success=true"):
        TianjiSequenceDataset(tmp_path)


def test_nonfinite_measured_feedback_rejected(tmp_path):
    paths = _make_root(tmp_path)
    for path in paths:
        with h5py.File(path, "r+") as handle:
            handle["observations/hands/qpos"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite actual feedback"):
        TianjiSequenceDataset(tmp_path)
