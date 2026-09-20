"""Causal SPD windows from Tianji feedback/JPEG recordings, never commands.

Only recorded cameras are decoded. The fixed three-slot validity mask distinguishes
an absent model view from a broken declared recording (which remains an error).
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from abc_minimal.config import TrainConfig
from abc_minimal.dataloader import DataParallelScope, GlobalStepSampler, MixtureDataset
from abc_minimal.preprocess import normalize, resize_pad_normalize

CAMERA_NAMES = ("top", "left_wrist", "right_wrist")
HISTORY_STEPS = 256
IMAGE_STEPS = 32
WINDOW_SPAN = 258
ROBOT_DOF = 54
_RATE_HZ = 30
_SECOND_NS = 1_000_000_000
_POLICY_COLUMNS = list(range(7)) + list(range(14, 34)) + list(range(7, 14)) + list(range(34, 54))
# A window starts with the previous measured row. Its 32 anchors are 1,9,...,249;
# each supervises future offsets 1..8, so the furthest label is row 257.
_ANCHORS = 1 + np.arange(IMAGE_STEPS, dtype=np.int64) * 8
_FUTURE = _ANCHORS[:, None] + np.arange(1, 9, dtype=np.int64)[None, :]


def load_tianji_config(root: str | Path) -> dict[str, Any]:
    """Validate schema v1 and retain the source's declared camera contract."""
    path = Path(root) / "dataset_config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read Tianji dataset configuration {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{path}: configuration must be a JSON object")
    required_values = {
        "schema_version": 1,
        "joint_unit": "rad",
        "policy_rate_hz": _RATE_HZ,
        "image_encoding": "jpeg",
        "decoded_color_order": "RGB",
        "image_width": 1280,
        "image_height": 720,
    }
    for key, expected in required_values.items():
        value = config.get(key)
        if isinstance(value, bool) or value != expected:
            raise ValueError(f"{path}: {key} must be {expected!r}, got {value!r}")
    quality = config.get("jpeg_quality")
    if isinstance(quality, bool) or not isinstance(quality, int) or not 0 <= quality <= 100:
        raise ValueError(f"{path}: jpeg_quality must be an integer in [0, 100]")
    if not isinstance(config.get("robot_config"), str) or not config["robot_config"].strip():
        raise ValueError(f"{path}: robot_config must be a nonempty string")
    names = config.get("joint_names")
    if (
        not isinstance(names, list)
        or len(names) != ROBOT_DOF
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != ROBOT_DOF
    ):
        raise ValueError(f"{path}: joint_names must contain 54 unique nonempty names in source order")
    cameras = config.get("camera_names")
    if (
        not isinstance(cameras, list)
        or not cameras
        or any(not isinstance(name, str) or name not in CAMERA_NAMES for name in cameras)
        or len(set(cameras)) != len(cameras)
    ):
        raise ValueError(f"{path}: camera_names must be a nonempty unique subset of {CAMERA_NAMES}")
    config["policy_joint_indices"] = _POLICY_COLUMNS.copy()
    config["policy_joint_names"] = [names[index] for index in _POLICY_COLUMNS]
    config["action_target"] = "future_actual_qpos"
    config["camera_contract"] = {
        "model_camera_keys": list(CAMERA_NAMES),
        "observed_camera_keys": list(cameras),
        "unavailable_camera_keys": [camera for camera in CAMERA_NAMES if camera not in cameras],
        "camera_validity": {
            "dtype": "bool",
            "sample_shape": [IMAGE_STEPS, len(CAMERA_NAMES)],
            "camera_order": list(CAMERA_NAMES),
            "recorded_camera_mask": [camera in cameras for camera in CAMERA_NAMES],
            "true_means": "recorded image available at this observation step",
            "false_means": "unavailable view; no image decoding or camera supervision",
            "images": "recorded cameras only; missing declared streams and invalid JPEGs are errors",
        },
    }
    return config


def _episode_header(handle: h5py.File, path: Path, config: Mapping[str, Any]) -> None:
    version = handle.attrs.get("schema_version")
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, Integral) or version != 1:
        raise ValueError(f"{path}: schema_version attribute must be integer 1")
    success = handle.attrs.get("success")
    if not isinstance(success, (bool, np.bool_)) or not success:
        raise ValueError(f"{path}: finalized episode must have success=true (operator confirmed)")
    robot = handle.attrs.get("robot_config")
    if isinstance(robot, bytes):
        robot = robot.decode("utf-8")
    if robot != config["robot_config"]:
        raise ValueError(f"{path}: robot_config {robot!r} differs from dataset configuration")
    task = handle.attrs.get("task")
    if not isinstance(task, (str, bytes)) or not task:
        raise ValueError(f"{path}: task attribute must be a nonempty string")
    if "images" not in handle or not isinstance(handle["images"], h5py.Group):
        raise ValueError(f"{path}: missing images group")
    cameras = set(handle["images"])
    if cameras != set(config["camera_names"]):
        raise ValueError(f"{path}: camera groups {sorted(cameras)} differ from configured {config['camera_names']}")


def _split_episodes(root: Path, config: Mapping[str, Any], seed: int) -> dict[str, list[Path]]:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Tianji split seed must be a nonnegative integer")
    paths = sorted(path for path in root.rglob("*.h5") if not path.name.endswith(".partial.h5"))
    if len(paths) < 2:
        raise ValueError(f"{root}: need at least two successful finalized .h5 episodes for train/val splitting")
    for path in paths:
        try:
            with h5py.File(path, "r") as handle:
                _episode_header(handle, path, config)
        except OSError as exc:
            raise ValueError(f"cannot open Tianji episode {path}: {exc}") from exc
    order = np.random.default_rng(int(seed)).permutation(len(paths))
    val_count = min(len(paths) - 1, max(1, math.ceil(len(paths) * 0.2)))
    return {
        "train": [paths[int(index)] for index in order[:-val_count]],
        "val": [paths[int(index)] for index in order[-val_count:]],
    }


def _timestamps(handle: h5py.File, name: str, path: Path) -> np.ndarray:
    if name not in handle or not isinstance(handle[name], h5py.Dataset):
        raise ValueError(f"{path}: missing dataset {name}")
    dataset = handle[name]
    if dataset.ndim != 1 or dataset.dtype.kind != "i" or dataset.dtype.itemsize != 8:
        raise ValueError(f"{path}: {name} must be a one-dimensional int64 vector")
    values = dataset[:]
    if not len(values) or np.any(values < 0) or np.any(values[1:] < values[:-1]):
        raise ValueError(f"{path}: {name} must be nonempty, nonnegative, and monotonically nondecreasing")
    return values


def _age_ns(value: float, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if value > np.iinfo(np.int64).max / 1_000_000:
        raise ValueError(f"{name} exceeds the timestamp range")
    return int(value * 1_000_000)


def _identity(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.relative_to(root)), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@dataclass
class _AlignedEpisode:
    path: Path
    qpos: np.ndarray
    image_indices: dict[str, np.ndarray]
    timestamp_ns: np.ndarray
    segments: list[tuple[int, int]]
    raw_counts: dict[str, int]


def _load_episode(
    path: Path, config: Mapping[str, Any], joint_max_age_ns: int, image_max_age_ns: int
) -> _AlignedEpisode:
    timestamps: dict[str, np.ndarray] = {}
    positions: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        _episode_header(handle, path, config)
        for stream, width in (("arms", 14), ("hands", 40)):
            prefix = f"observations/{stream}"
            timestamps[stream] = _timestamps(handle, f"{prefix}/timestamp_ns", path)
            name = f"{prefix}/qpos"
            if name not in handle or not isinstance(handle[name], h5py.Dataset):
                raise ValueError(f"{path}: missing dataset {name}")
            qpos = handle[name]
            if qpos.shape != (len(timestamps[stream]), width) or qpos.dtype.kind != "f" or qpos.dtype.itemsize != 4:
                raise ValueError(f"{path}: {name} must be float32 [{len(timestamps[stream])},{width}]")
            positions[stream] = qpos[:]
            if not np.all(np.isfinite(positions[stream])):
                raise ValueError(f"{path}: {name} contains non-finite actual feedback")
        for camera in config["camera_names"]:
            prefix = f"images/{camera}"
            timestamps[camera] = _timestamps(handle, f"{prefix}/timestamp_ns", path)
            name = f"{prefix}/jpeg"
            if name not in handle or not isinstance(handle[name], h5py.Dataset):
                raise ValueError(f"{path}: missing dataset {name}")
            jpeg = handle[name]
            if jpeg.shape != (len(timestamps[camera]),) or h5py.check_dtype(vlen=jpeg.dtype) != np.dtype("uint8"):
                raise ValueError(f"{path}: {name} must be vlen uint8 with one JPEG per timestamp")

    # Integer arithmetic avoids drift. Previous-available selection is causal and
    # searchsorted(right) makes the last observation win at duplicate timestamps.
    first = max(int(values[0]) for values in timestamps.values())
    last = min(int(values[-1]) for values in timestamps.values())
    first_step = (first * _RATE_HZ + _SECOND_NS - 1) // _SECOND_NS
    last_step = (last * _RATE_HZ + _RATE_HZ - 1) // _SECOND_NS
    if last_step < first_step:
        raise ValueError(f"{path}: recorded streams have no overlapping 30 Hz grid")
    steps = np.arange(first_step, last_step + 1, dtype=np.int64)
    grid = (steps // _RATE_HZ) * _SECOND_NS + (steps % _RATE_HZ) * _SECOND_NS // _RATE_HZ
    valid = np.ones(len(grid), dtype=np.bool_)
    indices = {}
    for stream, values in timestamps.items():
        index = np.searchsorted(values, grid, side="right") - 1
        max_age = joint_max_age_ns if stream in positions else image_max_age_ns
        valid &= (index >= 0) & (grid - values[np.maximum(index, 0)] <= max_age)
        indices[stream] = index
    valid_rows = np.flatnonzero(valid)
    if not len(valid_rows):
        raise ValueError(f"{path}: no causally aligned rows satisfy joint/image age limits")
    cuts = np.flatnonzero(np.diff(valid_rows) != 1) + 1
    boundaries = np.concatenate(([0], cuts, [len(valid_rows)]))
    segments = [(int(start), int(end)) for start, end in zip(boundaries[:-1], boundaries[1:])]
    arms = positions["arms"][indices["arms"][valid_rows]]
    hands = positions["hands"][indices["hands"][valid_rows]]
    qpos = np.concatenate((arms[:, :7], hands[:, :20], arms[:, 7:], hands[:, 20:]), axis=1)
    return _AlignedEpisode(
        path=path,
        qpos=qpos,
        image_indices={camera: indices[camera][valid_rows] for camera in config["camera_names"]},
        timestamp_ns=grid[valid_rows],
        segments=segments,
        raw_counts={stream: len(values) for stream, values in timestamps.items()},
    )


def validate_spd_norm_stats(value: Mapping[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Require finite ABC 54-vectors and positive, float32-representable scales."""
    if not isinstance(value, Mapping) or set(value) != {"state", "actions"}:
        raise ValueError("SPD norm_stats must contain state and actions")
    result = {}
    for key in ("state", "actions"):
        stats = value[key]
        if not isinstance(stats, Mapping) or set(stats) != {"mean", "std"}:
            raise ValueError(f"SPD norm_stats.{key} must contain mean and std")
        result[key] = {}
        for name in ("mean", "std"):
            try:
                raw = np.asarray(stats[name])
                if raw.dtype.kind not in "fiu":
                    raise ValueError("expected real numeric values")
                vector = raw.astype(np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"SPD norm_stats.{key}.{name} must be numeric") from exc
            if vector.shape != (ROBOT_DOF,) or not np.isfinite(vector).all():
                raise ValueError(f"SPD norm_stats.{key}.{name} must be a finite 54-vector")
            if np.any(np.abs(vector) > np.finfo(np.float32).max):
                raise ValueError(f"SPD norm_stats.{key}.{name} exceeds float32 range")
            if name == "std" and np.any(vector < np.nextafter(np.float32(0), np.float32(1))):
                raise ValueError(f"SPD norm_stats.{key}.std must be positive in float32")
            result[key][name] = vector.tolist()
    return result


class TianjiSequenceDataset(Dataset):
    """Whole-episode splits and gap-safe 258-row windows with lazy strict JPEGs.

    No geometric augmentation is applied: recordings have neither segmentation
    masks nor a calibrated camera/joint symmetry specification. Without supplied
    statistics, samples use ABC identity statistics; preparation fits training
    usable segments before constructing either loader.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        seed: int = 123,
        norm_stats: Mapping[str, Any] | None = None,
        joint_max_age_ms: float = 150.0,
        image_max_age_ms: float = 2000.0,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("Tianji split must be 'train' or 'val'")
        joint_age = _age_ns(joint_max_age_ms, "joint_max_age_ms")
        image_age = _age_ns(image_max_age_ms, "image_max_age_ms")
        self.root = Path(root).expanduser().resolve()
        self.config = load_tianji_config(self.root)
        self.camera_names = tuple(self.config["camera_names"])
        self.set_norm_stats(norm_stats if norm_stats is not None else {
            key: {"mean": [0.0] * ROBOT_DOF, "std": [1.0] * ROBOT_DOF}
            for key in ("state", "actions")
        })
        splits = _split_episodes(self.root, self.config, seed)
        self.episodes = splits[split]
        self._aligned = [_load_episode(path, self.config, joint_age, image_age) for path in self.episodes]
        self.samples: list[tuple[int, int]] = []
        episode_metadata = []
        for episode_index, episode in enumerate(self._aligned):
            count_before = len(self.samples)
            for start, end in episode.segments:
                self.samples.extend((episode_index, row) for row in range(start, end - WINDOW_SPAN + 1))
            episode_metadata.append({
                **_identity(episode.path, self.root),
                "raw_counts": episode.raw_counts,
                "aligned_rows": len(episode.qpos),
                "segments": [list(segment) for segment in episode.segments],
                "first_timestamp_ns": int(episode.timestamp_ns[0]),
                "last_timestamp_ns": int(episode.timestamp_ns[-1]),
                "windows": len(self.samples) - count_before,
            })
        if not self.samples:
            raise ValueError(
                f"{self.root}: {split} episodes contain no complete {WINDOW_SPAN}-row SPD windows; "
                "need at least 8.6 seconds of overlapping streams within configured age limits"
            )
        self.metadata = {
            "format": "tianji",
            "config": self.config,
            "camera_contract": self.config["camera_contract"],
            "split": split,
            "split_seed": int(seed),
            "train_episodes": [str(path.relative_to(self.root)) for path in splits["train"]],
            "val_episodes": [str(path.relative_to(self.root)) for path in splits["val"]],
            "alignment": {
                "rate_hz": _RATE_HZ,
                "grid_origin_ns": 0,
                "selection": "searchsorted_right_minus_one",
                "joint_max_age_ns": joint_age,
                "image_max_age_ns": image_age,
            },
            "episodes": episode_metadata,
            "windows": len(self.samples),
        }

    def set_norm_stats(self, norm_stats: Mapping[str, Any]) -> None:
        self.norm_stats = validate_spd_norm_stats(norm_stats)
        self._norm = {
            key: {name: np.asarray(vector, dtype=np.float32) for name, vector in stats.items()}
            for key, stats in self.norm_stats.items()
        }

    def __len__(self) -> int:
        return len(self.samples)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return self[int(rng.integers(len(self)))]

    def __getitem__(self, item: int) -> dict[str, Any]:
        episode_index, start = self.samples[item]
        episode = self._aligned[episode_index]
        images = {}
        with h5py.File(episode.path, "r") as handle:
            for camera in self.camera_names:
                jpeg = handle[f"images/{camera}/jpeg"]
                # Slow/stale-but-valid source cameras can repeat across anchors;
                # decode and preprocess each selected source JPEG only once.
                decoded = {}
                tensors = []
                for frame_index in episode.image_indices[camera][start + _ANCHORS]:
                    index = int(frame_index)
                    if index not in decoded:
                        try:
                            with Image.open(BytesIO(jpeg[index].tobytes())) as image:
                                if image.format != "JPEG" or image.mode != "RGB" or image.size != (1280, 720):
                                    raise ValueError("expected configured 1280x720 RGB JPEG encoding")
                                frame = np.array(image, dtype=np.uint8)
                            decoded[index] = resize_pad_normalize(
                                torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0,
                                preset="imagenet",
                            )
                        except (OSError, ValueError) as exc:
                            raise ValueError(f"{episode.path}: invalid JPEG images/{camera}/jpeg[{index}]: {exc}") from exc
                    tensors.append(decoded[index])
                images[camera] = torch.stack(tensors)
        qpos = episode.qpos
        return {
            "state": torch.from_numpy(normalize(qpos[start + 1:start + 257], self._norm["state"])),
            "previous_actions": torch.from_numpy(normalize(qpos[start:start + 256], self._norm["actions"])),
            "actions": torch.from_numpy(normalize(qpos[start + _FUTURE], self._norm["actions"])),
            "images": images,
            "camera_validity": torch.tensor(
                self.config["camera_contract"]["camera_validity"]["recorded_camera_mask"],
                dtype=torch.bool,
            ).expand(IMAGE_STEPS, -1),
        }


def _training_normalization(dataset: TianjiSequenceDataset) -> dict[str, dict[str, list[float]]]:
    """Population moments over training usable segments, counting each row once."""
    if dataset.metadata["split"] != "train":
        raise ValueError("normalization can only be fit to the training split")
    count = 0
    mean = np.zeros(ROBOT_DOF, dtype=np.float64)
    squared_deviation = np.zeros(ROBOT_DOF, dtype=np.float64)
    for episode in dataset._aligned:
        for start, end in episode.segments:
            if end - start < WINDOW_SPAN:
                continue
            values = episode.qpos[start:end].astype(np.float64)
            batch_count = len(values)
            batch_mean = values.mean(axis=0)
            batch_deviation = np.square(values - batch_mean).sum(axis=0)
            delta = batch_mean - mean
            total = count + batch_count
            squared_deviation += batch_deviation + np.square(delta) * (count * batch_count / total)
            mean += delta * (batch_count / total)
            count = total
    std = np.sqrt(np.maximum(squared_deviation / count, 1e-12))
    return validate_spd_norm_stats({
        key: {"mean": mean.tolist(), "std": std.tolist()} for key in ("state", "actions")
    })


def compute_tianji_normalization(
    root: str | Path,
    *,
    seed: int = 123,
    joint_max_age_ms: float = 150.0,
    image_max_age_ms: float = 2000.0,
) -> dict[str, dict[str, list[float]]]:
    return _training_normalization(TianjiSequenceDataset(
        root, split="train", seed=seed,
        joint_max_age_ms=joint_max_age_ms, image_max_age_ms=image_max_age_ms,
    ))


@dataclass
class SPDDataBundle:
    train_loader: DataLoader
    val_loaders: dict[str, DataLoader]
    norm_stats: dict[str, dict[str, list[float]]]
    contract: dict[str, Any]
    train_dataset: TianjiSequenceDataset
    val_dataset: TianjiSequenceDataset


def prepare_spd_data(
    config: TrainConfig,
    data_scope: DataParallelScope,
    resume_step: int,
    norm_stats=None,
) -> SPDDataBundle:
    """Prepare a single immutable recording root and step-keyed ABC loaders."""
    if not config.spd_data.root:
        raise ValueError("SPD requires spd_data.root")
    if not 0 <= data_scope.rank < data_scope.world:
        raise ValueError("invalid SPD data-parallel rank/world")
    if not 0 <= resume_step <= config.train_steps:
        raise ValueError("resume_step must be within the configured training steps")
    if tuple(config.spd_model.camera_keys) != CAMERA_NAMES:
        raise ValueError("SPD model must retain the three canonical camera slots")
    kwargs = {
        "seed": config.seed,
        "joint_max_age_ms": config.spd_data.joint_max_age_ms,
        "image_max_age_ms": config.spd_data.image_max_age_ms,
    }
    train = TianjiSequenceDataset(config.spd_data.root, split="train", **kwargs)
    stats = _training_normalization(train) if norm_stats is None else validate_spd_norm_stats(norm_stats)
    train.set_norm_stats(stats)
    val = TianjiSequenceDataset(config.spd_data.root, split="val", norm_stats=stats, **kwargs)
    mixture = MixtureDataset([train], [1.0], len(train), seed=config.seed)
    sampler = GlobalStepSampler(
        start_step=resume_step, stop_step=config.train_steps, batch_size=config.batch_size,
        data_rank=data_scope.rank, data_world=data_scope.world,
    )
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
    }
    # Dedicated generators isolate DataLoader worker seeding from model RNG on
    # resume. Samples themselves depend only on the global sampler key.
    train_loader = DataLoader(
        mixture, sampler=sampler, drop_last=True,
        generator=torch.Generator().manual_seed(config.seed), **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(val, range(data_scope.rank, len(val), data_scope.world)),
        shuffle=False, drop_last=False,
        generator=torch.Generator().manual_seed(config.seed), **loader_kwargs,
    )
    contract = {
        "format": "tianji-spd-v1",
        "root": str(train.root),
        "dataset_config_identity": _identity(train.root / "dataset_config.json", train.root),
        "train": train.metadata,
        "val": val.metadata,
        "labels": {
            "state": "measured_qpos",
            "previous_actions": "previous_measured_qpos_not_commands",
            "actions": "future_measured_qpos_not_commands",
            "policy_joint_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
            "window_rows": WINDOW_SPAN,
            "history_rows": [1, 256],
            "previous_rows": [0, 255],
            "chunk_anchor_rows": _ANCHORS.tolist(),
            "future_offsets": list(range(1, 9)),
        },
        "normalization": {
            "population": "training_aligned_segments_with_at_least_258_rows_each_row_once",
            "std_floor": 1e-6,
            "formula": "abc_minimal.preprocess.normalize: (x-mean)/(std+1e-6)",
            "norm_stats": stats,
        },
        "images": {"preprocess": "abc_minimal.preprocess.resize_pad_normalize", "preset": "imagenet", "size": [224, 224]},
        "sampling": {
            "seed": config.seed,
            "batch_size_per_rank": config.batch_size,
            "data_world": data_scope.world,
            "placement": data_scope.placement,
            "train": "ABC GlobalStepSampler and MixtureDataset; uniform usable windows",
            "val": "full split in index order strided by rank; no padding or drop_last",
        },
    }
    # Snapshot metadata and vectors into independent JSON-safe checkpoint values.
    contract = json.loads(json.dumps(contract, allow_nan=False))
    return SPDDataBundle(train_loader, {"tianji": val_loader}, stats, contract, train, val)


__all__ = [
    "TianjiSequenceDataset", "SPDDataBundle", "load_tianji_config",
    "compute_tianji_normalization", "validate_spd_norm_stats", "prepare_spd_data",
]
