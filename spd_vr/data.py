"""Atomic HDF5 recording and 30 Hz SPD training-window access.

The raw stream is written at the simulator/control cadence.  Training samples
only store integer references into that stream, so RGB, segmentation, robot
state/action, PICO tracking, objects, contacts, validity, and full MuJoCo state
retain one timestamped source of truth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import uuid
from numbers import Integral

import h5py
import numpy as np

from .contracts import (
    ACTION_CHUNK,
    CAMERA_NAMES,
    HISTORY_STEPS,
    IMAGE_STEPS,
    IMAGE_STRIDE,
    ROBOT_DOF,
)


SCHEMA_VERSION = "spd-vr-hdf5-v1"
RAW_HZ = 60
TRAIN_HZ = 30
NO_CONTACT_LIMIT_NS = 10_000_000_000
IMAGE_HEIGHT = 168
IMAGE_WIDTH = 224
WINDOW_SPAN = HISTORY_STEPS + 2  # prior row through furthest +8 label.
_MAX_SIGNED_TIMESTAMP_NS = 0x7FFFFFFFFFFFFFFF


def _timestamp_array(value: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    """Normalize an integer timestamp vector without allowing wraparound."""
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1-D integer vector")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must use an integer dtype")
    if raw.dtype.kind == "u" and raw.size and np.any(raw > _MAX_SIGNED_TIMESTAMP_NS):
        raise ValueError(f"{name} exceeds the signed 64-bit timestamp contract")
    timestamps = raw.astype(np.int64, copy=False)
    if timestamps.size and np.any(timestamps < 0):
        raise ValueError(f"{name} must be non-negative")
    return timestamps


def _array(value: Any, shape: tuple[int, ...], name: str, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if result.dtype.kind == "f" and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _bool_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Normalize boolean flags without truthiness-coercing malformed data."""
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {raw.shape}")
    if raw.dtype.kind == "b":
        return raw.astype(np.bool_, copy=False)
    if raw.dtype.kind in {"i", "u"} and np.all(np.isin(raw, (0, 1))):
        return raw.astype(np.bool_, copy=False)
    raise ValueError(f"{name} must contain boolean or 0/1 integer values")


def _bounded_integer(value: object, name: str, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result > upper:
        raise ValueError(f"{name} must be a non-negative bounded integer")
    return result


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest_bytes(value: Any) -> bytes:
    array = np.asarray(value)
    if array.dtype.kind in {"O", "U", "S"}:
        scalar = array.item()
        return scalar if isinstance(scalar, bytes) else str(scalar).encode("utf-8")
    return np.ascontiguousarray(array).tobytes()


@dataclass(frozen=True, slots=True)
class CameraFrame:
    rgb: np.ndarray
    segmentation: np.ndarray

    def __post_init__(self) -> None:
        rgb = _array(
            self.rgb, (IMAGE_HEIGHT, IMAGE_WIDTH, 3), "rgb", np.uint8
        )
        segmentation = _array(
            self.segmentation,
            (IMAGE_HEIGHT, IMAGE_WIDTH, 2),
            "segmentation",
            np.int32,
        )
        object.__setattr__(self, "rgb", rgb.copy())
        object.__setattr__(self, "segmentation", segmentation.copy())


@dataclass(frozen=True, slots=True)
class RawFrame:
    timestamp_ns: int
    qpos: np.ndarray
    qvel: np.ndarray
    qpos_target: np.ndarray
    mujoco_full_state: np.ndarray
    cameras: Mapping[str, CameraFrame]
    pico_hands: np.ndarray
    pico_timestamp_ns: int
    pico_sequence_id: int
    tracking_epoch: int
    pico_scale: np.ndarray
    validity: np.ndarray
    pico_bridge_monotonic_ns: int | None = None
    objects: Any = None
    contacts: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp_ns",
            _bounded_integer(self.timestamp_ns, "timestamp_ns", _MAX_SIGNED_TIMESTAMP_NS),
        )
        for name in ("qpos", "qvel", "qpos_target"):
            object.__setattr__(
                self, name, _array(getattr(self, name), (ROBOT_DOF,), name, np.float32).copy()
            )
        state = np.asarray(self.mujoco_full_state, dtype=np.float64)
        if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError("mujoco_full_state must be a non-empty finite vector")
        object.__setattr__(self, "mujoco_full_state", state.copy())
        if set(self.cameras) != set(CAMERA_NAMES):
            raise ValueError(f"cameras must contain exactly {CAMERA_NAMES}")
        cameras = dict(self.cameras)
        if any(not isinstance(cameras[name], CameraFrame) for name in CAMERA_NAMES):
            raise ValueError("cameras must contain CameraFrame values")
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(
            self,
            "pico_hands",
            _array(self.pico_hands, (2, 26, 7), "pico_hands", np.float32).copy(),
        )
        pico_timestamp = _bounded_integer(
            self.pico_timestamp_ns, "pico_timestamp_ns", _MAX_SIGNED_TIMESTAMP_NS
        )
        bridge_timestamp = (
            pico_timestamp
            if self.pico_bridge_monotonic_ns is None
            else _bounded_integer(
                self.pico_bridge_monotonic_ns,
                "pico_bridge_monotonic_ns",
                _MAX_SIGNED_TIMESTAMP_NS,
            )
        )
        object.__setattr__(self, "pico_timestamp_ns", pico_timestamp)
        object.__setattr__(self, "pico_bridge_monotonic_ns", bridge_timestamp)
        object.__setattr__(
            self,
            "pico_sequence_id",
            _bounded_integer(self.pico_sequence_id, "pico_sequence_id", 0xFFFFFFFFFFFFFFFF),
        )
        object.__setattr__(
            self,
            "tracking_epoch",
            _bounded_integer(self.tracking_epoch, "tracking_epoch", 0xFFFFFFFFFFFFFFFF),
        )
        scale = _array(self.pico_scale, (2,), "pico_scale", np.float32)
        if np.any(scale <= 0):
            raise ValueError("pico_scale must be positive")
        object.__setattr__(self, "pico_scale", scale.copy())
        object.__setattr__(self, "validity", _bool_array(self.validity, (2,), "validity").copy())


def build_training_index(
    timestamps_ns: Sequence[int] | np.ndarray,
    *,
    target_hz: int = TRAIN_HZ,
    tolerance_ns: int = 10_000_000,
) -> np.ndarray:
    """Map an irregular monotonic raw stream onto an exact-rate timeline."""
    timestamps = _timestamp_array(timestamps_ns, "timestamps")
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("timestamps must be a non-empty vector")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if (
        isinstance(target_hz, bool)
        or not isinstance(target_hz, Integral)
        or int(target_hz) <= 0
    ):
        raise ValueError("target_hz must be a positive integer")
    if (
        isinstance(tolerance_ns, bool)
        or not isinstance(tolerance_ns, Integral)
        or int(tolerance_ns) < 0
    ):
        raise ValueError("tolerance_ns must be a non-negative integer")
    target_hz = int(target_hz)
    tolerance_ns = int(tolerance_ns)
    period = 1_000_000_000 / target_hz
    count = int(np.floor((timestamps[-1] - timestamps[0]) / period)) + 1
    targets = timestamps[0] + np.rint(np.arange(count) * period).astype(np.int64)
    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 0, timestamps.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(timestamps[left] - targets) <= np.abs(timestamps[right] - targets)
    index = np.where(choose_left, left, right).astype(np.int64)
    error = np.abs(timestamps[index] - targets)
    keep = np.concatenate(([True], np.diff(index) > 0)) & (error <= tolerance_ns)
    return index[keep]


def _training_grid_steps(
    timestamps_ns: np.ndarray,
    index_30hz: np.ndarray,
    *,
    target_hz: int = TRAIN_HZ,
) -> np.ndarray:
    """Recover nominal grid ordinals so dropped 30 Hz rows remain visible."""
    selected = np.asarray(timestamps_ns, dtype=np.int64)[index_30hz]
    period = 1_000_000_000 / target_hz
    return np.rint((selected - int(timestamps_ns[0])) / period).astype(np.int64)


def _has_hand_object_contact(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "hand_object" in value:
            return bool(value["hand_object"])
        return any(_has_hand_object_contact(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_hand_object_contact(item) for item in value)
    return False


def filter_contact_mask(
    timestamps_ns: Sequence[int] | np.ndarray,
    contact_mask: Sequence[bool] | np.ndarray,
    *,
    threshold_ns: int = NO_CONTACT_LIMIT_NS,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Mark raw rows in continuous no-contact spans longer than the limit.

    The raw stream is retained verbatim.  The returned mask is consumed when
    constructing the 30 Hz training index, so a discarded span becomes an
    explicit segment boundary instead of silently being bridged by a sample.
    Duration is measured from the first to the last row in a run, matching the
    paper's continuous-time interpretation and remaining valid for irregular
    raw timestamps.
    """
    timestamps = _timestamp_array(timestamps_ns, "contact timestamps")
    contacts = _bool_array(contact_mask, timestamps.shape, "contact mask")
    if timestamps.ndim != 1 or contacts.shape != timestamps.shape:
        raise ValueError("timestamps and contact mask must have the same 1-D shape")
    if (
        isinstance(threshold_ns, bool)
        or not isinstance(threshold_ns, Integral)
        or int(threshold_ns) < 0
    ):
        raise ValueError("threshold_ns must be a non-negative integer")
    threshold_ns = int(threshold_ns)
    if timestamps.size and np.any(np.diff(timestamps) <= 0):
        raise ValueError("contact timestamps must be strictly increasing")
    keep = np.ones(timestamps.shape, dtype=np.bool_)
    audit: list[dict[str, int]] = []
    start: int | None = None

    def close(end: int) -> None:
        nonlocal start
        if start is None:
            return
        duration = int(timestamps[end - 1] - timestamps[start])
        if duration > threshold_ns:
            keep[start:end] = False
            audit.append(
                {
                    "start_ns": int(timestamps[start]),
                    "end_ns": int(timestamps[end - 1]),
                    "duration_ns": duration,
                    "samples": end - start,
                }
            )
        start = None

    for index, active in enumerate(contacts):
        if not bool(active) and start is None:
            start = index
        elif bool(active):
            close(index)
    close(int(timestamps.size))
    return keep, audit


def build_contact_segments(
    grid_step: Sequence[int] | np.ndarray,
    eligible: Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """Return ``[start, end)`` training-index segments safe for windowing.

    Segments split on either an ineligible row (for example a >10 s contact
    gap) or a missing nominal 30 Hz grid step.  Keeping this representation in
    the HDF5 file makes the no-crossing rule inspectable without re-parsing
    contact JSON.
    """
    raw_grid = np.asarray(grid_step)
    if raw_grid.ndim != 1 or raw_grid.dtype.kind not in {"i", "u"}:
        raise ValueError("grid_step must be a 1-D integer vector")
    if raw_grid.dtype.kind == "u" and raw_grid.size and np.any(
        raw_grid > np.iinfo(np.int64).max
    ):
        raise ValueError("grid_step exceeds signed 64-bit bounds")
    grid = raw_grid.astype(np.int64, copy=False)
    raw_mask = np.asarray(eligible)
    if raw_mask.ndim != 1 or raw_mask.shape != grid.shape:
        raise ValueError("grid_step and eligible must have the same 1-D shape")
    if raw_mask.dtype.kind == "b":
        mask = raw_mask.astype(np.bool_, copy=False)
    elif raw_mask.dtype.kind in {"i", "u"} and np.all(np.isin(raw_mask, (0, 1))):
        mask = raw_mask.astype(np.bool_, copy=False)
    else:
        raise ValueError("eligible must contain boolean or 0/1 integer values")
    if grid.size and np.any(np.diff(grid) <= 0):
        raise ValueError("grid_step must be strictly increasing")
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        contiguous = index > 0 and int(grid[index]) == int(grid[index - 1]) + 1
        if bool(value):
            if start is None:
                start = index
            elif not contiguous:
                segments.append((start, index))
                start = index
        elif start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, int(mask.size)))
    return np.asarray(segments, dtype=np.int64).reshape(-1, 2)


class EpisodeWriter:
    """Stream one episode to a staging file, validate it, then publish atomically."""

    def __init__(
        self,
        output_path: str | Path,
        manifest: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> None:
        self.output_path = Path(output_path)
        if self.output_path.suffix not in {".h5", ".hdf5"}:
            raise ValueError("episode output must end in .h5 or .hdf5")
        if self.output_path.exists():
            if not overwrite or not self.output_path.is_file():
                raise FileExistsError(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.staging_path = self.output_path.with_name(
            f".{self.output_path.name}.{uuid.uuid4().hex}.staging"
        )
        self._handle = h5py.File(self.staging_path, "w")
        self._handle.attrs.update(
            schema_version=SCHEMA_VERSION,
            raw_hz=RAW_HZ,
            training_hz=TRAIN_HZ,
            complete=False,
        )
        self._manifest = json.loads(_json(manifest))
        self._datasets: dict[str, h5py.Dataset] = {}
        self._digests: dict[str, Any] = {}
        self._count = 0
        self._last_timestamp = -1
        self._full_state_dim: int | None = None
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._count

    def _create(self, name: str, value: np.ndarray) -> h5py.Dataset:
        shape = value.shape
        kwargs: dict[str, Any] = {
            "shape": (0, *shape),
            "maxshape": (None, *shape),
            "dtype": value.dtype,
            "chunks": (1, *shape),
        }
        if value.size >= 32:
            kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
        dataset = self._handle.create_dataset(name, **kwargs)
        self._datasets[name] = dataset
        self._digests[name] = hashlib.sha256()
        return dataset

    def _append(self, name: str, value: Any, dtype: Any) -> None:
        array = np.asarray(value, dtype=dtype)
        dataset = self._datasets.get(name)
        if dataset is None:
            dataset = self._create(name, array)
        if dataset.shape[1:] != array.shape:
            raise ValueError(
                f"{name} changed shape from {dataset.shape[1:]} to {array.shape}"
            )
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = array
        self._digests[name].update(_digest_bytes(array))

    def append(self, frame: RawFrame) -> None:
        if self._closed:
            raise RuntimeError("episode writer is closed")
        if frame.timestamp_ns <= self._last_timestamp:
            raise ValueError("raw timestamps must be strictly increasing")
        if self._full_state_dim is None:
            self._full_state_dim = frame.mujoco_full_state.size
        elif frame.mujoco_full_state.size != self._full_state_dim:
            raise ValueError("MuJoCo full-state width changed within an episode")

        self._append("raw/timestamp_ns", frame.timestamp_ns, np.uint64)
        self._append("raw/observation/qpos", frame.qpos, np.float32)
        self._append("raw/observation/qvel", frame.qvel, np.float32)
        # Explicit action stream: the value actually realized by MuJoCo at
        # this tick.  Keep the command target beside it for audit only.
        self._append("raw/action/qpos", frame.qpos, np.float32)
        self._append("raw/action/qpos_target", frame.qpos_target, np.float32)
        self._append("raw/mujoco/full_state", frame.mujoco_full_state, np.float64)
        self._append("raw/pico/hands", frame.pico_hands, np.float32)
        self._append("raw/pico/source_timestamp_ns", frame.pico_timestamp_ns, np.uint64)
        self._append(
            "raw/pico/bridge_monotonic_ns",
            frame.pico_bridge_monotonic_ns,
            np.uint64,
        )
        self._append("raw/pico/sequence_id", frame.pico_sequence_id, np.uint64)
        self._append("raw/pico/tracking_epoch", frame.tracking_epoch, np.uint64)
        self._append("raw/pico/scale", frame.pico_scale, np.float32)
        self._append("raw/validity/sides", frame.validity, np.bool_)
        for camera in CAMERA_NAMES:
            self._append(f"raw/cameras/{camera}/rgb", frame.cameras[camera].rgb, np.uint8)
            self._append(
                f"raw/cameras/{camera}/segmentation",
                frame.cameras[camera].segmentation,
                np.int32,
            )
        string_dtype = h5py.string_dtype("utf-8")
        self._append("raw/objects/json", _json(frame.objects), string_dtype)
        self._append("raw/contacts/json", _json(frame.contacts), string_dtype)
        self._append(
            "raw/contacts/hand_object",
            _has_hand_object_contact(frame.contacts),
            np.bool_,
        )
        self._last_timestamp = frame.timestamp_ns
        self._count += 1

    def finish(self) -> Path:
        if self._closed:
            raise RuntimeError("episode writer is closed")
        if self._count == 0:
            self.abort()
            raise ValueError("cannot publish an empty episode")
        try:
            timestamps = self._datasets["raw/timestamp_ns"][:]
            train_index = build_training_index(timestamps)
            raw_contact = self._datasets["raw/contacts/hand_object"][:]
            contact_keep, removed_spans = filter_contact_mask(timestamps, raw_contact)
            grid_step = _training_grid_steps(timestamps, train_index)
            contact_eligible = contact_keep[train_index]
            segments = build_contact_segments(grid_step, contact_eligible)
            self._handle.create_dataset(
                "training/index_30hz", data=train_index, dtype=np.int64
            )
            self._handle.create_dataset(
                "training/grid_step", data=grid_step, dtype=np.int64
            )
            self._handle.create_dataset(
                "training/contact_eligible", data=contact_eligible, dtype=np.bool_
            )
            self._handle.create_dataset(
                "training/segments_30hz", data=segments, dtype=np.int64
            )
            dataset_sha256 = {
                name: digest.hexdigest()
                for name, digest in sorted(self._digests.items())
            }
            dataset_sha256["training/index_30hz"] = hashlib.sha256(
                np.ascontiguousarray(train_index).tobytes()
            ).hexdigest()
            dataset_sha256["training/grid_step"] = hashlib.sha256(
                np.ascontiguousarray(grid_step).tobytes()
            ).hexdigest()
            dataset_sha256["training/contact_eligible"] = hashlib.sha256(
                np.ascontiguousarray(contact_eligible).tobytes()
            ).hexdigest()
            dataset_sha256["training/segments_30hz"] = hashlib.sha256(
                np.ascontiguousarray(segments).tobytes()
            ).hexdigest()
            manifest = {
                **self._manifest,
                "schema_version": SCHEMA_VERSION,
                "created_unix_ns": time.time_ns(),
                "raw_frames": self._count,
                "training_frames": int(train_index.size),
                "policy_action_source": "raw/action/qpos (future actual MuJoCo state)",
                "teleop_target_source": "raw/action/qpos_target (audit only)",
                "canonical_cameras": list(CAMERA_NAMES),
                "contact_filter": {
                    "threshold_ns": NO_CONTACT_LIMIT_NS,
                    "raw_removed_frames": int(np.count_nonzero(~contact_keep)),
                    "removed_spans": removed_spans,
                    "training_eligible_frames": int(np.count_nonzero(contact_eligible)),
                    "training_segments": int(segments.shape[0]),
                },
                "dataset_sha256": dataset_sha256,
            }
            self._handle.create_dataset("manifest/json", data=_json(manifest))
            self._handle.attrs["complete"] = True
            self._handle.flush()
            self._handle.close()
            self._closed = True
            validate_episode(self.staging_path)
            os.replace(self.staging_path, self.output_path)
        except Exception:
            self.abort()
            raise
        return self.output_path

    def abort(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True
        if self.staging_path.exists():
            self.staging_path.unlink()

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None:
            self.abort()
        elif not self._closed:
            self.finish()


def _required_shapes(handle: h5py.File) -> dict[str, tuple[int, ...] | None]:
    return {
        "raw/timestamp_ns": (),
        "raw/observation/qpos": (ROBOT_DOF,),
        "raw/observation/qvel": (ROBOT_DOF,),
        "raw/action/qpos": (ROBOT_DOF,),
        "raw/action/qpos_target": (ROBOT_DOF,),
        "raw/mujoco/full_state": None,
        "raw/pico/hands": (2, 26, 7),
        "raw/pico/source_timestamp_ns": (),
        "raw/pico/bridge_monotonic_ns": (),
        "raw/pico/sequence_id": (),
        "raw/pico/tracking_epoch": (),
        "raw/pico/scale": (2,),
        "raw/validity/sides": (2,),
        **{
            f"raw/cameras/{camera}/rgb": (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
            for camera in CAMERA_NAMES
        },
        **{
            f"raw/cameras/{camera}/segmentation": (IMAGE_HEIGHT, IMAGE_WIDTH, 2)
            for camera in CAMERA_NAMES
        },
        "raw/objects/json": (),
        "raw/contacts/json": (),
        "raw/contacts/hand_object": (),
    }


def validate_episode(path: str | Path, *, verify_checksums: bool = False) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA_VERSION or not bool(
            handle.attrs.get("complete", False)
        ):
            raise ValueError("episode is incomplete or has an unsupported schema")
        if (
            "manifest/json" not in handle
            or "training/index_30hz" not in handle
            or "training/grid_step" not in handle
            or "training/contact_eligible" not in handle
            or "training/segments_30hz" not in handle
        ):
            raise ValueError("episode is missing manifest or derived training metadata")
        raw_manifest = handle["manifest/json"][()]
        manifest = json.loads(
            raw_manifest.decode() if isinstance(raw_manifest, bytes) else raw_manifest
        )
        if not isinstance(manifest, Mapping):
            raise ValueError("manifest/json must contain a JSON object")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("manifest schema_version does not match the HDF5 schema")
        if manifest.get("policy_action_source") != "raw/action/qpos (future actual MuJoCo state)":
            raise ValueError("manifest policy_action_source must identify future actual qpos")
        if manifest.get("teleop_target_source") != "raw/action/qpos_target (audit only)":
            raise ValueError("manifest teleop_target_source must identify audit-only targets")
        if manifest.get("canonical_cameras") != list(CAMERA_NAMES):
            raise ValueError("manifest canonical_cameras do not match the episode schema")
        required = _required_shapes(handle)
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"episode is missing datasets: {missing}")
        length = int(handle["raw/timestamp_ns"].shape[0])
        if length == 0:
            raise ValueError("episode contains no raw frames")
        for name, suffix in required.items():
            dataset = handle[name]
            if dataset.shape[0] != length:
                raise ValueError(f"{name} length differs from raw timestamps")
            if suffix is not None and dataset.shape[1:] != suffix:
                raise ValueError(f"{name} must have trailing shape {suffix}")
        timestamp_dataset = handle["raw/timestamp_ns"]
        if timestamp_dataset.dtype.kind not in {"i", "u"}:
            raise ValueError("raw timestamps must use an integer dtype")
        timestamps = _timestamp_array(timestamp_dataset[:], "raw timestamps")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("raw timestamps are not strictly increasing")
        for name in (
            "raw/pico/source_timestamp_ns",
            "raw/pico/bridge_monotonic_ns",
        ):
            dataset = handle[name]
            if dataset.dtype.kind not in {"i", "u"}:
                raise ValueError(f"{name} must use an integer dtype")
            _timestamp_array(dataset[:], name)
        for name in ("raw/pico/sequence_id", "raw/pico/tracking_epoch"):
            dataset = handle[name]
            if dataset.dtype.kind not in {"i", "u"}:
                raise ValueError(f"{name} must use an integer dtype")
            values = dataset[:]
            if values.size and dataset.dtype.kind == "i" and np.any(values < 0):
                raise ValueError(f"{name} must be non-negative")
        index_dataset = handle["training/index_30hz"]
        grid_dataset = handle["training/grid_step"]
        if index_dataset.dtype.kind not in {"i", "u"} or grid_dataset.dtype.kind not in {"i", "u"}:
            raise ValueError("training index/grid must use integer dtypes")
        raw_index = index_dataset[:]
        raw_grid = grid_dataset[:]
        signed_max = np.iinfo(np.int64).max
        if any(
            dataset.dtype.kind == "u"
            and dataset.size
            and np.any(values > signed_max)
            for dataset, values in ((index_dataset, raw_index), (grid_dataset, raw_grid))
        ):
            raise ValueError("training index/grid exceed signed 64-bit bounds")
        index = raw_index.astype(np.int64, copy=False)
        if index.ndim != 1 or index.size == 0 or np.any(np.diff(index) <= 0):
            raise ValueError("training index must be a strictly increasing vector")
        if index.size and (index[0] < 0 or index[-1] >= length):
            raise ValueError("training index points outside the raw stream")
        grid_step = raw_grid.astype(np.int64, copy=False)
        if grid_step.shape != index.shape or np.any(np.diff(grid_step) <= 0):
            raise ValueError("training grid steps must match and increase with the index")
        expected_grid_step = _training_grid_steps(timestamps, index)
        if not np.array_equal(grid_step, expected_grid_step):
            raise ValueError("training grid steps disagree with raw timestamps")
        # All numeric streams must remain finite after publication; this also
        # catches post-publication edits that a shape-only check would miss.
        finite_streams = (
            "raw/observation/qpos",
            "raw/observation/qvel",
            "raw/action/qpos",
            "raw/action/qpos_target",
            "raw/mujoco/full_state",
            "raw/pico/hands",
            "raw/pico/scale",
        )
        expected_dtypes = {
            "raw/observation/qpos": np.dtype(np.float32),
            "raw/observation/qvel": np.dtype(np.float32),
            "raw/action/qpos": np.dtype(np.float32),
            "raw/action/qpos_target": np.dtype(np.float32),
            "raw/mujoco/full_state": np.dtype(np.float64),
            "raw/pico/hands": np.dtype(np.float32),
            "raw/pico/scale": np.dtype(np.float32),
        }
        for name in finite_streams:
            dataset = handle[name]
            if dataset.dtype != expected_dtypes[name]:
                raise ValueError(f"{name} must use dtype {expected_dtypes[name]}")
            for start in range(0, length, 4096):
                if not np.all(np.isfinite(dataset[start : min(length, start + 4096)])):
                    raise ValueError(f"{name} contains non-finite values")
        if np.any(handle["raw/pico/scale"][:] <= 0):
            raise ValueError("raw/pico/scale must be positive")
        for camera in CAMERA_NAMES:
            if handle[f"raw/cameras/{camera}/rgb"].dtype != np.dtype(np.uint8):
                raise ValueError(f"raw/cameras/{camera}/rgb must be uint8")
            if handle[f"raw/cameras/{camera}/segmentation"].dtype != np.dtype(np.int32):
                raise ValueError(f"raw/cameras/{camera}/segmentation must be int32")
        if handle["raw/validity/sides"].dtype != np.dtype(np.bool_):
            raise ValueError("raw/validity/sides must be bool")
        # The explicit action stream is the actual MuJoCo qpos at the same
        # tick.  Check it in bounded chunks so large episodes stay streaming.
        action = handle["raw/action/qpos"]
        observation = handle["raw/observation/qpos"]
        for start in range(0, length, 4096):
            end = min(length, start + 4096)
            if not np.array_equal(action[start:end], observation[start:end]):
                raise ValueError("raw/action/qpos must equal actual raw/observation/qpos")
        eligible_dataset = handle["training/contact_eligible"]
        if eligible_dataset.dtype != np.dtype(np.bool_):
            raise ValueError("training/contact_eligible must be bool")
        eligible = eligible_dataset[:]
        if eligible.shape != index.shape or eligible.dtype.kind != "b":
            raise ValueError("training/contact_eligible must match training index")
        segments_dataset = handle["training/segments_30hz"]
        if segments_dataset.dtype.kind not in {"i", "u"}:
            raise ValueError("training/segments_30hz must use an integer dtype")
        raw_segments = segments_dataset[:]
        if (
            segments_dataset.dtype.kind == "u"
            and segments_dataset.size
            and np.any(raw_segments > signed_max)
        ):
            raise ValueError("training segments exceed signed 64-bit bounds")
        segments = raw_segments.astype(np.int64, copy=False)
        if segments.ndim != 2 or segments.shape[1:] != (2,):
            raise ValueError("training/segments_30hz must have shape [S,2]")
        expected_segments = build_contact_segments(grid_step, eligible)
        if not np.array_equal(segments, expected_segments):
            raise ValueError("training segments disagree with contact eligibility/grid")
        if segments.size and (
            np.any(segments[:, 0] < 0)
            or np.any(segments[:, 1] > index.size)
            or np.any(segments[:, 0] >= segments[:, 1])
        ):
            raise ValueError("training segments point outside the index")
        for name, expected in (("raw_frames", length),):
            value = manifest.get(name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) != expected:
                raise ValueError(f"manifest {name} does not match the HDF5 stream")
        contact_filter = manifest.get("contact_filter")
        if not isinstance(contact_filter, Mapping):
            raise ValueError("manifest contact_filter is missing")
        value = contact_filter.get("threshold_ns")
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError("contact_filter threshold_ns is invalid")
        threshold = int(value)
        expected_keep, _ = filter_contact_mask(
            timestamps, handle["raw/contacts/hand_object"][:], threshold_ns=threshold
        )
        if not np.array_equal(eligible, expected_keep[index]):
            raise ValueError("training contact eligibility disagrees with raw contacts")
        training_frames = manifest.get("training_frames")
        if (
            isinstance(training_frames, bool)
            or not isinstance(training_frames, Integral)
            or int(training_frames) != int(index.size)
        ):
            raise ValueError("manifest training_frames does not match the 30 Hz index")
        if verify_checksums:
            checksums = manifest.get("dataset_sha256")
            if not isinstance(checksums, Mapping):
                raise ValueError("manifest dataset_sha256 is missing")
            dataset_names: set[str] = set()

            def collect(name: str, value: h5py.Dataset | h5py.Group) -> None:
                if name != "manifest/json" and isinstance(value, h5py.Dataset):
                    dataset_names.add(name)

            handle.visititems(collect)
            if set(checksums) != dataset_names:
                raise ValueError("manifest dataset_sha256 does not cover every dataset")
            for name, expected in checksums.items():
                if (
                    not isinstance(name, str)
                    or not isinstance(expected, str)
                    or len(expected) != 64
                    or any(character not in "0123456789abcdefABCDEF" for character in expected)
                ):
                    raise ValueError(f"invalid dataset checksum entry: {name!r}")
                digest = hashlib.sha256()
                for row in handle[name]:
                    digest.update(_digest_bytes(row))
                if digest.hexdigest() != expected:
                    raise ValueError(f"checksum mismatch: {name}")
        return manifest


def scan_episodes(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted((*root.rglob("*.h5"), *root.rglob("*.hdf5")))


def sequence_indices(index_30hz: np.ndarray, start: int) -> dict[str, np.ndarray]:
    """Return the exact 258-row decomposition for one SPD training sample."""
    raw_index = np.asarray(index_30hz)
    if raw_index.ndim != 1 or raw_index.dtype.kind not in {"i", "u"}:
        raise ValueError("index_30hz must be a 1-D integer vector")
    if raw_index.dtype.kind == "u" and raw_index.size and np.any(
        raw_index > np.iinfo(np.int64).max
    ):
        raise ValueError("index_30hz exceeds signed 64-bit bounds")
    index = raw_index.astype(np.int64, copy=False)
    if index.size == 0 or np.any(index < 0) or np.any(np.diff(index) <= 0):
        raise ValueError("index_30hz must be a strictly increasing non-negative vector")
    if isinstance(start, bool) or not isinstance(start, Integral):
        raise TypeError("start must be an integer")
    start = int(start)
    if start < 1 or start + HISTORY_STEPS >= len(index):
        raise IndexError("SPD sample requires prior, 256 history, and +256 future rows")
    history = index[start : start + HISTORY_STEPS]
    previous = index[start - 1 : start + HISTORY_STEPS - 1]
    base = np.arange(0, HISTORY_STEPS, IMAGE_STRIDE)
    future = np.stack(
        [index[start + offset + 1 : start + offset + ACTION_CHUNK + 1] for offset in base]
    )
    return {
        "history": history,
        "previous": previous,
        "images": history[base],
        "future": future,
    }


class SPDSequenceDataset:
    """Map-style dataset whose labels are future actual qpos, never commands."""

    def __init__(
        self,
        root: str | Path,
        *,
        require_both_sides_valid: bool = True,
        normalization: Mapping[str, Sequence[float]] | None = None,
        symmetry_spec: Any | None = None,
        symmetry_probability: float = 0.0,
        visual_randomization_probability: float = 0.0,
        visual_randomization_strength: float = 0.65,
    ) -> None:
        self.episodes = scan_episodes(root)
        if not self.episodes:
            raise ValueError(f"no HDF5 episodes found under {root}")
        self.normalization = dict(normalization or {})
        if not 0.0 <= float(symmetry_probability) <= 1.0:
            raise ValueError("symmetry_probability must be in [0,1]")
        if symmetry_probability and symmetry_spec is None:
            raise ValueError("symmetry_spec is required when symmetry_probability is non-zero")
        if isinstance(symmetry_spec, Mapping):
            from .augment import SymmetrySpec

            symmetry_spec = SymmetrySpec.from_mapping(symmetry_spec)
        if not 0.0 <= float(visual_randomization_probability) <= 1.0:
            raise ValueError("visual_randomization_probability must be in [0,1]")
        if not 0.0 <= float(visual_randomization_strength) <= 1.0:
            raise ValueError("visual_randomization_strength must be in [0,1]")
        self.symmetry_spec = symmetry_spec
        self.symmetry_probability = float(symmetry_probability)
        self.visual_randomization_probability = float(visual_randomization_probability)
        self.visual_randomization_strength = float(visual_randomization_strength)
        self.samples: list[tuple[Path, int]] = []
        for path in self.episodes:
            validate_episode(path)
            with h5py.File(path, "r") as handle:
                index = handle["training/index_30hz"][:]
                grid_step = handle["training/grid_step"][:]
                validity = handle["raw/validity/sides"][:]
                if "training/segments_30hz" in handle:
                    segments = handle["training/segments_30hz"][:]
                else:
                    # Episodes written before contact-segment support are
                    # still safe because their grid is checked per window.
                    segments = build_contact_segments(
                        grid_step, np.ones(index.shape, dtype=np.bool_)
                    )
            for segment_start, segment_end in segments:
                # The previous row and +256 label must remain in one
                # eligible, nominally consecutive segment.
                for start in range(
                    int(segment_start) + 1, int(segment_end) - HISTORY_STEPS
                ):
                    window = index[start - 1 : start + HISTORY_STEPS + 1]
                    if len(window) != WINDOW_SPAN:
                        continue
                    if require_both_sides_valid and not bool(np.all(validity[window])):
                        continue
                    self.samples.append((path, start))
        if not self.samples:
            raise ValueError("episodes contain no complete 258-row SPD windows")

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize(self, value: np.ndarray, prefix: str) -> np.ndarray:
        mean = np.asarray(self.normalization.get(f"{prefix}_mean", 0.0), dtype=np.float32)
        std = np.asarray(self.normalization.get(f"{prefix}_std", 1.0), dtype=np.float32)
        return (value - mean) / np.maximum(std, 1e-6)

    def __getitem__(self, item: int) -> dict[str, Any]:
        import torch
        from abc_minimal.preprocess import resize_pad_normalize

        path, start = self.samples[item]
        rng = np.random.default_rng(int(item))
        with h5py.File(path, "r") as handle:
            index = handle["training/index_30hz"][:]
            indices = sequence_indices(index, start)
            history_index = indices["history"]
            previous_index = indices["previous"]
            future_index = indices["future"]
            qpos_ds = handle["raw/observation/qpos"]
            action_ds = handle["raw/action/qpos"]
            qpos_raw = qpos_ds[history_index]
            previous_raw = action_ds[previous_index]
            future_raw = action_ds[future_index.reshape(-1)].reshape(
                IMAGE_STEPS, ACTION_CHUNK, ROBOT_DOF
            )
            image_index = indices["images"]
            images_raw = {}
            segmentations_raw = {}
            for camera in CAMERA_NAMES:
                images_raw[camera] = handle[f"raw/cameras/{camera}/rgb"][image_index]
                if self.visual_randomization_probability:
                    segmentations_raw[camera] = handle[
                        f"raw/cameras/{camera}/segmentation"
                    ][image_index]
        if self.visual_randomization_probability and float(rng.random()) < self.visual_randomization_probability:
            from .visual import randomize_camera_frames

            images_raw = randomize_camera_frames(
                images_raw,
                segmentations_raw,
                rng,
                strength=self.visual_randomization_strength,
            )
        if self.symmetry_spec is not None and self.symmetry_probability:
            # Seed by global sample index: augmentation is reproducible across
            # worker processes and does not depend on DataLoader scheduling.
            if float(rng.random()) < self.symmetry_probability:
                from .augment import augment_trajectory

                qpos_raw, previous_raw, future_raw, images_raw = augment_trajectory(
                    qpos_raw, previous_raw, future_raw, images_raw, self.symmetry_spec
                )
        qpos = self._normalize(qpos_raw, "qpos")
        previous = self._normalize(previous_raw, "action")
        future = self._normalize(future_raw, "action")
        images = {}
        for camera in CAMERA_NAMES:
            frames = images_raw[camera]
            tensors = [
                resize_pad_normalize(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
                for frame in frames
            ]
            images[camera] = torch.stack(tensors)
        return {
            "qpos": torch.from_numpy(qpos.astype(np.float32)),
            "previous_action": torch.from_numpy(previous.astype(np.float32)),
            "future_action": torch.from_numpy(future.astype(np.float32)),
            "images": images,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an SPD-VR HDF5 episode")
    parser.add_argument("episode", type=Path)
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args(argv)
    manifest = validate_episode(args.episode, verify_checksums=args.checksums)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CameraFrame",
    "EpisodeWriter",
    "NO_CONTACT_LIMIT_NS",
    "RawFrame",
    "SCHEMA_VERSION",
    "SPDSequenceDataset",
    "build_training_index",
    "build_contact_segments",
    "filter_contact_mask",
    "scan_episodes",
    "sequence_indices",
    "validate_episode",
]
