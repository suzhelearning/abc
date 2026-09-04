"""Shape-checked public contracts shared by teleop, data, and policy code."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import numpy as np


ROBOT_DOF = 54
ARM_DOF_PER_SIDE = 7
HAND_DOF_PER_SIDE = 20
HISTORY_STEPS = 256
ACTION_CHUNK = 8
IMAGE_STRIDE = 8
IMAGE_STEPS = HISTORY_STEPS // IMAGE_STRIDE
CAMERA_NAMES = ("top", "left_wrist", "right_wrist")


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _timestamp(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= int(value) <= 0x7FFFFFFFFFFFFFFF
    ):
        raise ValueError(f"{name} must be a non-negative signed 64-bit integer")
    return int(value)


def _flag(value: object, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean or integer 0/1 flag")


@dataclass(frozen=True, slots=True)
class JointTarget:
    timestamp_ns: int
    qpos: np.ndarray
    left_valid: bool = True
    right_valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _timestamp(self.timestamp_ns, "timestamp_ns"))
        object.__setattr__(self, "left_valid", _flag(self.left_valid, "left_valid"))
        object.__setattr__(self, "right_valid", _flag(self.right_valid, "right_valid"))
        object.__setattr__(self, "qpos", _finite_array(self.qpos, (ROBOT_DOF,), "qpos"))


@dataclass(frozen=True, slots=True)
class Observation:
    timestamp_ns: int
    qpos: np.ndarray
    images: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _timestamp(self.timestamp_ns, "timestamp_ns"))
        object.__setattr__(self, "qpos", _finite_array(self.qpos, (ROBOT_DOF,), "qpos"))
        if self.images is not None and set(self.images) != set(CAMERA_NAMES):
            raise ValueError(f"images must contain exactly {CAMERA_NAMES}")


@dataclass(frozen=True, slots=True)
class ActionChunk:
    qpos: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qpos",
            _finite_array(self.qpos, (ACTION_CHUNK, ROBOT_DOF), "action qpos"),
        )


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    """One SPD training example.

    ``previous_action`` is the previous *observed* joint state.  The project
    intentionally trains on future actual qpos rather than teleop targets.
    """

    qpos: np.ndarray
    previous_action: np.ndarray
    future_action: np.ndarray
    images: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "qpos", _finite_array(self.qpos, (HISTORY_STEPS, ROBOT_DOF), "qpos")
        )
        object.__setattr__(
            self,
            "previous_action",
            _finite_array(
                self.previous_action,
                (HISTORY_STEPS, ROBOT_DOF),
                "previous_action",
            ),
        )
        object.__setattr__(
            self,
            "future_action",
            _finite_array(
                self.future_action,
                (IMAGE_STEPS, ACTION_CHUNK, ROBOT_DOF),
                "future_action",
            ),
        )
        if set(self.images) != set(CAMERA_NAMES):
            raise ValueError(f"images must contain exactly {CAMERA_NAMES}")
        checked = {}
        for camera in CAMERA_NAMES:
            value = np.asarray(self.images[camera])
            if value.shape != (IMAGE_STEPS, 168, 224, 3):
                raise ValueError(
                    f"{camera} images must have shape "
                    f"{(IMAGE_STEPS, 168, 224, 3)}, got {value.shape}"
                )
            if value.dtype != np.uint8:
                raise ValueError(f"{camera} images must be uint8")
            checked[camera] = value.copy()
        object.__setattr__(self, "images", checked)


__all__ = [
    "ACTION_CHUNK",
    "ARM_DOF_PER_SIDE",
    "CAMERA_NAMES",
    "HAND_DOF_PER_SIDE",
    "HISTORY_STEPS",
    "IMAGE_STEPS",
    "IMAGE_STRIDE",
    "ROBOT_DOF",
    "ActionChunk",
    "JointTarget",
    "Observation",
    "TrajectorySample",
]
