"""Independent PICO-wrist neutral alignment for one arm side.

The alignment object is deliberately transport-agnostic.  It consumes one
wrist pose at a time, requires a bounded stable window, and retains the last
safe target whenever the input becomes stale, invalid, or changes epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

import numpy as np


PICO_TO_ROBOT_ROTATION = np.asarray(
    ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64
)


def _strict_integer(value: object, name: str, *, upper: int) -> int:
    """Validate bounded counters without silently truncating malformed input."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result > upper:
        raise ValueError(f"{name} is outside its bounded integer contract")
    return result


def _strict_flag(value: object, name: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, Integral) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean or integer 0/1 flag")


def _pose_matrix(value: Any) -> np.ndarray:
    """Return a finite homogeneous pose from a 4×4 matrix or xyz+xyzw."""

    if hasattr(value, "position") and hasattr(value, "quaternion_xyzw"):
        value = (
            *np.asarray(value.position, dtype=np.float64),
            *np.asarray(value.quaternion_xyzw, dtype=np.float64),
        )
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (7,):
        position, quaternion = array[:3], array[3:]
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("pose quaternion must be finite and non-zero")
        x, y, z, w = quaternion / norm
        rotation = np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation
        result[:3, 3] = position
        array = result
    if array.shape != (4, 4) or not np.all(np.isfinite(array)):
        raise ValueError("pose must be a finite 4x4 transform or 7-vector")
    if not np.allclose(array[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("pose must be homogeneous")
    rotation = array[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise ValueError("pose rotation must be proper")
    return np.array(array, dtype=np.float64, copy=True)


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(first.T @ second) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


@dataclass(frozen=True, slots=True)
class AlignedPose:
    target_pose: np.ndarray
    aligned: bool
    hold_reason: str | None
    stable_count: int

    def __post_init__(self) -> None:
        target = np.array(self.target_pose, dtype=np.float64, copy=True)
        target.setflags(write=False)
        object.__setattr__(self, "target_pose", target)

    @property
    def pose(self) -> np.ndarray:
        return self.target_pose

    @property
    def valid(self) -> bool:
        return bool(self.aligned and self.hold_reason is None)

    @property
    def hold(self) -> bool:
        return self.hold_reason is not None


class SideAlignment:
    """Ten-frame, reset-on-jump neutral alignment for one wrist."""

    def __init__(
        self,
        *,
        neutral_robot: Any | None = None,
        stable_frames: int = 10,
        max_translation_step_m: float = 0.02,
        max_rotation_step_rad: float = 0.15,
        stale_after_ns: int = 50_000_000,
        position_scale: float = 1.0,
        pico_to_robot_rotation: Any | None = None,
    ) -> None:
        if isinstance(stable_frames, bool) or not isinstance(stable_frames, Integral) or int(stable_frames) <= 0:
            raise ValueError("stable_frames must be positive")
        try:
            max_translation_step_m = float(max_translation_step_m)
            max_rotation_step_rad = float(max_rotation_step_rad)
            position_scale = float(position_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("alignment limits must be numeric") from exc
        if not math.isfinite(max_translation_step_m) or max_translation_step_m <= 0:
            raise ValueError("max_translation_step_m must be finite and positive")
        if not math.isfinite(max_rotation_step_rad) or max_rotation_step_rad <= 0:
            raise ValueError("max_rotation_step_rad must be finite and positive")
        if isinstance(stale_after_ns, bool) or not isinstance(stale_after_ns, Integral) or int(stale_after_ns) < 0:
            raise ValueError("stale_after_ns must be non-negative")
        if not math.isfinite(position_scale) or position_scale <= 0:
            raise ValueError("position_scale must be finite and positive")
        self.stable_frames = int(stable_frames)
        self.max_translation_step_m = max_translation_step_m
        self.max_rotation_step_rad = max_rotation_step_rad
        self.stale_after_ns = int(stale_after_ns)
        self.position_scale = position_scale
        self.neutral_robot = _pose_matrix(np.eye(4) if neutral_robot is None else neutral_robot)
        rotation = np.asarray(
            np.eye(3) if pico_to_robot_rotation is None else pico_to_robot_rotation,
            dtype=np.float64,
        )
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("pico_to_robot_rotation must be a proper 3x3 rotation")
        self.pico_to_robot_rotation = rotation.copy()
        self._epoch: int | None = None
        self._last_timestamp_ns: int | None = None
        self._candidate: np.ndarray | None = None
        self._stable_count = 0
        self._aligned = False
        self._pico_neutral: np.ndarray | None = None
        self._last_target: np.ndarray | None = None

    @property
    def aligned(self) -> bool:
        return self._aligned

    @property
    def stable_count(self) -> int:
        return self._stable_count

    @property
    def epoch(self) -> int | None:
        return self._epoch

    @property
    def transform(self) -> np.ndarray | None:
        if self._pico_neutral is None:
            return None
        return self.neutral_robot @ np.linalg.inv(self._pico_neutral)

    @property
    def last_target(self) -> np.ndarray | None:
        return None if self._last_target is None else self._last_target.copy()

    def _clear_alignment(self) -> None:
        self._candidate = None
        self._stable_count = 0
        self._aligned = False
        self._pico_neutral = None

    def _target_or_neutral(self) -> np.ndarray:
        return self._last_target if self._last_target is not None else self.neutral_robot

    def _held(self, reason: str) -> AlignedPose:
        return AlignedPose(self._target_or_neutral(), False, reason, self._stable_count)

    def accept(
        self,
        wrist_pose: Any,
        active: bool,
        epoch: int,
        timestamp_ns: int,
        *,
        now_ns: int | None = None,
    ) -> AlignedPose:
        try:
            timestamp = _strict_integer(
                timestamp_ns, "timestamp_ns", upper=0x7FFFFFFFFFFFFFFF
            )
            epoch = _strict_integer(epoch, "epoch", upper=0xFFFFFFFFFFFFFFFF)
            active = _strict_flag(active, "active")
            if now_ns is not None:
                now_ns = _strict_integer(
                    now_ns, "now_ns", upper=0x7FFFFFFFFFFFFFFF
                )
        except ValueError:
            return self._held("invalid_metadata")
        if self._epoch is None:
            self._epoch = epoch
        elif epoch != self._epoch:
            self._epoch = epoch
            self._last_timestamp_ns = timestamp
            self._clear_alignment()
            return self._held("epoch_change")
        duplicate = self._last_timestamp_ns is not None and timestamp == self._last_timestamp_ns
        if self._last_timestamp_ns is not None and timestamp < self._last_timestamp_ns:
            self._last_timestamp_ns = timestamp
            self._clear_alignment()
            return self._held("timestamp_rollback")
        if now_ns is not None and not 0 <= now_ns - timestamp <= self.stale_after_ns:
            self._last_timestamp_ns = timestamp
            self._clear_alignment()
            return self._held("stale")
        if not active:
            self._last_timestamp_ns = timestamp
            self._clear_alignment()
            return self._held("inactive")
        if duplicate:
            return AlignedPose(
                self._target_or_neutral(),
                self._aligned,
                None if self._aligned else "aligning",
                self._stable_count,
            )
        self._last_timestamp_ns = timestamp
        try:
            current = _pose_matrix(wrist_pose)
        except (TypeError, ValueError):
            self._clear_alignment()
            return self._held("invalid_pose")

        if self._aligned:
            assert self._candidate is not None and self._pico_neutral is not None
            translation_step = float(np.linalg.norm(current[:3, 3] - self._candidate[:3, 3]))
            rotation_step = _rotation_distance(self._candidate[:3, :3], current[:3, :3])
            if translation_step > self.max_translation_step_m or rotation_step > self.max_rotation_step_rad:
                self._clear_alignment()
                self._candidate = current
                self._stable_count = 1
                return self._held("aligning")
            basis = self.pico_to_robot_rotation
            target = self.neutral_robot.copy()
            target[:3, 3] += self.position_scale * basis @ (
                current[:3, 3] - self._pico_neutral[:3, 3]
            )
            target[:3, :3] = (
                basis
                @ current[:3, :3]
                @ self._pico_neutral[:3, :3].T
                @ basis.T
                @ self.neutral_robot[:3, :3]
            )
            self._candidate = current
            self._last_target = target
            return AlignedPose(target, True, None, self._stable_count)

        if self._candidate is None:
            self._candidate = current
            self._stable_count = 1
        else:
            translation_step = float(np.linalg.norm(current[:3, 3] - self._candidate[:3, 3]))
            rotation_step = _rotation_distance(self._candidate[:3, :3], current[:3, :3])
            if translation_step > self.max_translation_step_m or rotation_step > self.max_rotation_step_rad:
                self._stable_count = 1
            else:
                self._stable_count += 1
            self._candidate = current
        if self._stable_count >= self.stable_frames:
            self._pico_neutral = current.copy()
            self._aligned = True
            self._last_target = self.neutral_robot.copy()
            return AlignedPose(self._last_target, True, None, self._stable_count)
        return self._held("aligning")

    def stale(self, now_ns: int) -> AlignedPose:
        try:
            now_ns = _strict_integer(now_ns, "now_ns", upper=0x7FFFFFFFFFFFFFFF)
        except ValueError:
            return self._held("invalid_metadata")
        # No accepted sample, a future-dated clock, or a gap beyond the
        # freshness budget must remain fail-closed.  A recent sample is still
        # usable; report its alignment state without manufacturing a HOLD.
        if self._last_timestamp_ns is None:
            return self._held("stale")
        age_ns = now_ns - self._last_timestamp_ns
        if age_ns < 0 or age_ns > self.stale_after_ns:
            return self._held("stale")
        return AlignedPose(
            self._target_or_neutral(),
            self._aligned,
            None if self._aligned else "aligning",
            self._stable_count,
        )

    def realign(self) -> None:
        self._clear_alignment()

    def reset(self) -> None:
        self._clear_alignment()
        self._epoch = None
        self._last_timestamp_ns = None
        self._last_target = None


__all__ = ["AlignedPose", "PICO_TO_ROBOT_ROTATION", "SideAlignment"]
