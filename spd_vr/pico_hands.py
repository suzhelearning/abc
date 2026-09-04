"""PICO 26-joint input helpers and the canonical 26→21 hand projection.

The vendor stream contains position plus quaternion for each of 26 joints.
Wuji's public retargeter consumes MediaPipe's 21 position keypoints, so this
module owns the index selection, scale application and wrist-frame transform
in one place.  The core contract is transport-independent; a small decoder
also accepts XRoboToolkit/ROS-style messages for replay and bridge migration,
without adding ROS as a runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .teleop import PICO_JOINT_COUNT, PICO_TO_MEDIAPIPE

PICO_HAND_JOINT_COUNT = PICO_JOINT_COUNT
MEDIAPIPE_JOINT_COUNT = 21


class HandFrameError(ValueError):
    """Raised when a PICO hand cannot be converted to a usable keypoint set."""


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (PICO_HAND_JOINT_COUNT, 3):
        quaternion = np.zeros((PICO_HAND_JOINT_COUNT, 4), dtype=np.float64)
        quaternion[:, 3] = 1.0
        array = np.concatenate((array, quaternion), axis=1)
    if array.shape != (PICO_HAND_JOINT_COUNT, 7):
        raise HandFrameError(f"{name} must have shape (26,3) or (26,7), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise HandFrameError(f"{name} must contain finite values")
    return array.copy()


def _scale(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HandFrameError(f"{name} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise HandFrameError(f"{name} must be finite and positive")
    return result


def _flag(value: Any, name: str) -> bool:
    """Validate a transport active flag without Python truthiness coercion."""
    if type(value) is bool:
        return value
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise HandFrameError(f"{name} must be a boolean or integer 0/1 flag")


def _counter(value: Any, name: str, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise HandFrameError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= upper:
        raise HandFrameError(f"{name} is outside its bounded integer contract")
    return result


@dataclass(frozen=True)
class PicoHandFrame:
    """ROS-independent representation of one atomic dual-hand frame.

    This is intentionally a transport container rather than the validation
    boundary.  A callback may contain one malformed side; ``PicoHandsInput``
    and ``WujiRetargetPair`` validate each side independently and turn only
    that side into HOLD.  Keeping construction permissive also lets replay
    tools preserve the original malformed sample for audit.
    """

    left_hand: np.ndarray
    right_hand: np.ndarray
    left_active: bool = True
    right_active: bool = True
    tracking_epoch: int = 0
    sequence_id: int = 0
    timestamp_ns: int = 0
    left_scale: float = 1.0
    right_scale: float = 1.0

    def __post_init__(self) -> None:
        # Copy arrays when possible, but do not reject malformed values here;
        # the resilient adapter below needs to retain the good side of a
        # partially corrupt callback.  ``PicoHandsInput.update`` is the strict
        # API used by normal consumers.
        for name in ("left_hand", "right_hand"):
            try:
                object.__setattr__(self, name, np.array(getattr(self, name), copy=True))
            except (TypeError, ValueError):
                # Preserve the original object so the consuming validation can
                # report/hold it without hiding the source sample.
                pass
        for name in ("left_active", "right_active"):
            # Preserve malformed flags for the resilient adapter.  Coercing
            # arbitrary strings/NaNs to ``bool`` would turn a corrupt callback
            # into an apparently active hand and could let it reach the
            # retargeter.  Valid transport flags are normalized to bool; all
            # other values remain visible so the consumer can HOLD that side.
            value = getattr(self, name)
            if type(value) is bool:
                normalized = value
            elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
                normalized = bool(value)
            else:
                normalized = value
            object.__setattr__(self, name, normalized)
        bounds = {
            "tracking_epoch": 0xFFFFFFFFFFFFFFFF,
            "sequence_id": 0xFFFFFFFFFFFFFFFF,
            "timestamp_ns": 0x7FFFFFFFFFFFFFFF,
        }
        for name, upper in bounds.items():
            value = getattr(self, name)
            # Normalize only an already-valid integer.  In particular, do not
            # turn 1.5 or "1" into a valid sequence/timestamp: the resilient
            # adapter must be able to recognize and HOLD malformed metadata.
            if (
                not isinstance(value, bool)
                and isinstance(value, (int, np.integer))
                and 0 <= int(value) <= upper
            ):
                object.__setattr__(self, name, int(value))


class PicoHandsInput:
    """Hold the latest frame and expose transformed MediaPipe keypoints."""

    def __init__(self, frame: PicoHandFrame | Mapping[str, Any] | None = None) -> None:
        self._frame: PicoHandFrame | None = None
        if frame is not None:
            self.update(frame)

    # Public-by-convention helpers are also used by the resilient dual-hand
    # adapter.  Keeping them on the class avoids callers reimplementing the
    # exact PICO shape/scale contract.
    _array = staticmethod(_array)
    _scale = staticmethod(_scale)

    @classmethod
    def _from_message(cls, message: Any) -> PicoHandFrame:
        """Decode an XRoboToolkit/ROS-style ``PicoHands`` message.

        The vendor bridge normally emits the versioned wire frame directly,
        but accepting the native message shape keeps offline replay and a
        ROS-to-Zenoh migration from creating a second hand mapping.
        """

        def pose_array(values: Any, name: str) -> np.ndarray:
            rows: list[list[float]] = []
            for pose in values:
                position = getattr(pose, "position", pose)
                if all(hasattr(position, axis) for axis in ("x", "y", "z")):
                    row = [float(position.x), float(position.y), float(position.z)]
                    orientation = getattr(pose, "orientation", None)
                    if orientation is not None and all(
                        hasattr(orientation, axis) for axis in ("x", "y", "z", "w")
                    ):
                        row.extend(
                            [
                                float(orientation.x),
                                float(orientation.y),
                                float(orientation.z),
                                float(orientation.w),
                            ]
                        )
                    else:
                        row.extend([0.0, 0.0, 0.0, 1.0])
                else:
                    row = [float(item) for item in position]
                    if len(row) == 3:
                        row.extend([0.0, 0.0, 0.0, 1.0])
                rows.append(row)
            return cls._array(rows, name)

        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        timestamp_ns = (
            int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            if stamp is not None
            else int(getattr(message, "timestamp_ns", 0))
        )
        return PicoHandFrame(
            left_hand=pose_array(message.left_joints, "left_joints"),
            right_hand=pose_array(message.right_joints, "right_joints"),
            left_active=_flag(message.left_active, "left_active"),
            right_active=_flag(message.right_active, "right_active"),
            tracking_epoch=_counter(
                getattr(message, "tracking_epoch", 0),
                "tracking_epoch",
                0xFFFFFFFFFFFFFFFF,
            ),
            sequence_id=_counter(
                getattr(message, "sequence_id", 0),
                "sequence_id",
                0xFFFFFFFFFFFFFFFF,
            ),
            timestamp_ns=_counter(timestamp_ns, "timestamp_ns", 0x7FFFFFFFFFFFFFFF),
            left_scale=cls._scale(getattr(message, "left_scale", 1.0), "left_scale"),
            right_scale=cls._scale(getattr(message, "right_scale", 1.0), "right_scale"),
        )

    def update(
        self,
        frame: PicoHandFrame | Mapping[str, Any] | Any,
        right_hand: Any | None = None,
        *,
        left_active: bool = True,
        right_active: bool = True,
        tracking_epoch: int = 0,
        sequence_id: int = 0,
        timestamp_ns: int = 0,
        left_scale: float = 1.0,
        right_scale: float = 1.0,
    ) -> None:
        if right_hand is not None:
            value = PicoHandFrame(
                _array(frame, "left_hand"),
                _array(right_hand, "right_hand"),
                left_active,
                right_active,
                tracking_epoch,
                sequence_id,
                timestamp_ns,
                left_scale,
                right_scale,
            )
        elif isinstance(frame, PicoHandFrame):
            value = PicoHandFrame(
                _array(frame.left_hand, "left_hand"),
                _array(frame.right_hand, "right_hand"),
                _flag(frame.left_active, "left_active"),
                _flag(frame.right_active, "right_active"),
                _counter(frame.tracking_epoch, "tracking_epoch", 0xFFFFFFFFFFFFFFFF),
                _counter(frame.sequence_id, "sequence_id", 0xFFFFFFFFFFFFFFFF),
                _counter(frame.timestamp_ns, "timestamp_ns", 0x7FFFFFFFFFFFFFFF),
                _scale(frame.left_scale, "left_scale"),
                _scale(frame.right_scale, "right_scale"),
            )
        elif isinstance(frame, Mapping):
            value = PicoHandFrame(
                _array(frame["left_hand"], "left_hand"),
                _array(frame["right_hand"], "right_hand"),
                _flag(frame.get("left_active", True), "left_active"),
                _flag(frame.get("right_active", True), "right_active"),
                _counter(frame.get("tracking_epoch", 0), "tracking_epoch", 0xFFFFFFFFFFFFFFFF),
                _counter(frame.get("sequence_id", 0), "sequence_id", 0xFFFFFFFFFFFFFFFF),
                _counter(frame.get("timestamp_ns", 0), "timestamp_ns", 0x7FFFFFFFFFFFFFFF),
                _scale(frame.get("left_scale", 1.0), "left_scale"),
                _scale(frame.get("right_scale", 1.0), "right_scale"),
            )
        else:
            # Accept both a TrackingFrame-like object and native ROS message.
            # The latter is detected by its left_joints/right_joints fields.
            if hasattr(frame, "left_joints") and hasattr(frame, "right_joints"):
                value = self._from_message(frame)
            else:
                value = PicoHandFrame(
                    _array(getattr(frame, "left_hand"), "left_hand"),
                    _array(getattr(frame, "right_hand"), "right_hand"),
                    _flag(getattr(frame, "left_active", True), "left_active"),
                    _flag(getattr(frame, "right_active", True), "right_active"),
                    _counter(getattr(frame, "tracking_epoch", 0), "tracking_epoch", 0xFFFFFFFFFFFFFFFF),
                    _counter(getattr(frame, "sequence", getattr(frame, "sequence_id", 0)), "sequence_id", 0xFFFFFFFFFFFFFFFF),
                    _counter(getattr(frame, "bridge_monotonic_ns", getattr(frame, "timestamp_ns", 0)), "timestamp_ns", 0x7FFFFFFFFFFFFFFF),
                    _scale(getattr(frame, "left_scale", 1.0), "left_scale"),
                    _scale(getattr(frame, "right_scale", 1.0), "right_scale"),
                )
        self._frame = value

    @property
    def frame(self) -> PicoHandFrame:
        if self._frame is None:
            raise HandFrameError("PICO hand frame is not initialized")
        return self._frame

    def get_side_fingers_data(self, side: str) -> np.ndarray:
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        frame = self.frame
        if side == "left":
            hand, active, scale = frame.left_hand, frame.left_active, frame.left_scale
        else:
            hand, active, scale = frame.right_hand, frame.right_active, frame.right_scale
        if not active:
            return np.zeros((MEDIAPIPE_JOINT_COUNT, 3), dtype=np.float64)
        points = hand[PICO_TO_MEDIAPIPE, :3] * float(scale)
        try:
            from wuji_retargeting.mediapipe import apply_mediapipe_transformations

            transformed = np.asarray(
                apply_mediapipe_transformations(points, side), dtype=np.float64
            )
        except Exception as exc:
            raise HandFrameError(f"{side} MediaPipe transform failed") from exc
        if transformed.shape != (MEDIAPIPE_JOINT_COUNT, 3) or not np.all(np.isfinite(transformed)):
            raise HandFrameError(f"{side} transformed keypoints are invalid")
        return transformed

    def get_side_raw_fingers_data(self, side: str) -> np.ndarray:
        """Return scaled PICO points selected into MediaPipe's 21-joint order.

        ``wuji_retargeting.Retargeter.retarget`` applies the wrist-frame
        transformation itself.  This raw view is therefore the correct input
        for the high-level retargeter, while :meth:`get_side_fingers_data`
        remains available for callers that need the transformed view directly.
        """

        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        frame = self.frame
        if side == "left":
            hand, active, scale = frame.left_hand, frame.left_active, frame.left_scale
        else:
            hand, active, scale = frame.right_hand, frame.right_active, frame.right_scale
        if not active:
            return np.zeros((MEDIAPIPE_JOINT_COUNT, 3), dtype=np.float64)
        points = _array(hand, f"{side}_hand")[PICO_TO_MEDIAPIPE, :3] * _scale(
            scale, f"{side}_scale"
        )
        return np.asarray(points, dtype=np.float64)

    def get_fingers_data(self) -> dict[str, np.ndarray]:
        return {
            "left_fingers": self.get_side_fingers_data("left"),
            "right_fingers": self.get_side_fingers_data("right"),
        }


__all__ = [
    "HandFrameError",
    "MEDIAPIPE_JOINT_COUNT",
    "PICO_HAND_JOINT_COUNT",
    "PICO_TO_MEDIAPIPE",
    "PicoHandFrame",
    "PicoHandsInput",
]
