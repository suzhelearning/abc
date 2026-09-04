"""Incremental framing and typed decoding for the PICO inner byte stream."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from numbers import Integral

import numpy as np

FRAME_MAGIC = 0xAB
FRAME_TYPE_HEAD = 0x05
FRAME_TYPE_WORLD_RESET = 0x06
FRAME_TYPE_HAND_LEFT = 0x38
FRAME_TYPE_HAND_RIGHT = 0x39
HAND_PAYLOAD_BYTES = 733
MAX_PAYLOAD_BYTES = HAND_PAYLOAD_BYTES
MAX_INNER_FRAME_BYTES = 14 + MAX_PAYLOAD_BYTES

_HEADER = struct.Struct("<BBqI")
_FLOAT = struct.Struct("<f")
_HAND_SHAPE = (26, 7)
_HEAD_PAYLOAD_BYTES = 7 * 4
_WORLD_RESET_PAYLOAD_BYTES = 4
_QUATERNION_TOLERANCE = 1e-3

if _HEADER.size != 14 or MAX_INNER_FRAME_BYTES != 747:
    raise RuntimeError("PICO inner frame layout is inconsistent")


class PicoFrameError(ValueError):
    """Raised when framing or typed payload validation fails."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _readonly_array(
    value: object, shape: tuple[int, ...], name: str
) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float32, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise PicoFrameError("invalid_array", name) from exc
    if array.shape != shape:
        raise PicoFrameError("wrong_shape", name)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PicoFrame:
    frame_type: int
    timestamp_ms: int
    payload: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_type, bool)
            or not isinstance(self.frame_type, Integral)
            or not 0 <= int(self.frame_type) <= 0xFF
        ):
            raise PicoFrameError("invalid_frame_type")
        if (
            isinstance(self.timestamp_ms, bool)
            or not isinstance(self.timestamp_ms, Integral)
            or not -0x8000000000000000
            <= int(self.timestamp_ms)
            <= 0x7FFFFFFFFFFFFFFF
        ):
            raise PicoFrameError("invalid_timestamp")
        try:
            payload = bytes(self.payload)
        except (TypeError, ValueError) as exc:
            raise PicoFrameError("invalid_payload") from exc
        object.__setattr__(self, "frame_type", int(self.frame_type))
        object.__setattr__(self, "timestamp_ms", int(self.timestamp_ms))
        object.__setattr__(self, "payload", payload)


@dataclass(frozen=True)
class PicoPose:
    position: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", _readonly_array(self.position, (3,), "position")
        )
        object.__setattr__(
            self,
            "quaternion_xyzw",
            _readonly_array(self.quaternion_xyzw, (4,), "quaternion_xyzw"),
        )


@dataclass(frozen=True)
class PicoHand:
    active: bool
    scale: float
    joints: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "active", bool(self.active))
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(
            self, "joints", _readonly_array(self.joints, _HAND_SHAPE, "joints")
        )


@dataclass(frozen=True)
class PairedHands:
    timestamp_ms: int
    epoch: int
    left: PicoHand
    right: PicoHand


class PicoStreamDecoder:
    """Incrementally split callback chunks without consuming partial frames."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes | bytearray | memoryview) -> list[PicoFrame]:
        self._buffer.extend(data)
        frames: list[PicoFrame] = []
        while len(self._buffer) >= _HEADER.size:
            magic, frame_type, timestamp_ms, payload_size = _HEADER.unpack_from(
                self._buffer
            )
            if magic != FRAME_MAGIC:
                self.reset()
                raise PicoFrameError("bad_magic", hex(magic))
            if payload_size > MAX_PAYLOAD_BYTES:
                self.reset()
                raise PicoFrameError("payload_too_large", str(payload_size))
            frame_size = _HEADER.size + payload_size
            if len(self._buffer) < frame_size:
                break
            payload = bytes(self._buffer[_HEADER.size:frame_size])
            del self._buffer[:frame_size]
            frames.append(PicoFrame(frame_type, timestamp_ms, payload))
        return frames


def decode_hand(frame: PicoFrame) -> PicoHand:
    if frame.frame_type not in (FRAME_TYPE_HAND_LEFT, FRAME_TYPE_HAND_RIGHT):
        raise PicoFrameError("wrong_frame_type", hex(frame.frame_type))
    if len(frame.payload) != HAND_PAYLOAD_BYTES:
        raise PicoFrameError("wrong_hand_payload_size", str(len(frame.payload)))
    active_value = frame.payload[0]
    if active_value not in (0, 1):
        raise PicoFrameError("invalid_active", str(active_value))
    scale = _FLOAT.unpack_from(frame.payload, 1)[0]
    if not math.isfinite(scale) or scale <= 0.0:
        raise PicoFrameError("invalid_scale")
    joints = np.frombuffer(frame.payload, dtype="<f4", count=26 * 7, offset=5).reshape(
        _HAND_SHAPE
    )
    if not np.all(np.isfinite(joints)):
        raise PicoFrameError("non_finite_value", "joints")
    if active_value:
        quaternion_norms = np.linalg.norm(
            joints[:, 3:7].astype(np.float64, copy=False), axis=1
        )
        if np.any(
            np.abs(quaternion_norms - 1.0) > _QUATERNION_TOLERANCE
        ):
            raise PicoFrameError("invalid_quaternion", "hand_joints")
    return PicoHand(bool(active_value), scale, joints)


def decode_head_pose(frame: PicoFrame) -> PicoPose:
    if frame.frame_type != FRAME_TYPE_HEAD:
        raise PicoFrameError("wrong_frame_type", hex(frame.frame_type))
    if len(frame.payload) != _HEAD_PAYLOAD_BYTES:
        raise PicoFrameError("wrong_head_payload_size", str(len(frame.payload)))
    values = np.frombuffer(frame.payload, dtype="<f4", count=7)
    if not np.all(np.isfinite(values)):
        raise PicoFrameError("non_finite_value", "head_pose")
    quaternion = values[3:7]
    norm = float(np.linalg.norm(quaternion.astype(np.float64, copy=False)))
    if abs(norm - 1.0) > _QUATERNION_TOLERANCE:
        raise PicoFrameError("invalid_quaternion", "head_pose")
    return PicoPose(values[:3], quaternion)


def decode_world_reset_yaw(frame: PicoFrame) -> float:
    if frame.frame_type != FRAME_TYPE_WORLD_RESET:
        raise PicoFrameError("wrong_frame_type", hex(frame.frame_type))
    if len(frame.payload) != _WORLD_RESET_PAYLOAD_BYTES:
        raise PicoFrameError("wrong_world_reset_payload_size", str(len(frame.payload)))
    yaw = _FLOAT.unpack(frame.payload)[0]
    if not math.isfinite(yaw):
        raise PicoFrameError("non_finite_value", "world_reset_yaw")
    return yaw


class HandPairer:
    """Pair left/right hand payloads only when timestamp and caller epoch match."""

    def __init__(self) -> None:
        self._epoch: int | None = None
        self._timestamp_ms: int | None = None
        self._left: PicoHand | None = None
        self._right: PicoHand | None = None

    def reset(self) -> None:
        self._timestamp_ms = None
        self._left = None
        self._right = None

    def accept(self, frame: PicoFrame, epoch: int) -> PairedHands | None:
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, Integral)
            or not 0 < int(epoch) <= 0xFFFFFFFFFFFFFFFF
        ):
            raise PicoFrameError("invalid_epoch")
        epoch = int(epoch)
        if self._epoch is not None and epoch < self._epoch:
            raise PicoFrameError("epoch_rollback")
        if self._epoch is None or epoch > self._epoch:
            self.reset()
            self._epoch = epoch

        if frame.frame_type == FRAME_TYPE_WORLD_RESET:
            decode_world_reset_yaw(frame)
            self.reset()
            return None
        if frame.frame_type not in (FRAME_TYPE_HAND_LEFT, FRAME_TYPE_HAND_RIGHT):
            return None

        hand = decode_hand(frame)
        if self._timestamp_ms != frame.timestamp_ms:
            self.reset()
            self._timestamp_ms = frame.timestamp_ms
        if frame.frame_type == FRAME_TYPE_HAND_LEFT:
            self._left = hand
        else:
            self._right = hand

        if self._left is None or self._right is None:
            return None
        pair = PairedHands(
            timestamp_ms=frame.timestamp_ms,
            epoch=epoch,
            left=self._left,
            right=self._right,
        )
        self.reset()
        return pair


__all__ = [
    "FRAME_MAGIC",
    "FRAME_TYPE_HAND_LEFT",
    "FRAME_TYPE_HAND_RIGHT",
    "FRAME_TYPE_HEAD",
    "FRAME_TYPE_WORLD_RESET",
    "HAND_PAYLOAD_BYTES",
    "MAX_INNER_FRAME_BYTES",
    "MAX_PAYLOAD_BYTES",
    "HandPairer",
    "PairedHands",
    "PicoFrame",
    "PicoFrameError",
    "PicoHand",
    "PicoPose",
    "PicoStreamDecoder",
    "decode_hand",
    "decode_head_pose",
    "decode_world_reset_yaw",
]
