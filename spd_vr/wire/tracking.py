"""Canonical 1,540-byte little-endian tracking wire codec."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .crc import crc32

TRACKING_PACKET_SIZE = 1540
TRACKING_VERSION = 1
TRACKING_MAGIC = b"SVT1"

_LEFT_ACTIVE = 1 << 0
_RIGHT_ACTIVE = 1 << 1
_HEAD_VALID = 1 << 2
_KNOWN_FLAGS = _LEFT_ACTIVE | _RIGHT_ACTIVE | _HEAD_VALID
_HEADER = struct.Struct("<IHHIIQQqqff")
_HEAD_FLOATS = 7
_HAND_SHAPE = (26, 7)
_HEAD_OFFSET = _HEADER.size
_LEFT_HAND_OFFSET = _HEAD_OFFSET + _HEAD_FLOATS * 4
_RIGHT_HAND_OFFSET = _LEFT_HAND_OFFSET + 26 * 7 * 4
_MAGIC_U32 = int.from_bytes(TRACKING_MAGIC, "little")
_QUATERNION_TOLERANCE = 1e-3

if _HEADER.size != 56 or _RIGHT_HAND_OFFSET + 26 * 7 * 4 != TRACKING_PACKET_SIZE:
    raise RuntimeError("tracking wire layout does not total 1540 bytes")


class TrackingProtocolError(ValueError):
    """Raised when a tracking frame violates the wire contract."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _array(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float32, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise TrackingProtocolError("invalid_array", name) from exc
    if array.shape != shape:
        raise TrackingProtocolError(
            "wrong_shape", f"{name} must have shape {shape}, got {array.shape}"
        )
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TrackingFrame:
    sequence: int
    tracking_epoch: int
    source_timestamp_ns: int
    bridge_monotonic_ns: int
    left_active: bool
    right_active: bool
    head_valid: bool
    left_scale: float
    right_scale: float
    head_pose: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_pose", _array("head_pose", self.head_pose, (7,)))
        object.__setattr__(
            self, "left_hand", _array("left_hand", self.left_hand, _HAND_SHAPE)
        )
        object.__setattr__(
            self, "right_hand", _array("right_hand", self.right_hand, _HAND_SHAPE)
        )


def _check_uint64(name: str, value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 < int(value) <= 0xFFFFFFFFFFFFFFFF
    ):
        raise TrackingProtocolError("invalid_metadata", name)


def _check_int64(name: str, value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 < int(value)
        or int(value) > 0x7FFFFFFFFFFFFFFF
    ):
        raise TrackingProtocolError("invalid_metadata", name)


def _validate_quaternions(name: str, poses: np.ndarray) -> None:
    quaternions = poses[..., 3:7]
    norms = np.linalg.norm(quaternions.astype(np.float64, copy=False), axis=-1)
    if np.any(np.abs(norms - 1.0) > _QUATERNION_TOLERANCE):
        raise TrackingProtocolError("invalid_quaternion", name)


def _validate_frame(frame: TrackingFrame) -> None:
    _check_uint64("sequence", frame.sequence)
    _check_uint64("tracking_epoch", frame.tracking_epoch)
    _check_int64("source_timestamp_ns", frame.source_timestamp_ns)
    _check_int64("bridge_monotonic_ns", frame.bridge_monotonic_ns)
    if any(
        type(value) is not bool
        for value in (frame.left_active, frame.right_active, frame.head_valid)
    ):
        raise TrackingProtocolError("invalid_flag")

    scalars = np.asarray((frame.left_scale, frame.right_scale), dtype=np.float64)
    if not np.all(np.isfinite(scalars)):
        raise TrackingProtocolError("non_finite_value", "scale")
    if np.any(scalars <= 0.0):
        raise TrackingProtocolError("invalid_scale")

    for name in ("head_pose", "left_hand", "right_hand"):
        if not np.all(np.isfinite(getattr(frame, name))):
            raise TrackingProtocolError("non_finite_value", name)

    if frame.head_valid:
        _validate_quaternions("head_pose", frame.head_pose.reshape(1, 7))
    if frame.left_active:
        _validate_quaternions("left_hand", frame.left_hand)
    if frame.right_active:
        _validate_quaternions("right_hand", frame.right_hand)


def _write_f32(packet: bytearray, offset: int, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    raw = memoryview(contiguous).cast("B")
    memoryview(packet)[offset : offset + raw.nbytes] = raw


def encode_tracking(frame: TrackingFrame) -> bytes:
    if not isinstance(frame, TrackingFrame):
        raise TrackingProtocolError("invalid_frame_type")
    _validate_frame(frame)

    flags = (
        (_LEFT_ACTIVE if frame.left_active else 0)
        | (_RIGHT_ACTIVE if frame.right_active else 0)
        | (_HEAD_VALID if frame.head_valid else 0)
    )
    packet = bytearray(TRACKING_PACKET_SIZE)
    _HEADER.pack_into(
        packet,
        0,
        _MAGIC_U32,
        TRACKING_VERSION,
        flags,
        TRACKING_PACKET_SIZE,
        0,
        int(frame.sequence),
        int(frame.tracking_epoch),
        int(frame.source_timestamp_ns),
        int(frame.bridge_monotonic_ns),
        float(frame.left_scale),
        float(frame.right_scale),
    )
    _write_f32(packet, _HEAD_OFFSET, frame.head_pose)
    _write_f32(packet, _LEFT_HAND_OFFSET, frame.left_hand)
    _write_f32(packet, _RIGHT_HAND_OFFSET, frame.right_hand)
    struct.pack_into("<I", packet, 12, crc32(memoryview(packet)[16:]))
    return bytes(packet)


def decode_tracking(packet: bytes | bytearray | memoryview) -> TrackingFrame:
    packet = bytes(packet)
    if len(packet) != TRACKING_PACKET_SIZE:
        raise TrackingProtocolError("wrong_size", str(len(packet)))
    (
        magic,
        version,
        flags,
        declared_size,
        expected_crc,
        sequence,
        tracking_epoch,
        source_timestamp_ns,
        bridge_monotonic_ns,
        left_scale,
        right_scale,
    ) = _HEADER.unpack_from(packet)
    if magic != _MAGIC_U32:
        raise TrackingProtocolError("wrong_magic")
    if version != TRACKING_VERSION:
        raise TrackingProtocolError("wrong_version", str(version))
    if flags & ~_KNOWN_FLAGS:
        raise TrackingProtocolError("non_zero_reserved", hex(flags & ~_KNOWN_FLAGS))
    if declared_size != TRACKING_PACKET_SIZE:
        raise TrackingProtocolError("wrong_declared_size", str(declared_size))
    if expected_crc != crc32(memoryview(packet)[16:]):
        raise TrackingProtocolError("crc_mismatch")

    head_pose = np.frombuffer(packet, dtype="<f4", count=7, offset=_HEAD_OFFSET)
    left_hand = np.frombuffer(
        packet, dtype="<f4", count=26 * 7, offset=_LEFT_HAND_OFFSET
    ).reshape(_HAND_SHAPE)
    right_hand = np.frombuffer(
        packet, dtype="<f4", count=26 * 7, offset=_RIGHT_HAND_OFFSET
    ).reshape(_HAND_SHAPE)
    frame = TrackingFrame(
        sequence=sequence,
        tracking_epoch=tracking_epoch,
        source_timestamp_ns=source_timestamp_ns,
        bridge_monotonic_ns=bridge_monotonic_ns,
        left_active=bool(flags & _LEFT_ACTIVE),
        right_active=bool(flags & _RIGHT_ACTIVE),
        head_valid=bool(flags & _HEAD_VALID),
        left_scale=left_scale,
        right_scale=right_scale,
        head_pose=head_pose,
        left_hand=left_hand,
        right_hand=right_hand,
    )
    _validate_frame(frame)
    return frame


class TrackingStreamGate:
    """Reject rollback and duplicate frames while allowing a newer epoch reset."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_epoch: int | None = None
        self.last_sequence: int | None = None
        self.last_source_timestamp_ns: int | None = None
        self.last_bridge_monotonic_ns: int | None = None

    def accept(self, frame: TrackingFrame) -> bool:
        _validate_frame(frame)
        if self.last_epoch is not None and frame.tracking_epoch < self.last_epoch:
            raise TrackingProtocolError("epoch_rollback")
        if self.last_epoch == frame.tracking_epoch:
            if self.last_sequence is not None and frame.sequence <= self.last_sequence:
                raise TrackingProtocolError("out_of_order")
            if (
                self.last_source_timestamp_ns is not None
                and frame.source_timestamp_ns <= self.last_source_timestamp_ns
            ):
                raise TrackingProtocolError("timestamp_rollback", "source")
            if (
                self.last_bridge_monotonic_ns is not None
                and frame.bridge_monotonic_ns <= self.last_bridge_monotonic_ns
            ):
                raise TrackingProtocolError("timestamp_rollback", "bridge")

        self.last_epoch = frame.tracking_epoch
        self.last_sequence = frame.sequence
        self.last_source_timestamp_ns = frame.source_timestamp_ns
        self.last_bridge_monotonic_ns = frame.bridge_monotonic_ns
        return True


__all__ = [
    "TRACKING_PACKET_SIZE",
    "TRACKING_VERSION",
    "TrackingFrame",
    "TrackingProtocolError",
    "TrackingStreamGate",
    "decode_tracking",
    "encode_tracking",
]
