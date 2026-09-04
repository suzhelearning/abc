"""Canonical 272-byte dual-arm target wire codec.

The arm-IK process publishes this packet and the MuJoCo viewer consumes it.
The packet deliberately carries both sides, but validity and HOLD reasons are
independent so one failed arm never causes the other arm to be zeroed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from enum import IntEnum
from math import isfinite
from numbers import Integral, Real
from typing import Iterable, Mapping

from .crc import crc32


ARM_TARGET_PACKET_SIZE = 272
ARM_TARGET_VERSION = 2
ARM_TARGET_MAGIC = b"SPDA"
ARM_TARGET_CRC_OFFSET = ARM_TARGET_PACKET_SIZE - 4
LEFT_VALID = 1
RIGHT_VALID = 2
LEFT_Q_OFFSET = 44
RIGHT_Q_OFFSET = LEFT_Q_OFFSET + 7 * 8
LEFT_QDOT_OFFSET = RIGHT_Q_OFFSET + 7 * 8
RIGHT_QDOT_OFFSET = LEFT_QDOT_OFFSET + 7 * 8
# Legacy constant names kept for C++/Python fixture compatibility.
CRC_OFFSET = ARM_TARGET_CRC_OFFSET


class ArmTargetHoldReason(IntEnum):
    NONE = 0
    INPUT_STALE = 1
    SOLVER_FAILURE = 2
    PAUSED = 3
    INACTIVE = 4
    ALIGNING = 5
    DISCONNECTED = 6
    EPOCH_CHANGE = 7


class ArmTargetProtocolError(ValueError):
    """Raised when an arm target violates its structural/semantic contract."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _strict_integer(value: object, name: str) -> int:
    """Validate an integer without silently truncating strings or floats."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ArmTargetProtocolError("invalid_metadata", name)
    return int(value)


def _strict_vector(value: object, name: str) -> tuple[float, ...]:
    """Normalize a seven-element numeric vector without truthiness coercion."""
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ArmTargetProtocolError("invalid_vector", name) from exc
    if len(values) != 7:
        raise ArmTargetProtocolError("wrong_vector_size", name)
    normalized: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ArmTargetProtocolError("invalid_vector", name)
        number = float(item)
        if not isfinite(number):
            raise ArmTargetProtocolError("non_finite_value", name)
        normalized.append(number)
    return tuple(normalized)


# Keep a normal dataclass (rather than slots) for compatibility with existing
# fixture/tools that clone frames via ``frame.__dict__``.
@dataclass(frozen=True)
class ArmTargetFrame:
    sequence: int
    tracking_epoch: int
    source_timestamp_ns: int
    control_timestamp_ns: int
    valid_mask: int
    left_hold_reason: ArmTargetHoldReason
    right_hold_reason: ArmTargetHoldReason
    left_q: tuple[float, ...]
    right_q: tuple[float, ...]
    left_qdot: tuple[float, ...]
    right_qdot: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_hold_reason", _coerce_reason(self.left_hold_reason))
        object.__setattr__(self, "right_hold_reason", _coerce_reason(self.right_hold_reason))
        for name in ("left_q", "right_q", "left_qdot", "right_qdot"):
            object.__setattr__(self, name, _strict_vector(getattr(self, name), name))


def _coerce_reason(value: ArmTargetHoldReason | int) -> ArmTargetHoldReason:
    if isinstance(value, ArmTargetHoldReason):
        return value
    try:
        return ArmTargetHoldReason(_strict_integer(value, "hold_reason"))
    except (TypeError, ValueError, ArmTargetProtocolError) as exc:
        raise ArmTargetProtocolError("invalid_hold_reason", str(value)) from exc


def _coerce_frame(frame: ArmTargetFrame | Mapping[str, object]) -> ArmTargetFrame:
    if isinstance(frame, ArmTargetFrame):
        return frame
    if not isinstance(frame, Mapping):
        raise ArmTargetProtocolError("invalid_frame_type")
    try:
        return ArmTargetFrame(
            sequence=_strict_integer(frame["sequence"], "sequence"),
            tracking_epoch=_strict_integer(frame["tracking_epoch"], "tracking_epoch"),
            source_timestamp_ns=_strict_integer(frame["source_timestamp_ns"], "source_timestamp_ns"),
            control_timestamp_ns=_strict_integer(frame["control_timestamp_ns"], "control_timestamp_ns"),
            valid_mask=_strict_integer(frame["valid_mask"], "valid_mask"),
            left_hold_reason=_coerce_reason(frame["left_hold_reason"]),
            right_hold_reason=_coerce_reason(frame["right_hold_reason"]),
            left_q=_strict_vector(frame["left_q"], "left_q"),
            right_q=_strict_vector(frame["right_q"], "right_q"),
            left_qdot=_strict_vector(frame["left_qdot"], "left_qdot"),
            right_qdot=_strict_vector(frame["right_qdot"], "right_qdot"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ArmTargetProtocolError):
            raise
        raise ArmTargetProtocolError("invalid_frame", str(exc)) from exc


def _finite(values: Iterable[float]) -> bool:
    return all(isfinite(float(value)) for value in values)


def _valid_mask_and_reason(
    valid_mask: int,
    left_reason: ArmTargetHoldReason,
    right_reason: ArmTargetHoldReason,
) -> bool:
    if isinstance(valid_mask, bool) or not isinstance(valid_mask, Integral):
        return False
    if int(valid_mask) & ~0x03:
        return False
    return (
        bool(int(valid_mask) & LEFT_VALID) == (left_reason is ArmTargetHoldReason.NONE)
        and bool(int(valid_mask) & RIGHT_VALID) == (right_reason is ArmTargetHoldReason.NONE)
    )


def _validate_frame(frame: ArmTargetFrame) -> None:
    for name in (
        "sequence",
        "tracking_epoch",
        "source_timestamp_ns",
        "control_timestamp_ns",
    ):
        value = getattr(frame, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ArmTargetProtocolError("invalid_metadata", name)
        if name == "source_timestamp_ns" or name == "control_timestamp_ns":
            if int(value) > 0x7FFFFFFFFFFFFFFF:
                raise ArmTargetProtocolError("invalid_metadata", name)
        elif int(value) > 0xFFFFFFFFFFFFFFFF:
            raise ArmTargetProtocolError("invalid_metadata", name)
    if not _valid_mask_and_reason(
        frame.valid_mask, frame.left_hold_reason, frame.right_hold_reason
    ):
        raise ArmTargetProtocolError("invalid_valid_mask_or_hold_reason")
    for name in ("left_q", "right_q", "left_qdot", "right_qdot"):
        if not _finite(getattr(frame, name)):
            raise ArmTargetProtocolError("non_finite_value", name)


def encode_arm_target(frame: ArmTargetFrame | Mapping[str, object]) -> bytes:
    frame = _coerce_frame(frame)
    _validate_frame(frame)
    packet = bytearray(ARM_TARGET_PACKET_SIZE)
    packet[:4] = ARM_TARGET_MAGIC
    struct.pack_into(
        "<HHQQQQBBBB",
        packet,
        4,
        ARM_TARGET_VERSION,
        ARM_TARGET_PACKET_SIZE,
        int(frame.sequence),
        int(frame.tracking_epoch),
        int(frame.source_timestamp_ns),
        int(frame.control_timestamp_ns),
        int(frame.valid_mask),
        int(frame.left_hold_reason),
        int(frame.right_hold_reason),
        0,
    )
    for offset, values in (
        (LEFT_Q_OFFSET, frame.left_q),
        (RIGHT_Q_OFFSET, frame.right_q),
        (LEFT_QDOT_OFFSET, frame.left_qdot),
        (RIGHT_QDOT_OFFSET, frame.right_qdot),
    ):
        struct.pack_into("<7d", packet, offset, *values)
    struct.pack_into(
        "<I", packet, ARM_TARGET_CRC_OFFSET, crc32(memoryview(packet)[:ARM_TARGET_CRC_OFFSET])
    )
    return bytes(packet)


def decode_arm_target(
    packet: bytes | bytearray | memoryview,
    *,
    now_ns: int | None = None,
    max_age_ns: int | None = None,
    last_sequence: int | None = None,
    last_epoch: int | None = None,
) -> ArmTargetFrame:
    packet = bytes(packet)
    if len(packet) != ARM_TARGET_PACKET_SIZE:
        raise ArmTargetProtocolError("wrong_size", str(len(packet)))
    if packet[:4] != ARM_TARGET_MAGIC:
        raise ArmTargetProtocolError("wrong_magic")
    version, declared_size = struct.unpack_from("<HH", packet, 4)
    if version != ARM_TARGET_VERSION:
        raise ArmTargetProtocolError("wrong_version", str(version))
    if declared_size != ARM_TARGET_PACKET_SIZE:
        raise ArmTargetProtocolError("wrong_declared_size", str(declared_size))
    expected_crc = struct.unpack_from("<I", packet, ARM_TARGET_CRC_OFFSET)[0]
    if expected_crc != crc32(memoryview(packet)[:ARM_TARGET_CRC_OFFSET]):
        raise ArmTargetProtocolError("crc_mismatch")
    sequence, epoch, source_ns, control_ns = struct.unpack_from("<QQQQ", packet, 8)
    valid_mask, left_value, right_value, reserved = struct.unpack_from("<BBBB", packet, 40)
    if reserved:
        raise ArmTargetProtocolError("non_zero_reserved")
    left_reason = _coerce_reason(left_value)
    right_reason = _coerce_reason(right_value)
    if not _valid_mask_and_reason(valid_mask, left_reason, right_reason):
        raise ArmTargetProtocolError("invalid_valid_mask_or_hold_reason")
    frame = ArmTargetFrame(
        sequence=sequence,
        tracking_epoch=epoch,
        source_timestamp_ns=source_ns,
        control_timestamp_ns=control_ns,
        valid_mask=valid_mask,
        left_hold_reason=left_reason,
        right_hold_reason=right_reason,
        left_q=struct.unpack_from("<7d", packet, LEFT_Q_OFFSET),
        right_q=struct.unpack_from("<7d", packet, RIGHT_Q_OFFSET),
        left_qdot=struct.unpack_from("<7d", packet, LEFT_QDOT_OFFSET),
        right_qdot=struct.unpack_from("<7d", packet, RIGHT_QDOT_OFFSET),
    )
    _validate_frame(frame)
    if last_epoch is not None:
        last_epoch = _strict_integer(last_epoch, "last_epoch")
        if last_epoch < 0:
            raise ArmTargetProtocolError("invalid_metadata", "last_epoch")
    if last_sequence is not None:
        last_sequence = _strict_integer(last_sequence, "last_sequence")
        if last_sequence < 0:
            raise ArmTargetProtocolError("invalid_metadata", "last_sequence")
    if last_epoch is not None:
        if epoch < last_epoch:
            raise ArmTargetProtocolError("epoch_rollback")
        if epoch == last_epoch and last_sequence is not None:
            if sequence <= last_sequence:
                raise ArmTargetProtocolError("out_of_order")
    if max_age_ns is not None:
        max_age_ns = _strict_integer(max_age_ns, "max_age_ns")
        if max_age_ns < 0:
            raise ArmTargetProtocolError("invalid_max_age")
    if now_ns is not None:
        now_ns = _strict_integer(now_ns, "now_ns")
        if now_ns < 0 or max_age_ns is None:
            raise ArmTargetProtocolError("invalid_max_age")
        age_ns = now_ns - int(control_ns)
        if not 0 <= age_ns <= max_age_ns:
            frame = replace(
                frame,
                valid_mask=0,
                left_hold_reason=ArmTargetHoldReason.INPUT_STALE,
                right_hold_reason=ArmTargetHoldReason.INPUT_STALE,
            )
    return frame


class ArmTargetStreamDecoder:
    """Stateful epoch/sequence/freshness gate used by the viewer process."""

    def __init__(self, max_age_ns: int = 50_000_000) -> None:
        max_age_ns = _strict_integer(max_age_ns, "max_age_ns")
        if max_age_ns < 0:
            raise ValueError("max_age_ns must be non-negative")
        self.max_age_ns = max_age_ns
        self.last_epoch: int | None = None
        self.last_sequence: int | None = None

    def reset(self) -> None:
        self.last_epoch = None
        self.last_sequence = None

    def decode(self, packet: bytes, *, now_ns: int | None = None) -> ArmTargetFrame:
        frame = decode_arm_target(
            packet,
            now_ns=now_ns,
            max_age_ns=self.max_age_ns if now_ns is not None else None,
            last_sequence=self.last_sequence,
            last_epoch=self.last_epoch,
        )
        if self.last_epoch != frame.tracking_epoch:
            self.last_sequence = None
        self.last_epoch = frame.tracking_epoch
        self.last_sequence = frame.sequence
        return frame


# Compatibility aliases used by callers that imported the old SPD module.
PACKET_SIZE = ARM_TARGET_PACKET_SIZE
encode_packet = encode_arm_target
decode_packet = decode_arm_target
encode_arm_target_packet = encode_arm_target
decode_arm_target_packet = decode_arm_target


__all__ = [
    "ARM_TARGET_CRC_OFFSET",
    "ARM_TARGET_MAGIC",
    "ARM_TARGET_PACKET_SIZE",
    "ARM_TARGET_VERSION",
    "ArmTargetFrame",
    "ArmTargetHoldReason",
    "ArmTargetProtocolError",
    "ArmTargetStreamDecoder",
    "CRC_OFFSET",
    "LEFT_Q_OFFSET",
    "LEFT_VALID",
    "PACKET_SIZE",
    "RIGHT_Q_OFFSET",
    "LEFT_QDOT_OFFSET",
    "RIGHT_QDOT_OFFSET",
    "RIGHT_VALID",
    "crc32",
    "decode_arm_target",
    "decode_arm_target_packet",
    "decode_packet",
    "encode_arm_target",
    "encode_arm_target_packet",
    "encode_packet",
]
