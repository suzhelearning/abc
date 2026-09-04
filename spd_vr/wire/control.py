"""Canonical 40-byte little-endian teleoperation control codec."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral

from .crc import crc32

CONTROL_PACKET_SIZE = 40
CONTROL_VERSION = 1
CONTROL_MAGIC = b"SVC1"

_PACKET = struct.Struct("<IHHIIQq8s")
_MAGIC_U32 = int.from_bytes(CONTROL_MAGIC, "little")
_ZERO_RESERVED = bytes(8)

if _PACKET.size != CONTROL_PACKET_SIZE:
    raise RuntimeError("control wire layout does not total 40 bytes")


class ControlCommand(IntEnum):
    START = 1
    PAUSE = 2
    RESUME = 3
    REALIGN = 4
    RESET = 5
    SHUTDOWN = 6


class ControlProtocolError(ValueError):
    """Raised when a control frame violates the wire contract."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


@dataclass(frozen=True)
class ControlFrame:
    sequence: int
    monotonic_timestamp_ns: int
    command: ControlCommand


def _validate_frame(frame: ControlFrame) -> None:
    if not isinstance(frame, ControlFrame):
        raise ControlProtocolError("invalid_frame_type")
    if (
        isinstance(frame.sequence, bool)
        or not isinstance(frame.sequence, Integral)
        or not 0 < int(frame.sequence) <= 0xFFFFFFFFFFFFFFFF
        or isinstance(frame.monotonic_timestamp_ns, bool)
        or not isinstance(frame.monotonic_timestamp_ns, Integral)
        or not 0 < int(frame.monotonic_timestamp_ns) <= 0x7FFFFFFFFFFFFFFF
    ):
        raise ControlProtocolError("invalid_metadata")
    if not isinstance(frame.command, ControlCommand):
        raise ControlProtocolError("invalid_command")


def encode_control(frame: ControlFrame) -> bytes:
    _validate_frame(frame)
    packet = bytearray(
        _PACKET.pack(
            _MAGIC_U32,
            CONTROL_VERSION,
            int(frame.command),
            CONTROL_PACKET_SIZE,
            0,
            int(frame.sequence),
            int(frame.monotonic_timestamp_ns),
            _ZERO_RESERVED,
        )
    )
    struct.pack_into("<I", packet, 12, crc32(memoryview(packet)[16:]))
    return bytes(packet)


def decode_control(packet: bytes | bytearray | memoryview) -> ControlFrame:
    packet = bytes(packet)
    if len(packet) != CONTROL_PACKET_SIZE:
        raise ControlProtocolError("wrong_size", str(len(packet)))
    (
        magic,
        version,
        command_value,
        declared_size,
        expected_crc,
        sequence,
        monotonic_timestamp_ns,
        reserved,
    ) = _PACKET.unpack(packet)
    if magic != _MAGIC_U32:
        raise ControlProtocolError("wrong_magic")
    if version != CONTROL_VERSION:
        raise ControlProtocolError("wrong_version", str(version))
    try:
        command = ControlCommand(command_value)
    except ValueError as exc:
        raise ControlProtocolError("invalid_command", str(command_value)) from exc
    if declared_size != CONTROL_PACKET_SIZE:
        raise ControlProtocolError("wrong_declared_size", str(declared_size))
    if expected_crc != crc32(memoryview(packet)[16:]):
        raise ControlProtocolError("crc_mismatch")
    if reserved != _ZERO_RESERVED:
        raise ControlProtocolError("non_zero_reserved")

    frame = ControlFrame(
        sequence=sequence,
        monotonic_timestamp_ns=monotonic_timestamp_ns,
        command=command,
    )
    _validate_frame(frame)
    return frame


class ControlSequenceGate:
    """Apply new commands once and recognize exact retransmissions as idempotent."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_frame: ControlFrame | None = None

    @property
    def last_sequence(self) -> int | None:
        return None if self.last_frame is None else self.last_frame.sequence

    def accept(self, frame: ControlFrame) -> bool:
        _validate_frame(frame)
        if self.last_frame is None:
            self.last_frame = frame
            return True
        if frame.sequence < self.last_frame.sequence:
            raise ControlProtocolError("sequence_rollback")
        if frame.sequence == self.last_frame.sequence:
            if frame != self.last_frame:
                raise ControlProtocolError("duplicate_conflict")
            return False
        self.last_frame = frame
        return True


__all__ = [
    "CONTROL_PACKET_SIZE",
    "CONTROL_VERSION",
    "ControlCommand",
    "ControlFrame",
    "ControlProtocolError",
    "ControlSequenceGate",
    "decode_control",
    "encode_control",
]
