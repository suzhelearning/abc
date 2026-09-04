"""CRC-32 primitives shared by the teleoperation wire codecs."""

from __future__ import annotations

import binascii


def crc32(data: bytes | bytearray | memoryview) -> int:
    """Return the unsigned IEEE CRC-32 used by all teleoperation packets."""

    return binascii.crc32(data) & 0xFFFFFFFF


__all__ = ["crc32"]
