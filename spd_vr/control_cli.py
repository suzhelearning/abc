"""Publish one versioned control command to the SPD-VR process graph."""

from __future__ import annotations

import argparse
import time

from .wire import CONTROL_KEY, ControlCommand, ControlFrame, encode_control


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=tuple(command.name.lower() for command in ControlCommand),
    )
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument(
        "--sequence",
        type=int,
        help="strictly increasing command sequence; defaults to monotonic_ns",
    )
    args = parser.parse_args(argv)
    sequence = int(time.monotonic_ns() if args.sequence is None else args.sequence)
    if sequence <= 0:
        parser.error("--sequence must be positive")
    command = ControlCommand[args.command.upper()]
    from .zenoh_transport import ZenohNode, peer_config

    frame = ControlFrame(sequence, max(1, time.monotonic_ns()), command)
    with ZenohNode(peer_config(listen=False, endpoint=args.endpoint)) as node:
        node.declare_publisher(CONTROL_KEY).put(encode_control(frame))
        # Let the transport enqueue the sample before closing the short-lived
        # command session.  This is bounded and does not affect the control
        # loop's real-time path.
        time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
