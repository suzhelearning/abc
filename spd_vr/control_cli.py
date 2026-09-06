"""Publish versioned control commands to the SPD-VR process graph.

The positional form sends one command and exits.  ``--pedal`` keeps one
terminal attached to the process graph and maps a three-button foot pedal to
the common collection controls: ``R`` realigns, ``S`` toggles pause/resume,
and ``D`` shuts down the episode.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

from .wire import CONTROL_KEY, ControlCommand, ControlFrame, encode_control


@dataclass(frozen=True, slots=True)
class PedalAction:
    """Result of interpreting one pedal/keyboard byte."""

    command: ControlCommand | None
    paused: bool
    stop: bool


def pedal_action(key: str, *, paused: bool) -> PedalAction:
    """Map a pedal key to a control command and the next loop state.

    The helper is intentionally independent of Zenoh and terminal handling so
    the safety-critical key mapping is directly testable.  Unknown keys are
    ignored; ``q``, Escape, and Ctrl-C leave the controller without sending a
    shutdown command.
    """

    normalized = key.lower()
    if normalized == "r":
        return PedalAction(ControlCommand.REALIGN, paused, False)
    if normalized == "s":
        return PedalAction(
            ControlCommand.RESUME if paused else ControlCommand.PAUSE,
            not paused,
            False,
        )
    if normalized == "d":
        return PedalAction(ControlCommand.SHUTDOWN, paused, True)
    if normalized in {"q", "\x1b", "\x03", "\x04"}:
        return PedalAction(None, paused, True)
    return PedalAction(None, paused, False)


def _sequence_seed(sequence: int | None) -> int:
    value = time.monotonic_ns() if sequence is None else int(sequence)
    if value <= 0:
        raise ValueError("control sequence must be positive")
    return value


def _publish_command(publisher: object, sequence: int, command: ControlCommand) -> int:
    frame = ControlFrame(sequence, max(1, time.monotonic_ns()), command)
    publisher.put(encode_control(frame))
    # Manual sequence seeds are supported, while monotonic_ns keeps the
    # default path naturally increasing between pedal presses.
    return max(sequence + 1, time.monotonic_ns())


def _run_pedal_mode(*, endpoint: str, sequence: int | None) -> int:
    """Run the persistent single-terminal foot-pedal controller."""

    if not sys.stdin.isatty():
        raise RuntimeError("--pedal requires a TTY; run it in an interactive terminal")

    fd = sys.stdin.fileno()
    original_attributes = termios.tcgetattr(fd)
    next_sequence = _sequence_seed(sequence)
    paused = False
    from .zenoh_transport import ZenohNode, peer_config

    print(
        "Pedal control: R=realign, S=pause/resume, D=finish + shutdown. "
        "q/Esc/Ctrl-C exits without shutdown.",
        flush=True,
    )
    try:
        tty.setcbreak(fd)
        with ZenohNode(peer_config(listen=False, endpoint=endpoint)) as node:
            publisher = node.declare_publisher(CONTROL_KEY)
            while True:
                readable, _, _ = select.select([sys.stdin], [], [], 0.25)
                if not readable:
                    continue
                key = sys.stdin.read(1)
                if not key:
                    break
                action = pedal_action(key, paused=paused)
                paused = action.paused
                if action.command is None:
                    if action.stop:
                        print("Pedal controller exited without shutdown.", flush=True)
                        break
                    continue

                next_sequence = _publish_command(
                    publisher, next_sequence, action.command
                )
                if action.command is ControlCommand.REALIGN:
                    print("realign sent", flush=True)
                elif action.command is ControlCommand.PAUSE:
                    print("pause sent", flush=True)
                elif action.command is ControlCommand.RESUME:
                    print("resume sent", flush=True)
                else:
                    print("shutdown sent; waiting for viewer to publish the HDF5 episode", flush=True)
                if action.stop:
                    # Give Zenoh a bounded opportunity to enqueue the final
                    # shutdown sample before the publisher/session closes.
                    time.sleep(0.05)
                    break
    except KeyboardInterrupt:
        print("Pedal controller interrupted without shutdown.", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_attributes)
        print("", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=tuple(command.name.lower() for command in ControlCommand),
    )
    parser.add_argument(
        "--pedal",
        "--interactive",
        action="store_true",
        dest="pedal",
        help="keep this terminal open and read the R/S/D foot-pedal keys",
    )
    parser.add_argument("--endpoint", default="tcp/127.0.0.1:7447")
    parser.add_argument(
        "--sequence",
        type=int,
        help="strictly increasing command sequence; defaults to monotonic_ns",
    )
    args = parser.parse_args(argv)
    if args.pedal and args.command is not None:
        parser.error("--pedal cannot be combined with a positional command")
    if not args.pedal and args.command is None:
        parser.error("a command is required unless --pedal is used")
    if args.sequence is not None and args.sequence <= 0:
        parser.error("--sequence must be positive")
    if args.pedal:
        try:
            return _run_pedal_mode(endpoint=args.endpoint, sequence=args.sequence)
        except RuntimeError as exc:
            parser.error(str(exc))

    sequence = _sequence_seed(args.sequence)
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


__all__ = ["PedalAction", "main", "pedal_action"]
