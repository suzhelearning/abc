from spd_vr.control_cli import PedalAction, pedal_action
from spd_vr.wire import ControlCommand


def test_pedal_keys_map_to_lifecycle_commands() -> None:
    assert pedal_action("r", paused=False) == PedalAction(
        command=ControlCommand.REALIGN,
        paused=False,
        stop=False,
    )
    assert pedal_action("S", paused=False) == PedalAction(
        command=ControlCommand.PAUSE,
        paused=True,
        stop=False,
    )
    assert pedal_action("s", paused=True) == PedalAction(
        command=ControlCommand.RESUME,
        paused=False,
        stop=False,
    )
    assert pedal_action("d", paused=False) == PedalAction(
        command=ControlCommand.SHUTDOWN,
        paused=False,
        stop=True,
    )


def test_pedal_exit_keys_do_not_publish_a_control_command() -> None:
    for key in ("q", "\x1b", "\x03"):
        assert pedal_action(key, paused=True) == PedalAction(
            command=None,
            paused=True,
            stop=True,
        )


def test_unknown_pedal_key_is_ignored() -> None:
    assert pedal_action("x", paused=False) == PedalAction(
        command=None,
        paused=False,
        stop=False,
    )
