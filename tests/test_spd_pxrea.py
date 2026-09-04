import ctypes
import json
import struct

import pytest

from spd_vr.pico_frames import FRAME_TYPE_HAND_LEFT, FRAME_TYPE_HAND_RIGHT
from spd_vr.pxrea_bridge import BridgeCore
from spd_vr.pxrea_sdk import (
    BoundedCallbackQueue,
    PXREAClient,
    PXREADevCustomMessage,
    PXREA_DEVICE_CUSTOM,
)
from spd_vr.wire import TRACKING_PACKET_SIZE, decode_tracking


def _hand_frame(frame_type, timestamp):
    payload = bytes([0, 0, 0x80, 0x3F]) + bytes(729)
    return struct.pack("<BBqI", 0xAB, frame_type, timestamp, len(payload)) + payload


def test_bridge_pairs_vendor_frames_without_a_zenoh_listener():
    core = BridgeCore(selected_device_id="FAKE", clock_ns=lambda: 100)
    assert core.accept_event(("FAKE", _hand_frame(FRAME_TYPE_HAND_LEFT, 10))) == []
    packets = core.accept_event(("FAKE", _hand_frame(FRAME_TYPE_HAND_RIGHT, 10)))
    assert len(packets) == 1
    assert len(packets[0]) == TRACKING_PACKET_SIZE
    frame = decode_tracking(packets[0])
    assert frame.bridge_monotonic_ns == 100
    assert frame.tracking_epoch == core.epoch


@pytest.mark.parametrize("timestamp", [0, -1, 9_223_372_036_855])
def test_bridge_rejects_non_positive_or_overflowing_pico_timestamp(timestamp):
    core = BridgeCore(selected_device_id="FAKE")
    core.accept_event(("FAKE", _hand_frame(FRAME_TYPE_HAND_LEFT, timestamp)))
    assert core.accept_event(("FAKE", _hand_frame(FRAME_TYPE_HAND_RIGHT, timestamp))) == []
    assert core.status()["invalid_payloads"] == 1


def test_bridge_drops_malformed_callback_event_without_worker_failure():
    core = BridgeCore(selected_device_id="FAKE")
    assert core.accept_event({"device_id": object(), "data": b"bad"}) == []
    assert core.status()["invalid_payloads"] == 1


def test_fake_source_can_pin_selected_device_id(tmp_path, capsys):
    source = tmp_path / "events.jsonl"
    source.write_text("", encoding="utf-8")
    from spd_vr.pxrea_bridge import _run_fake_source

    assert _run_fake_source(
        source,
        publisher=lambda payload: None,
        status_publisher=lambda payload: None,
        device_id="PICO-2",
    ) == 0
    assert json.loads(capsys.readouterr().out)["device_id"] == "PICO-2"


class _FakeFunction:
    def __init__(self):
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return 0


class _FakeLibrary:
    def __init__(self):
        self.PXREAInit = _FakeFunction()
        self.PXREADeinit = _FakeFunction()
        self.PXREASendCustomMessage = _FakeFunction()


def test_pxrea_callback_rejects_oversize_before_copying():
    queue = BoundedCallbackQueue()
    client = PXREAClient(_FakeLibrary(), queue=queue)
    payload = ctypes.create_string_buffer(b"x" * 2049)
    message = PXREADevCustomMessage(
        b"FAKE", 2049, ctypes.cast(payload, ctypes.POINTER(ctypes.c_char))
    )
    client._callback(None, PXREA_DEVICE_CUSTOM, 0, ctypes.byref(message))
    assert queue.qsize() == 0
    assert client.status()["dropped_oversize"] == 1
