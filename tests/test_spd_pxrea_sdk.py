import ctypes
import gc
import weakref

import pytest

from spd_vr.pxrea_sdk import (
    BoundedCallbackQueue,
    CallbackEvent,
    PXREAClient,
    PXREADevCustomMessage,
    PXREADevStateJson,
    PXREAError,
    PXREA_CALLBACK_MASK,
    PXREA_DEVICE_CONNECT,
    PXREA_DEVICE_CUSTOM,
    PXREA_DEVICE_STATE_JSON,
)


class _FakeFunction:
    def __init__(self, result=0):
        self.result = result
        self.argtypes = None
        self.restype = None
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


class _FakeLibrary:
    def __init__(self, init_result=0, deinit_result=0):
        self.PXREAInit = _FakeFunction(init_result)
        self.PXREADeinit = _FakeFunction(deinit_result)


def test_custom_message_matches_vendor_sdk_abi():
    assert PXREADevCustomMessage.devID.offset == 0
    assert PXREADevCustomMessage.dataSize.offset == 32
    assert PXREADevCustomMessage.dataPtr.offset == 40
    assert ctypes.sizeof(PXREADevCustomMessage) == 48


def test_state_json_matches_vendor_sdk_abi():
    assert PXREADevStateJson.devID.offset == 0
    assert PXREADevStateJson.stateJson.offset == 32
    assert ctypes.sizeof(PXREADevStateJson) == 16384


def test_init_uses_full_tracking_callback_mask():
    library = _FakeLibrary()
    client = PXREAClient(library)
    assert client.flags == PXREA_CALLBACK_MASK
    assert PXREA_CALLBACK_MASK & PXREA_DEVICE_CUSTOM
    assert PXREA_CALLBACK_MASK & PXREA_DEVICE_STATE_JSON
    with client:
        assert library.PXREAInit.calls[0][2] == PXREA_CALLBACK_MASK
        assert library.PXREAInit.argtypes == [
            ctypes.c_void_p,
            client.callback_type,
            ctypes.c_uint,
        ]
        assert library.PXREAInit.restype is ctypes.c_int
        assert library.PXREADeinit.argtypes == []
        assert library.PXREADeinit.restype is ctypes.c_int


def test_init_failure_does_not_call_deinit():
    library = _FakeLibrary(init_result=7)
    client = PXREAClient(library)
    with pytest.raises(PXREAError, match="PXREAInit failed: 7"):
        client.__enter__()
    assert library.PXREADeinit.calls == []


def test_context_deinits_once_and_keeps_callback_alive_until_close():
    library = _FakeLibrary()
    client = PXREAClient(library)
    with client:
        callback_ref = weakref.ref(client.callback)
        assert callback_ref() is not None
    assert client.callback is None
    client.close()
    gc.collect()
    assert len(library.PXREAInit.calls) == 1
    assert len(library.PXREADeinit.calls) == 1


def test_deinit_failure_keeps_client_open_for_retry():
    library = _FakeLibrary(deinit_result=9)
    client = PXREAClient(library)
    client.__enter__()
    with pytest.raises(PXREAError, match="PXREADeinit failed: 9"):
        client.close()
    assert client.callback is not None
    assert client._initialized is True
    library.PXREADeinit.result = 0
    client.close()
    assert client.callback is None
    assert len(library.PXREADeinit.calls) == 2


def test_callback_copies_state_json_and_lifecycle_events():
    queue = BoundedCallbackQueue(max_bytes=16352)
    client = PXREAClient(_FakeLibrary(), queue=queue)
    message = PXREADevStateJson(b"FAKE", b'{"value":"{}"}')
    client._callback(None, PXREA_DEVICE_STATE_JSON, 0, ctypes.byref(message))
    assert queue.get() == CallbackEvent(
        "FAKE", b'{"value":"{}"}', PXREA_DEVICE_STATE_JSON
    )
    client._callback(None, PXREA_DEVICE_CONNECT, 0, None)
    lifecycle = queue.get()
    assert lifecycle == CallbackEvent("", b"", PXREA_DEVICE_CONNECT)


def test_callback_rejects_oversize_before_copying():
    queue = BoundedCallbackQueue()
    client = PXREAClient(_FakeLibrary(), queue=queue)
    oversized = ctypes.create_string_buffer(b"x" * 2049)
    message = PXREADevCustomMessage(
        b"FAKE", 2049, ctypes.cast(oversized, ctypes.POINTER(ctypes.c_char))
    )
    client._callback(None, PXREA_DEVICE_CUSTOM, 0, ctypes.byref(message))
    assert queue.qsize() == 0
    assert client.status()["dropped_oversize"] == 1


def test_queue_drops_oldest_on_slot_overflow():
    queue = BoundedCallbackQueue(max_items=64, max_bytes=2048)
    for index in range(65):
        assert queue.put(CallbackEvent("FAKE", bytes([index])))
    assert queue.qsize() == 64
    assert queue.get().data == b"\x01"
    assert queue.dropped_overflow == 1
