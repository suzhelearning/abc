"""Minimal ctypes adapter for PXREA tracking callbacks."""

from __future__ import annotations

import ctypes
import threading
from collections import deque
from dataclasses import dataclass
from os import PathLike
from typing import Any


class PXREAError(RuntimeError):
    """Raised when the vendor SDK rejects setup or is unavailable."""


class PXREADevCustomMessage(ctypes.Structure):
    _fields_ = [
        ("devID", ctypes.c_char * 32),
        ("dataSize", ctypes.c_uint64),
        ("dataPtr", ctypes.POINTER(ctypes.c_char)),
    ]


class PXREADevStateJson(ctypes.Structure):
    _fields_ = [
        ("devID", ctypes.c_char * 32),
        ("stateJson", ctypes.c_char * 16352),
    ]


CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
)

PXREA_SERVER_CONNECT = 1 << 2
PXREA_SERVER_DISCONNECT = 1 << 3
PXREA_DEVICE_FIND = 1 << 4
PXREA_DEVICE_MISSING = 1 << 5
PXREA_DEVICE_CONNECT = 1 << 9
PXREA_DEVICE_STATE_JSON = 1 << 25
PXREA_DEVICE_CUSTOM = 1 << 26
PXREA_CALLBACK_MASK = (
    PXREA_SERVER_CONNECT
    | PXREA_SERVER_DISCONNECT
    | PXREA_DEVICE_FIND
    | PXREA_DEVICE_MISSING
    | PXREA_DEVICE_CONNECT
    | PXREA_DEVICE_STATE_JSON
    | PXREA_DEVICE_CUSTOM
)
_LIFECYCLE_TYPES = {
    PXREA_SERVER_CONNECT,
    PXREA_SERVER_DISCONNECT,
    PXREA_DEVICE_FIND,
    PXREA_DEVICE_MISSING,
    PXREA_DEVICE_CONNECT,
}


@dataclass(frozen=True)
class CallbackEvent:
    device_id: str
    data: bytes
    event_type: int = PXREA_DEVICE_CUSTOM


class BoundedCallbackQueue:
    """Thread-safe FIFO with at most 64 callback slots and 2 KiB per slot."""

    def __init__(self, max_items: int = 64, max_bytes: int = 2048) -> None:
        if max_items <= 0 or max_bytes <= 0:
            raise ValueError("queue bounds must be positive")
        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self._items: deque[CallbackEvent] = deque()
        self._cv = threading.Condition()
        self.dropped_overflow = 0

    def put(self, event: CallbackEvent) -> bool:
        if not isinstance(event, CallbackEvent):
            raise TypeError("event must be CallbackEvent")
        if len(event.data) > self.max_bytes:
            return False
        with self._cv:
            if len(self._items) >= self.max_items:
                self._items.popleft()
                self.dropped_overflow += 1
            self._items.append(event)
            self._cv.notify()
        return True

    def get(self, timeout: float | None = None) -> CallbackEvent | None:
        with self._cv:
            if not self._items and timeout is not None:
                self._cv.wait(timeout)
            if not self._items:
                return None
            return self._items.popleft()

    def drain(self) -> list[CallbackEvent]:
        with self._cv:
            events = list(self._items)
            self._items.clear()
            return events

    def qsize(self) -> int:
        with self._cv:
            return len(self._items)


class PXREAClient:
    """Own a PXREA callback and its exact init/deinit lifecycle."""

    callback_type = CALLBACK

    def __init__(
        self,
        library: Any,
        *,
        queue: BoundedCallbackQueue | None = None,
        user_data: int | None = None,
        flags: int | None = None,
    ) -> None:
        self.library = library
        self.queue = queue or BoundedCallbackQueue()
        self.user_data = user_data
        self.flags = PXREA_CALLBACK_MASK if flags is None else int(flags)
        self._initialized = False
        self._closed = False
        self._status_lock = threading.Lock()
        self._status_counts = {"dropped_oversize": 0, "dropped_queue": 0}
        self._callback = CALLBACK(self._on_callback)
        self._configure_symbols()

    @classmethod
    def load_library(cls, path: str | bytes | PathLike[str]) -> "PXREAClient":
        try:
            library = ctypes.CDLL(path)
        except OSError as exc:
            raise PXREAError(f"failed to load PXREA library: {path}") from exc
        return cls(library)

    def _configure_symbols(self) -> None:
        try:
            init = self.library.PXREAInit
            deinit = self.library.PXREADeinit
        except AttributeError as exc:
            raise PXREAError(f"missing PXREA symbol: {exc.args[0]}") from exc
        init.argtypes = [ctypes.c_void_p, CALLBACK, ctypes.c_uint]
        init.restype = ctypes.c_int
        deinit.argtypes = []
        deinit.restype = ctypes.c_int
        self._init_fn = init
        self._deinit_fn = deinit

    @property
    def callback(self) -> CALLBACK | None:
        return self._callback

    def __enter__(self) -> "PXREAClient":
        if self._closed:
            raise PXREAError("PXREA client is closed")
        if self._initialized:
            return self
        result = int(self._init_fn(self.user_data, self._callback, self.flags))
        if result != 0:
            raise PXREAError(f"PXREAInit failed: {result}")
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        if self._initialized:
            result = int(self._deinit_fn())
            if result != 0:
                raise PXREAError(f"PXREADeinit failed: {result}")
            self._initialized = False
        self._closed = True
        self._callback = None

    def status(self) -> dict[str, int]:
        with self._status_lock:
            return dict(self._status_counts)

    def _on_callback(
        self,
        _user: ctypes.c_void_p,
        callback_type: int,
        _status: int,
        message_ptr: ctypes.c_void_p,
    ) -> None:
        callback_type = int(callback_type)
        if callback_type in _LIFECYCLE_TYPES:
            self.queue.put(CallbackEvent("", b"", callback_type))
            return
        if not message_ptr:
            return
        if callback_type == PXREA_DEVICE_STATE_JSON:
            message = ctypes.cast(
                message_ptr, ctypes.POINTER(PXREADevStateJson)
            ).contents
            raw = bytes(message.stateJson).split(b"\0", 1)[0]
            device_id = bytes(message.devID).split(b"\0", 1)[0].decode(
                "utf-8", "replace"
            )
            if not self.queue.put(CallbackEvent(device_id, raw, callback_type)):
                with self._status_lock:
                    self._status_counts["dropped_queue"] += 1
            return
        if callback_type != PXREA_DEVICE_CUSTOM:
            return
        message = ctypes.cast(
            message_ptr, ctypes.POINTER(PXREADevCustomMessage)
        ).contents
        size = int(message.dataSize)
        if size > self.queue.max_bytes:
            with self._status_lock:
                self._status_counts["dropped_oversize"] += 1
            return
        if size and not message.dataPtr:
            with self._status_lock:
                self._status_counts["dropped_oversize"] += 1
            return
        raw = ctypes.string_at(message.dataPtr, size)
        device_id = bytes(message.devID).split(b"\0", 1)[0].decode(
            "utf-8", "replace"
        )
        if not self.queue.put(CallbackEvent(device_id, raw, callback_type)):
            with self._status_lock:
                self._status_counts["dropped_queue"] += 1

__all__ = [
    "BoundedCallbackQueue",
    "CALLBACK",
    "CallbackEvent",
    "PXREAClient",
    "PXREADevCustomMessage",
    "PXREADevStateJson",
    "PXREAError",
    "PXREA_CALLBACK_MASK",
    "PXREA_DEVICE_CONNECT",
    "PXREA_DEVICE_CUSTOM",
    "PXREA_DEVICE_FIND",
    "PXREA_DEVICE_MISSING",
    "PXREA_DEVICE_STATE_JSON",
    "PXREA_SERVER_CONNECT",
    "PXREA_SERVER_DISCONNECT",
]
