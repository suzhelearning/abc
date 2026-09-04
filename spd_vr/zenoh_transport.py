from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any, Generic, TypeVar, cast

import zenoh


T = TypeVar("T")


class LatestSample(Generic[T]):
    def __init__(self) -> None:
        self._lock = Lock()
        self._value: T | None = None
        self._generation = 0
        self._consumed_generation = 0
        self._dropped = 0

    def put(self, value: T) -> None:
        with self._lock:
            if self._value is not None and self._generation > self._consumed_generation:
                self._dropped += 1
            self._value = value
            self._generation += 1
    def take_new(self, last_generation: int) -> tuple[int, T] | None:
        with self._lock:
            if self._value is None or self._generation <= last_generation:
                return None
            self._consumed_generation = self._generation
            return self._generation, cast(T, self._value)
    def invalidate(self) -> None:
        with self._lock:
            self._value = None
            self._generation += 1
            self._consumed_generation = self._generation

    @property
    def storage_size(self) -> int:
        with self._lock:
            return int(self._value is not None)

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped


def peer_config(*, listen: bool, endpoint: str) -> zenoh.Config:
    config = zenoh.Config()
    config.insert_json5("mode", '"router"' if listen else '"client"')
    config.insert_json5("scouting/multicast/enabled", "false")
    config.insert_json5("listen/endpoints", f'["{endpoint}"]' if listen else "[]")
    config.insert_json5("connect/endpoints", "[]" if listen else f'["{endpoint}"]')
    return config

CONTROL_CONGESTION_CONTROL = zenoh.CongestionControl.BLOCK


class ZenohNode:
    def __init__(self, config: zenoh.Config) -> None:
        self._session: zenoh.Session | None = zenoh.open(config)
        self._subscribers: list[zenoh.Subscriber] = []
        self._publishers: list[zenoh.Publisher] = []

    def __enter__(self) -> ZenohNode:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
    def declare_publisher(
        self,
        key: str,
        *,
        congestion_control: Any | None = None,
        reliability: Any | None = None,
        express: bool | None = None,
    ) -> zenoh.Publisher:
        if self._session is None:
            raise RuntimeError("ZenohNode is closed")
        kwargs: dict[str, Any] = {}
        if congestion_control is not None:
            kwargs["congestion_control"] = congestion_control
        if reliability is not None:
            kwargs["reliability"] = reliability
        if express is not None:
            kwargs["express"] = express
        publisher = self._session.declare_publisher(key, **kwargs)
        self._publishers.append(publisher)
        return publisher

    def declare_latest_subscriber(
        self,
        key: str,
        decoder: Callable[[bytes], T],
        mailbox: LatestSample[T],
    ) -> zenoh.Subscriber:
        if self._session is None:
            raise RuntimeError("ZenohNode is closed")

        def callback(sample: zenoh.Sample) -> None:
            # A malformed wire sample is a dropped sample, not a process
            # failure.  The consumer's last-safe target remains in place and
            # the next valid sequence can still be accepted.
            try:
                value = decoder(bytes(sample.payload))
            except (TypeError, ValueError, OSError):
                return
            mailbox.put(value)

        subscriber = self._session.declare_subscriber(key, callback)
        self._subscribers.append(subscriber)
        return subscriber

    def close(self) -> None:
        if self._session is None:
            return
        session, self._session = self._session, None
        subscribers, self._subscribers = self._subscribers, []
        publishers, self._publishers = self._publishers, []
        first_error: Exception | None = None

        for resource in (*reversed(subscribers), *reversed(publishers)):
            try:
                resource.undeclare()
            except Exception as error:
                if first_error is None:
                    first_error = error
        try:
            session.close()
        except Exception as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error
