"""Shared rectified-flow primitives used by ABC and SPD policies."""

from __future__ import annotations

from typing import TypeVar


TensorT = TypeVar("TensorT")


def flow_interpolate(
    data: TensorT,
    noise: TensorT,
    time: TensorT,
    *,
    data_at_one: bool,
) -> tuple[TensorT, TensorT]:
    """Return a point on a linear flow path and its constant velocity.

    ABC's released checkpoint uses ``data_at_one=False`` (data at t=0 and
    noise at t=1).  SPD follows the paper notation and uses
    ``data_at_one=True`` (noise at t=0 and data at t=1).  Keeping the
    direction explicit prevents an easy-to-miss sign error while allowing
    both policies to share the same primitive.
    """

    if data_at_one:
        point = (1 - time) * noise + time * data
        velocity = data - noise
    else:
        point = (1 - time) * data + time * noise
        velocity = noise - data
    return point, velocity


__all__ = ["flow_interpolate"]
