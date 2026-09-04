"""Explicit, auditable visual and left/right symmetry augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import math
from typing import Any, Mapping

import numpy as np

from .contracts import CAMERA_NAMES, ROBOT_DOF


@dataclass(frozen=True, slots=True)
class SymmetrySpec:
    """A calibrated mirror transform for the canonical 54-D joint contract.

    The transform is intentionally data-driven.  Joint-axis signs cannot be
    guessed safely from names, so a project must provide a verified
    permutation/sign table before enabling augmentation.  Image frames are
    horizontally mirrored and the wrist-camera streams are exchanged.
    """

    permutation: tuple[int, ...]
    sign: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.permutation, (list, tuple)) or not isinstance(self.sign, (list, tuple)):
            raise ValueError("symmetry permutation and sign must be arrays")
        if (
            len(self.permutation) != ROBOT_DOF
            or any(isinstance(value, bool) or not isinstance(value, Integral) for value in self.permutation)
            or sorted(int(value) for value in self.permutation) != list(range(ROBOT_DOF))
        ):
            raise ValueError("symmetry permutation must contain each canonical index exactly once")
        if (
            len(self.sign) != ROBOT_DOF
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) not in {-1.0, 1.0}
                for value in self.sign
            )
        ):
            raise ValueError("symmetry signs must be +/-1 for all 54 joints")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SymmetrySpec":
        try:
            raw_permutation = value["permutation"]
            raw_sign = value["sign"]
            if not isinstance(raw_permutation, (list, tuple)) or not isinstance(raw_sign, (list, tuple)):
                raise ValueError("permutation and sign must be arrays")
            if any(
                isinstance(item, bool) or not isinstance(item, Integral)
                for item in raw_permutation
            ):
                raise ValueError("permutation entries must be integers")
            if any(
                isinstance(item, bool) or not isinstance(item, Real)
                for item in raw_sign
            ):
                raise ValueError("sign entries must be numbers")
            permutation = tuple(int(item) for item in raw_permutation)
            sign = tuple(float(item) for item in raw_sign)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("symmetry spec requires permutation and sign arrays") from exc
        return cls(permutation, sign)

    def to_mapping(self) -> dict[str, list[int] | list[float]]:
        return {"permutation": list(self.permutation), "sign": list(self.sign)}

    def joints(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        if array.shape[-1] != ROBOT_DOF:
            raise ValueError(f"joint array must end in {ROBOT_DOF}")
        return array[..., self.permutation] * np.asarray(self.sign, dtype=array.dtype)

    def cameras(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(images) != set(CAMERA_NAMES):
            raise ValueError(f"images must contain exactly {CAMERA_NAMES}")
        # Mirror the workspace and exchange the two wrist viewpoints.  Copy
        # so callers can safely mutate the augmented sample.
        return {
            "top": np.flip(np.asarray(images["top"]), axis=-2).copy(),
            "left_wrist": np.flip(np.asarray(images["right_wrist"]), axis=-2).copy(),
            "right_wrist": np.flip(np.asarray(images["left_wrist"]), axis=-2).copy(),
        }


def augment_trajectory(
    qpos: np.ndarray,
    previous_action: np.ndarray,
    future_action: np.ndarray,
    images: Mapping[str, np.ndarray],
    spec: SymmetrySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Apply one calibrated mirror consistently to state, labels, and RGB."""
    return (
        spec.joints(qpos),
        spec.joints(previous_action),
        spec.joints(future_action),
        spec.cameras(images),
    )


__all__ = ["SymmetrySpec", "augment_trajectory"]
