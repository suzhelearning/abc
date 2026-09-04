"""Segmentation-aware visual randomization for SPD-VR training frames."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .contracts import CAMERA_NAMES


def randomize_instance_colors(
    rgb: np.ndarray,
    segmentation: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float = 0.65,
) -> np.ndarray:
    """Recolor visible instance IDs while preserving geometry and background.

    MuJoCo segmentation stores object type and object ID in the final axis.
    The object-ID channel is used as the stable key; ID zero is treated as
    background.  A fixed RNG supplied by the Dataset makes this transform
    reproducible across DataLoader workers.
    """
    image = np.asarray(rgb, dtype=np.uint8)
    seg = np.asarray(segmentation, dtype=np.int32)
    if image.ndim == 4:
        if seg.shape != (image.shape[0], image.shape[1], image.shape[2], 2):
            raise ValueError("batched segmentation must have shape [T,H,W,2]")
        return np.stack(
            [
                randomize_instance_colors(image[index], seg[index], rng, strength=strength)
                for index in range(image.shape[0])
            ],
            axis=0,
        )
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("rgb must have shape [H,W,3]")
    if seg.shape != (*image.shape[:2], 2):
        raise ValueError("segmentation must have shape [H,W,2]")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be in [0,1]")
    result = image.copy().astype(np.float32)
    object_ids = seg[..., 1]
    for object_id in np.unique(object_ids):
        if int(object_id) <= 0:
            continue
        mask = object_ids == object_id
        color = rng.uniform(0.0, 255.0, size=(3,)).astype(np.float32)
        result[mask] = (1.0 - strength) * result[mask] + strength * color
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def randomize_camera_frames(
    images: Mapping[str, np.ndarray],
    segmentations: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    *,
    strength: float = 0.65,
) -> dict[str, np.ndarray]:
    if set(images) != set(CAMERA_NAMES) or set(segmentations) != set(CAMERA_NAMES):
        raise ValueError(f"images and segmentations must contain exactly {CAMERA_NAMES}")
    return {
        camera: randomize_instance_colors(
            images[camera], segmentations[camera], rng, strength=strength
        )
        for camera in CAMERA_NAMES
    }


__all__ = ["randomize_camera_frames", "randomize_instance_colors"]
