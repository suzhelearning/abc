"""Recorder compatibility layer backed by the canonical atomic HDF5 writer.

``EpisodeWriter`` is the low-level 60 Hz row writer used by the live viewer.
``EpisodeRecorder`` adds the lifecycle methods needed by
``EpisodeController`` while preserving exactly the same on-disk schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .data import (
    CameraFrame,
    EpisodeWriter,
    RawFrame,
    SCHEMA_VERSION,
    SPDSequenceDataset,
    build_contact_segments,
    build_training_index,
    filter_contact_mask,
    scan_episodes,
    sequence_indices,
    validate_episode,
)


def _episode_path(root: Path, episode_id: str | int) -> Path:
    if root.suffix.lower() in {".h5", ".hdf5"}:
        return root
    return root / "episodes" / str(episode_id) / "episode.hdf5"


class EpisodeRecorder:
    """Lifecycle adapter around :class:`spd_vr.data.EpisodeWriter`.

    Live code should normally call :meth:`append` with a complete ``RawFrame``
    because all streams are required to share one 60 Hz tick.  The explicit
    lifecycle methods make checkpoint/skip control safe without introducing a
    second recorder schema.
    """

    def __init__(
        self,
        output_root: str | Path,
        *,
        overwrite: bool = False,
        manifest_defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.overwrite = bool(overwrite)
        self.manifest_defaults = dict(manifest_defaults or {})
        self._writer: EpisodeWriter | None = None
        self.episode_path: Path | None = None

    @property
    def frame_count(self) -> int:
        return 0 if self._writer is None else self._writer.frame_count

    def start_episode(self, episode_id: str | int, task_manifest: Mapping[str, Any]) -> None:
        if self._writer is not None:
            raise RuntimeError("an episode is already recording")
        path = _episode_path(self.output_root, episode_id)
        manifest = {**self.manifest_defaults, **dict(task_manifest), "episode_id": str(episode_id)}
        self._writer = EpisodeWriter(path, manifest, overwrite=self.overwrite)
        self.episode_path = path

    def append(self, frame: RawFrame) -> None:
        if self._writer is None:
            raise RuntimeError("an episode is not recording")
        self._writer.append(frame)

    append_raw = append

    def submit(self, frame: RawFrame) -> None:
        """Alias used by collection loops that submit one complete raw row."""

        self.append(frame)

    def finish_episode(self) -> Path:
        if self._writer is None:
            raise RuntimeError("no episode is recording")
        writer, self._writer = self._writer, None
        try:
            return writer.finish()
        except Exception:
            writer.abort()
            raise

    def discard_episode(self, reason: str = "operator_skip") -> None:
        del reason  # The reason is kept by the episode state machine/audit log.
        if self._writer is not None:
            self._writer.abort()
            self._writer = None

    def abort(self) -> None:
        self.discard_episode("abort")


def validate_episode_path(path: str | Path, *, verify_checksums: bool = True) -> dict[str, Any]:
    """Resolve an episode file/directory and delegate to the canonical validator."""

    candidate = Path(path)
    if candidate.is_dir():
        direct = candidate / "episode.hdf5"
        if direct.is_file():
            candidate = direct
        else:
            matches = sorted((*candidate.glob("*.h5"), *candidate.glob("*.hdf5")))
            if len(matches) != 1:
                raise FileNotFoundError(f"expected one episode HDF5 under {candidate}")
            candidate = matches[0]
    return validate_episode(candidate, verify_checksums=verify_checksums)


__all__ = [
    "CameraFrame",
    "EpisodeRecorder",
    "EpisodeWriter",
    "RawFrame",
    "SCHEMA_VERSION",
    "SPDSequenceDataset",
    "build_contact_segments",
    "build_training_index",
    "filter_contact_mask",
    "scan_episodes",
    "sequence_indices",
    "validate_episode",
    "validate_episode_path",
]
