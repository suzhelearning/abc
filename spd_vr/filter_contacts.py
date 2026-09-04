"""Offline contact-gap filtering for SPD-VR HDF5 episodes.

Filtering is deliberately non-destructive: all 60 Hz raw rows remain in the
published file.  The derived ``training/contact_eligible`` mask and
``training/segments_30hz`` index tell :class:`spd_vr.data.SPDSequenceDataset`
which windows are legal, while the manifest records every removed span.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import h5py
import numpy as np

from .data import (
    NO_CONTACT_LIMIT_NS,
    build_contact_segments,
    filter_contact_mask,
    validate_episode,
    _digest_bytes,
    _json,
)


class ContactFilterError(ValueError):
    """Raised when an episode cannot be filtered without losing provenance."""


def _dataset_sha256(dataset: h5py.Dataset) -> str:
    digest = hashlib.sha256()
    if dataset.ndim == 0:
        digest.update(_digest_bytes(dataset[()]))
    else:
        for row in dataset:
            digest.update(_digest_bytes(row))
    return digest.hexdigest()


def filter_episode(
    input_h5: str | Path,
    output_h5: str | Path,
    *,
    threshold_ns: int = NO_CONTACT_LIMIT_NS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a provenance-preserving episode with contact-safe train segments."""
    input_path, output_path = Path(input_h5), Path(output_h5)
    if input_path.resolve() == output_path.resolve():
        raise ContactFilterError("input and output episodes must be different files")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    source_manifest = validate_episode(input_path, verify_checksums=True)
    with h5py.File(input_path, "r") as source:
        timestamps = source["raw/timestamp_ns"][:]
        contact = source["raw/contacts/hand_object"][:]
        index = source["training/index_30hz"][:]
        grid = source["training/grid_step"][:]
        raw_keep, removed_spans = filter_contact_mask(
            timestamps, contact, threshold_ns=threshold_ns
        )
        eligible = raw_keep[index]
        segments = build_contact_segments(grid, eligible)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, staging_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.staging-", dir=output_path.parent
        )
        os.close(fd)
        staging = Path(staging_name)
        try:
            with h5py.File(staging, "w") as target:
                for name, value in source.attrs.items():
                    target.attrs[name] = value
                for name in source:
                    source.copy(name, target, name=name)
                training = target.require_group("training")
                for name in ("contact_eligible", "segments_30hz"):
                    if name in training:
                        del training[name]
                training.create_dataset("contact_eligible", data=eligible, dtype=np.bool_)
                training.create_dataset("segments_30hz", data=segments, dtype=np.int64)

                manifest_raw = target["manifest/json"][()]
                manifest = json.loads(
                    manifest_raw.decode()
                    if isinstance(manifest_raw, bytes)
                    else manifest_raw
                )
                manifest["contact_filter"] = {
                    "threshold_ns": int(threshold_ns),
                    "raw_removed_frames": int(np.count_nonzero(~raw_keep)),
                    "removed_spans": removed_spans,
                    "training_eligible_frames": int(np.count_nonzero(eligible)),
                    "training_segments": int(segments.shape[0]),
                    "derived_from": str(input_path.resolve()),
                }
                checksums: dict[str, str] = {}
                def collect(name: str, value: h5py.Dataset | h5py.Group) -> None:
                    if name != "manifest/json" and isinstance(value, h5py.Dataset):
                        checksums[name] = _dataset_sha256(value)
                target.visititems(collect)
                manifest["dataset_sha256"] = checksums
                del target["manifest/json"]
                target.create_dataset("manifest/json", data=_json(manifest))
                target.flush()
            os.replace(staging, output_path)
        except Exception:
            staging.unlink(missing_ok=True)
            raise
    # Validate after the atomic rename, including every copied/raw checksum.
    validate_episode(output_path, verify_checksums=True)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_raw_frames": int(source_manifest["raw_frames"]),
        "raw_removed_frames": int(np.count_nonzero(~raw_keep)),
        "training_eligible_frames": int(np.count_nonzero(eligible)),
        "training_segments": int(segments.shape[0]),
        "removed_spans": removed_spans,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", type=Path)
    parser.add_argument("output_h5", type=Path)
    parser.add_argument(
        "--threshold-seconds", type=float, default=NO_CONTACT_LIMIT_NS / 1e9
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold_seconds) or args.threshold_seconds < 0:
        parser.error("--threshold-seconds must be finite and non-negative")
    report = filter_episode(
        args.input_h5,
        args.output_h5,
        threshold_ns=int(round(args.threshold_seconds * 1e9)),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


__all__ = ["ContactFilterError", "filter_episode", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
