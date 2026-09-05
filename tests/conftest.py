"""Shared pytest fixtures for tests that need local vendor assets."""

import os
from pathlib import Path

import pytest

from spd_vr.config import TeleopConfig


@pytest.fixture
def vendor_urdf() -> Path:
    """Return an authorized local URDF or skip asset-dependent tests.

    The public branch intentionally excludes Tianji/Wuji2 bytes.  CI or a
    researcher with an authorized bundle can point tests at another copy with
    ``SPD_VR_TEST_URDF``; a clean checkout should still run all non-vendor
    contract tests instead of failing during collection/runtime.
    """

    path = Path(os.environ.get("SPD_VR_TEST_URDF", TeleopConfig().urdf_path)).expanduser()
    if not path.is_file():
        pytest.skip(
            "requires a local authorized Tianji-Wuji2 URDF; "
            "set SPD_VR_TEST_URDF to run this asset-dependent test"
        )
    return path.resolve()
