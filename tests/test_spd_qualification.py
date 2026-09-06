import hashlib
import json
import os

import pytest

from spd_vr.model_compiler.qualification import (
    QualificationError,
    verify_contact_qualification_receipt,
    write_contact_qualification_receipt,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fast_qualification_accepts_unchanged_bundle_and_rejects_changed_piece(tmp_path):
    output = tmp_path / "generated"
    output.mkdir()
    urdf = tmp_path / "robot.urdf"
    sealed_contents = {
        "model_manifest.yaml": b"model-manifest",
        "collision_manifest.yaml": b"collision-manifest",
        "unified_plant.xml": b"full-model",
        "arm_ik.xml": b"arm-model",
    }
    urdf.write_bytes(b"robot-urdf")
    for relative, content in sealed_contents.items():
        (output / relative).write_bytes(content)
    proxy = output / "collision" / "piece.mesh"
    proxy.parent.mkdir()
    proxy.write_bytes(b"proxy-a")
    proxy_stat = proxy.stat()
    receipt = {
        "schema_version": 1,
        "kind": "spd-vr-contact-qualification",
        "qualified_files": {
            "urdf": {
                "path": str(urdf.resolve()),
                "sha256": _sha256(b"robot-urdf"),
            },
            **{
                relative: {"path": relative, "sha256": _sha256(content)}
                for relative, content in sealed_contents.items()
            },
        },
        "collision_pieces": [
            {
                "path": "collision/piece.mesh",
                "size": proxy_stat.st_size,
                "mtime_ns": proxy_stat.st_mtime_ns,
            }
        ],
        "full_verification": {"contact_records": 1},
    }
    (output / "contact_qualification.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    report = verify_contact_qualification_receipt(output, urdf)
    assert report["collision_pieces"] == 1
    assert report["verification"] == "cached_contact_qualification"

    proxy.write_bytes(b"proxy-b")
    os.utime(
        proxy,
        ns=(proxy_stat.st_atime_ns, proxy_stat.st_mtime_ns + 1_000_000_000),
    )
    with pytest.raises(QualificationError, match="changed since qualification"):
        verify_contact_qualification_receipt(output, urdf)


def test_full_qualification_atomically_writes_a_runtime_receipt(tmp_path, monkeypatch):
    output = tmp_path / "generated"
    output.mkdir()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot/>", encoding="utf-8")
    for name in (
        "model_manifest.yaml",
        "unified_plant.xml",
        "arm_ik.xml",
    ):
        (output / name).write_text(name, encoding="utf-8")
    proxy = output / "collision" / "piece.stl"
    proxy.parent.mkdir()
    proxy.write_bytes(b"mesh")
    (output / "collision_manifest.yaml").write_text(
        json.dumps({"records": [{"pieces": [{"file": "collision/piece.stl"}]}]}),
        encoding="utf-8",
    )

    import spd_vr.model_compiler.qualification as qualification

    monkeypatch.setattr(
        qualification,
        "verify_artifacts",
        lambda manifest, source: type("Verified", (), {"output_dir": output})(),
    )
    monkeypatch.setattr(
        qualification,
        "verify_contact_qualified",
        lambda manifest, **kwargs: {"records": 1, "surface_gate": "passed"},
    )

    receipt = write_contact_qualification_receipt(output, urdf)

    assert receipt == output / "contact_qualification.json"
    assert verify_contact_qualification_receipt(output, urdf)["collision_pieces"] == 1
