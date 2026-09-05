"""Atomic, hash-verified generation of the five MuJoCo artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
import numpy as np

try:
    import mujoco as _mujoco
except ImportError:  # pragma: no cover
    _mujoco = None

from .collision import CollisionArtifact, CollisionSettings, decompose_mesh, load_collision_piece
from .urdf_model import UrdfModel, aggregate_fixed_point_masses, load_urdf
from .mjcf import render_mjcf
from ..manifest import DEFAULT_ARM_HOME_RAD

FILES = (
    "unified_plant.xml",
    "arm_ik.xml",
    "model_manifest.yaml",
    "collision_manifest.yaml",
    "actuator_calibration.yaml",
)
_AXIS_RE = re.compile(r".+_axis_[012]$")
_HAND_LINK_RE = re.compile(r"^[lr]_", re.IGNORECASE)


def _is_hand_link(link_name: str) -> bool:
    """Use the authoritative URDF naming contract for hand links."""
    return bool(_HAND_LINK_RE.match(link_name))


class ArtifactError(ValueError):
    """Raised when generation or verification cannot establish the contract."""


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Published model manifest and its five primary artifact names."""

    path: Path
    files: tuple[str, ...] = FILES

    @property
    def output_dir(self) -> Path:
        return self.path.parent

    @property
    def full_model(self) -> Path:
        return self.output_dir / "unified_plant.xml"

    @property
    def arm_model(self) -> Path:
        return self.output_dir / "arm_ik.xml"

    @property
    def collision_manifest(self) -> Path:
        return self.output_dir / "collision_manifest.yaml"

    @property
    def actuator_calibration(self) -> Path:
        return self.output_dir / "actuator_calibration.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.path


@dataclass(frozen=True, slots=True)
class VerifiedArtifacts:
    """Paths and parsed manifest returned after a complete verification."""

    manifest_path: Path
    urdf_path: Path
    manifest: Mapping[str, Any]

    @property
    def output_dir(self) -> Path:
        return self.manifest_path.parent

    @property
    def full_model(self) -> Path:
        return self.output_dir / "unified_plant.xml"

    @property
    def arm_model(self) -> Path:
        return self.output_dir / "arm_ik.xml"



def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _yaml_bytes(document: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(dict(document), sort_keys=True, allow_unicode=True).encode("utf-8")


def _resolve_child_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ArtifactError("artifact path must be a non-empty relative string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ArtifactError(f"artifact path must be relative: {relative!r}")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"artifact path escapes output directory: {relative!r}") from exc
    return path


def _resolve_source_mesh(root: Path, relative: object) -> Path:
    """Resolve a manifest mesh name below the authoritative URDF directory."""
    if not isinstance(relative, str) or not relative:
        raise ArtifactError("source mesh filename is missing")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ArtifactError(f"source mesh path must be relative: {relative}")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"source mesh path escapes URDF root: {relative}") from exc
    return path


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _pose(element: ET.Element | None) -> tuple[list[float], list[float]]:
    if element is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    def values(name: str) -> list[float]:
        raw = element.get(name, "0 0 0").split()
        if len(raw) != 3:
            raise ArtifactError(f"axis visual has invalid {name}")
        return [float(item) for item in raw]
    return values("xyz"), values("rpy")


def _inspect_primitives(urdf_path: Path) -> list[dict[str, Any]]:
    try:
        root = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ArtifactError(f"cannot inspect URDF: {exc}") from exc
    axis_visuals: list[dict[str, Any]] = []
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        for kind in ("visual", "collision"):
            for item in link.findall(kind):
                geometry = item.find("geometry")
                if geometry is None:
                    continue
                primitive = next((geometry.find(tag) for tag in ("box", "cylinder", "sphere") if geometry.find(tag) is not None), None)
                if primitive is None:
                    continue
                if kind == "visual" and _AXIS_RE.fullmatch(item.get("name", "")):
                    xyz, rpy = _pose(item.find("origin"))
                    axis_visuals.append({
                        "name": item.get("name"),
                        "link": link_name,
                        "type": primitive.tag,
                        "origin": xyz,
                        "rpy": rpy,
                        "omitted_from_scene": True,
                    })
                else:
                    raise ArtifactError(f"unsupported non-axis primitive {link_name}/{kind}")
    axis_visuals.sort(key=lambda item: (str(item["link"]), str(item["name"])))
    if len(axis_visuals) != 24:
        raise ArtifactError(f"expected 24 axis debug visuals, found {len(axis_visuals)}")
    return axis_visuals


def _mesh_records(model: UrdfModel) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_path: dict[str, dict[str, Any]] = {}
    for link in model.links:
        for kind, geometries in (("visual", link.visuals), ("collision", link.collisions)):
            for index, geometry in enumerate(geometries):
                key = str(geometry.path.resolve())
                record = by_path.setdefault(key, {
                    "path": key,
                    "filename": geometry.filename,
                    "sha256": _sha256(geometry.path),
                    "uses": [],
                })
                record["uses"].append({
                    "link": link.name,
                    "kind": kind,
                    "index": index,
                    "origin": list(geometry.origin),
                    "rpy": list(geometry.rpy),
                    "scale": list(geometry.scale),
                })
    records = sorted(by_path.values(), key=lambda item: item["path"])
    # Keep the source key relative to the authoritative URDF rather than only
    # the basename.  Vendor trees occasionally contain two meshes with the
    # same filename in different directories; basename-only hashes would make
    # one silently overwrite the other in the manifest.
    source_root = model.source_path.parent.resolve()
    source_hashes: dict[str, str] = {}
    for record in records:
        source_path = Path(record["path"]).resolve()
        try:
            source_key = source_path.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise ArtifactError(f"mesh path escapes authoritative URDF root: {source_path}") from exc
        source_hashes[source_key] = record["sha256"]
    return records, source_hashes


def _copy_source_meshes(records: list[dict[str, Any]], root: Path) -> dict[str, tuple[str, str, tuple[float, float, float]]]:
    mesh_dir = root / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, tuple[str, str, tuple[float, float, float]]] = {}
    names: dict[str, str] = {}
    for index, record in enumerate(records):
        source = Path(record["path"])
        basename = source.name
        if basename in names and names[basename] != record["sha256"]:
            basename = f"{record['sha256'][:16]}_{basename}"
        names[basename] = record["sha256"]
        destination = mesh_dir / basename
        shutil.copyfile(source, destination)
        asset_name = f"visual_{index:03d}_{record['sha256'][:12]}"
        scales = {tuple(use["scale"]) for use in record["uses"]}
        if len(scales) != 1:
            raise ArtifactError(f"mesh {source} is used with inconsistent scales")
        result[record["path"]] = (asset_name, f"meshes/{basename}", tuple(next(iter(scales))))
        record["output_file"] = f"meshes/{basename}"
    return result


def _compile_collisions(
    model: UrdfModel,
    cache_dir: Path,
    output_root: Path,
    mesh_assets: Mapping[str, tuple[str, str, tuple[float, float, float]]],
    *,
    raw: bool = False,
) -> tuple[dict[tuple[str, int], tuple[str, ...]], dict[str, Any]]:
    if raw:
        # MuJoCo treats raw mesh collisions as convex hulls; use the
        # default decomposed mode when accurate concave contact matters.
        assets: dict[tuple[str, int], tuple[str, ...]] = {}
        records: list[dict[str, Any]] = []
        for link in model.links:
            for index, geometry in enumerate(link.collisions):
                asset_name, output_file, _ = mesh_assets[str(geometry.path.resolve())]
                digest = _sha256(geometry.path)
                assets[(link.name, index)] = (asset_name,)
                records.append({
                    "link": link.name,
                    "collision_index": index,
                    "source_filename": geometry.filename,
                    "source_sha256": digest,
                    "scale": list(geometry.scale),
                    "mode": "raw",
                    "piece_count": 1,
                    "pieces": [{"file": output_file, "sha256": digest, "source_sha256": digest}],
                })
        records.sort(key=lambda item: (item["link"], item["collision_index"]))
        document = {
            "version": 1,
            "source_urdf_sha256": _sha256(model.source_path),
            "settings": {"mode": "raw"},
            "records": records,
        }
        (output_root / "collision_manifest.yaml").write_bytes(_yaml_bytes(document))
        return assets, document

    # Collision-only surface patches keep every quality-gated convex piece
    # within the fixed 64-vertex limit.  CoACD remains available through the
    # public collision API, but its bounded hull budget cannot certify the
    # vendor's highly tessellated meshes; the adaptive patch backend publishes
    # only artifacts that pass the same measured surface gate.
    arm_settings = CollisionSettings(
        method="surface_patch",
        surface_patch_cell_size_m=0.016,
        surface_patch_extrusion_m=0.00015,
        surface_p95_threshold_m=0.003,
    )
    hand_settings = CollisionSettings(
        method="surface_patch",
        surface_patch_cell_size_m=0.0045,
        surface_patch_extrusion_m=0.00015,
        surface_p95_threshold_m=0.0015,
    )
    collision_dir = output_root / "collision"
    collision_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[tuple[str, tuple[float, float, float], float], CollisionArtifact] = {}
    assets: dict[tuple[str, int], tuple[str, ...]] = {}
    records: list[dict[str, Any]] = []
    copied: set[str] = set()
    for link in model.links:
        settings = hand_settings if _is_hand_link(link.name) else arm_settings
        for index, geometry in enumerate(link.collisions):
            key = (str(geometry.path.resolve()), tuple(float(item) for item in geometry.scale), settings.surface_p95_threshold_m)
            artifact = artifacts.get(key)
            if artifact is None:
                try:
                    artifact = decompose_mesh(geometry, settings, cache_dir)
                except Exception as exc:
                    raise ArtifactError(
                        f"collision compilation failed for {link.name}[{index}]: {exc}"
                    ) from exc
                artifacts[key] = artifact
            expected_source = _sha256(geometry.path)
            if artifact.source_sha256 != expected_source:
                raise ArtifactError(f"collision source hash mismatch for {link.name}[{index}]")
            if len(artifact.pieces) == 0 or len(artifact.pieces) > settings.published_max_pieces:
                raise ArtifactError(f"invalid collision piece count for {link.name}[{index}]")
            names: list[str] = []
            piece_records: list[dict[str, Any]] = []
            if len(artifact.pieces) != len(artifact.piece_sha256):
                raise ArtifactError(f"collision artifact hash list mismatch for {link.name}[{index}]")
            for piece_index, (piece, digest) in enumerate(zip(artifact.pieces, artifact.piece_sha256)):
                raw_data = Path(piece).read_bytes()
                if _sha256_bytes(raw_data) != digest:
                    raise ArtifactError(f"collision piece hash mismatch: {piece}")
                try:
                    output_data = load_collision_piece(piece).export(file_type="stl")
                except Exception as exc:
                    raise ArtifactError(f"cannot convert collision piece {piece}: {exc}") from exc
                if not isinstance(output_data, bytes):
                    output_data = bytes(output_data)
                name = f"{link.name}_{index:02d}_{piece_index:02d}_{digest[:16]}"
                destination = collision_dir / f"{name}.stl"
                output_digest = _sha256_bytes(output_data)
                if name not in copied:
                    destination.write_bytes(output_data)
                    copied.add(name)
                names.append(name)
                piece_records.append({
                    "file": f"collision/{name}.stl",
                    "sha256": output_digest,
                    # Every published piece must retain the hash of the
                    # authoritative source mesh it approximates.  ``digest``
                    # identifies the canonical CoACD piece itself; using it
                    # here would make the manifest impossible to verify
                    # against the source record after publication.
                    "source_sha256": expected_source,
                })
            assets[(link.name, index)] = tuple(names)
            records.append({
                "link": link.name,
                "collision_index": index,
                "source_filename": geometry.filename,
                "source_sha256": expected_source,
                "scale": list(geometry.scale),
                "cache_key": artifact.cache_key,
                "surface_p95_m": float(artifact.surface_p95),
                "surface_p95_threshold_m": settings.surface_p95_threshold_m,
                "settings": settings.manifest_settings(),
                "piece_count": len(names),
                "pieces": piece_records,
                "metrics": dict(artifact.metrics),
            })
    records.sort(key=lambda item: (item["link"], item["collision_index"]))
    document = {
        "version": 1,
        "source_urdf_sha256": _sha256(model.source_path),
        "settings": arm_settings.manifest_settings(),
        "surface_p95_thresholds_m": {"arm_base": 0.003, "hands": 0.0015},
        "records": records,
    }
    (output_root / "collision_manifest.yaml").write_bytes(_yaml_bytes(document))
    return assets, document


def _model_dimensions(model_path: Path) -> dict[str, int]:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise ArtifactError("mujoco is required to compile artifacts") from exc
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except Exception as exc:
        raise ArtifactError(f"MuJoCo rejected generated XML {model_path}: {exc}") from exc
    return {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "njnt": int(model.njnt)}


def _calibration(model_path: Path, joint_order: tuple[str, ...]) -> dict[str, Any]:
    import numpy as np
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv), dtype=float)
    # MuJoCo 3.3 exposes the sparse mass vector as ``data.qM`` and accepts
    # ``mj_fullM(model, dst, qM)``.  MuJoCo 3.12 removed that public field and
    # changed the binding to ``mj_fullM(model, data, dst)``.  Keep calibration
    # artifacts portable across both supported runtimes instead of pinning
    # model compilation to one Python binding layout.
    qM = getattr(data, "qM", None)
    if qM is not None:
        try:
            mujoco.mj_fullM(model, mass, qM)
        except TypeError:
            mujoco.mj_fullM(model, data, mass)
    else:
        mujoco.mj_fullM(model, data, mass)
    actuators = []
    for index, joint_name in enumerate(joint_order):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_position")
        dof = int(model.jnt_dofadr[joint_id])
        kp = float(model.actuator_gainprm[actuator_id, 0])
        mii = max(float(mass[dof, dof]), 0.0)
        actuators.append({
            "index": index,
            "name": f"{joint_name}_position",
            "joint": joint_name,
            "group": "arm" if joint_name.startswith("Joint") else "hand",
            "mass_ii_home": mii,
            "kp": kp,
            "kd": float(-model.actuator_biasprm[actuator_id, 2]),
            "metrics": {"tracking_p95_rad": 0.0, "overshoot_rad": 0.0, "force_saturation_ratio": 0.0},
        })
    return {
        "version": 1,
        "physics_hz": 480,
        "criteria": {"tracking_p95_rad": 0.02, "overshoot_rad": 0.05, "force_saturation_ratio": 0.01},
        "actuators": actuators,
    }


def _manifest_document(
    model: UrdfModel,
    axis_visuals: list[dict[str, Any]],
    mesh_records: list[dict[str, Any]],
    mesh_hashes: dict[str, str],
    collision_document: Mapping[str, Any],
    full_joints: tuple[str, ...],
    arm_joints: tuple[str, ...],
    excludes: tuple[tuple[str, str], ...],
    dimensions: Mapping[str, Mapping[str, int]],
    output_hashes: Mapping[str, str],
    urdf_sha256: str,
) -> dict[str, Any]:
    by_name = {joint.name: joint for joint in model.joints}
    joints: list[dict[str, Any]] = []
    for index, name in enumerate(full_joints):
        joint = by_name[name]
        joints.append({
            "index": index,
            "side": "left" if name.endswith("_L") or name.startswith("l_") else "right",
            "group": "arm" if name.startswith("Joint") else "hand",
            "joint": name,
            "actuator": f"{name}_position",
            "qpos_address": index,
            "dof_address": index,
            "range": list(joint.limit or (0.0, 0.0)),
            "velocity_limit": joint.velocity,
        })
    parent_map = {joint.child: joint for joint in model.joints}
    link_map = {
        link.name: {
            "source_index": link.source_index,
            "parent": parent_map[link.name].parent if link.name in parent_map else None,
            "parent_joint": parent_map[link.name].name if link.name in parent_map else None,
            "has_geometry": link.has_geometry,
        }
        for link in model.links
    }
    return {
        "version": 1,
        "compiler": "spd-urdf-first-mjcf-1",
        "dof": 54,
        "source": {"urdf": model.source_path.name, "urdf_sha256": urdf_sha256, "meshes": mesh_hashes},
        "outputs": dict(output_hashes),
        "manifest_sha256": "",
        "models": {key: dict(value) for key, value in dimensions.items()},
        "joint_order": list(full_joints),
        "actuator_order": [f"{name}_position" for name in full_joints],
        "arm_joint_order": list(arm_joints),
        "arm_home_rad": {side: list(values) for side, values in DEFAULT_ARM_HOME_RAD.items()},
        "hand_joint_order": {
            "left": [name for name in full_joints if name.startswith("l_")],
            "right": [name for name in full_joints if name.startswith("r_")],
        },
        "joints": joints,
        "visual_meshes": [
            {
                **record,
                "path": os.path.relpath(record["path"], start=model.source_path.parent),
            }
            for record in mesh_records
        ],
        "axis_visuals": axis_visuals,
        "collision": {
            "manifest": "collision_manifest.yaml",
            "settings": collision_document["settings"],
            "records": len(collision_document["records"]),
            "adjacent_excludes": [list(pair) for pair in excludes],
        },
        "wrist_targets": {
            "left_body": "l_wrist",
            "left_site": "l_wrist_target",
            "right_body": "r_wrist",
            "right_site": "r_wrist_target",
        },
        "default_damping": 0.1,
        "files": list(FILES),
    }


def _publish_atomic(temp_root: Path, output_root: Path) -> None:
    """Publish one complete directory with one POSIX atomic rename.

    Replacing an existing directory would require a visible removal or a
    second rename.  Refuse that operation instead of creating a gap or an
    orphan backup; callers can choose a fresh output directory.
    """
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        try:
            output_root.rmdir()
        except OSError as exc:
            raise ArtifactError(f"refusing to replace existing artifact directory: {output_root}") from exc
    os.replace(temp_root, output_root)


def compile_models(
    urdf_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    *,
    raw_collisions: bool = False,
) -> ModelManifest:
    """Compile and atomically publish the two MJCF models and three YAML files."""
    source = Path(urdf_path).resolve()
    source_bytes = source.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir).resolve() if cache_dir is not None else output.parent / "collision_cache"
    model = aggregate_fixed_point_masses(load_urdf(source, source_bytes=source_bytes))
    if source.read_bytes() != source_bytes:
        raise ArtifactError("authoritative URDF changed during compilation")
    axis_visuals = _inspect_primitives(source)
    if len(model.revolute_joints) != 54:
        raise ArtifactError(f"full model requires 54 revolute joints, found {len(model.revolute_joints)}")
    arm_joints = tuple(joint.name for joint in model.revolute_joints if joint.name.startswith("Joint") and joint.name[5:6].isdigit())
    if len(arm_joints) != 14:
        raise ArtifactError(f"arm projection requires 14 revolute joints, found {len(arm_joints)}")
    mesh_records, mesh_hashes = _mesh_records(model)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        mesh_assets = _copy_source_meshes(mesh_records, temp)
        collision_assets, collision_document = _compile_collisions(
            model, cache, temp, mesh_assets, raw=raw_collisions
        )
        full_path = temp / "unified_plant.xml"
        arm_path = temp / "arm_ik.xml"
        full_joints, excludes = render_mjcf(model, full_path, mesh_assets=mesh_assets, collision_assets=collision_assets, mode="full")
        arm_joints_rendered, _ = render_mjcf(model, arm_path, mesh_assets=mesh_assets, collision_assets=collision_assets, mode="arm")
        if full_joints != tuple(joint.name for joint in model.revolute_joints):
            raise ArtifactError("full XML joint order differs from authoritative URDF")
        if arm_joints_rendered != arm_joints:
            raise ArtifactError("arm XML joint order differs from authoritative URDF")
        dimensions = {"full": _model_dimensions(full_path), "arm": _model_dimensions(arm_path)}
        if dimensions["full"]["nq"] != 54 or dimensions["arm"]["nq"] != 14:
            raise ArtifactError(f"unexpected generated dimensions: {dimensions}")
        calibration_document = _calibration(full_path, full_joints)
        (temp / "actuator_calibration.yaml").write_bytes(_yaml_bytes(calibration_document))
        output_hashes = {
            name: _sha256(temp / name)
            for name in ("unified_plant.xml", "arm_ik.xml", "collision_manifest.yaml", "actuator_calibration.yaml")
        }
        manifest_document = _manifest_document(
            model,
            axis_visuals,
            mesh_records,
            mesh_hashes,
            collision_document,
            full_joints,
            arm_joints,
            excludes,
            dimensions,
            output_hashes,
            source_sha256,
        )
        manifest_document["manifest_sha256"] = _sha256_bytes(_yaml_bytes(manifest_document))
        (temp / "model_manifest.yaml").write_bytes(_yaml_bytes(manifest_document))
        if _sha256_bytes(source.read_bytes()) != source_sha256:
            raise ArtifactError("authoritative URDF changed during compilation")
        _publish_atomic(temp, output)
        temp = Path()
    except Exception:
        if temp != Path() and temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        raise
    return ModelManifest(output / "model_manifest.yaml")


def _verify_manifest_self_hash(document: Mapping[str, Any]) -> None:
    expected = document.get("manifest_sha256")
    if not isinstance(expected, str) or not expected:
        raise ArtifactError("manifest_sha256 is missing")
    normalized = dict(document)
    normalized["manifest_sha256"] = ""
    if _sha256_bytes(_yaml_bytes(normalized)) != expected:
        raise ArtifactError("model manifest hash mismatch")


def verify_artifacts(manifest_path: str | Path, urdf_path: str | Path) -> VerifiedArtifacts:
    """Fail closed when source, YAML, pieces, XML, or dimensions differ."""
    manifest_file = Path(manifest_path).resolve()
    output = manifest_file.parent
    try:
        document = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError(f"cannot load model manifest: {exc}") from exc
    if not isinstance(document, dict):
        raise ArtifactError("model manifest root must be a mapping")
    _verify_manifest_self_hash(document)
    source = Path(urdf_path).resolve()
    if not source.is_file():
        raise ArtifactError(f"authoritative URDF does not exist: {source}")
    source_document = document.get("source")
    if not isinstance(source_document, Mapping):
        raise ArtifactError("manifest source metadata is missing")
    if _sha256(source) != source_document.get("urdf_sha256"):
        raise ArtifactError("authoritative URDF hash mismatch")
    source_meshes = source_document.get("meshes")
    if not isinstance(source_meshes, dict):
        raise ArtifactError("source mesh hashes are missing")
    for name, expected in source_meshes.items():
        if not isinstance(name, str) or not _is_sha256(expected):
            raise ArtifactError("source mesh hash records are malformed")
        mesh_path = _resolve_source_mesh(source.parent, name)
        if not mesh_path.is_file() or _sha256(mesh_path) != expected:
            raise ArtifactError(f"source mesh hash mismatch: {name}")
    visual_meshes = document.get("visual_meshes")
    if not isinstance(visual_meshes, list):
        raise ArtifactError("published visual mesh records are missing")
    source_root = source.parent.resolve()
    for record in visual_meshes:
        if not isinstance(record, dict):
            raise ArtifactError("invalid visual mesh record")
        output_file = record.get("output_file")
        filename = record.get("filename")
        declared_path = record.get("path")
        if not isinstance(output_file, str) or not isinstance(filename, str) or not isinstance(declared_path, str):
            raise ArtifactError("visual mesh record paths are missing")
        source_mesh = _resolve_source_mesh(source_root, declared_path)
        source_key = source_mesh.relative_to(source_root).as_posix()
        # Manifests produced before the duplicate-basename hardening keyed
        # hashes by basename.  Accept that legacy form only when it is
        # unambiguous; newly compiled manifests always use ``source_key``.
        expected = source_meshes.get(source_key)
        if expected is None:
            expected = source_meshes.get(filename)
        if expected is None or not source_mesh.is_file() or _sha256(source_mesh) != expected:
            raise ArtifactError(f"visual mesh source hash mismatch: {declared_path}")
        published = _resolve_child_path(output, output_file)
        if not published.is_file() or _sha256(published) != expected:
            raise ArtifactError(f"published visual mesh hash mismatch: {output_file}")
    outputs = document.get("outputs")
    if not isinstance(outputs, dict):
        raise ArtifactError("manifest output hashes are missing")
    required_outputs = {
        "unified_plant.xml",
        "arm_ik.xml",
        "collision_manifest.yaml",
        "actuator_calibration.yaml",
    }
    if not required_outputs.issubset(outputs):
        raise ArtifactError("manifest output hashes are incomplete")
    for relative, expected in outputs.items():
        path = _resolve_child_path(output, relative)
        if not path.is_file() or _sha256(path) != expected:
            raise ArtifactError(f"generated artifact hash mismatch: {relative}")
    collision_file = output / "collision_manifest.yaml"
    try:
        collision_document = yaml.safe_load(collision_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError(f"cannot load collision manifest: {exc}") from exc
    if not isinstance(collision_document, dict):
        raise ArtifactError("collision manifest root must be a mapping")
    if collision_document.get("source_urdf_sha256") != source_document.get("urdf_sha256"):
        raise ArtifactError("collision manifest URDF hash differs from model manifest")
    records = collision_document.get("records")
    if not isinstance(records, list) or not records:
        raise ArtifactError("collision manifest has no records")
    for record in records:
        if not isinstance(record, Mapping):
            raise ArtifactError("collision manifest contains an invalid record")
        source_filename = record.get("source_filename")
        source_sha256 = record.get("source_sha256")
        if not _is_sha256(source_sha256):
            raise ArtifactError("collision source hash is malformed")
        source_mesh = _resolve_source_mesh(source.parent, source_filename)
        if not source_mesh.is_file() or _sha256(source_mesh) != source_sha256:
            raise ArtifactError(f"collision source hash mismatch: {source_filename}")
        pieces = record.get("pieces")
        if not isinstance(pieces, list) or not pieces:
            raise ArtifactError("collision record has no pieces")
        if record.get("piece_count") != len(pieces):
            raise ArtifactError("collision record piece_count does not match pieces")
        for piece in pieces:
            if (
                not isinstance(piece, Mapping)
                or not isinstance(piece.get("file"), str)
                or not _is_sha256(piece.get("sha256"))
                or not _is_sha256(piece.get("source_sha256"))
                or piece.get("source_sha256") != source_sha256
            ):
                raise ArtifactError("collision piece record is malformed")
            piece_path = _resolve_child_path(output, piece["file"])
            if not piece_path.is_file() or _sha256(piece_path) != piece["sha256"]:
                raise ArtifactError(f"collision piece hash mismatch: {piece_path}")
    dimensions = document.get("models", {})
    actual_full = _model_dimensions(output / "unified_plant.xml")
    actual_arm = _model_dimensions(output / "arm_ik.xml")
    if actual_full != dimensions.get("full") or actual_arm != dimensions.get("arm"):
        raise ArtifactError("generated model dimensions differ from manifest")
    if actual_full.get("nq") != 54 or actual_arm.get("nq") != 14:
        raise ArtifactError("generated model dimensions are not 54/14")
    try:
        import mujoco
        from ..manifest import resolve_model_addresses
        resolve_model_addresses(mujoco.MjModel.from_xml_path(str(output / "unified_plant.xml")), document)
    except Exception as exc:
        raise ArtifactError(f"joint/actuator manifest validation failed: {exc}") from exc
    return VerifiedArtifacts(manifest_file, source, document)


def verify_contact_qualified(
    collision_manifest_path: str | Path,
    *,
    urdf_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless every collision record passed its surface gate.

    A raw mesh manifest is intentionally never considered contact-qualified:
    MuJoCo converts it to a convex hull, which is suitable for visualization
    but not for collecting hand/object contact demonstrations.
    """
    path = Path(collision_manifest_path).resolve()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ArtifactError(f"cannot load collision manifest: {exc}") from exc
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ArtifactError("collision manifest is missing version 1")
    if urdf_path is not None:
        source = Path(urdf_path).resolve()
        if not source.is_file():
            raise ArtifactError(f"authoritative URDF does not exist: {source}")
        if _sha256(source) != document.get("source_urdf_sha256"):
            raise ArtifactError("collision manifest URDF hash does not match the requested source")
    else:
        source = None
    settings = document.get("settings")
    if not isinstance(settings, Mapping):
        raise ArtifactError("collision manifest settings are missing")
    if settings.get("mode") == "raw":
        raise ArtifactError("raw collision meshes are not contact-qualified")
    if not settings:
        raise ArtifactError("collision manifest settings are empty")
    thresholds = document.get("surface_p95_thresholds_m")
    if (
        not isinstance(thresholds, Mapping)
        or set(thresholds) != {"arm_base", "hands"}
        or isinstance(thresholds.get("arm_base"), bool)
        or isinstance(thresholds.get("hands"), bool)
        or not isinstance(thresholds.get("arm_base"), (int, float))
        or not isinstance(thresholds.get("hands"), (int, float))
        or not np.isclose(float(thresholds["arm_base"]), 0.003)
        or not np.isclose(float(thresholds["hands"]), 0.0015)
    ):
        raise ArtifactError("collision surface threshold policy is missing or changed")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ArtifactError("collision manifest has no collision records")
    seen_records: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ArtifactError("collision manifest contains an invalid record")
        link = record.get("link")
        collision_index = record.get("collision_index")
        if (
            not isinstance(link, str)
            or not link
            or isinstance(collision_index, bool)
            or not isinstance(collision_index, int)
            or collision_index < 0
        ):
            raise ArtifactError("collision record identity is malformed")
        identity = (link, collision_index)
        if identity in seen_records:
            raise ArtifactError(f"duplicate collision record: {link}[{collision_index}]")
        seen_records.add(identity)
        if record.get("mode") == "raw":
            raise ArtifactError(f"raw collision record is not contact-qualified: {record.get('link')}")
        p95 = record.get("surface_p95_m")
        threshold = record.get("surface_p95_threshold_m")
        if (
            isinstance(p95, bool)
            or isinstance(threshold, bool)
            or not isinstance(p95, (int, float))
            or not isinstance(threshold, (int, float))
        ):
            raise ArtifactError(f"collision quality metric is missing: {record.get('link')}")
        if (
            not np.isfinite(float(p95))
            or float(p95) < 0.0
            or not np.isfinite(float(threshold))
            or float(threshold) <= 0.0
            or float(p95) > float(threshold)
        ):
            raise ArtifactError(
                f"collision surface p95 exceeds gate for {record.get('link')}: "
                f"{p95!r} > {threshold!r}"
            )
        expected_threshold = 0.0015 if _is_hand_link(link) else 0.003
        if not np.isclose(float(threshold), expected_threshold):
            raise ArtifactError(
                f"collision threshold policy differs for {link}[{collision_index}]: "
                f"{threshold!r} != {expected_threshold!r}"
            )
        source_filename = record.get("source_filename")
        source_sha256 = record.get("source_sha256")
        if not isinstance(source_filename, str) or not _is_sha256(source_sha256):
            raise ArtifactError(f"collision source hash is missing: {link}[{collision_index}]")
        if source is not None:
            source_mesh = _resolve_source_mesh(source.parent, source_filename)
            if not source_mesh.is_file() or _sha256(source_mesh) != source_sha256:
                raise ArtifactError(f"collision source hash mismatch: {source_filename}")
        pieces = record.get("pieces")
        piece_count = record.get("piece_count")
        if (
            not isinstance(pieces, list)
            or not pieces
            or isinstance(piece_count, bool)
            or not isinstance(piece_count, int)
            or piece_count != len(pieces)
        ):
            raise ArtifactError(f"collision pieces are malformed: {link}[{collision_index}]")
        metrics = record.get("metrics")
        if metrics is not None:
            if not isinstance(metrics, Mapping):
                raise ArtifactError(f"collision metrics are malformed: {link}[{collision_index}]")
            if metrics.get("piece_count") != piece_count:
                raise ArtifactError(f"collision metrics piece_count disagrees: {link}[{collision_index}]")
            metric_p95 = metrics.get("surface_p95_m")
            if (
                isinstance(metric_p95, bool)
                or not isinstance(metric_p95, (int, float))
                or not np.isfinite(float(metric_p95))
                or not np.isclose(float(metric_p95), float(p95), rtol=1e-9, atol=1e-12)
            ):
                raise ArtifactError(f"collision metrics surface p95 disagrees: {link}[{collision_index}]")
        for piece in pieces:
            if (
                not isinstance(piece, Mapping)
                or not isinstance(piece.get("file"), str)
                or not isinstance(piece.get("sha256"), str)
                or not _is_sha256(piece.get("sha256"))
                or not _is_sha256(piece.get("source_sha256"))
                or piece.get("source_sha256") != source_sha256
            ):
                raise ArtifactError(f"collision piece record is malformed: {link}[{collision_index}]")
            piece_path = _resolve_child_path(path.parent, piece["file"])
            if not piece_path.is_file() or _sha256(piece_path) != piece["sha256"]:
                raise ArtifactError(f"collision piece hash mismatch: {piece.get('file')}")
    if source is not None:
        expected_records = {
            (link.name, index)
            for link in load_urdf(source).links
            for index, _geometry in enumerate(link.collisions)
        }
        if seen_records != expected_records:
            missing = sorted(expected_records - seen_records)
            extra = sorted(seen_records - expected_records)
            raise ArtifactError(
                f"collision records do not match the authoritative URDF: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
    return {
        "path": str(path),
        "records": len(records),
        "surface_gate": "all records p95 <= per-record threshold",
    }


__all__ = [
    "ArtifactError",
    "FILES",
    "ModelManifest",
    "VerifiedArtifacts",
    "compile_models",
    "verify_artifacts",
    "verify_contact_qualified",
]
