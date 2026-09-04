"""Deterministic, fail-closed CoACD collision compilation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import threading
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Iterable
import coacd
import fcntl
import numpy as np
import trimesh

from .urdf_model import MeshGeometry




class CollisionError(RuntimeError):
    """Raised when a collision artifact cannot be safely compiled or loaded."""


@dataclass(frozen=True, slots=True)
class CollisionSettings:
    """The complete, reproducible CoACD invocation and quality policy."""

    seed: int = 0
    max_pieces: int = 16
    max_vertices: int = 64
    threshold: float = 0.05
    preprocess_mode: str = "auto"
    preprocess_resolution: int = 50
    resolution: int = 2000
    mcts_nodes: int = 20
    mcts_iterations: int = 150
    mcts_max_depth: int = 3
    pca: bool = False
    merge: bool = True
    decimate: bool = False
    extrude: bool = False
    extrude_margin: float = 0.01
    apx_mode: str = "ch"
    real_metric: bool = False
    surface_samples: int = 2048
    surface_p95_threshold_m: float = 0.0015
    _extra_coacd_params: tuple[tuple[str, Any], ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        if self.seed != 0 or self.max_pieces != 16 or self.max_vertices != 64:
            raise ValueError("collision builds require seed=0, 16 pieces, and 64 vertices")
        overridden = {"seed", "max_convex_hull", "max_ch_vertex"} & dict(self._extra_coacd_params).keys()
        if overridden:
            raise ValueError(f"cannot override fixed CoACD parameters: {sorted(overridden)}")
        if self.surface_samples <= 0 or not np.isfinite(self.surface_p95_threshold_m):
            raise ValueError("surface quality settings must be finite and positive")
        if self.surface_p95_threshold_m <= 0:
            raise ValueError("surface_p95_threshold_m must be positive")

    def coacd_kwargs(self) -> dict[str, Any]:
        """Return exactly the keyword arguments accepted by CoACD 1.0.14."""
        params = {
            "threshold": self.threshold,
            "max_convex_hull": self.max_pieces,
            "preprocess_mode": self.preprocess_mode,
            "preprocess_resolution": self.preprocess_resolution,
            "resolution": self.resolution,
            "mcts_nodes": self.mcts_nodes,
            "mcts_iterations": self.mcts_iterations,
            "mcts_max_depth": self.mcts_max_depth,
            "pca": self.pca,
            "merge": self.merge,
            "decimate": self.decimate,
            "max_ch_vertex": self.max_vertices,
            "extrude": self.extrude,
            "extrude_margin": self.extrude_margin,
            "apx_mode": self.apx_mode,
            "seed": self.seed,
            "real_metric": self.real_metric,
        }
        params.update(dict(self._extra_coacd_params))
        return params


@dataclass(frozen=True, slots=True)
class CollisionArtifact:
    cache_key: str
    pieces: tuple[Path, ...]
    piece_sha256: tuple[str, ...]
    surface_p95: float
    source_sha256: str
    scale: tuple[float, float, float]
    cache_hit: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def piece_paths(self) -> tuple[Path, ...]:
        return self.pieces

    @property
    def piece_metrics(self) -> dict[str, Any]:
        return self.metrics

    @property
    def quality_p95(self) -> float:
        return self.surface_p95
    @property
    def source_mesh_sha256(self) -> str:
        return self.source_sha256

_KEY_LOCKS: dict[str, threading.Lock] = {}
_KEY_LOCKS_GUARD = threading.Lock()


def _cache_lock(root: Path, key: str):
    with _KEY_LOCKS_GUARD:
        thread_lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    return _CacheLock(root / f".{key}.lock", thread_lock)


class _CacheLock:
    def __init__(self, path: Path, thread_lock: threading.Lock) -> None:
        self.path = path
        self.thread_lock = thread_lock
        self.handle = None

    def __enter__(self):
        self.thread_lock.acquire()
        try:
            self.handle = self.path.open("a+")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            self.thread_lock.release()
            raise
        return self

    def __exit__(self, *_exc):
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        finally:
            self.thread_lock.release()


_MESH_MAGIC = b"SPD_COLLISION_MESH_V1\n"
_MANIFEST = "manifest.json"


def _coacd_version() -> str:
    try:
        return importlib.metadata.version("coacd")
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(coacd, "__version__", "unknown"))


def _locked_decompose(func):
    @wraps(func)
    def wrapped(mesh, settings, cache_root):
        source, source_hash, scale = _source_and_mesh(mesh)
        key = _cache_key(source_hash, scale, settings)
        root = Path(cache_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with _cache_lock(root, key):
            cached = _artifact_from_cache(final_dir=root / key, key=key, source=source,
                                          source_hash=source_hash, scale=scale, settings=settings)
            if cached is not None:
                return cached
            return func(mesh, settings, root, _prepared=(source, source_hash, scale))
    return wrapped


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_and_mesh(mesh: MeshGeometry | trimesh.Trimesh) -> tuple[trimesh.Trimesh, str, tuple[float, float, float]]:
    if isinstance(mesh, MeshGeometry):
        path = mesh.resolved_path.resolve()
        try:
            source_bytes = path.read_bytes()
            source_hash = _hash_bytes(source_bytes)
            file_type = path.suffix.lstrip(".") or None
            loaded = trimesh.load_mesh(
                io.BytesIO(source_bytes), file_type=file_type, force="mesh", process=False
            )
            if _hash_bytes(path.read_bytes()) != source_hash:
                raise CollisionError("source mesh changed while loading")
        except CollisionError:
            raise
        except Exception as exc:
            raise CollisionError(f"cannot load source mesh {path}: {exc}") from exc
        scale = tuple(float(item) for item in mesh.scale)
    elif isinstance(mesh, trimesh.Trimesh):
        loaded = mesh.copy()
        scale = (1.0, 1.0, 1.0)
        source_hash = _hash_bytes(_canonical_mesh_bytes(loaded))
    else:
        raise TypeError("mesh must be MeshGeometry or trimesh.Trimesh")
    vertices = np.asarray(loaded.vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise CollisionError("source mesh must have vertices (N,3) and triangular faces (M,3)")
    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
        raise CollisionError("source mesh is empty or non-finite")
    if (faces < 0).any() or (faces >= len(vertices)).any():
        raise CollisionError("source mesh has out-of-range face indices")
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not np.isfinite(source.volume) or abs(float(source.volume)) <= 0:
        raise CollisionError("source mesh has non-positive volume")
    return source, source_hash, scale

def _canonical_mesh_arrays(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype="<f8")
    raw_faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise CollisionError("CoACD piece vertices must have shape (N,3)")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not np.isfinite(raw_faces).all():
        raise CollisionError("CoACD piece faces must be finite triangles")
    faces = np.asarray(raw_faces, dtype=np.int64)
    if not np.array_equal(raw_faces, faces):
        raise CollisionError("CoACD piece face indices must be integers")
    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
        raise CollisionError("CoACD piece is empty or non-finite")
    if (faces < 0).any() or (faces >= len(vertices)).any():
        raise CollisionError("CoACD piece has out-of-range face indices")
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    canonical_vertices = np.ascontiguousarray(vertices[order], dtype="<f8")
    # Preserve winding while normalizing cyclic triangle start indices.
    remapped_faces = remap[faces]
    starts = np.argmin(remapped_faces, axis=1)
    offsets = (starts[:, None] + np.arange(3)[None, :]) % 3
    canonical_faces = np.take_along_axis(remapped_faces, offsets, axis=1)
    face_order = np.lexsort((canonical_faces[:, 2], canonical_faces[:, 1], canonical_faces[:, 0]))
    return canonical_vertices, np.ascontiguousarray(canonical_faces[face_order], dtype="<i4")
def _canonical_mesh_bytes(mesh: trimesh.Trimesh | tuple[np.ndarray, np.ndarray]) -> bytes:
    if isinstance(mesh, trimesh.Trimesh):
        vertices, faces = _canonical_mesh_arrays(mesh.vertices, mesh.faces)
    else:
        vertices, faces = _canonical_mesh_arrays(*mesh)
    return (
        _MESH_MAGIC
        + struct.pack("<QQ", len(vertices), len(faces))
        + vertices.tobytes(order="C")
        + faces.tobytes(order="C")
    )


def _read_mesh_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    if not data.startswith(_MESH_MAGIC) or len(data) < len(_MESH_MAGIC) + 16:
        raise CollisionError("invalid collision piece")
    offset = len(_MESH_MAGIC)
    vertex_count, face_count = struct.unpack_from("<QQ", data, offset)
    offset += 16
    vertex_bytes = vertex_count * 3 * 8
    face_bytes = face_count * 3 * 4
    if len(data) != offset + vertex_bytes + face_bytes:
        raise CollisionError("truncated collision piece")
    vertices = np.frombuffer(data, dtype="<f8", count=vertex_count * 3, offset=offset).reshape((-1, 3)).copy()
    offset += vertex_bytes
    faces = np.frombuffer(data, dtype="<i4", count=face_count * 3, offset=offset).reshape((-1, 3)).copy()
    return vertices, faces


def _sample_surface(mesh: trimesh.Trimesh, samples: int, rng: np.random.Generator) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if len(triangles) == 0 or not np.isfinite(areas).all() or float(areas.sum()) <= 0:
        raise CollisionError("cannot sample a degenerate surface")
    probabilities = areas / areas.sum()
    selected = rng.choice(len(triangles), size=samples, p=probabilities)
    uv = rng.random((samples, 2))
    over = (uv[:, 0] + uv[:, 1]) > 1.0
    uv[over] = 1.0 - uv[over]
    a = triangles[selected, 0]
    b = triangles[selected, 1]
    c = triangles[selected, 2]
    return a + uv[:, :1] * (b - a) + uv[:, 1:] * (c - a)


def _distance_to_mesh(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    try:
        _, distance, _ = trimesh.proximity.closest_point(mesh, points)
    except Exception:
        # trimesh's naive implementation avoids an optional rtree dependency.
        _, distance, _ = trimesh.proximity.closest_point_naive(mesh, points)
    distance = np.asarray(distance, dtype=np.float64)
    if not np.isfinite(distance).all():
        raise CollisionError("surface distance is non-finite")
    return distance


def bidirectional_surface_p95(
    source: trimesh.Trimesh | tuple[np.ndarray, np.ndarray],
    pieces: Iterable[trimesh.Trimesh | tuple[np.ndarray, np.ndarray]],
    samples: int,
) -> float:
    """Return the worst (p95) of source→proxy and proxy→source distances."""
    if samples <= 0:
        raise ValueError("samples must be positive")

    def as_mesh(item: trimesh.Trimesh | tuple[np.ndarray, np.ndarray]) -> trimesh.Trimesh:
        if isinstance(item, trimesh.Trimesh):
            return item
        vertices, faces = _canonical_mesh_arrays(*item)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    source_mesh = as_mesh(source)
    proxy_parts = tuple(as_mesh(piece) for piece in pieces)
    if not proxy_parts:
        raise CollisionError("collision proxy is empty")
    proxy = trimesh.util.concatenate(proxy_parts)
    rng = np.random.default_rng(0)
    source_points = _sample_surface(source_mesh, samples, rng)
    proxy_points = _sample_surface(proxy, samples, rng)
    source_to_proxy = np.percentile(_distance_to_mesh(source_points, proxy), 95)
    proxy_to_source = np.percentile(_distance_to_mesh(proxy_points, source_mesh), 95)
    return float(max(source_to_proxy, proxy_to_source))


def _cache_key(source_hash: str, scale: tuple[float, float, float], settings: CollisionSettings) -> str:
    payload = {
        "source_mesh_sha256": source_hash,
        "scale": scale,
        "coacd_version": _coacd_version(),
        "seed": settings.seed,
        "max_pieces": settings.max_pieces,
        "max_vertices": settings.max_vertices,
        "coacd_params": settings.coacd_kwargs(),
        "surface_samples": settings.surface_samples,
        "surface_p95_threshold_m": settings.surface_p95_threshold_m,
    }
    return _hash_bytes(_canonical_json(payload).encode("utf-8"))


def _artifact_from_cache(
    final_dir: Path,
    key: str,
    source: trimesh.Trimesh,
    source_hash: str,
    scale: tuple[float, float, float],
    settings: CollisionSettings,
) -> CollisionArtifact | None:
    manifest_path = final_dir / _MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("cache_key") != key
            or manifest.get("source_mesh_sha256") != source_hash
            or tuple(manifest.get("scale", ())) != scale
            or manifest.get("coacd_version") != _coacd_version()
            or manifest.get("settings") != settings.coacd_kwargs()
            or int(manifest.get("surface_samples", settings.surface_samples)) != settings.surface_samples
        ):
            return None
        records = manifest["pieces"]
        if not isinstance(records, list) or not 1 <= len(records) <= settings.max_pieces:
            return None
        paths: list[Path] = []
        hashes: list[str] = []
        proxy_parts: list[trimesh.Trimesh] = []
        for index, item in enumerate(records):
            filename = item["file"]
            digest = item["sha256"]
            path = (final_dir / filename).resolve()
            if path.parent != final_dir.resolve() or filename != f"piece_{index:02d}_{digest[:16]}.mesh":
                return None
            data = path.read_bytes()
            if _hash_bytes(data) != digest:
                return None
            vertices, faces = _read_mesh_bytes(data)
            if len(vertices) > settings.max_vertices or not np.isfinite(vertices).all():
                return None
            piece = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            volume = abs(float(piece.volume))
            if not np.isfinite(volume) or volume <= 0 or not np.isclose(volume, float(item["volume"])):
                return None
            proxy_parts.append(piece)
            paths.append(path)
            hashes.append(digest)
        if hashes != sorted(hashes):
            return None
        metrics = manifest["metrics"]
        if not isinstance(metrics, dict):
            return None
        if (
            metrics.get("piece_count") != len(records)
            or not np.isfinite(float(manifest["surface_p95"]))
            or float(manifest["surface_p95"]) > settings.surface_p95_threshold_m
            or not np.isclose(float(metrics["surface_p95_m"]), float(manifest["surface_p95"]))
        ):
            return None
        measured = bidirectional_surface_p95(source, proxy_parts, settings.surface_samples)
        if not np.isclose(measured, float(manifest["surface_p95"]), rtol=1e-9, atol=1e-12):
            return None
        return CollisionArtifact(
            cache_key=key,
            pieces=tuple(paths),
            piece_sha256=tuple(hashes),
            surface_p95=measured,
            source_sha256=source_hash,
            scale=scale,
            cache_hit=True,
            metrics=dict(metrics),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, CollisionError):
        return None


@_locked_decompose
def decompose_mesh(
    mesh: MeshGeometry | trimesh.Trimesh,
    settings: CollisionSettings,
    cache_root: str | Path,
    *,
    _prepared=None,
) -> CollisionArtifact:
    """Compile one validated source mesh, or raise without any fallback proxy."""
    if _prepared is None:
        source, source_hash, scale = _source_and_mesh(mesh)
    else:
        source, source_hash, scale = _prepared
    key = _cache_key(source_hash, scale, settings)
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / key
    cached = _artifact_from_cache(final_dir, key, source, source_hash, scale, settings)
    if cached is not None:
        return cached
    if final_dir.exists():
        shutil.rmtree(final_dir)

    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        result = coacd.run_coacd(
            coacd.Mesh(
                vertices=np.ascontiguousarray(source.vertices, dtype=np.float64),
                indices=np.ascontiguousarray(source.faces, dtype=np.int32),
            ),
            **settings.coacd_kwargs(),
        )
    except Exception as exc:
        raise CollisionError(f"CoACD decomposition failed: {exc}") from exc

    if not result:
        raise CollisionError("CoACD returned empty decomposition")
    canonical: list[tuple[str, bytes, np.ndarray, np.ndarray, float]] = []
    for piece in result:
        try:
            vertices, faces = _canonical_mesh_arrays(piece[0], piece[1])
            if len(vertices) > settings.max_vertices:
                raise CollisionError("CoACD piece exceeds max vertices")
            piece_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            volume = abs(float(piece_mesh.volume))
            if not np.isfinite(volume) or volume <= 0:
                raise CollisionError("CoACD piece has non-positive volume")
            data = _canonical_mesh_bytes((vertices, faces))
            canonical.append((_hash_bytes(data), data, vertices, faces, volume))
        except CollisionError:
            raise
        except Exception as exc:
            raise CollisionError(f"invalid CoACD output: {exc}") from exc
    if len(canonical) > settings.max_pieces:
        raise CollisionError("CoACD output exceeds max pieces")
    canonical.sort(key=lambda item: item[0])
    proxy_meshes = [trimesh.Trimesh(vertices=item[2], faces=item[3], process=False) for item in canonical]
    surface_p95 = bidirectional_surface_p95(source, proxy_meshes, settings.surface_samples)
    if surface_p95 > settings.surface_p95_threshold_m:
        raise CollisionError(
            f"collision surface p95 {surface_p95:.9g} m exceeds {settings.surface_p95_threshold_m:.9g} m"
        )

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
    try:
        piece_records = []
        for index, (digest, data, vertices, faces, volume) in enumerate(canonical):
            filename = f"piece_{index:02d}_{digest[:16]}.mesh"
            (temp_dir / filename).write_bytes(data)
            piece_records.append({"file": filename, "sha256": digest, "vertices": len(vertices), "volume": volume})
        manifest = {
            "cache_key": key,
            "source_mesh_sha256": source_hash,
            "scale": scale,
            "coacd_version": _coacd_version(),
            "settings": settings.coacd_kwargs(),
            "surface_p95": surface_p95,
            "metrics": {"piece_count": len(canonical), "surface_p95_m": surface_p95},
            "pieces": piece_records,
        }
        manifest_bytes = (_canonical_json(manifest) + "\n").encode("utf-8")
        manifest_path = temp_dir / _MANIFEST
        manifest_path.write_bytes(manifest_bytes)
        with manifest_path.open("rb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_dir, final_dir)
        temp_dir = Path()
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as exc:
        if temp_dir != Path() and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise CollisionError(f"cannot publish collision cache: {exc}") from exc

    return CollisionArtifact(
        cache_key=key,
        pieces=tuple(final_dir / item["file"] for item in piece_records),
        piece_sha256=tuple(item["sha256"] for item in piece_records),
        surface_p95=surface_p95,
        source_sha256=source_hash,
        scale=scale,
        cache_hit=False,
        metrics=dict(manifest["metrics"]),
    )


def load_collision_piece(path: str | Path) -> trimesh.Trimesh:
    """Load a deterministic cache piece for downstream MJCF generation."""
    vertices, faces = _read_mesh_bytes(Path(path).read_bytes())
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
