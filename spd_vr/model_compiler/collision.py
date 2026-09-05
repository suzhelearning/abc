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
from scipy.spatial import ConvexHull, QhullError

from .urdf_model import MeshGeometry




class CollisionError(RuntimeError):
    """Raised when a collision artifact cannot be safely compiled or loaded."""


@dataclass(frozen=True, slots=True)
class CollisionSettings:
    """The complete, reproducible CoACD invocation and quality policy."""

    method: str = "coacd"
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
    surface_patch_cell_size_m: float | None = None
    surface_patch_extrusion_m: float = 0.00015
    surface_patch_max_pieces: int = 20_000
    surface_patch_refinements: int = 3
    _extra_coacd_params: tuple[tuple[str, Any], ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        if self.method not in {"coacd", "surface_patch"}:
            raise ValueError("collision method must be 'coacd' or 'surface_patch'")
        if self.seed != 0 or self.max_pieces != 16 or self.max_vertices != 64:
            raise ValueError("CoACD builds require seed=0, 16 pieces, and 64 vertices")
        overridden = {"seed", "max_convex_hull", "max_ch_vertex"} & dict(self._extra_coacd_params).keys()
        if overridden:
            raise ValueError(f"cannot override fixed CoACD parameters: {sorted(overridden)}")
        if self.surface_samples <= 0 or not np.isfinite(self.surface_p95_threshold_m):
            raise ValueError("surface quality settings must be finite and positive")
        if self.surface_p95_threshold_m <= 0:
            raise ValueError("surface_p95_threshold_m must be positive")
        if self.surface_patch_cell_size_m is not None and (
            not np.isfinite(self.surface_patch_cell_size_m) or self.surface_patch_cell_size_m <= 0
        ):
            raise ValueError("surface_patch_cell_size_m must be positive and finite")
        if not np.isfinite(self.surface_patch_extrusion_m) or self.surface_patch_extrusion_m <= 0:
            raise ValueError("surface_patch_extrusion_m must be positive and finite")
        if self.surface_patch_max_pieces <= 0 or self.surface_patch_refinements < 0:
            raise ValueError("surface patch limits must be positive")

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

    def manifest_settings(self) -> dict[str, Any]:
        """Return the full deterministic backend policy stored in manifests."""
        return {
            "method": self.method,
            "coacd": self.coacd_kwargs(),
            "surface_patch": {
                "cell_size_m": self.surface_patch_cell_size_m,
                "extrusion_m": self.surface_patch_extrusion_m,
                "max_pieces": self.surface_patch_max_pieces,
                "refinements": self.surface_patch_refinements,
            },
        }

    @property
    def published_max_pieces(self) -> int:
        return self.surface_patch_max_pieces if self.method == "surface_patch" else self.max_pieces


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


def _convex_hull_piece(points: np.ndarray, max_vertices: int) -> trimesh.Trimesh | None:
    """Build an oriented convex piece, or return ``None`` for a degenerate hull."""
    try:
        hull = ConvexHull(points, qhull_options="Qc Qx Q12")
    except QhullError:
        return None
    if len(hull.vertices) > max_vertices:
        return None
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[hull.vertices] = np.arange(len(hull.vertices), dtype=np.int64)
    faces = remap[np.asarray(hull.simplices, dtype=np.int64)]
    if (faces < 0).any():
        return None
    piece = trimesh.Trimesh(
        vertices=np.asarray(points[hull.vertices], dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    # scipy does not promise a consistent winding for the simplex facets;
    # MuJoCo and the manifest validator both require a real positive-volume
    # mesh, so orient the closed hull before measuring it.
    piece.fix_normals()
    try:
        volume = abs(float(piece.volume))
    except Exception:
        return None
    if not np.isfinite(volume) or volume <= 1e-15:
        return None
    return piece


def _patch_normal(triangles: np.ndarray, indices: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """Return a deterministic normal for a planar/near-planar triangle patch."""
    normals = np.cross(
        triangles[indices, 1] - triangles[indices, 0],
        triangles[indices, 2] - triangles[indices, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-15
    if valid.any():
        normal = normals[valid].sum(axis=0)
        length = float(np.linalg.norm(normal))
        if length > 1e-12:
            return normal / length
    try:
        _, singular_values, vectors = np.linalg.svd(
            points - points.mean(axis=0), full_matrices=False
        )
    except np.linalg.LinAlgError:
        return None
    if len(singular_values) < 2 or float(singular_values[0]) <= 1e-15:
        return None
    normal = np.asarray(vectors[-1], dtype=np.float64)
    length = float(np.linalg.norm(normal))
    return normal / length if length > 1e-12 else None


def _surface_patch_candidate(
    triangles: np.ndarray,
    indices: np.ndarray,
    settings: CollisionSettings,
) -> tuple[trimesh.Trimesh, bool] | None:
    """Make one convex patch, extruding planar patches by a small fixed margin."""
    points = np.unique(np.asarray(triangles[indices], dtype=np.float64).reshape(-1, 3), axis=0)
    if len(points) < 4:
        # A single triangle is still a valid surface patch; the extrusion path
        # below turns it into a thin positive-volume convex prism.
        if len(points) < 3:
            return None
    direct = _convex_hull_piece(points, settings.max_vertices)
    if direct is not None:
        return direct, False
    normal = _patch_normal(triangles, indices, points)
    if normal is None:
        return None
    extruded = np.vstack(
        (
            points + normal[None, :] * settings.surface_patch_extrusion_m,
            points - normal[None, :] * settings.surface_patch_extrusion_m,
        )
    )
    piece = _convex_hull_piece(extruded, settings.max_vertices)
    return (piece, True) if piece is not None else None


def _surface_patch_pieces(
    source: trimesh.Trimesh,
    settings: CollisionSettings,
    cell_size_m: float,
) -> tuple[list[trimesh.Trimesh], dict[str, Any]]:
    """Partition welded connected surfaces into bounded convex patches."""
    clean = source.copy()
    clean.merge_vertices(digits_vertex=8)
    clean.update_faces(clean.unique_faces())
    clean.update_faces(clean.nondegenerate_faces())
    clean.remove_unreferenced_vertices()
    if len(clean.faces) == 0:
        raise CollisionError("surface patch source has no non-degenerate faces")

    pieces: list[trimesh.Trimesh] = []
    component_count = 0
    split_count = 0
    extruded_count = 0
    for component in clean.split(only_watertight=False):
        component_count += 1
        triangles = np.asarray(component.triangles, dtype=np.float64)
        if len(triangles) == 0 or not np.isfinite(triangles).all():
            raise CollisionError("surface patch component is empty or non-finite")
        centroids = triangles.mean(axis=1)
        cell_index = np.floor(centroids / cell_size_m).astype(np.int64)
        grouped: dict[tuple[int, int, int], list[int]] = {}
        for index, key in enumerate(map(tuple, cell_index)):
            grouped.setdefault(key, []).append(index)
        # Sorted keys make the recursive partition independent of hash order.
        stack = [np.asarray(grouped[key], dtype=np.int64) for key in sorted(grouped)]
        while stack:
            indices = stack.pop()
            candidate = _surface_patch_candidate(triangles, indices, settings)
            if candidate is not None:
                piece, was_extruded = candidate
                pieces.append(piece)
                extruded_count += int(was_extruded)
                if len(pieces) > settings.surface_patch_max_pieces:
                    raise CollisionError(
                        "surface patch decomposition exceeds "
                        f"{settings.surface_patch_max_pieces} pieces"
                    )
                continue
            if len(indices) <= 1:
                raise CollisionError("surface patch contains an irreducibly degenerate triangle")
            axis = int(np.argmax(np.ptp(centroids[indices], axis=0)))
            ordered = indices[np.argsort(centroids[indices, axis], kind="mergesort")]
            middle = len(ordered) // 2
            if middle <= 0 or middle >= len(ordered):
                raise CollisionError("surface patch cannot be split deterministically")
            stack.append(ordered[:middle])
            stack.append(ordered[middle:])
            split_count += 1
    if not pieces:
        raise CollisionError("surface patch decomposition returned no pieces")
    return pieces, {
        "method": "surface_patch",
        "cell_size_m": float(cell_size_m),
        "component_count": component_count,
        "split_count": split_count,
        "extruded_piece_count": extruded_count,
        "piece_count": len(pieces),
        "max_piece_vertices": max(len(piece.vertices) for piece in pieces),
    }


def _decompose_surface_patches(
    source: trimesh.Trimesh,
    settings: CollisionSettings,
) -> tuple[list[trimesh.Trimesh], float, dict[str, Any]]:
    """Find a deterministic patch resolution that passes the surface gate."""
    cell_size = settings.surface_patch_cell_size_m
    if cell_size is None:
        cell_size = settings.surface_p95_threshold_m * 3.0
    last_p95 = float("inf")
    last_metrics: dict[str, Any] = {}
    for refinement in range(settings.surface_patch_refinements + 1):
        pieces, metrics = _surface_patch_pieces(source, settings, cell_size)
        surface_p95 = bidirectional_surface_p95(source, pieces, settings.surface_samples)
        metrics = {
            **metrics,
            "refinement": refinement,
            "surface_p95_m": surface_p95,
        }
        if surface_p95 <= settings.surface_p95_threshold_m:
            return pieces, surface_p95, metrics
        last_p95 = surface_p95
        last_metrics = metrics
        cell_size *= 0.5
    raise CollisionError(
        f"surface patch p95 {last_p95:.9g} m exceeds "
        f"{settings.surface_p95_threshold_m:.9g} m after "
        f"{settings.surface_patch_refinements + 1} resolutions; "
        f"last metrics={last_metrics}"
    )


def _cache_key(source_hash: str, scale: tuple[float, float, float], settings: CollisionSettings) -> str:
    payload = {
        "source_mesh_sha256": source_hash,
        "scale": scale,
        "coacd_version": _coacd_version(),
        "seed": settings.seed,
        "max_pieces": settings.max_pieces,
        "max_vertices": settings.max_vertices,
        "settings": settings.manifest_settings(),
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
            or manifest.get("settings") != settings.manifest_settings()
            or int(manifest.get("surface_samples", settings.surface_samples)) != settings.surface_samples
        ):
            return None
        records = manifest["pieces"]
        if not isinstance(records, list) or not 1 <= len(records) <= settings.published_max_pieces:
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
            piece.fix_normals()
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

    patch_metrics: dict[str, Any] = {}
    if settings.method == "surface_patch":
        proxy_meshes, surface_p95, patch_metrics = _decompose_surface_patches(source, settings)
        result = [(piece.vertices, piece.faces) for piece in proxy_meshes]
    else:
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
            piece_mesh.fix_normals()
            volume = abs(float(piece_mesh.volume))
            if not np.isfinite(volume) or volume <= 0:
                raise CollisionError("CoACD piece has non-positive volume")
            data = _canonical_mesh_bytes((vertices, faces))
            canonical.append((_hash_bytes(data), data, vertices, faces, volume))
        except CollisionError:
            raise
        except Exception as exc:
            raise CollisionError(f"invalid CoACD output: {exc}") from exc
    if len(canonical) > settings.published_max_pieces:
        raise CollisionError(
            f"{settings.method} output exceeds {settings.published_max_pieces} pieces"
        )
    canonical.sort(key=lambda item: item[0])
    # Measure the canonical, hash-sorted pieces that will actually be cached;
    # sampling a concatenated mesh is order-sensitive, so measuring an
    # unsorted in-memory list would make cache validation nondeterministic.
    proxy_meshes = [
        trimesh.Trimesh(vertices=item[2], faces=item[3], process=False)
        for item in canonical
    ]
    surface_p95 = bidirectional_surface_p95(source, proxy_meshes, settings.surface_samples)
    if surface_p95 > settings.surface_p95_threshold_m:
        raise CollisionError(
            f"collision surface p95 {surface_p95:.9g} m exceeds "
            f"{settings.surface_p95_threshold_m:.9g} m"
        )
    metrics = {
        **patch_metrics,
        "method": settings.method,
        "piece_count": len(canonical),
        "surface_p95_m": surface_p95,
    }

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
            "settings": settings.manifest_settings(),
            "surface_p95": surface_p95,
            "metrics": metrics,
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
    piece = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    piece.fix_normals()
    return piece
