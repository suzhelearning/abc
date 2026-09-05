from __future__ import annotations

import numpy as np
import trimesh

from spd_vr.model_compiler.collision import CollisionSettings, decompose_mesh, load_collision_piece


def test_surface_patch_decomposition_publishes_quality_gated_convex_pieces(tmp_path):
    # Two offset boxes make a small concave/disconnected proxy case without
    # depending on the vendor mesh or on a cache from another machine.
    source = trimesh.util.concatenate(
        [
            trimesh.creation.box(extents=(0.04, 0.02, 0.02)),
            trimesh.creation.box(extents=(0.02, 0.04, 0.02), transform=trimesh.transformations.translation_matrix((0.03, 0.03, 0.0))),
        ]
    )
    settings = CollisionSettings(
        method="surface_patch",
        surface_patch_cell_size_m=0.006,
        surface_patch_extrusion_m=0.0001,
        surface_patch_max_pieces=256,
        surface_samples=256,
        surface_p95_threshold_m=0.003,
    )

    artifact = decompose_mesh(source, settings, tmp_path / "cache")
    cached = decompose_mesh(source, settings, tmp_path / "cache")

    assert artifact.metrics["method"] == "surface_patch"
    assert cached.cache_hit is True
    assert cached.piece_sha256 == artifact.piece_sha256
    assert artifact.surface_p95 <= settings.surface_p95_threshold_m
    assert 1 < len(artifact.pieces) <= settings.surface_patch_max_pieces
    for path in artifact.pieces:
        piece = load_collision_piece(path)
        assert len(piece.vertices) <= settings.max_vertices
        assert np.isfinite(piece.vertices).all()
        assert abs(float(piece.volume)) > 0.0
