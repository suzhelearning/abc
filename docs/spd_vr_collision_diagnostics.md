# SPD-VR collision-quality diagnostics

This note records bounded CoACD experiments and the contact-qualified adaptive
surface-patch build for the locally authorized `Link_Base.STL`. CoACD remains
diagnostic evidence when its approximation misses the gate; the compiler now
has a separate deterministic convex-patch backend that publishes only after
the same measured surface gate passes. Vendor bytes and generated artifacts
are still local until redistribution terms are confirmed.

## Source and metric

- CoACD: `1.0.14`
- source mesh: `476,946` vertices / `158,982` triangles
- source SHA-256: `95b3f71bb8c0c1236a62dd622a2973252be0817880d13bdda1daabc6e1e11433`
- metric: `bidirectional_surface_p95`, sampling source and proxy surfaces with
  deterministic NumPy RNG seed `0`; the reported value is the larger of
  source→proxy and proxy→source p95 distances.
- gate: `0.003 m` (`3 mm`)

## Contact-qualified surface-patch build

The default compiler backend welds exact duplicate vertices, splits connected
components, bins triangle centroids into deterministic cells, and recursively
bisects any bin whose convex hull would exceed 64 vertices. Planar bins are
extruded by `0.15 mm` to produce positive-volume MuJoCo meshes. Arm links start
at a `16 mm` cell and hand links at `4.5 mm`; the backend halves the cell on a
quality miss, up to the recorded refinement limit. `rtree` supplies the
spatial index used by the surface-distance measurement.

The local authorized bundle produced the following artifact (`spd-model`
compile followed by `spd-model --verify --verify-contact`):

| quantity | result |
| --- | ---: |
| collision identities | 62 |
| total convex pieces | 23,099 |
| maximum record p95 | 2.5855 mm |
| `Link_Base.STL` pieces / p95 | 4,617 / 2.1596 mm |
| maximum piece vertices | 64 |
| generated directory size | about 95.7 MB |

The generated full plant and arm model both retained the 54/14 joint
contracts. A `0.125 s` MuJoCo benchmark on this CPU completed 64 physics steps
and 8 control ticks with step p95 about `9.10 ms`, below the `16.67 ms` control
budget. This is a local simulation diagnostic, not a hardware or physical
contact validation.

## Bounded trials

| max hull argument | CoACD threshold | preprocess resolution | merge | observed hulls | measured p95 |
| ---: | ---: | ---: | :---: | ---: | ---: |
| 4,096 | 0.001 | 100 | false | 10,138 | 6.66 mm (8,192 samples) |
| 16,384 | 0.0005 | 200 | false | 18,875 | 7.42 mm (8,192 samples) |

Both trials used `mcts_nodes=2`, `mcts_iterations=20`/`10`,
`mcts_max_depth=2`, `resolution=4,000`/`8,000`, `max_ch_vertex=64`,
`decimate=true`, `real_metric=true`, and `seed=0`.  The 4,096 trial was
also measured with 2,048 samples (`6.62 mm`); the 16,384 trial was measured
with 32,768 samples (`7.38 mm`).  Every returned piece passed the finite,
positive-volume checks used by the diagnostic, but the quality gate did not.

The result is not monotonic in the requested hull budget because the source
contains many disconnected/non-manifold regions and CoACD preprocessing is
itself a lossy voxelization.  Raising the budget or lowering the threshold
alone is therefore not evidence that a 3 mm proxy exists.  The adaptive patch
backend is a separate explicit path; it records its settings and rejects any
artifact whose measured p95 exceeds the manifest gate.

## Topology-repair trial

Exact vertex welding and removal of unreferenced vertices reduce the source to
`79,405` vertices while preserving `158,982` faces and exposing `46`
connected components.  Processing those components independently did not
solve the approximation error:

- one convex hull per component measured `112.7 mm` p95 at 2,048 samples;
- representative component-isolated CoACD runs measured about `5.8 mm`,
  `7.99 mm`, and `2.02 mm` p95 at 2,048/4,096 samples;
- some thin components returned zero-volume hulls, which are rejected by the
  production artifact validator.

The repair is now an explicit, hashed compiler transformation used only to
construct convex surface patches; it never replaces the authoritative source
mesh in the visual model or source manifest. Any vendor-approved collision
asset can still supersede it, but must carry its own source hash and manifest.

## Reproducibility

Run the bounded experiments from the repository root with the synchronized
`.venv` and the local authorized asset:

```bash
sha256sum assets/tianji_wuji2/meshes/Link_Base.STL
.venv/bin/python -c 'import importlib.metadata; print(importlib.metadata.version("coacd"))'
```

The full CoACD stdout and temporary piece arrays used for this note were kept
outside the repository under `/tmp/spd_coacd*`; they are intentionally not
treated as release artifacts. A future published contact build must provide a
fresh manifest that lists the source hash, exact patch/CoACD settings, all
pieces, and a measured p95 at or below the per-link gate before recording
contact data.
