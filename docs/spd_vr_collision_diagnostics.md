# SPD-VR collision-quality diagnostics

This note records bounded CoACD experiments for the locally authorized
`Link_Base.STL`.  It is diagnostic evidence only: no result below is a
published collision artifact, and the 3 mm contact gate remains fail-closed.

## Source and metric

- CoACD: `1.0.14`
- source mesh: `476,946` vertices / `158,982` triangles
- source SHA-256: `95b3f71bb8c0c1236a62dd622a2973252be0817880d13bdda1daabc6e1e11433`
- metric: `bidirectional_surface_p95`, sampling source and proxy surfaces with
  deterministic NumPy RNG seed `0`; the reported value is the larger of
  source→proxy and proxy→source p95 distances.
- gate: `0.003 m` (`3 mm`)

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
alone is therefore not evidence that a 3 mm proxy exists.  The default
compiler still uses its fixed reproducible settings and rejects any artifact
whose measured p95 exceeds the manifest gate.

## Reproducibility

Run the bounded experiments from the repository root with the synchronized
`.venv` and the local authorized asset:

```bash
sha256sum assets/tianji_wuji2/meshes/Link_Base.STL
.venv/bin/python -c 'import importlib.metadata; print(importlib.metadata.version("coacd"))'
```

The full CoACD stdout and temporary piece arrays used for this note were kept
outside the repository under `/tmp/spd_coacd*`; they are intentionally not
treated as release artifacts.  A future contact-qualified build must provide
a fresh manifest that lists the source hash, exact CoACD settings, all pieces,
and a measured p95 at or below 3 mm before recording contact data.
