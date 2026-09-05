# SPD-VR architecture contract

## Scope and safety boundary

SPD-VR accepts PICO tracking as hardware input and emits targets only to a
MuJoCo Tianji-Wuji2 plant. There is intentionally no Tianji or Wuji2 follower,
fieldbus client, motor command, safety PLC, or physical emergency-stop API in
this repository.

## Runtime data flow

```text
PICO / XRoboToolkit / PXREA
  -> 1540-byte versioned + CRC tracking frame over Zenoh
  -> 10 stable-frame neutral calibration per side
  -> left/right wrist IK (7 + 7) -> 272-byte arm target
  -> left/right Wuji retargeting (20 + 20) in the viewer
  -> canonical 54-D position target
  -> one MuJoCo plant at 480 Hz, controlled at 60 Hz
  -> atomic HDF5 raw stream + exact 30 Hz row index
```

An epoch change resets both calibration windows and hand filters. A stale,
invalid, or failed side holds its last safe 27-D arm/hand target; it never
zeros the robot and never invalidates the other side.
Accepted targets are also rate-limited from URDF joint velocity limits at the
60 Hz control cadence before they reach MuJoCo.

## Ownership seams

- `abc_minimal`: unchanged ABC baseline plus a shared direction-explicit flow
  interpolation primitive.
- `spd_vr/model_compiler`: deterministic URDF-first MJCF, collision, actuator,
  and manifest generation.
- `spd_vr/model_builder.py`: compatibility wrapper for older ABC/SPD scripts;
  it delegates to the same compiler and never maintains a second model path.
- `assets/tianji_wuji2/SHA256SUMS`: byte-level provenance for the supplied
  URDF/STL bundle; it does not substitute for vendor redistribution permission.
- `spd_vr/teleop.py`, `spd_vr/ik.py`, `spd_vr/pxrea_bridge.py`: hardware-input
  adaptation and simulation target production.
- `spd_vr/alignment.py`, `spd_vr/qp_arm.py`, `spd_vr/arm_ik.py`: 200 Hz
  dual-arm joint[1] neutral alignment and one persistent OSQP Jacobian-QP per
  side; the process publishes only the versioned arm-target packet.
- `spd_vr/viewer.py`: 60 Hz Wuji hand retargeting, full MuJoCo ownership and
  recording process. `spd_vr/live.py` remains a combined smoke entry point.
- `spd_vr/simulation.py`: canonical-name MuJoCo address resolution and the
  optional ABC `MJWarpSim` stepping adapter.
- `spd_vr/data.py`: the only episode schema and the only 30 Hz training sampler.
- `spd_vr/episode.py`, `spd_vr/recorder.py`: checkpoint/pause/revert/skip
  state machine and lifecycle adapter around the same atomic HDF5 writer;
  `EpisodeController(collection_identity=...)` writes the validated formal
  collection identity into that manifest instead of creating a second schema.
- `spd_vr/filter_contacts.py`, `spd_vr/visual.py`, `spd_vr/augment.py`: offline
  contact-gap filtering plus opt-in segmentation/color and calibrated mirror
  transforms.
- `spd_vr/policy.py`, `spd_vr/training.py`: pure SPD policy and optimizer loop.

## HDF5 invariants

- All raw datasets have exactly the same leading length and refer to one 60 Hz
  control tick.
- `training/index_30hz` is strictly increasing and points into raw rows.
- `raw/observation/qpos` is actual simulated state.
- `raw/action/qpos` is the same-tick actual joint action stream; future rows
  are the policy labels.
- `raw/action/qpos_target` is operator-command audit data, not a policy label.
- The training dataset derives future actions from actual qpos.
- PICO source time and PC bridge monotonic time are separate datasets.
- Object poses and contact forces come from the same authoritative MuJoCo data;
  the caller names task-object bodies explicitly.
- A sample requires 258 indexed rows, including the true prior row and the
  farthest future `+256` row.
- `training/grid_step` prevents a sample from crossing a missing 30 Hz tick.
- `training/contact_eligible` and `training/segments_30hz` remove only
  continuous hand-object-free spans longer than 10 seconds from training;
  raw rows and an audit list remain intact.
- The persisted normalization artifact is either absent (identity for smoke
  use) or exactly four finite 54-D vectors (`qpos/action` mean/std); standard
  deviations must be positive and float32-representable.
- Vendor signed-millisecond timestamps are accepted only after conversion to
  a positive, non-overflowing tracking nanosecond timestamp; corrupt values
  are dropped rather than clamped into a fresh-looking sample.
- Publication is staging-file → structural validation → atomic rename.
- `EpisodeWriter(require_usable_training=True)` adds a formal-data gate before
  the atomic rename; it rejects a recording with no complete contact-eligible
  258-row SPD window. The live viewer, combined smoke entry, lifecycle recorder, and
  three-window launcher expose this as `--require-usable-training`.

Zenoh process keys have fixed payload sizes: `spd/vr/v1/tracking` is 1,540
bytes, `spd/vr/v1/arm_targets` is 272 bytes, and `spd/vr/v1/control` is 40
bytes. All include version/CRC metadata plus sequence and epoch checks.

The `spd_vr.scenes` package reuses the deterministic procedural reset layer:
six scenes and 17 Table-2 tasks, seeded object placement, mass, friction,
color, and explicit free-body names. `spd-scene` writes a JSON reset/model
manifest; `spd-vr-live --scene-manifest` carries that provenance into HDF5.
This is a scene/data layer, not a physical robot-control output.
The live entry validates the task name and object-body list against the reset
manifest before accepting it.
`SPDVRSim.reset_scene()` then restores each free-body pose and the manifest's
mass/friction/color parameters before the first control tick.
When present, builder source hashes in the scene manifest are checked as well.

Live HDF5 recording currently requires the vanilla `mujoco` backend. The
optional MJWarp backend is available for stepping/viewer work, but recording
fails closed until constraint/contact forces can be synchronized into the same
authoritative `MjData` used for the full-physics snapshot.

## Policy and inference

The 54-D policy is approximately 223M parameters: a frozen DINOv3 ViT-B/16,
shared visual pooling with four learned queries per camera, an eight-block
observation trunk, and an independently parameterized eight-block action
expert. Training denoises 32 independent chunks in parallel. Its
prefix-parallel mask lets each chunk read the causal observation window and
its own eight action tokens, never another chunk's independent noise.

Deployment uses a real per-layer rolling KV cache. Every 30 Hz control tick
calls `SPDPolicy.append_observation`; image tensors are included on the 8-tick
subsample cadence. On a chunk boundary, `sample_actions_cached` reuses the
cached observation/action projections for ten Euler steps and returns one
`[batch, 8, 54]` chunk.

## Collision qualification

The compiler supports `--raw-collisions` only for fast visualization and
control-chain tests. Contact-data collection uses the deterministic adaptive
convex-surface-patch backend by default: welded connected components are
spatially partitioned, convex pieces are recursively bounded to 64 vertices,
and every published record is measured against the bidirectional surface p95
gate. The lower-level CoACD backend remains available for diagnostics and
explicit callers, but a CoACD miss never falls back silently. A successful
raw build is not evidence of contact accuracy.
`spd-model --verify --verify-contact` and any live run with `--record-to`
enforce this gate; raw or unqualified artifacts are rejected before the
recorder opens.

`spd-vr-preflight` provides the corresponding read-only process-graph check.
It reports JSON results for ADB/RoboticsService, the vendor SDK, Python
dependencies, display, generated artifacts, Zenoh endpoint, tmux session, and
optionally the contact gate.  The launcher accepts `--preflight` and
`--serial <device-id>`; it never creates an ADB reverse entry itself.

`spd-vr-acceptance` is the bounded P6 acceptance command.  It validates the
six-scene/17-task deterministic registry, model and optional contact gate, and
an explicitly supplied HDF5 file/directory (including checksums).  With
`--replay` it restores full MuJoCo state and checks the recorded model hash;
without `--episodes` it reports data acceptance as not requested rather than
inventing a pass.  Formal data acceptance should add
`--require-usable-training`, which rejects a structurally valid but entirely
idle episode with no complete contact-eligible 258-row SPD window; synthetic idle smoke may
omit that opt-in gate but is never equivalent to a demonstration dataset.

`spd-vr-dataset-audit` is the aggregate collection gate. It keeps per-episode
acceptance separate from collection-level task/scene coverage, PICO source
timestamp provenance, two-sided validity, raw/source/qualified duration and
the planned 75-hour target. Formal mode requires the manifest collection
identity (`run_id`, `operator_id`, `pico_serial`) and complete 258-row windows;
it requires scene/task seed plus model/URDF/contact-manifest hashes and counts
only contact-qualified duration. Mixed artifact hashes fail the collection
gate.

`spd-vr-collection-plan` is the pre-recording schedule artifact. It expands
the same registry into all 17 Table-2 task quotas and 1,916 deterministic
episode IDs (75.25 qualified hours), but remains explicitly
`data_collected: false`; only HDF5 episodes accepted by the collection audit
can contribute to the formal target.
The viewer, direct live loop and three-window launcher can bind a recording to
one plan episode; task, reset seed and (when present) collection identity are
checked before the writer opens, and the plan path/hash are preserved in the
episode manifest.

`spd-vr-evaluation` converts reviewed binary rollout outcomes into a strict
17-task report covering the full policy and the five planned ablations. It
computes Wilson 95% intervals and binds the report to the dataset split, model
config, DINO hash, commit and seed; it never generates outcomes itself.

`spd-sim-benchmark` measures the MuJoCo/MJWarp control tick (480 Hz physics,
60 Hz control) with optional camera rendering.  It is a timing diagnostic only;
it does not publish HDF5 or turn an unqualified collision model into a data
collection artifact.

`spd-policy-benchmark` is the P4 full-model diagnostic. It requires a supplied
official DINOv3 checkpoint, records its SHA-256, measures streaming KV append
and ten-step cached action sampling plus steady-state peak memory on the
requested device, and never falls back to an ABC or tiny checkpoint.

Training checkpoints retain the rank-zero compatibility RNG record plus a
per-rank Torch/NumPy/Python/CUDA snapshot. Resume selects the current rank and
rejects a world-size or CPU/CUDA mismatch before stochastic training continues.

The hardware, vendor-license, target-CUDA, 8-GPU-resume, formal data, and
release gates that cannot be established by these local diagnostics are
defined in [`spd_vr_external_validation.md`](spd_vr_external_validation.md).
Those gates remain fail-closed until their external evidence is archived. Once
the records exist, `spd-vr-release-audit` checks the versioned evidence bundle
and the exact DINO checkpoint without publishing or authorizing actuation.

`spd-replay-episode` restores each recorded full-physics state and checks the
same model hash, qpos/qvel tolerance, optional hand-object contact bit, and
camera rendering. It never sends commands to a physical follower.

Symmetry augmentation is opt-in through `spd_vr.augment.SymmetrySpec`: a
calibrated 54-joint permutation/sign table is required, then the Dataset
mirrors RGB and swaps wrist cameras while applying the same transform to
qpos, previous actual action, and future labels. Missing calibration fails
closed; no anatomical sign convention is guessed from joint names.
