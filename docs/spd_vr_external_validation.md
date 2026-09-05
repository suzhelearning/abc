# SPD-VR external validation runbook

This runbook is the execution boundary for the gates that cannot be proven by
the repository's CPU/fake-source tests. It is intentionally fail-closed:
synthetic idle episodes, a CPU timing result, an unlicensed vendor bundle, or a
different DINO checkpoint are evidence of a smoke test only, not a release.

Every lab run should archive the command output, the exact git commit, the
asset/checkpoint hashes, and the resulting JSON beside the run's manifest. Do
not put vendor URDF/STL bytes, PXREA `.so` files, gated DINO weights, or raw
human recordings in the public repository.

## Gate ledger

| Gate | Required evidence | Pass condition | Owner / blocker |
| --- | --- | --- | --- |
| Vendor assets | Written Tianji/Wuji2 terms, asset IDs, scope, and the hashes in `assets/tianji_wuji2/SHA256SUMS` | The exact local URDF/STL bundle is permitted for the intended publication and use | Vendor confirmation is still outstanding |
| PICO teleoperation | Preflight JSON, process logs, and a hardware episode for every fault case below | Continuous operation and side-local HOLD/re-alignment behavior are observed without process or data-contract violations | Requires PICO 4 Ultra, RoboticsService, and the vendor SDK `.so` |
| Contact model | `model_manifest.yaml`, `collision_manifest.yaml`, and `spd-vr-acceptance` JSON | Every contact record passes the configured surface gate and the model hashes agree | Local authorized URDF/STL required |
| DINO/GPU | Checkpoint provenance JSON, SHA-256, policy benchmark JSON, and environment capture | Official ViT-B/16 checkpoint loads completely; target CUDA run meets the 30 Hz chunk p95 after compile review | Gated weights and target GPU are not present in the current environment |
| Training | 8-GPU smoke/resume logs, overfit/long-run curves, checkpoints and configs | Resume restores model/EMA/optimizers/RNG and the formal run completes the agreed 170k-step protocol | Requires the target CUDA host and usable demonstrations |
| Data | Per-episode acceptance JSON, collection ledger, task/scene coverage, and source-time audit | Every published episode passes checksums, contact/258-row gate, replay, task coverage and human-source audit; the aggregate reaches the approved 75 h plan | Formal collection has not started |
| Release | Notice, licenses, checkpoint provenance, evaluation/ablation report and safety review | All above gates are green and no artifact boundary is ambiguous | Do not call a partial run an SPD-75h release |

## 0. Capture the environment and source boundary

Run this from the repository root before each gate. Keep the output outside
the repository if it contains device identifiers or local paths.

```bash
git rev-parse --show-toplevel
git rev-parse HEAD
git status --short
uv lock --check
uv run python -VV
uv run python -c 'import torch, mujoco; print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "cuda_devices": torch.cuda.device_count(), "mujoco": mujoco.__version__})'
cd assets/tianji_wuji2 && sha256sum -c SHA256SUMS
```

The public branch intentionally contains only the asset instructions and
hash list. A clean public checkout should be tested with an absent
`SPD_VR_TEST_URDF`; only explicitly vendor-dependent tests may skip.

## 1. Vendor asset and checkpoint provenance

### Tianji/Wuji2 bundle

Before a public release, obtain written terms that name the supplied
Tianji+Wuji2 combination, not just the public Wuji retargeting repository. The
record must answer:

1. Which URDF, STL, textures and generated derivatives are covered (record
   each SHA-256 and the declared robot name).
2. Whether research use, modification, internal sharing, publication in a
   public Git repository, and commercial use are each allowed.
3. Whether the license survives conversion to MJCF/contact proxies and whether
   those derived artifacts may be published.
4. Required copyright/notice wording, attribution, expiry, and revocation.

Until all four answers are attached to the lab record, keep the bundle local.
The MIT notices on the public Wuji repositories are provenance for those
upstreams only; they do not establish rights for the Tianji arm or the local
combined bundle.

### DINOv3 checkpoint

Use the official `facebook/dinov3-vitb16-pretrain-lvd1689m` source and accept
its gated terms with the account that will run the experiment. Create a local
JSON record next to (not inside) the checkpoint, for example:

```json
{
  "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
  "checkpoint_filename": "dinov3_vitb16_pretrain_lvd1689m.pth",
  "source_url": "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m",
  "license": "dinov3-license",
  "license_url": "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
  "terms_accepted": true,
  "access_date": "YYYY-MM-DD",
  "sha256": "<64 lowercase hex characters>"
}
```

The SHA-256 must be calculated from the exact file passed to
`spd-policy-benchmark`; `terms_accepted` is a human attestation, not an
automatic license grant. The benchmark already records the file hash and
rejects missing/incompatible weights. Do not substitute ABC's 14-DoF
checkpoint or a tiny test backbone.

## 2. Model and contact qualification

With a locally authorized bundle, build and verify the same artifact that will
be used for recording. Keep the generated directory and JSON output together:

```bash
uv run spd-model \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf \
  --output generated/spd_vr \
  --cache cache/spd_collision

uv run spd-model --verify --verify-contact \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf \
  --output generated/spd_vr

uv run spd-vr-acceptance \
  --manifest generated/spd_vr/model_manifest.yaml \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf \
  --require-contact --seed-count 3 > acceptance-model.json
```

The acceptance command is read-only. A raw convex-hull manifest, a missing
record, an asset hash mismatch, or a model outside the manifest is a failure.
`--raw-collisions` remains a display/control smoke mode and must not be used
for contact-qualified recording.

## 3. PICO 4 Ultra hardware matrix

First run the read-only preflight with the same model and SDK that the
launcher will use. `--serial` selects a device; it does not create an ADB
reverse entry.

```bash
uv run spd-vr-preflight \
  --sdk-library /path/to/libPXREARobotSDK.so \
  --serial <pico-device-id> \
  --manifest generated/spd_vr/model_manifest.yaml \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf \
  --require-contact > preflight-pico.json
```

Then launch the real three-window graph. Use a new output path for every run;
never overwrite a failed episode while diagnosing it.

```bash
scripts/start_spd_vr.sh --detach --preflight \
  --serial <pico-device-id> \
  --sdk-library /path/to/libPXREARobotSDK.so \
  --model generated/spd_vr/unified_plant.xml \
  --arm-model generated/spd_vr/arm_ik.xml \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf \
  --record-to /lab-runs/<run-id>/episode.hdf5 \
  --collection-run-id <run-id> \
  --operator-id <operator-id> \
  --pico-serial <pico-device-id> \
  --require-usable-training

scripts/start_spd_vr.sh --status
# send pause/resume/realign/reset/shutdown with spd-control as required
scripts/stop_spd_vr.sh
```

For each row below, preserve the bridge, arm-IK and viewer logs together with
the HDF5 file. The expected result is behavioral, not a suggested synthetic
fixture:

| Case | Fault injection / observation | Required result |
| --- | --- | --- |
| Continuous | Run the agreed duration with both hands visible and repeat a known bilateral motion | All managed processes remain alive; tracking sequence/epoch and raw row lengths remain valid; no unexplained zero target or model-hash change |
| Left occlusion | Cover or remove only the left hand while the right hand continues | Left arm/hand holds its last safe target; right side continues; no global zeroing or cross-side invalidation |
| Right occlusion | Repeat for the right hand | Symmetric to left occlusion |
| SDK disconnect | Stop the PXREA stream or remove the device | Affected side enters stale/HOLD within the 50 ms freshness policy; no jump to zero; the run is marked failed until recovery is recorded |
| Reconnect / epoch | Restore the stream and observe the new tracking epoch | Both affected calibration windows require ten stable neutral frames before targets resume; the first post-reconnect frames are not treated as demonstrations |
| Arm-IK failure | Induce a bounded solver failure or invalid wrist sample on one side | Only that side holds its previous safe target; the opposite side and viewer remain live |
| Lifecycle | Exercise pause, resume, realign and reset, then shutdown | Control frames are acknowledged without changing wire sizes; reset/epoch invalidates old calibration; shutdown exits all three managed processes |

After every candidate episode, run the structural and formal gates. A failed
case is evidence to retain, not an episode to publish.

```bash
uv run spd-validate-episode /lab-runs/<run-id>/episode.hdf5 --checksums
uv run spd-vr-acceptance \
  --manifest generated/spd_vr/model_manifest.yaml \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf \
  --episodes /lab-runs/<run-id>/episode.hdf5 \
  --require-contact --require-usable-training --replay
```

The fake-source mode and the repository's local regression suite cover the
wire/process graph only; they do not close this hardware matrix.

## 4. Official DINO and target-GPU gate

On the target CUDA host, record the GPU, driver, CUDA runtime, PyTorch and
MuJoCo versions before measuring. Run both an eager and a reviewed compile
case with enough ticks to avoid reporting only warm-up behavior:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())'

uv run spd-policy-benchmark \
  --dino-checkpoint /lab-runs/<run-id>/dinov3_vitb16_pretrain_lvd1689m.pth \
  --checkpoint-provenance /lab-runs/<run-id>/dinov3-provenance.json \
  --device cuda:0 --warmup-ticks 32 --measure-ticks 256 \
  > policy-eager.json

uv run spd-policy-benchmark \
  --dino-checkpoint /lab-runs/<run-id>/dinov3_vitb16_pretrain_lvd1689m.pth \
  --checkpoint-provenance /lab-runs/<run-id>/dinov3-provenance.json \
  --device cuda:0 --warmup-ticks 32 --measure-ticks 256 --compile \
  > policy-compile.json
```

Review all of `dino_missing_keys`, `dino_unexpected_keys`, the checkpoint
hash, peak memory, append p95/p99 and cached chunk p95. Only after the target
GPU's memory headroom and numerical comparison are reviewed may the run be
repeated with `--enforce-deadline`; the command's deadline is the 30 Hz
16.67 ms chunk budget, not a guarantee of end-to-end hardware latency.
`--enforce-deadline` also requires `--checkpoint-provenance`; the command
checks that the JSON model ID, source URL, license, accepted terms, access
date, filename and SHA-256 match the actual checkpoint.

After the review, the formal machine-readable deadline result is:

```bash
uv run spd-policy-benchmark \
  --dino-checkpoint /lab-runs/<run-id>/dinov3_vitb16_pretrain_lvd1689m.pth \
  --checkpoint-provenance /lab-runs/<run-id>/dinov3-provenance.json \
  --device cuda:0 --warmup-ticks 32 --measure-ticks 256 --compile \
  --enforce-deadline > policy-qualified.json
```

## 5. 8-GPU smoke, resume, and formal training

Use a dataset directory whose episodes have passed the formal 258-row gate.
Keep the exact command/config and checkpoint in the run record. First run a
short distributed smoke, stop after a checkpoint, then resume into a new
output directory and verify the step/epoch, DINO hash, normalization, model,
EMA, Muon/AdamW and RNG state.

```bash
uv run torchrun --standalone --nproc-per-node 8 train_spd.py \
  --dataset-root /lab-runs/<dataset> \
  --dino-checkpoint /lab-runs/<run-id>/dinov3_vitb16_pretrain_lvd1689m.pth \
  --train-steps 20 --output-dir /lab-runs/<run-id>/smoke-a

uv run torchrun --standalone --nproc-per-node 8 train_spd.py \
  --dataset-root /lab-runs/<dataset> \
  --dino-checkpoint /lab-runs/<run-id>/dinov3_vitb16_pretrain_lvd1689m.pth \
  --resume /lab-runs/<run-id>/smoke-a/last.pt \
  --train-steps 40 --output-dir /lab-runs/<run-id>/smoke-b

uv run torchrun --standalone --nproc-per-node 8 train_spd.py \
  --dataset-root /lab-runs/<dataset> \
  --dino-checkpoint /lab-runs/<run-id>/dinov3_vitb16_pretrain_lvd1689m.pth \
  --train-steps 170000 --output-dir /lab-runs/<run-id>/formal
```

The repository's console entry point is `spd-train`; the `train_spd.py`
commands above are retained for compatibility with the ABC training launcher.
If the target checkout does not expose that compatibility script, use the
equivalent `uv run torchrun ... -m spd_vr.training` invocation and record it.
Do not count a CPU/tiny-DINO run as this gate.

Required diagnostics before accepting the formal checkpoint:

- the local deterministic tiny optimization-step smoke passes (`uv run pytest
  -q tests/test_spd_training.py`); the target environment must repeat the
  tiny-data overfit with a documented decreasing flow loss and no NaN/Inf;
- a long-run sample has stable loss, gradient norms, EMA and validation;
- the resumed run is numerically continuous at the checkpoint boundary;
- the final checkpoint has no embedded frozen DINO weights and its slim-state
  completeness checks pass.

## 6. Formal data collection and 75-hour ledger

Approve the task/scene schedule before recording. The ledger must include
operator, PICO serial, scene/task/seed, start/end source timestamps, raw and
30 Hz row counts, valid/contact-eligible frames, dropped/occluded spans,
episode checksum, model/URDF/contact manifest hashes and acceptance JSON.

Do not aggregate hours from idle or rejected recordings. For each episode:

```bash
uv run spd-vr-acceptance \
  --manifest generated/spd_vr/model_manifest.yaml \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf \
  --episodes /lab-runs/<dataset>/episode_<id>.hdf5 \
  --require-contact --require-usable-training --replay
```

After the per-episode checks, aggregate the same directory with the collection
auditor. The first command is a report; the second is the formal 75-hour,
all-task gate and is expected to fail until the collection is actually
complete.

```bash
uv run spd-vr-dataset-audit /lab-runs/<dataset> \
  --require-metadata --require-usable-training

uv run spd-vr-dataset-audit /lab-runs/<dataset> \
  --target-hours 75 --require-target --require-all-tasks \
  --require-metadata --require-usable-training
```

The report separates raw wall-clock duration, PICO source duration, and
contact-qualified duration. Only the latter is used for the formal target;
missing scene/task/seed identity, source timestamps, collection identity,
model/URDF/contact-manifest hashes, checksums or complete 258-row windows make
an episode fail rather than silently counting its file size as data. A formal
collection also rejects mixed artifact hashes.

The 75-hour target is an experiment-plan quantity, not a property inferred
from file size. Publish an aggregate only after the approved scene/task
coverage, contact/258-row yield, source-time audit and human review are all
complete. Keep raw 60 Hz streams; filtering may only update training metadata
and must preserve the audit copy.

## 7. Evaluation, ablation and future hardware review

Before releasing a checkpoint, archive the exact dataset split, normalization,
model config, DINO hash, seed and evaluation code. At minimum compare the
full SPD configuration against the planned ablations (visual input, history,
contact filtering, actual-qpos labels and streaming-vs-batched inference),
and report task-level success with confidence intervals rather than training
loss alone.

Any future Tianji/Wuji2 fine-tune is a separate safety review. This repository
does not emit follower commands, motor bus packets, emergency-stop calls or
collision-safe physical trajectories; a successful simulation acceptance JSON
cannot authorize hardware actuation.
