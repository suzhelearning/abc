# ABC Training (`abc_minimal`)

**Contents**

- [Training data](#training-data)
  - [Exporting a task from ABC-130k](#exporting-a-task-from-abc-130k)
  - [Converting local MCAPs & the episode format](#converting-local-mcaps--the-episode-format)
- [Multi-node training](#multi-node-training)
- [Subtask & operator conditioning](#subtask--operator-conditioning)
- [Visualizing episodes and policies](#visualizing-episodes-and-policies)
- [Tianji/Wuji2 SPD](#tianjiwuji2-spd)


`abc_minimal` is the training and inference package for ABC-DiT: the model and
its DINOv3/CLIP backbones (`dit.py`), the episode dataloader (`dataloader.py`,
`episode_io.py`, `preprocess.py`), the training loop and checkpointing
(`train_loop.py`, `checkpointing.py`), DiT and VLA policy inference
(`policy.py`), the sim-eval glue (`sim_env.py`, `eval_policy.py`), and
the episode/policy visualizers. The scripts at the repository root —
`train.py`, `eval_policy.py`, `viz_episode.py`, `viz_policy.py` — are thin
wrappers around this package.

The [top-level README](../README.md) covers Tianji/Wuji2 SPD setup, data,
training and simulation. For the original ABC-DiT/VLA quickstart, see the
[upstream README](https://github.com/amazon-far/abc). Simulator environments and
the sim-eval task catalogue are covered in the
[abc_sim README](../abc_sim/README.md). This document is the training-side
reference: multi-node jobs, the episode data format and converters, and prompt
conditioning options.

## Training data

While we host a single task in training format, there are many more in the ABC
Dataset. The ABC-130k MCAPs are hosted on Hugging Face at
[`XDOF/ABC-130k`](https://huggingface.co/datasets/XDOF/ABC-130k). The dataset
is gated, so accept access on the dataset page and set `HF_TOKEN` before
downloading.

### Exporting a task from ABC-130k

Download all MCAPs for one task and convert them in place:

```bash
uv run scripts/export_hf_task.py --task organize_the_condiment_bottles
```

By default this downloads both `train` and `val`, stages raw MCAPs under
`$ABC_CACHE/hf_tasks/<task>/`, runs the MCAP converter, writes converted episodes
to `$ABC_CACHE/train_real/` and `$ABC_CACHE/val_real/`, then deletes the staged
raw MCAPs after each successful split conversion. For a quick smoke test:

```bash
uv run scripts/export_hf_task.py --task organize_the_condiment_bottles --split train --max-episodes 1
```

### Converting local MCAPs & the episode format

If you already have local MCAPs, call the lower-level converter directly:

```bash
uv run scripts/export_mcap.py ./train_run_1 ./out
```

The input is expected to look like:

```text
train_run_1/
  <task_name>/
    episode_<uuid>/
      episode.mcap
```

You can also pass the number of worker processes:

```bash
uv run scripts/export_mcap.py ./train_run_1 ./out 8
```

Each output episode is written to `./out/episode_<uuid>/` in the same format
the trainer reads:

```text
episode_<uuid>/
  states_actions.bin               # (num_steps, 28) float64: 14 states + 14 actions
  combined_camera-images-rgb.mp4   # 30 fps vertical stack of 224x224 camera views
  episode_metadata.json            # task name, cameras, resolutions, timing, num_steps
```

The mp4 is encoded in a manner that allows for efficient dataloading. For details, see the ABC paper.

## Multi-node training

For multi-node jobs without a shared filesystem, predownload a deterministic
node-local shard on each node before launching training:

```bash
ABC_CACHE=/local_nvme/abc_cache HF_TOKEN=... \
uv run scripts/prepare_hf_shards.py \
  --tasks organize_the_condiment_bottles \
  --num-nodes 8 --node-rank $NODE_RANK --workers 8

ABC_CACHE=/local_nvme/abc_cache \
uv run torchrun --nnodes 8 --node-rank $NODE_RANK --nproc-per-node 8 train.py
```

The predownload step writes this node's converted episodes into the usual
`train_real/` and `val_real/` directories. Training auto-detects the
`hf_status/shard.json` marker and uses local-rank sampling, so validation
metrics are accumulated across different validation shards on different GPUs.

The revision is pinned to a commit SHA at run time and training verifies at
startup that all nodes sharded the same snapshot. If the dataset may change
while nodes prepare, pass the same explicit `--revision <sha>` to every node.

## Subtask & operator conditioning

In addition to the task prompt, our policies can condition  **subtask** labels and on
the episode's **operator** id. The MCAP converter (`scripts/export_mcap.py`)
extracts both from the release MCAPs when present and writes two optional
extra files next to the episode:

```text
episode_<uuid>/
  subtasks.json      # {"<frame_idx>": "<subtask label>", ...}  — per-frame subtask
  operator.json      # {"operator_id": "<uuid>"}                — the teleoperator id
```

- **Subtasks** Enable at train time with `--prompt.use-subtask-as-prompt`,
  choosing `--prompt.subtask-mode {replace,append}` (`replace` swaps the task prompt for the subtask label;
  `append` formats both via `--prompt.subtask-append-format`).
- **Operators** We map UUIDs to short deterministic labels (`operator 0`, … or names). The labels are computed
  per-task such that operators with more hours (proxy for quality) have lower numbers.
  Enable with `--prompt.use-operator-id-as-prompt` and choose `--prompt.operator-prompting-mode {text_indexed,text_name}`; the label is appended as `"{prompt}. {operator}"`.

  The per-task map must be built prior to training and passed via
  `--prompt.operator-label-map-path`. Build it with:

  ```bash
  ABC_CACHE=cache/tshirt uv run scripts/build_operator_label_map.py \
      --out cache/tshirt/operator_label_map.json
  ```

  then pass `--prompt.operator-label-map-path cache/tshirt/operator_label_map.json`.

  On a node-sharded multi-node cache no single machine holds every episode, and
  a map built from one shard would mis-rank operators and miss those on other
  nodes. Instead, build one manifest per node from its local shard, gather the
  shard manifests on one machine, and fold them into a global ranking —
  hours and episode counts sum exactly across shards, so the result matches a
  full single-machine scan:

  ```bash
  uv run scripts/build_operator_label_map.py --out shard_$NODE.json   # on each node
  uv run scripts/build_operator_label_map.py \
      --combine shard_0.json shard_1.json ... --out operator_label_map.json
  ```

  then copy the combined manifest to every node at the same path and pass it
  via `--prompt.operator-label-map-path`.

Both are off by default. Some episodes do not have eg. subtask annotations and
for these training will drop back to task prompt only.

(We intend to release the global manifest in future but this is TODO.)

## Visualizing episodes and policies

`viz_policy.py` rolls out a checkpoint live in a viser window, and
`viz_episode.py` plays back dataset episodes the same way, no checkpoint
needed — browse a pool (`--root cache/train_sim`) or open one episode
directly; `--mode physics` re-simulates the recorded actions instead of posing
the arms. The default pose mode replays the whole recorded scene, objects
included, for episodes that ship `scene_qpos.npy` (the current sim_224
release), and falls back to posing the arms alone — objects held at their
start pose — for episodes without it:

```bash
uv run viz_policy.py --sim.checkpoint cache/bottles_75k.pt --port 8080
uv run viz_episode.py --root cache/train_sim --port 8080
```

## Tianji/Wuji2 SPD

This branch adds `--policy spd` to **the existing ABC training loop**, not a
separate `train_spd.py` framework. It implements the policy described in
[SPD, Sections 3.3 and A.5–A.6](https://arxiv.org/html/2608.15917v1#A1.SS5),
adapted from the paper's 56 joints to Tianji/Wuji2's 54. The old `spd_vr`
package is neither imported nor required.

### Reused ABC interfaces

| Interface | SPD integration |
| --- | --- |
| `TrainConfig`, `train.py` | `policy=spd`, `spd_model`, `spd_data`; common logging, optimization controls and resume flags |
| `dit.py` | Existing frozen `DinoVisionBackbone`, initial `AttentionPoolBlock`, sinusoidal embedding helper |
| `preprocess.py` | Aspect-preserving resize/pad, ImageNet normalization, state/action normalize and unnormalize |
| `dataloader.py` | Existing `MixtureDataset`, `GlobalStepSampler`, `DataParallelScope` |
| `checkpointing.py` | Existing atomic save/load, topology metadata, `last.pt` handling, with SPD EMA/RNG/provenance extensions |
| `train_loop.py` | Same distributed loop, clipping, scheduler, logging and validation; Muon/AdamW exposed as one optimizer |
| `policy.py` | `SPDPolicyConfig` and stateful `SPDInferencePolicy.observe/infer/reset` |

SPD-specific temporal experts, masks and rolling K/V live in `spd.py`.
`tianji_data.py` reads collector HDF5 directly; `dino_weights.py` maps official HF
DINO weights without depending on Transformers. `spd_optim.py` holds the Muon
adapter and EMA. ABC-DiT remains the default and ABC-VLA remains selectable.

### Architecture and boundaries

Defaults: 768 hidden width, 12 heads, MLP ratio 4, eight paired attention levels,
256 observations at 30 Hz, image stride 8, four visual queries per camera,
eight-action chunks, and a 32-timestep causal window. Each action layer reads
its corresponding observation layer's projected normalized-input K/V directly.
Training uses independent per-chunk flow times, noise-to-data velocity targets,
and 0.03 noise on state/previous-action conditioning. Sampling uses ten Euler
steps. DINO remains both frozen and in eval mode during policy training.

The paper does not fully specify every projection or parameter-sharing choice.
This implementation uses a shared initial ABC pool with camera-specific queries,
and independent ordinary camera cross-attention after observation blocks 2/4/6.
The terminal observation level retains its normalized K/V projection but drops
Q/O/FFN and final visual refresh outputs with no action-loss consumer. It does
not add redundant linear projections around `MultiheadAttention`.
The actual default count is **224,277,558** parameters (85,669,632 frozen DINO;
56,702,976 action-expert blocks), not the paper's rounded 222M. This is a
paper-based implementation with explicit assumptions, not verified author code.

### Tianji recordings and batch contract

Pass one collector root containing `dataset_config.json` and finalized `.h5`
episodes. For example, `/home/current/Documents/TianjiData/20260914_compressed`
contains the original 21 two-camera recordings at JPEG quality 50. Quality
metadata accepts integers in `[0,100]`; declared image dimensions, RGB JPEG
decoding, successful episode headers and recorded camera groups remain strict.
The separate `20260915_compressed` collection declares three cameras and can
be selected as its own root. Different collection contracts are not merged.

The loader aligns to 30 Hz using only previously available samples, resolves
duplicates by taking the last, and breaks windows at stale streams. Default
maximum ages are 150 ms for joints and 2 s for images. Whole episodes are split
80/20 by seed; normalization is fitted only on usable training segments.

| Batch field | Shape | Meaning |
| --- | --- | --- |
| `state` | `[B,256,54]` | Normalized measured joint positions |
| `previous_actions` | `[B,256,54]` | Previous measured positions, **not recorded commands** |
| `actions` | `[B,32,8,54]` | Future measured positions at anchors 0,8,…,248 |
| `images[camera]` | `[B,32,3,224,224]` | Recorded views, resized/padded and ImageNet-normalized |
| `camera_validity` | `[B,32,3]` bool | Ordered `top,left_wrist,right_wrist` availability |

Joint order is left arm 7, left hand 20, right arm 7, right hand 20.
Missing right-camera frames are masked, omitted from the backbone, and do not
supervise that branch. Without a mask all three cameras are required. No
language, contact masks, or unrecorded command targets are fabricated.

### Train, resume, and log

After the upstream setup (`uv sync --extra dev`), a bounded training command is:

```bash
uv run train.py --policy spd \
  --spd-data.root /home/current/Documents/TianjiData/20260914_compressed \
  --spd-data.dino-checkpoint /home/current/syz/abc/model/model.safetensors \
  --output-dir cache/tianji_spd \
  --batch-size 1 --num-workers 0 --no-compile \
  --flow.max-action-prefix 0 --flow.mask-state-ratio 0 \
  --optim.learning-rate 0.001 --optim.weight-decay 0.1 \
  --optim.lr-warmup-steps 0 \
  --train-steps 10000 --log-every 10 \
  --val-every 250 --val-batches 16 \
  --ckpt-every 100 --keep-last-checkpoint-only
```

The explicit common optimizer flags select the paper's constant-LR recipe;
ABC-DiT's defaults are unchanged. Do not use the DiT/VLA action-prefix or state
dropout knobs for SPD. `--ema-half-life-steps` defaults to 20. The paper's
170k-step/batch-64 recipe is not a claim about what this workstation can run.
Measure capacity before increasing batch size, enabling a third recorded view,
or changing compilation settings.

Native DINO `.pth` and official HF `.safetensors` are supported. Keep the official
HF `config.json` beside its weights. Frozen weights are not embedded in SPD
checkpoints; the checkpoint records their SHA-256 and loading verifies it.

Add `--resume-from cache/tianji_spd/last.pt` to resume. A final checkpoint is saved
even between checkpoint intervals. `--keep-last-checkpoint-only` bounds storage
to one persistent file but needs space for another file during atomic replacement.
Resume restores both optimizers, scheduler, EMA, rank RNG and step-keyed sampling.
Changed source identities, split, normalization, optimizer recipe or batch/world
topology are rejected. `--load-pretrained --pretrained-ckpt-name /path/to/last.pt`
instead initializes compatible SPD weights for a new run/dataset; it is not
optimizer resume. Old `spd-paired-kv-v2` weights require the explicit one-time
conversion below; they cannot be loaded directly. DiT/VLA weights are not SPD imports.

W&B uses the normal `--log-wandb --wandb-project spd` flags and
`WANDB_ENTITY=yizhesun-current-robotics`. Authenticate outside source code.
Use a **new run ID** for this architecture, not the running legacy SPD experiment.
Requested SPD W&B initialization errors stop training instead of silently
dropping logs. Checkpoints stay local; metrics, configuration and validation
results are logged. Validation uses EMA, reports aligned chunk reconstruction
and flow losses, and evaluates up to `val_batches` per rank, not the entire split
unless configured accordingly.

For DDP, launch this same entry point with `torchrun`. Every rank must see the
complete, unchanged collection; ABC's node-sharded converted-data cache is not
the SPD HDF5 input format. Two-rank CPU/Gloo execution is verified; multi-GPU
throughput and FSDP for SPD are not qualified. FSDP remains VLA-only.

### Inference

`SPDPolicy.forward(batch, noise=None, t=None)` returns a scalar flow loss.
`sample_action_chunks` returns normalized `[B,32,8,54]` predictions aligned with
training labels. `sample_actions` conditions on the latest state (index 255),
not anchor 248, and returns one normalized `[B,8,54]` chunk. Rolling cache methods
are `append_observation`, `predict_cached_velocity`, and `sample_actions_cached`.

The ABC-style `SPDInferencePolicy` accepts a checkpoint, `SPDPolicyConfig`, and
device. Supply the matching DINO path and model config; EMA loading is the
default. `observe(obs)` takes physical-unit `state` and `previous_actions`
(`[54]` or `[B,54]`), CHW/BCHW RGB images (uint8 or float `[0,1]`), and a bool
camera mask (`[3]` or `[B,3]`). Call it every 30 Hz tick; images are subsampled
at stride 8. Then `infer()` samples from the cache and returns **radian-valued**
`[8,54]` or `[B,8,54]` actions. `infer(obs)` appends once before sampling; do not
append the same tick twice. `reset()` is required between episodes. Explicit
`noise` makes sampling repeatable. This engine rejects DiT RTC prefixes.

This engine is a prediction interface, not hardware actuator control. The
Tianji simulation adapter below adds named 54-joint position control, target
position/rate limiting, and task measurement. Existing YAM hardware deployment
and interactive YAM viewer entry points reject SPD rather than misrouting it
to a 14-D robot. No physical task-success qualification is implied.

### Verification

CPU verification loaded official DINO into the **full default 224.28M model**,
ran `train.py --policy spd` on actual compressed recordings through one
Muon/AdamW/EMA update, validation, and checkpoint save. Reloaded EMA inference
processed actual RGB frames and produced finite `[8,54]` radian actions with
ten Euler steps; masked right-camera pixels did not change the result.
Separate small-model shared-CLI checks exercised two-process CPU/Gloo training,
and continuous versus resumed training produced bitwise-identical policy/EMA
weights. Original ABC-DiT forward/backward/sampling also passed a CPU smoke.
These checks do not establish policy convergence, new-branch GPU performance,
VLA model execution, or physical task success.

### Tianji simulation rollout

`eval_policy.py --embodiment tianji_wuji2 --policy spd` uses the existing ABC
rollout loop, camera rendering, video writer, and summary format. The supported
Tianji scene is `tianji_pick_hammer`, matching the recorded task rather than
pretending that a 54-D policy can drive the YAM catalogue unchanged.

`abc_sim.tianji_env.TianjiTaskEnv` resolves joints, DOFs and actuators by name.
The hammer adds a free joint, so the model has 61 qpos entries but observations
and actions remain exactly 54 joint angles. Position servo targets are limited
to URDF bounds and per-tick target-rate limits; the robot is stepped by real
CPU MuJoCo dynamics, not posed kinematically for each action. Actual joint
velocities are still determined by dynamics, not guaranteed by target limiting.
Physics bad-state warnings and nonfinite states fail the rollout.

SPD receives measured feedback **every 30 Hz control tick**, including the seven
ticks between image updates; inference uses its current rolling cache. Cache and
previous-feedback state reset at each episode. DiT action-prefix RTC and CUDA
graph warmup are not substituted for this history mechanism.

#### Migrate the already-trained weights

The old 251M graph has extra affine query/output projections around visual
attention. The converter composes them into the new attention matrices and
removes only observation outputs that never had a consumer. For example,
`Wq_new = Wq_attention @ Wq_outer`; biases and output projections are composed
as well. It converts raw and EMA weights separately, retains positional
buffers, and adjusts normalization statistics for ABC's denominator epsilon.
It does not import the old package at runtime.

```bash
uv run scripts/convert_spd_checkpoint.py \
  --source-path /home/current/syz/abc/cache/tianji_b4_20260918/last.pt \
  --output-path cache/tianji_b4_converted/last.pt \
  --dino-checkpoint /home/current/syz/abc/model/model.safetensors
```

The converter refuses existing output paths, unsupported architectures, invalid
tensors and mismatched DINO hashes. The result is **weights-only**, for inference
or a new `--load-pretrained` training run. Old Muon/AdamW state is not equivalent
under matrix composition and is deliberately not migrated; `--resume-from` is
not supported for these converted snapshots.

A real held-out 256-step window was checked against the original 10k-step EMA:
all 32 anchor velocities differed by at most `9.54e-7`; two ten-Euler-step action
samples differed by at most `1.19e-7` radians. These are measured FP32 expert
reference errors, not a claim of bitwise equivalence for every possible input.
The measured report is `cache/tianji_b4_converted/equivalence.json`.

#### Build the local scene

Robot/scan assets remain external and are not redistributed by this branch.
`initial_qpos.json` contains a `qpos` array of 54 radians and matching
`joint_names`, ordered left arm, left hand, right arm, right hand. The verified
local file was extracted from a recorded validation episode and was checked
against URDF limits; its source episode/row/timestamp are included.

```bash
uv run scripts/build_tianji_scene.py \
  --robot-xml /home/current/syz/abc/generated/spd_vr/unified_plant.xml \
  --hammer-mesh /home/current/Documents/objects/hammer_m.obj \
  --output cache/tianji_sim/pick_hammer_views.xml \
  --initial-qpos-path cache/tianji_sim/initial_qpos.json \
  --fit-wrist-cameras
```

The builder preserves robot collision masks, meshes, inertias and servo gains.
Collision-only geometry is assigned a hidden visualization group, not removed
from physics. It adds a 0.90 m tabletop and the provided hammer visual mesh,
four longitudinal convex collision regions, and an explicit 0.30 kg fixture
mass with box-approximation inertia. These object properties are assumptions,
not measured contact calibration.

The source wrist mounts were partly occluded by hand/table geometry.
`--fit-wrist-cameras` chooses nominal workspace-facing poses at the supplied
initial robot posture. They are then **fixed relative to each wrist**, never
tracking the object at runtime. These are simulation mounts, not recovered
hardware extrinsics. Omit the flag to retain source mounts. The adjacent scene
JSON records source hashes, collision assumptions and the chosen camera poses.

#### Run the trained model

```bash
MUJOCO_GL=egl uv run eval_policy.py \
  --checkpoint cache/tianji_b4_converted/last.pt \
  --policy spd --embodiment tianji_wuji2 --task tianji_pick_hammer \
  --tianji.model-path cache/tianji_sim/pick_hammer_views.xml \
  --tianji.urdf-path /home/current/syz/abc/assets/tianji_wuji2/tianji_wuji2.urdf \
  --tianji.initial-qpos-path cache/tianji_sim/initial_qpos.json \
  --spd-dino-checkpoint /home/current/syz/abc/model/model.safetensors \
  --camera-backend mujoco --camera-height 360 --camera-width 640 \
  --no-rtc --no-fast-inference --prefix-length 0 --execute-chunk-dim 8 \
  --num-worlds 1 --num-chunks 16 --device cuda \
  --save-video --log-every-chunk --output-dir outputs/tianji_spd_10000
```

The default active policy cameras remain `top,left_wrist`; all three cameras
are rendered for the video. Enabling a camera absent from the checkpoint's
training provenance is rejected. Right-camera rendering is not evidence of a
trained right-camera policy branch.

The task success predicate is a height gain of at least 0.05 m plus hand/object
contact sustained for six control ticks. Floor/arm contact does not count.
The threshold is configurable; this is a simulation-specific lift criterion,
not a calibrated real-world success metric. Resets use the fixed provided
scene, not the YAM randomizers.

Outputs include `world_000.mp4` and `summary.json`, containing actual physics
timestep/decimation, lift/contact status, tracking error and command clipping.
The verified run executed 128 control ticks (4.267 simulated seconds), with
zero BADQPOS/BADQVEL/BADQACC events. **It did not lift the hammer.** A decoded
five-frame visual contact sheet is retained as `contact_sheet.png`.
Steady cached action sampling took about 35–38 ms per chunk. Per-tick visual
updates, physics and rendering are included in `steps_s`, not `infer_s`.
The 30 Hz value describes simulation control time, not a demonstrated wall-clock
real-time guarantee. Both installed MuJoCo 3.12.0 and upstream-pinned 3.8.0
completed the 128-tick rollout; the latter was tested in an isolated temporary
package directory without changing the existing environment. The full suite
passed 97 tests on 3.8.0, including closed-loop feedback across episode resets
and fail-closed handling of contact-buffer overflow.

The inherited surface-patch robot assets produce narrow-hull Qhull precision
warnings during compilation. They were not suppressed or “fixed” by dropping
collision geometry. Together with uncalibrated camera mounts, approximate
hammer contacts and the real-to-simulation visual gap, this limits what can be
concluded from the rollout. A functioning inference loop is not task mastery.
