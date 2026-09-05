# **Scalable Behavior Cloning with Open Data, Training, and Evaluation**

<p align="center">
  <strong>
    <a href="https://abc.bot">Project Website</a> |
    <a href="https://abc.bot/abc.pdf">Paper</a> |
    <a href="https://huggingface.co/datasets/XDOF/ABC-130k">Raw Data</a>
  </strong>
</p>

![](assets/teaser.jpg)


Code for the ABC project.

> Note: we have released a minimal training pipeline for ABC-DiT & conversion scripts for the data. We also re-host a small subset of the sim data and real data for 1 task to allow users to get started. Please check back later for the full code release, including VLA training, real deployment infra & pretrained checkpoints.


## Release Roadmap
- [x] June 17 -- Release Minimal Training Pipeline
- [ ] End of June -- Release all sim data
- [ ] By end of July -- full code release

## Setup

```bash
# Install uv if you don't have it.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Install ffmpeg.
sudo apt-get install -y ffmpeg     # on Linux
```

```bash
# Pin Python and create the project venv. uv reads pyproject.toml here.
cd abc
uv python pin 3.12
uv sync
# Run the SPD-VR regression suite (pytest is in the dev dependency group).
uv run pytest -q tests
```

## SPD-VR branch: PICO to a 54-DoF digital twin

The `spd-vr` branch adds a simulation-only adaptation of *Pre-training Visual
Dexterity in Simulation* for PICO 4 Ultra, Tianji dual arms, and Wuji2 dual
hands. It keeps ABC's DINOv3 implementation, image preprocessing,
flow-matching primitive, distributed training structure, and MuJoCo-Warp
adapter. It does not load an ABC 14-DoF checkpoint and has no language input or
real-robot output.

The arm process uses one persistent OSQP Jacobian velocity QP per side with
URDF position/velocity bounds. The PICO wrist's joint[1] pose must remain
stable for ten frames before either QP may publish a target. The policy order
is fixed and independent of URDF/MJCF storage order:

```text
left arm 7 + left hand 20 + right arm 7 + right hand 20 = 54 DoF
```

Build the MuJoCo plant from the authoritative URDF. The normal command uses a
deterministic, adaptive convex-surface-patch backend and fails closed if any
proxy misses its surface gate. It welds each connected source component,
partitions it spatially, recursively bounds every convex piece to 64 vertices,
and measures the bidirectional surface p95 before publishing. The locally
authorized vendor bundle now compiles to 62 contact records / 23,099 pieces;
the worst record is about 2.59 mm p95 (below the 3 mm arm gate), and
`Link_Base.STL` is about 2.16 mm with 4,617 pieces. `rtree` is a declared
dependency so large-mesh quality checks use a bounded spatial index. CoACD
remains available through the collision API and its bounded trials are kept as
diagnostic evidence, but they do not silently become a contact proxy when they
miss the gate. `--raw-collisions` is only a display/control smoke mode because
MuJoCo treats each raw mesh as a convex hull.

The generated contact artifact is still a local validation result; vendor
assets remain excluded from the public branch until their redistribution terms
are confirmed. Additional CoACD trials, source hash, patch settings, and the
exact quality-measurement definition are recorded in
[`docs/spd_vr_collision_diagnostics.md`](docs/spd_vr_collision_diagnostics.md).

The public branch does not redistribute the vendor Tianji-Wuji2 URDF or STL
bytes.  Place an authorized local bundle under `assets/tianji_wuji2/` and
verify `SHA256SUMS` before running model compilation or simulation tests; the
directory's README records the publication restriction.

```bash
uv run spd-model \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf \
  --output generated/spd_vr \
  --cache cache/spd_collision

# Development smoke build only:
uv run spd-model \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf \
  --output /tmp/spd_vr_raw --raw-collisions

# Contact-data gate (raw builds fail here by design):
uv run spd-model --verify --verify-contact \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf \
  --output generated/spd_vr
```

For a quick two-terminal smoke test, run the PICO bridge and the combined live
loop below. The production-shaped three-window launcher separates the 200 Hz
arm-IK process from the 60 Hz viewer: only the viewer owns the complete
MuJoCo plant, while the arm process publishes the 272-byte arm-target frame.

```bash
uv run spd-vr-live \
  --model generated/spd_vr/unified_plant.xml \
  --arm-model generated/spd_vr/arm_ik.xml \
  --viewer --record-to cache/spd/train/episode_0001.hdf5

uv run spd-pico-bridge --sdk-library /path/to/libpxrea.so

# Three windows: viewer (Zenoh router + plant), arm_ik, pxrea_bridge.
scripts/start_spd_vr.sh --detach \
  --model generated/spd_vr/unified_plant.xml \
  --arm-model generated/spd_vr/arm_ik.xml \
  --sdk-library /path/to/libpxrea.so
# If multiple PICO devices are online, add --serial <device-id>.
scripts/start_spd_vr.sh --status
scripts/stop_spd_vr.sh
```

For a repeatable three-process protocol smoke, replace the SDK option with
`--fake-source-jsonl <events.jsonl> --wait-for-shutdown`; the bridge then stays
alive after replay until `uv run spd-control shutdown` is sent. This mode is
for simulation/CI only and does not represent PICO hardware availability.

The optional `spd-control pause|resume|realign|reset|shutdown` command sends
the 40-byte versioned control frame to all three processes. Pause freezes the
viewer physics and recorder; reset/epoch changes require a fresh neutral
alignment before arm targets become valid.

Each HDF5 episode contains the full raw 60 Hz stream: actual 54-D qpos/qvel
and `raw/action/qpos`, the 54-D teleoperation target, three 224×168 RGB and instance-segmentation
views, raw PICO hands and source/bridge timestamps plus scale/epoch/sequence
metadata, MuJoCo-derived object/contact records and hand-object contact flags,
per-side validity, and MuJoCo `mjSTATE_FULLPHYSICS`. A 30 Hz index and its
nominal grid ordinal reference those rows without duplicating or re-encoding
the source data; dropped grid rows cannot be crossed by a training window.
Policy labels are always future actual MuJoCo qpos; teleoperation targets are
audit data only. Training normalization is train-only and fail-closed: a
persisted file must contain finite, positive-std 54-D qpos/action vectors;
malformed statistics are rejected before Dataset or training access.

```bash
uv run spd-validate-episode cache/spd/train/episode_0001.hdf5 --checksums
uv run spd-filter-contacts \
  cache/spd/train/episode_0001.hdf5 \
  cache/spd/train/episode_0001.contact-filtered.hdf5
uv run torchrun --standalone --nproc-per-node 8 train_spd.py \
  --dataset-root cache/spd
# Resume a saved SPD run (model/EMA/optimizers/RNG and DINO hash are checked):
uv run torchrun --standalone --nproc-per-node 8 train_spd.py \
  --dataset-root cache/spd --resume cache/spd_checkpoints/last.pt
```

Contact filtering never deletes the raw stream.  It adds
`training/contact_eligible` and `[start,end)` `training/segments_30hz`; any
continuous hand-object-free span longer than ten seconds becomes a hard
segment boundary.  The manifest records the removed raw spans and their
timestamps.  The Dataset consumes these segments, so a 258-row sample cannot
bridge a filtered interval.

The paper's six scenes and 17 task registry is deterministic and produces a
reset manifest with seeded object pose, mass, friction, color, and contact
metadata.  Build a scene model on top of a verified plant, then pass its scene
manifest to the live recorder:

```bash
uv run spd-scene --list
uv run spd-scene --scene jenga --task hollow_tower --seed 7 \
  --base-model generated/spd_vr/unified_plant.xml \
  --output-model generated/spd_vr/jenga_hollow_tower.xml
uv run spd-vr-live \
  --model generated/spd_vr/jenga_hollow_tower.xml \
  --arm-model generated/spd_vr/arm_ik.xml \
  --scene-manifest generated/spd_vr/jenga_hollow_tower.scene.json \
  --record-to cache/spd/train/jenga_0001.hdf5
```

For recording, keep the scene XML beside the verified compiler artifacts
(`model_manifest.yaml` and `collision_manifest.yaml`); the live preflight
checks both manifests before opening HDF5.

Before opening the three-window process graph, an optional read-only preflight
checks the PICO/RoboticsService ADB reverse entry, vendor SDK, Python
dependencies, display, generated artifacts, Zenoh endpoint, and managed tmux
session.  Add `--require-contact` (or use it automatically with recording) to
also enforce the contact surface-quality gate:

```bash
uv run spd-vr-preflight --manifest generated/spd_vr/model_manifest.yaml \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf --require-contact
scripts/start_spd_vr.sh --preflight --record-to cache/spd/train/episode.hdf5
```

The command is read-only with respect to ADB: it uses `adb reverse --list` and
never mutates device forwarding.  `--fake-source` replaces the hardware SDK
for protocol/CI smoke, but does not waive artifact or contact checks.

The bounded acceptance set is also read-only.  It validates all registered
scene resets, the generated model/contact gate, and (when supplied) every HDF5
episode with checksums; `--replay` additionally restores full MuJoCo state:

```bash
uv run spd-vr-acceptance --seed-count 3 --episodes cache/spd/train \
  --manifest generated/spd_vr/model_manifest.yaml \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf --require-contact --replay
```

No `--episodes` means the data portion is reported as not requested.  Passing
an empty episode directory, a malformed episode, an unqualified collision
manifest, or a replay/model hash mismatch returns non-zero.

For a simulation-only real-time diagnostic (no HDF5 publication), run:

```bash
uv run spd-sim-benchmark --model generated/spd_vr/unified_plant.xml \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf --duration 5 --render
```

It reports p50/p95/p99 control-tick latency against the 60 Hz budget and
explicitly does not waive the collision, hardware, or policy gates.

Once the licensed official DINOv3 checkpoint and target GPU are available, the
full streaming policy benchmark records the checkpoint hash and cached-action
latency. It does not download weights or fall back to ABC/tiny checkpoints:

```bash
uv run spd-policy-benchmark \
  --dino-checkpoint cache/dinov3_vitb16_pretrain_lvd1689m.pth \
  --device cuda:0 --measure-ticks 64 --compile
```

Add `--enforce-deadline` only after reviewing the target-GPU run; a missing
checkpoint or unavailable CUDA device returns a structured non-zero error.

Replay restores the recorded `mjSTATE_FULLPHYSICS` rows and checks qpos/qvel,
state tolerance, optional hand-object contacts, and (with `--render`) camera
availability against the same model hash:

```bash
uv run spd-replay-episode cache/spd/train/jenga_0001.hdf5 \
  --model generated/spd_vr/jenga_hollow_tower.xml \
  --urdf assets/tianji_wuji2/tianji_wuji2.urdf
```

One SPD training example spans 258 consecutive 30 Hz rows: one true previous
row, 256 observation rows, and the farthest future label at `+256`. The model
trains all 32 eight-action chunks in parallel with a causal 32-timestep window,
a frozen DINOv3 ViT-B/16, four pooled tokens per camera, an 8-block observation
trunk, an independently parameterized 8-block action expert, Muon/AdamW, and a
20-step-half-life EMA. The 54-D model is about 223M parameters. Deployment
appends observation/projection KV at every 30 Hz tick, retains the matching
32-timestep window, and uses 10 Euler steps to emit each eight-action chunk.

The main implementation lives in `spd_vr/`; `abc_minimal/` remains the released
ABC baseline except for the shared direction-explicit flow helper.

Training-time visual randomization is disabled by default and can be enabled
with `--visual-randomization-probability` (instance-segmentation keyed color
replacement). Left/right symmetry is also opt-in with
`--symmetry-probability` plus a reviewed `--symmetry-spec-path`; the required
joint-axis sign table is never inferred automatically.

## Training

First we need to download the requisite data (norm stats and either a sample or full data.)
```bash
uv run prepare.py            # preview (a few episodes of data, ~130MB)
uv run prepare.py --full     # all data for bottles in bin (~35GB)
uv run prepare.py --checkpoint  # add to also pull the pretrained 75k policy (~7.7GB)
```

This populates the cache dir (default `cache/`, or `ABC_CACHE` if set) with:

```
cache/
  norm_stats.json                       # state/action z-score stats
  train_real/episode_<uuid>/{states_actions.bin, combined_camera-images-rgb.mp4, episode_metadata.json}
  val_real/...
  train_sim/...
  val_sim/...
```

Set `ABC_CACHE=/path/to/cache` before running commands if you want the cache
outside the repository.

:warning: Note: `prepare.py` does not download DINO weights. Review and follow the DINO license terms, then download the weights from [Meta](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) or [Hugging Face](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m). Save the file as `dinov3_vitb16_pretrain_lvd1689m.pth` in the cache dir. :warning:

The command to run training is below. Note that this is for single node training with 8 GPUs, change `nproc-per-node` if you want.

```bash
uv run torchrun --standalone --nproc-per-node 8 train.py
```
The dataclass config is exposed as CLI flags; `uv run python train.py --help`
shows training, optimizer, flow, CLIP asset, and model options such as
`--model.hidden-size`, `--model.depth`, and `--model.camera-keys`. The default
model config is the checkpoint-compatible ABC-DiT XL shape.

If you pulled with `--full` above, this checkpoint is expected to work for
the bottles in bin task in both sim and real. The performance should be similar
to [this](assets/bottles_real.mp4).

Training defaults in `abc_minimal/config.py` match the production reference
finetune (lr 1e-4 with a 1k-step linear warmup, AdamW(0.9, 0.95), wd 0.01,
grad clip 10, prefix conditioning max 4 with noise 0.05, 10% state masking,
batch 90/GPU, 75k steps, hours-weighted 2-component mixture).
The dataclass config is exposed as CLI flags; `uv run python train.py --help`
shows training, optimizer, flow, CLIP asset, and model options such as
`--model.hidden-size`, `--model.depth`, and `--model.camera-keys`. The default
model config is the checkpoint-compatible ABC-DiT XL shape. If you have fewer GPUs
than 8 you may need to reduce nproc per node or if you have less than 80Gb of
VRAM you may need to reduce `--batch-size`.

The above training yields ~2.6-3 iterations / sec on H100/H200. It achieves a training
loss of ~`0.048` after 75k steps.

## Evaluation

You can either evaluate a checkpoint you trained yourself (drops into
`cache/finetune_checkpoints/last.pt`) or download our public
pretrained 75k-step bottles policy:

```bash
# Pulls cache/bottles_75k.pt (~7.7 GB) from the public bucket,
# alongside norm_stats.json and the preview tar.
uv run prepare.py --checkpoint
```

To visualize the policy live:

```bash
uv run viz_policy.py --sim.checkpoint cache/bottles_75k.pt --port 8080
```

opens a viser window at `localhost:8080`. It should look like this:

![](assets/sim_eval.gif)

`eval_policy.py` runs a more systematic evaluation:

```bash
# 20 worlds, save a video of each rollout, log per-chunk progress.
uv run eval_policy.py \
    --checkpoint cache/bottles_75k.pt \
    --num-worlds 20 \
    --save-video --log-every-chunk

# Output: $REPO/outputs/sim_eval_put_bottles/
#   summary.json     — success_rate, num_success, mean_max_bottles_in_bin
#   world_*.mp4      — per-world rollout videos (with --save-video)
```

Useful flags:

- `--num-worlds N` — independent random scenes (default 5).
- `--num-chunks N` — action chunks per rollout; each chunk is
`--execute-chunk-dim` actions (defaults: 120 chunks × 15 = 1800 sim steps).
- `--diffusion-steps N` — flow-matching Euler steps per inference
(default 10, matches production).
- `--checkpoint` — accepts a local `.pt` path or `s3://…/<file>.pt`.
- `--norm-stats-path` — explicit `norm_stats.json` (otherwise uses the
one bundled in the checkpoint).
- `--fast-inference` / `--no-fast-inference` (default on) — bf16 +
torch.compile + CUDA-graph captured `sample_actions`. ~5× faster
inference; first call pays a one-time ~25 s compile cost.
- `--vanilla-physics` / `--no-vanilla-physics` (default off) — enable
vanilla CPU `mujoco.mj_step` for env physics instead of the default
single-world mjwarp path.  This is because for single environments, it's
faster to use vanilla mujoco. Rendering still happens in MJWarp.

Note that the first launch compiles MJWarp's CUDA kernels (~1 min).

## Episode exports & training data format

While we host a single task in training format, there are many more in the ABC Dataset.
The ABC-130k MCAPs are hosted on Hugging Face at
[`XDOF/ABC-130k`](https://huggingface.co/datasets/XDOF/ABC-130k). The dataset
is gated, so accept access on the dataset page and set `HF_TOKEN` before
downloading.

Download all MCAPs for one task and convert them in place:

```bash
uv run export_hf_task.py --task organize_the_condiment_bottles
```

By default this downloads both `train` and `val`, stages raw MCAPs under
`$ABC_CACHE/hf_tasks/<task>/`, runs `export_mcap.py`, writes converted episodes
to `$ABC_CACHE/train_real/` and `$ABC_CACHE/val_real/`, then deletes the staged
raw MCAPs after each successful split conversion. For a quick smoke test:

```bash
uv run export_hf_task.py --task organize_the_condiment_bottles --split train --max-episodes 1
```

If you already have local MCAPs, call the lower-level converter directly:

```bash
uv run export_mcap.py ./train_run_1 ./out
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
uv run export_mcap.py ./train_run_1 ./out 8
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

## Licenses

This repository includes and adapts code from the following third-party
projects. Original license files and copyright headers are retained in all
cases. Bundled license texts live under `abc_minimal/third_party/`.

| Project | License | License file | Inclusion | What we use/adapt |
| --- | --- | --- | --- | --- |
| [DINOv3](https://github.com/facebookresearch/dinov3) | DINOv3 License (Meta) | [`abc_minimal/third_party/dinov3/LICENSE.md`](abc_minimal/third_party/dinov3/LICENSE.md) | Adapted (`abc_minimal/dit.py`); pretrained weights downloaded by the user | ViT-B/16 vision backbone (`DinoRope`, `DinoAttention`, `DinoMlp`, etc.) |
| [OpenAI CLIP](https://github.com/openai/CLIP) | MIT | [`abc_minimal/third_party/clip/LICENSE`](abc_minimal/third_party/clip/LICENSE) | Adapted (`abc_minimal/dit.py`); ViT-B/32 text weights + BPE vocab downloaded at runtime | CLIP text encoder + BPE tokenizer (`CLIPBPETokenizer`, `CLIPTextTower`, `CLIPTextEmbedder`) |
| [i2rt YAM](https://github.com/i2rt-robotics) | MIT | [`assets/put_bottles/assets/i2rt_yam/LICENSE`](assets/put_bottles/assets/i2rt_yam/LICENSE) | Vendored under `assets/put_bottles/assets/i2rt_yam/` | YAM robot MuJoCo model, meshes, and scene assets |
| Wuji retargeting | MIT | [`abc_minimal/third_party/wuji-retargeting/LICENSE`](abc_minimal/third_party/wuji-retargeting/LICENSE) | Vendored as the small Python runtime under `wuji_retargeting/` | PICO hand keypoints to Wuji2 20-DoF joint targets |

The Tianji-Wuji2 URDF/STL bundle under `assets/tianji_wuji2/` is hardware-vendor
material supplied for this research integration. Its redistribution terms are
not declared in this repository; confirm them before publishing or
redistributing the asset bundle. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
for the publication gate and the complete provenance table.

### DINOv3 use restrictions

The DINOv3 License prohibits use of the DINO Materials (including weights and
derivatives) for: military purposes; activities subject to ITAR or other
export-control regimes covering defense articles; nuclear applications;
espionage; and the development, manufacture, or use of weapons. Downstream
users who load DINOv3 weights through this codebase are bound by these
restrictions; see `abc_minimal/third_party/dinov3/LICENSE.md` for the full
license text.


## Citation

Please cite this work as

```
@misc{abc2026,
  title         = {Scalable Behavior Cloning with Open Data, Training, and Evaluation},
  author        = {Arthur Allshire and Himanshu Gaurav Singh and Ritvik Singh and Adam Rashid and Hongsuk Choi and David McAllister and Justin Yu and Yiyuan Chen and Huang Huang and Pieter Abbeel and Xi Chen and Rocky Duan and Phillip Isola and Jitendra Malik and Fred Shentu and Guanya Shi and Philipp Wu and Angjoo Kanazawa},
  year          = {2026},
  eprint        = {2606.27375},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2606.27375},
  url           = {https://arxiv.org/abs/2606.27375},
}
```
