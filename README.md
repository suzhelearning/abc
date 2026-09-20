# ABC-SPD：Tianji / Wuji2 双臂灵巧手策略

本项目基于 [ABC](https://github.com/amazon-far/abc)，面向 **Tianji 双臂 + Wuji2 双灵巧手的 54 自由度机器人**，
提供真实示范数据加载、SPD 策略训练、有状态推理和 MuJoCo 闭环仿真。
SPD 接入现有 `train.py --policy spd`，不依赖旧版 `spd_vr` 包，也不使用独立的 `train_spd.py`。

**当前边界：训练、推理和仿真链路已接通，但尚未证明抓锤任务成功；没有经过验证的 Tianji 真机控制适配器。**
本仓库不附带 SPD 作者发布的 checkpoint、Tianji 示范数据或完整机器人/扫描资产。

## 目录

- [项目结构](#项目结构)
- [SPD 模型与数据流](#spd-模型与数据流)
- [环境安装](#环境安装)
- [数据与权重准备](#数据与权重准备)
- [训练与续训](#训练与续训)
- [策略推理接口](#策略推理接口)
- [Tianji 仿真使用](#tianji-仿真使用)
- [验证与已知限制](#验证与已知限制)
- [常见问题](#常见问题)
- [上游 ABC 功能](#上游-abc-功能)
- [许可与引用](#licenses)

## 项目结构

```text
abc-spd/
├── train.py                       # 共用训练入口，SPD 必须指定 --policy spd
├── eval_policy.py                 # 离线闭环仿真、视频和评估报告
├── pyproject.toml                 # Python 依赖与 uv 索引配置
├── abc_minimal/
│   ├── config.py                  # SPD 模型、数据、训练及仿真配置
│   ├── spd.py                     # 观测/动作专家、时序注意力、缓存与采样
│   ├── tianji_data.py             # Tianji HDF5 读取、对齐、切窗与归一化
│   ├── dino_weights.py            # 官方 DINOv3 权重加载与格式映射
│   ├── spd_optim.py               # Muon/AdamW 适配与 EMA
│   ├── train_loop.py              # 共用训练、验证、分布式与日志
│   ├── checkpointing.py           # checkpoint 原子保存与恢复
│   ├── spd_conversion.py          # 旧版 SPD 权重转换
│   └── policy.py                  # SPDInferencePolicy 推理接口
├── abc_sim/
│   ├── tianji_scene.py            # 机器人、桌面、锤子和相机的场景构建
│   └── tianji_env.py              # 54 关节反馈、位置控制、接触和抬升判定
├── scripts/
│   ├── build_tianji_scene.py      # 构建 Tianji 抓锤场景
│   └── convert_spd_checkpoint.py  # 旧版 checkpoint 转换入口
├── tests/                         # 模型、数据、训练、推理、转换和仿真测试
├── cache/                         # 本地数据、权重、场景等；不是随仓库发布的资产
├── outputs/                       # rollout 视频和 summary.json
└── deploy/                        # 上游 YAM 部署工具，不是 Tianji 真机控制器
```

详细参考：[训练与模型](abc_minimal/README.md#tianjiwuji2-spd)、
[仿真环境](abc_sim/README.md#tianjiwuji2-embodiment)、
[上游部署](deploy/README.md)。

## SPD 模型与数据流

### 机器人与动作空间

策略关节顺序固定为：

| 部位 | 维数 | 单位 |
| --- | ---: | --- |
| 左臂 | 7 | rad |
| 左手 | 20 | rad |
| 右臂 | 7 | rad |
| 右手 | 20 | rad |
| 合计 | **54** | rad |

模型输出是关节位置目标，不是力矩或末端位姿。
数据加载器将采集器的源关节顺序映射为上述策略顺序；不能仅按长度拼接任意 54 个关节。

### 模型结构

```text
顶部/腕部 RGB 图像 → 冻结 DINOv3 → 视觉特征
                                      ↓
关节位置 + 上一时刻实测位置 → 时序观测专家 → 各层 K/V
                                                   ↓
噪声动作块 → 动作专家 + Flow Matching / Euler 采样 → 8 × 54 关节目标
                                                   ↓
                         MuJoCo 位置伺服 → 实测反馈 → 下一控制时刻
```

默认隐藏维度 768、12 个注意力头、8 层配对专家；总参数 **224,277,558**，其中 DINO 冻结参数为 **85,669,632**。
训练窗口包含 256 个观测时刻，控制时间为 30 Hz；图像每 8 个时刻更新，注意力因果窗口为 32 个时刻。
推理默认使用 10 步 Euler 采样，每次产生 8 步动作。

当前 SPD **不接收语言提示词**，行为来自示范数据及观测历史。
`--task tianji_pick_hammer` 选择仿真场景与评分逻辑，并不是送入模型的文字指令。
本实现依据 [SPD 论文](https://arxiv.org/html/2608.15917v1#A1.SS5) 将 56 关节适配为 54 关节，
并明确记录了论文未完全指定部分的实现假设；不是经过作者代码核验的官方复现。

## 环境安装

### 系统要求

- 推荐 **Linux + Python 3.12 + NVIDIA GPU**；包声明支持 Python ≥ 3.10。
- Linux PyTorch 固定为 **2.11.0+cu128**，需要兼容 CUDA 12.8 的 NVIDIA 驱动。
- MuJoCo 固定在 **3.8.x**，另包含上游所需的 MuJoCo Warp 依赖。
- Tianji 路径使用 **CPU MuJoCo 动力学**；策略可用 CUDA，EGL 相机渲染需要相应的图形驱动。
- 未建立通用的最低显存要求。先用 batch 1、不编译的配置验证，再按实际容量调整；不要套用上游 DiT 的 8 卡训练规模。

以下命令在 Bash 中执行。Ubuntu / Debian 系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y git curl ffmpeg libegl1 libgl1
nvidia-smi
```

`libegl1` 不替代 NVIDIA EGL 驱动。如果无头渲染失败，需要安装与现有驱动版本匹配的 NVIDIA GL/EGL 软件包，
不要照抄其他机器的驱动版本号。

### 安装项目

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone --branch abc-spd https://github.com/suzhelearning/abc.git abc-spd
cd abc-spd
uv python pin 3.12
uv sync --extra dev
```

已有本地仓库时，在其根目录执行最后两条命令即可。下文所有命令均从仓库根目录运行；
`uv run` 会使用该项目环境，不要直接借用另一个 checkout 的 Python 环境。
SPD 训练和仿真不需要 `uv sync --extra deploy`，安装该 extra 也不会增加 Tianji 真机控制能力。

### 安装检查

```bash
uv run python -c "import torch, mujoco, h5py; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); print('mujoco:', mujoco.__version__)"
uv run train.py --help
uv run eval_policy.py --help
uv run scripts/build_tianji_scene.py --help
```

计划使用 CUDA 时，第一条应显示 `cuda: True`。检查本身不会下载数据、加载 SPD checkpoint 或启动机器人。

## 数据与权重准备

### Tianji 录制数据

准备采集器产生的一个完整集合，至少包含两个成功结束的 episode：

```text
TianjiData/
├── dataset_config.json
├── episode_001.h5
└── episode_002.h5
```

这是已有采集器的 schema v1，不是通用 HDF5 格式。关键约束：

- `joint_unit=rad`、`policy_rate_hz=30`，54 个唯一源关节名。
- 图像为 JPEG，解码颜色顺序为 RGB，声明尺寸 1280 × 720；JPEG quality 为 0–100 的整数。
- `.h5` 必须已完成、具有 `success=true` 等合法头信息，摄像头组必须与配置一致；`.partial.h5` 不参与训练。
- 相机名称为 `top`、`left_wrist`、`right_wrist` 的非空子集。缺失视角通过掩码处理，已声明但损坏或缺失的流仍报错。
- 不直接混合不同采集契约的两相机和三相机集合。

加载器按 30 Hz 因果对齐；默认关节最大样本年龄 150 ms、图像最大年龄 2 s。
过期数据会切断窗口，完整 episode 按种子做约 80/20 划分，归一化仅拟合训练片段。

| 训练字段 | 默认形状 | 含义 |
| --- | --- | --- |
| `state` | `[B,256,54]` | 归一化实测关节位置 |
| `previous_actions` | `[B,256,54]` | 上一时刻实测位置，**不是控制命令** |
| `actions` | `[B,32,8,54]` | 32 个锚点各自后续 8 步的实测位置 |
| `images[camera]` | `[B,32,3,224,224]` | 等比例缩放、补边并归一化的图像 |
| `camera_validity` | `[B,32,3]` | 顶部、左腕、右腕顺序的 bool 掩码 |

仅训练不需要机器人 URDF 或仿真资产。`prepare.py` 的 ABC 数据下载流程不能替代 Tianji HDF5 数据准备。
完整契约见 [Tianji recordings](abc_minimal/README.md#tianji-recordings-and-batch-contract)。

### DINOv3 权重

先阅读并接受 [DINOv3 许可与访问条件](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)。
获准访问后，可使用项目环境中的 Hugging Face CLI 下载：

```bash
uv run hf auth login
uv run hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  model.safetensors config.json \
  --local-dir cache/dinov3-vitb16
```

官方 HF `config.json` 必须保留在 `model.safetensors` 同目录。加载器也支持原生 DINO `.pth`；
本文命令统一使用 HF 格式。`prepare.py` 不下载 DINO 权重。

SPD checkpoint **不内嵌冻结的 DINO 权重**，训练时记录 SHA-256，推理时必须提供同一份权重。
不要把不同版本的 DINO 文件替换到已有 checkpoint 下。

在同一个终端设置后续路径；将 `/absolute/path/...` 替换为自己机器上真实存在的路径：

```bash
export TIANJI_DATA=/absolute/path/to/TianjiData
export DINO_WEIGHTS="$PWD/cache/dinov3-vitb16/model.safetensors"
export SPD_RUN="$PWD/cache/tianji_spd"
```

## 训练与续训

### 单卡训练

```bash
uv run train.py --policy spd \
  --spd-data.root "$TIANJI_DATA" \
  --spd-data.dino-checkpoint "$DINO_WEIGHTS" \
  --output-dir "$SPD_RUN" \
  --batch-size 1 --num-workers 0 --no-compile \
  --flow.max-action-prefix 0 --flow.mask-state-ratio 0 \
  --optim.learning-rate 0.001 --optim.weight-decay 0.1 \
  --optim.lr-warmup-steps 0 \
  --train-steps 10000 --log-every 10 \
  --val-every 250 --val-batches 16 \
  --ckpt-every 100 --keep-last-checkpoint-only
```

**不要省略 `--policy spd`**：默认策略仍是 ABC-DiT。上述命令显式关闭 SPD 不支持的动作前缀和状态 dropout，
采用 Muon/AdamW、常量学习率和默认半衰期 20 步的 EMA。batch 1 是容量起点，不是收敛或成功率保证。

首次只检查链路时，将上述命令的 `--train-steps` 改为 `1`、`--val-every` 和 `--val-batches` 改为 `1`，
并使用单独的输出目录；这仍运行完整默认模型，不是轻量假数据测试。
训练产物包括 `last.pt` 和 `run_metadata.json`。只保留最后一个 checkpoint 时，原子替换仍需要另一个文件的临时空间。

### 续训与新任务初始化

在原训练命令中追加以下参数可恢复中断的训练：

```bash
--resume-from "$SPD_RUN/last.pt"
```

这是追加参数，不是独立 Shell 命令。续训恢复优化器、调度器、EMA、随机状态和采样进度。
保持原始数据、划分、归一化、优化配置和 batch/world 拓扑不变；若训练已完成，需要提高目标 `--train-steps` 才会继续更新。

要用兼容 SPD 权重开始一个新训练，而不是恢复优化器，在完整训练命令中改用：

```bash
--load-pretrained --pretrained-ckpt-name /absolute/path/to/spd/last.pt
```

新训练应选择新的输出目录。不能把 DiT/VLA checkpoint 当作 SPD 权重使用。
启用 W&B 时，在终端完成 `uv run wandb login`，再追加 `--log-wandb --wandb-project spd`；
按自己的账户设置 `WANDB_ENTITY`，不要把令牌写入代码。SPD 请求启用 W&B 后，初始化失败会终止训练。

多进程可用 `uv run torchrun --standalone --nproc-per-node N train.py` 替换命令入口并保留 SPD 参数。
每个 rank 必须看到完整且未变的同一集合；不使用 ABC 的节点分片缓存。
已记录两进程 CPU/Gloo 验证，但未验证多 GPU 吞吐；SPD 不支持 `--fsdp`，该选项仅用于 VLA。

## 策略推理接口

`abc_minimal.policy.SPDInferencePolicy` 提供物理单位的有状态预测接口，不直接驱动执行器。
构造时提供 checkpoint、`SPDPolicyConfig` 和 device，匹配训练模型配置及 DINO 文件；
配置必须关闭 `fast_inference`，并保持 `rtc_prefix_length=None`。默认加载 EMA。

| 接口 | 用法 |
| --- | --- |
| `observe(obs)` | 每个控制时刻追加一次实测反馈；模型按 stride 8 使用图像 |
| `infer()` | 基于当前缓存预测 `[8,54]`，批量时为 `[B,8,54]`，单位 rad |
| `infer(obs)` | 先追加一次该观测，再采样；不要和同一时刻的 `observe(obs)` 重复使用 |
| `reset()` | 每个 episode 开始前清除历史 |

`obs` 包含 `state`、`previous_actions`（均为 `[54]` 或 `[B,54]` 的实测 rad 值）、
`images`（CHW/BCHW RGB，uint8 或 `[0,1]` 浮点）及 `camera_validity`（bool `[3]` 或 `[B,3]`）。
没有掩码时必须提供全部三路相机；默认两相机输入应明确使用 `[True, True, False]`。
接口内部执行预处理，不要再次传入训练时已归一化的图像或关节值。

不必自己编写硬件循环即可验证策略，使用下面的仿真入口。

## Tianji 仿真使用

### 准备外部资产

除了 SPD checkpoint 和 DINO，还需要：

| 资产 | 要求 |
| --- | --- |
| 机器人 MJCF/XML | 已可加载的 Tianji/Wuji2 机器人模型及其引用的 mesh/include、关节、位置执行器和相机 |
| 匹配的 URDF | 用于解析策略关节名称、位置范围和目标变化率限制 |
| 锤子 OBJ | 用于构建抓锤场景的外部扫描网格 |
| 初始姿态 JSON | `qpos` 为 54 个 rad 数值，`joint_names` 与策略顺序匹配，姿态在 URDF 限位内 |

这些资产**不随本分支发布**，`build_tianji_scene.py` 也不是 URDF 到完整机器人 MJCF 的转换器。
必须先提供可用的机器人模型；初始姿态建议来自对应示范的实测状态，不要随意填零。

```bash
export TIANJI_ROBOT_XML=/absolute/path/to/unified_plant.xml
export TIANJI_URDF=/absolute/path/to/tianji_wuji2.urdf
export HAMMER_MESH=/absolute/path/to/hammer_m.obj
export TIANJI_INITIAL_QPOS=/absolute/path/to/initial_qpos.json
export SPD_CHECKPOINT="$SPD_RUN/last.pt"
```

### 旧版 SPD 权重转换（仅旧模型需要）

当前训练输出可以直接推理。若持有受支持的旧版 `spd-paired-kv-v2` checkpoint，先转换：

```bash
uv run scripts/convert_spd_checkpoint.py \
  --source-path /absolute/path/to/legacy/last.pt \
  --output-path cache/tianji_converted/last.pt \
  --dino-checkpoint "$DINO_WEIGHTS"

export SPD_CHECKPOINT="$PWD/cache/tianji_converted/last.pt"
```

仅加载可信来源的 checkpoint，旧格式包含 pickle。转换器拒绝覆盖已有目标文件。
转换结果为 **weights-only**：支持推理或 `--load-pretrained` 新训练，不支持 `--resume-from`；
旧优化器状态不会迁移。转换范围及数值对照见 [转换说明](abc_minimal/README.md#migrate-the-already-trained-weights)。

### 构建抓锤场景

```bash
uv run scripts/build_tianji_scene.py \
  --robot-xml "$TIANJI_ROBOT_XML" \
  --hammer-mesh "$HAMMER_MESH" \
  --output cache/tianji_sim/pick_hammer_views.xml \
  --initial-qpos-path "$TIANJI_INITIAL_QPOS" \
  --fit-wrist-cameras
```

构建器保留机器人碰撞、惯量和伺服配置，添加桌面及锤子，输出场景 XML 和相邻 JSON 说明。
默认桌高 0.90 m、锤子质量 0.30 kg，接触几何与惯量为明确的仿真假设。
`--fit-wrist-cameras` 按初始姿态选择固定腕部相机安装位姿，不是标定真机外参，也不会在运行时跟踪物体；
省略该选项可保留源相机安装位姿。

### 闭环评估与视频

```bash
MUJOCO_GL=egl uv run eval_policy.py \
  --checkpoint "$SPD_CHECKPOINT" \
  --policy spd --embodiment tianji_wuji2 --task tianji_pick_hammer \
  --tianji.model-path cache/tianji_sim/pick_hammer_views.xml \
  --tianji.urdf-path "$TIANJI_URDF" \
  --tianji.initial-qpos-path "$TIANJI_INITIAL_QPOS" \
  --spd-dino-checkpoint "$DINO_WEIGHTS" \
  --camera-backend mujoco --camera-height 360 --camera-width 640 \
  --no-rtc --no-fast-inference --prefix-length 0 --execute-chunk-dim 8 \
  --num-worlds 1 --num-chunks 16 --device cuda \
  --save-video --log-every-chunk --output-dir outputs/tianji_spd
```

这里 `--device cuda` 用于策略，`--camera-backend mujoco` 对应 Tianji 支持的渲染路径，动力学仍由 CPU MuJoCo 执行。
不要启用 DiT 的 RTC、CUDA-graph 快速推理或 MJWarp 并行世界参数。
`--device cpu` 可改用 CPU 策略推理，但不意味着 EGL 渲染不再需要图形环境。

默认参与策略的相机为 `top,left_wrist`，视频可显示三路相机。
只有 checkpoint 训练来源包含右腕视角时，才可追加 `--tianji.active-cameras top left_wrist right_wrist`；
视频中出现右腕画面，不代表模型接受过右腕图像训练。

输出位于 `outputs/tianji_spd/`：

- `world_000.mp4`：闭环 rollout 视频。
- `summary.json`：任务结果、物理步长、接触/抬升、跟踪误差和目标裁剪等。

16 个动作块、每块 8 步对应 128 个控制时刻，约 4.267 秒仿真时间。
成功判据为抬升至少 0.05 m 且手/物体接触持续 6 个控制时刻；地面或手臂接触不计入。
每个 episode 重置到提供的固定场景，并清空策略历史；不使用 YAM 的场景随机化。

## 验证与已知限制

```bash
MUJOCO_GL=egl uv run python -m pytest -q
```

仓库已有验证覆盖实际 HDF5 数据训练更新、EMA 保存/重载、缺失相机掩码、续训一致性及仿真反馈。
最近一次上游合并后的现有测试结果为 **97 passed**；该结果不等于策略收敛或真机验证。

已有完整模型 rollout 记录执行了 128 个控制时刻，没有 BADQPOS/BADQVEL/BADQACC 事件，
但**没有抬起锤子**。30 Hz 指仿真控制时间，不是已经证明的墙钟实时性能。
相机未标定、接触近似、碰撞网格编译警告及真实到仿真的视觉差异限制了结果解释。
详细历史记录见 [验证](abc_minimal/README.md#verification) 和 [仿真结果](abc_minimal/README.md#run-the-trained-model)。

**不要将 SPD 输出直接接入上游 YAM 控制器。** `deploy/deploy_policy.py`、策略服务器及现有 YAM 交互 viewer 会拒绝 SPD。
本项目尚无经过验证的 Tianji 硬件控制适配器，仿真位置/目标变化率限制不构成真机安全认证。

## 常见问题

| 问题 | 检查方式 |
| --- | --- |
| 下载 DINO 返回 401/403 | 在模型页面接受条款并取得权限，再登录有访问权的 HF 账户 |
| DINO SHA-256 不匹配 | 使用训练时同一份权重，不能仅靠文件名判断 |
| 找不到 `abc_sim` / `abc_minimal` | 在本仓库根目录完成 `uv sync --extra dev`，使用本项目的 `uv run` |
| `libtorchcodec` / FFmpeg 共享库加载失败 | 安装系统 FFmpeg 共享库；检查是否混用了其他 Conda/venv 的库路径。Tianji HDF5 JPEG 解码不走 TorchCodec，但 ABC MP4 加载会使用它 |
| CUDA 不可用或 EGL 初始化失败 | 分别检查 PyTorch/驱动和 NVIDIA GL/EGL；只安装 CUDA 计算运行时不能保证相机渲染 |
| 显存不足 | 从 batch 1、`--no-compile` 开始；可减小 `--spd-model.dino-frame-batch-size`，但不保证任意显卡都能运行完整模型 |
| 数据无法形成有效窗口 | 检查 episode 数量、时间戳、流过期和采集 schema；不要通过伪造帧或掩码绕过损坏数据 |
| 旧模型无法续训 | 转换的权重不含可恢复的优化器状态，使用 `--load-pretrained` 新训练 |
| 运行正常却没有抓起锤子 | 执行链路验证不等于任务成功；结合训练覆盖、相机/场景一致性、接触和跟踪指标分析 |

## 上游 ABC 功能

本分支仍保留 ABC-DiT、ABC-VLA、原有数据转换和 YAM 仿真/部署。
上游最新的逐帧 `prompt_timeline` 修复用于 ABC 的语言条件数据路径，不直接改变无语言条件的 SPD。

- [ABC 项目与论文](https://abc.bot)、[上游代码及完整 ABC 快速入门](https://github.com/amazon-far/abc)。
- [ABC 数据格式、导出和训练参考](abc_minimal/README.md)。
- [YAM 任务目录、评估与渲染](abc_sim/README.md)。
- [YAM 真机部署与遥操作](deploy/README.md)。

下方保留上游第三方许可与引用信息；Tianji 外部数据、机器人和扫描资产还需遵守各自来源的授权条件。

## Licenses

The project code is Apache-2.0 ([`LICENSE`](LICENSE)), with third-party
components covered by the licenses listed below. The published DiT checkpoints
`bottles_75k.pt` and `abc_dit_xl_200k_model.pt` are Apache-2.0.
Both checkpoints embed a DINOv3-derived vision backbone, so the DINOv3 use
restrictions below apply to the weights as well as to the code that loads them.

The released VLA checkpoints (including `vla_abc130k_200000_v2.pt`, the
`abc130k` and `200k` families, and their task finetunes) contain Gemma-derived
weights and are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms),
including its use and redistribution conditions. See [Gemma weights terms of
use](#gemma-weights-terms-of-use) below.

This repository includes and adapts code from the following third-party
projects. Original license files and copyright headers are retained in all
cases. Bundled license texts live under `assets/third_party/`.

| Project | License | License file | Inclusion | What we use/adapt |
| --- | --- | --- | --- | --- |
| [DINOv3](https://github.com/facebookresearch/dinov3) | DINOv3 License (Meta) | [`assets/third_party/dinov3/LICENSE.md`](assets/third_party/dinov3/LICENSE.md) | Adapted (`abc_minimal/dit.py`); pretrained weights downloaded by the user | ViT-B/16 vision backbone (`DinoRope`, `DinoAttention`, `DinoMlp`, etc.) |
| [OpenAI CLIP](https://github.com/openai/CLIP) | MIT | [`assets/third_party/clip/LICENSE`](assets/third_party/clip/LICENSE) | Adapted (`abc_minimal/dit.py`); ViT-B/32 text weights + BPE vocab downloaded at runtime | CLIP text encoder + BPE tokenizer (`CLIPBPETokenizer`, `CLIPTextTower`, `CLIPTextEmbedder`) |
| [openpi](https://github.com/Physical-Intelligence/openpi) | Apache 2.0 | [`assets/third_party/openpi/LICENSE`](assets/third_party/openpi/LICENSE) | Adapted (`deploy/client/websocket_client_policy.py`, `deploy/client/msgpack_numpy.py`) | Websocket inference client skeleton + msgpack NumPy serialization |
| [msgpack-numpy](https://github.com/lebedov/msgpack-numpy) | BSD 3-Clause | [`assets/third_party/msgpack_numpy/LICENSE.md`](assets/third_party/msgpack_numpy/LICENSE.md) | Adapted (`deploy/client/msgpack_numpy.py`, via openpi) | NumPy serialization strategy for msgpack |
| [Gemma](https://ai.google.dev/gemma) | Apache 2.0 (code); Gemma Terms of Use (weights) | [`assets/third_party/gemma/LICENSE`](assets/third_party/gemma/LICENSE) | Adapted (`abc_minimal/gemma/`); base checkpoint supplied by the user | Gemma 3 model + SentencePiece tokenizer for ABC-VLA |
| [SigLIP](https://github.com/google-research/big_vision) | Apache 2.0 | [`assets/third_party/gemma/LICENSE`](assets/third_party/gemma/LICENSE) | Adapted (`abc_minimal/gemma/siglip_vision/`) | SigLIP vision encoder for ABC-VLA |

### Gemma weights terms of use

Gemma model code is Apache-2.0, but Gemma *weights* are additionally governed by
the Google Gemma Terms of Use (https://ai.google.dev/gemma/terms), including its
Prohibited Use Policy. Training from the base Gemma 3 checkpoint requires a
user-supplied file; the base checkpoint is not bundled in this repository.
The released VLA checkpoints downloaded by `prepare.py` contain Gemma-derived
weights, so those terms also apply when finetuning or evaluating them without
a separate base checkpoint. See `assets/third_party/gemma/NOTICE` and the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) for details.

### Simulator asset licenses

The simulator asset packages installed by `prepare.py --sim` (and the RoboCasa
packs fetched by `--sim-robocasa`) bundle meshes and textures from the sources
below. abc-side processing — recentering, rescaling, trimesh re-export, resized
textures, custom convex collision decompositions, and retuned MJCF wrappers —
does not change the upstream licenses. Per-object credits ship inside the
packages where noted.

| Source | License | Where it is used |
| --- | --- | --- |
| [RoboCasa](https://github.com/robocasa/robocasa) objaverse pack: objects curated from [Objaverse 1.0](https://objaverse.allenai.org/objaverse-1.0), originally by individual Sketchfab creators | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (RoboCasa asset release) | `mug`; the object libraries in `task_put_relative`, `task_grab_clutter`, and `task_conveyor_pick`; mug variants in `task_mug_flip` and `task_mug_tree`; plates in `task_dishrack`; bottles in `task_water_bottles`; objects in `task_multi_drawer_search` and `task_inhand_transfer`; the verbatim pack via `--sim-robocasa` |
| RoboCasa lightwheel pack: objects by [LightWheel AI](https://www.lightwheel.ai/) | CC BY 4.0 (RoboCasa asset release) | object variants in `task_put_relative`, `task_grab_clutter`, `task_conveyor_pick`, `task_inhand_transfer`, and `task_multi_drawer_search`; dish racks in `task_dishrack`; the verbatim pack via `--sim-robocasa` |
| RoboCasa aigen pack: AI-generated objects ([Luma.ai](https://lumalabs.ai/)) | CC BY 4.0 (RoboCasa asset release) | `bowl` |
| [Google Scanned Objects](https://github.com/kevinzakka/mujoco_scanned_objects) | CC BY 4.0 | Two office objects in `task_multi_drawer_search`; per-object credits in the package's `OFFICE_ATTRIBUTIONS.md` |
| Sketchfab creators, individually credited | CC BY 4.0 | Six office objects in `task_multi_drawer_search` (see `OFFICE_ATTRIBUTIONS.md`); the box and crate in `task_bins` (see the package `README.md`); the Fujiya tin in `task_chess` by [Vision Fountain](https://sketchfab.com/visionfountain); [`bin`](https://sketchfab.com/3d-models/bin-8984db4f15284436ab704919327ca251) by [AlaPasta](https://sketchfab.com/alapasta); [`blocks`](https://sketchfab.com/3d-models/wooden-alphabet-blocks-5f8dfddbbc7d468784ca014378f7e5fe) by [Cherryvania](https://sketchfab.com/mikequeen123); [`dustpan`](https://sketchfab.com/3d-models/dustpan-d91eae20c02a4741aeb889246b417ae4) by [c_irby_paint](https://sketchfab.com/cirby2180); [`garbage_can`](https://sketchfab.com/3d-models/garbage-can-trashcan-bin-926826667ff04fb09a0907bbec54c766) by [BlackCube](https://sketchfab.com/blackcube4), including its copy in `task_water_bottles`; [`paper_ball`](https://sketchfab.com/3d-models/paper-ball-8afd2bfbe8c14fad937f768617d55f9e) by [ianshanewise](https://sketchfab.com/ianshanewise); [`tray`](https://sketchfab.com/3d-models/plastic-tray-e9b536258ae4499abec7b31ebd231daf) by [Aullwen](https://sketchfab.com/Aullwen) |
| [freepoly.org](https://freepoly.org/) | CC0 | The yellow tin in `task_chess` |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | MIT | The `short_cabinet` drawer fixture in `task_multi_drawer_search`; license copy in the package |
| [i2rt YAM](https://github.com/i2rt-robotics) | MIT | The `i2rt_yam` robot model package; license copy in the package |
| This project | Apache-2.0 | Everything else, modeled or scanned in-house: `ball_sorting_toy`, `brush`, `brush_flat`, `building_blocks`, `chess`, `cup_stacking`, `dishrack`, `drawer`, `flexible_gripper`, `hand_brush_smooth`, `jenga`, `letters`, `marker`, `mug_tree`, `new_brush`, `plate`, the generated `task_nuts_bolts` meshes, the conveyor fixture in `task_conveyor_pick`, the baked plate/rack originals in `task_dishrack`, `tin_2` in `task_chess`, and all task-tuned MJCF wrappers and collision meshes |

### DINOv3 use restrictions

The DINOv3 License prohibits use of the DINO Materials (including weights and
derivatives) for: military purposes; activities subject to ITAR or other
export-control regimes covering defense articles; nuclear applications;
espionage; and the development, manufacture, or use of weapons. Downstream
users who load DINOv3 weights through this codebase are bound by these
restrictions; see `assets/third_party/dinov3/LICENSE.md` for the full
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
