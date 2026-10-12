# 白模(White-box)微调 · 双向模型 推理包

本目录是从 `GameRender` 项目中整理出的**独立、可直接下载迁移**的推理包，内容为：

> 原始（双向 Self-Attention，`architecture = original_bidirectional`）Wan2.2-Animate-2-14B
> 底座模型，在"白模驱动视频 → 渲染风格视频"配对数据上做 LoRA 微调后的权重，
> 以及配套的推理脚本、依赖代码、演示数据。

对应原项目实验名：`animate2_base_scene00`（不含 `causal/`、`anyflow/` 的因果化/蒸馏模型，那是另一套，本次未包含）。

**本包不含底座大模型权重**（Wan2.2-Animate-2-14B，约47GB），只包含 LoRA 微调权重和代码，
体积很小，使用前需要你自行准备好底座模型并通过 `--model-dir` 指定其路径（见下文）。

## 目录结构

```
gamerender-whitebox-bidirectional-inference/
├── README.md                  本文件
├── pyproject.toml             推理依赖（uv 用户）
├── requirements.txt           推理依赖（pip 用户）
├── scripts/
│   └── infer.py               推理脚本（需通过 --model-dir 指定底座模型路径）
├── diffsynth-studio/           推理依赖的本地库（WanVideoPipeline 等模型实现，来自本地可编辑依赖）
├── checkpoints/animate2_base_scene00_lora/   白模微调 LoRA 权重（本次迁移的核心产物），共 76MB
│   ├── step-2000.safetensors                               (75MB，rank32；后续 causal/AR 训练也是以此步初始化的，故只保留这一个)
│   ├── training_args.json                                  训练时的完整超参数留档
│   └── loss.csv                                             训练 loss 曲线
└── demo_data/                  内置 2 条真实验证样本，用于快速验证环境是否跑通
    ├── metadata_val.jsonl
    └── val/*.mp4, *.jpg
```

**本包总大小约 97MB**（LoRA 权重 + 代码 + demo 数据），不含底座模型。

## 底座大模型（需自行准备，不在本包内）

推理必须依赖以下底座模型文件（原始 Wan2.2-Animate-2-14B，与原项目完全相同，未做任何改动），
请单独从源服务器拷贝或自行下载，保持以下目录结构：

```
<底座模型目录>/
├── wan_animate_2/wan_animate_2_bf16.safetensors                       (31GB, DiT 主干)
└── videomodel/Wan-AI/
    ├── models_t5_umt5-xxl-enc-bf16.pth                                (11GB, T5 文本编码器)
    ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth        (4.5GB, CLIP)
    ├── Wan2.1_VAE.pth                                                 (485MB, VAE)
    └── umt5-xxl/                                                      (T5 分词器, ~20MB)
```
（源服务器路径：`/data/xiyaodeng/model/Wan2.2-Animate-2-14B/`，共约47GB）

准备好后，运行推理脚本时用 `--model-dir <底座模型目录>` 指定即可，无需放入本包内。

## 迁移方式

直接把整个 `gamerender-whitebox-bidirectional-inference/` 目录（97MB）拷贝到新服务器即可：

```bash
rsync -avzP --progress /data/xiyaodeng/release/gamerender-whitebox-bidirectional-inference/ \
    user@新服务器:/path/to/gamerender-whitebox-bidirectional-inference/
```

底座大模型（47GB）需要单独传输（因为体积较大，建议用 rsync 支持断点续传），例如：

```bash
rsync -avzP --progress /data/xiyaodeng/model/Wan2.2-Animate-2-14B/ \
    user@新服务器:/path/to/Wan2.2-Animate-2-14B/
```

拷贝完成后**无需修改任何代码**——`scripts/infer.py` 的 `--lora`、`--data-dir`（demo 数据）
会基于脚本自身位置自动定位包内文件；`--model-dir` 按你实际放置底座模型的路径传入即可。

## 硬件 / 环境要求

- GPU：建议显存 ≥ 24GB（脚本内置了计算/卸载分离配置，权重会在 CPU/GPU 间按需搬运；
  源机器为 NVIDIA H200，驱动 570.133.20，CUDA 12.8 测试通过）。
- CPU 内存：建议 ≥ 64GB（卸载到 CPU 的权重需要常驻内存）。
- Python ≥ 3.10.1。
- 源机器参考版本：`torch==2.11.0+cu128`、`transformers==5.16.1`、`pillow==12.3.0`。

## 安装依赖

### 方式1：使用 uv（推荐，与源项目一致）

```bash
cd gamerender-whitebox-bidirectional-inference
uv sync
```

> `pyproject.toml` 中把 `diffsynth` 声明为本地可编辑依赖（指向同目录下的 `diffsynth-studio/`），
> `uv sync` 会自动装好。若新机器 CUDA 驱动版本不是 12.8，请编辑 `pyproject.toml` 末尾的
> `pytorch-cu128` 索引地址（改成 `cu121`/`cu124` 等匹配版本，或删掉该索引配置让 uv 自动选择默认版本）。

### 方式2：使用 pip

```bash
cd gamerender-whitebox-bidirectional-inference
python -m venv .venv && source .venv/bin/activate
# 先按新机器 CUDA 版本装 torch，例如 CUDA 12.8：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 运行推理

### 1）先跑内置 demo，验证环境是否正常

```bash
cd gamerender-whitebox-bidirectional-inference
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
    --model-dir /path/to/Wan2.2-Animate-2-14B \
    --lora checkpoints/animate2_base_scene00_lora/step-2000.safetensors \
    --out-dir outputs/demo
```
输出在 `outputs/demo/` 下：`{样本名}_gen.mp4`（生成结果）、`_gt.mp4`（GT渲染，用于对比）、
`_whitebox.mp4`（输入白模视频）。

### 2）真实使用：传入自己的白模视频 + 外观参考图

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
    --model-dir /path/to/Wan2.2-Animate-2-14B \
    --lora checkpoints/animate2_base_scene00_lora/step-2000.safetensors \
    --out-dir outputs/custom \
    --whitebox /path/to/your_whitebox.mp4 \
    --ref-image /path/to/ref_appearance.jpg \
    --prompt "人物外观描述：... 背景描述：..." \
    --prompt-ref "白盒渲染的游戏画面，画面中的角色在移动，相机视角持续变化。" \
    --num-inference-steps 40 --cfg-scale 3.0 --seed 0
```

参数说明：
- `--model-dir`：底座大模型目录路径（必填，本包不自带，需按上文说明自行准备）。
- `--lora`：LoRA checkpoint 路径，本包只保留了 `checkpoints/animate2_base_scene00_lora/step-2000.safetensors`
  （这也是后续 causal/AR 模型训练时用作初始化的那一步）。传 `none` 或不传则使用未微调的原始底座权重。
- `--whitebox`：你自己的白模/线框驱动视频（mp4），决定动作和场景几何结构。
- `--ref-image`：外观参考图（决定人物/画面的外观风格，通常用渲染画面首帧）。
- `--prompt`：对外观/背景的文字描述；`--prompt-ref`：对驱动视频(白模)内容的描述。
- `--num-inference-steps` / `--cfg-scale` / `--seed` / `--num-frames` / `--height` / `--width`：
  采样超参，默认 40步/cfg3.0/81帧/480×832，与训练时一致，不建议改分辨率。
- 更多参数见 `python scripts/infer.py --help`。

## 关于 LoRA 权重的说明

- 本包内 `checkpoints/animate2_base_scene00_lora/` 下的权重**只是 LoRA 适配器**（几十MB级别），
  必须搭配完整的底座大模型（`--model-dir` 指定）一起加载使用，二者缺一不可。
- `training_args.json` 记录了原始训练时的完整超参数（学习率、LoRA rank/target modules、
  数据路径等），仅作留档参考，推理时不会读取它。
- 这套权重是在**原始双向 Self-Attention**底座上训练的（而非项目中另一套因果/AR化改造的模型），
  对应 `inference_config.json` 中 `architecture: original_bidirectional`。

## 已知限制

- 本包不包含完整训练集（原始验证集约 196GB，训练集更大），仅内置 2 条真实验证样本用于
  快速验证环境；日常使用请通过 `--whitebox`/`--ref-image` 传入自己的素材。
- 不包含 `causal/`、`anyflow/` 目录下的因果化/少步蒸馏模型，那是另一套实验产物。
- `diffsynth-studio/` 为推理所需的最小代码集合（已去除 `.git`、`__pycache__`、`egg-info`），
  如需查看完整开发历史，请参考原项目 `/data/xiyaodeng/project/diffsynth-studio`。
