# MACORAG

Multi-agent RAG data processing and teacher trajectory construction for HotpotQA, 2Wiki, and MuSiQue.

## First Milestone

- Build canonical `data/processed/<dataset>/examples.*.jsonl` and `corpus.jsonl`.
- Build LinearRAG-compatible `data/retrieval_env/<dataset>/questions.json` and `chunks.json`.
- Generate full-API teacher trajectories with fixed tags:
  `<plan>`, `<retrieval>`, `<update-evidence>`, `<answer>`.
- Filter to 3K SFT trajectories for Qwen2.5-7B warm-up:
  1K HotpotQA, 1K 2Wiki, 1K MuSiQue.

## Safety Rule

Gold answers and gold evidence are used only by verifiers and filters. They must not appear in teacher prompts.

## macorag 环境依赖说明

当前执行 `pip` 安装时，如果网络受限会在构建环境里报 `BackendUnavailable: Cannot import 'setuptools.build_meta'`。
这是因为 pip 的 PEP517 隔离构建会尝试从索引重新拉取 `setuptools`。

建议在 `conda activate macorag` 后用以下方式安装依赖：

```bash
PIP_NO_BUILD_ISOLATION=1 python -m pip install --no-deps -e .
PIP_NO_BUILD_ISOLATION=1 python -m pip install -r LinearRAG/requirements.txt
```

也可直接执行仓库脚本（同样会带上该配置）：

```bash
bash scripts/setup_macorag_env.sh
```

## Qwen2.5-7B LoRA SFT（教师轨迹）

项目已提供基于 `data/sft/teacher_qwen_plus_trajectory_train` 的训练入口：

```bash
bash scripts/run_train_sft_lora_gpu1.sh \
  --check-only \
  --max-samples 20
```

上面命令用于先做数据检查，不会开始训练。
正式训练（固定 GPU1，默认不含 `state` 和 `observation`）：

```bash
bash scripts/run_train_sft_lora_gpu1.sh \
  --model-path model/Qwen2.5-7B-Instruct \
  --data-root data/sft/teacher_qwen_plus_trajectory_train \
  --output-dir outputs/lora_qwen2.5-7b_trajectory \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 3.0 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --max-length 4096 \
  --bf16 \
  --save-steps 100
```

首次运行前请先补齐依赖（示例）：

```bash
PIP_NO_BUILD_ISOLATION=1 /data/conda/envs/macorag/bin/pip install \
  torch torchvision torchaudio \
  transformers peft datasets accelerate trl bitsandbytes
```

默认脚本会把每轮轨迹转换为 `<plan> / <retrieval> / <update-evidence> / <answer>`，
并在转换时**丢弃 `state` 与 `observation` 字段**，避免模型学习维护状态文本。
