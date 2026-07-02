from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "Follow the role-specific prompt. Output exactly the requested XML-style tag with valid JSON."
)

# 基础路径：模型、SFT 数据和运行输出根目录。
PATH_DEFAULTS: dict[str, Any] = {
    "model_path": "model/Qwen2.5-7B-Instruct",
    "data_root": "data/sft/teacher_qwen_plus_trajectory_train",
    "output_root": "outputs/lora_qwen2.5-7b_trajectory",
}

# 数据与样本：控制样本上限和单条训练样本最大 token 长度。
DATA_DEFAULTS: dict[str, Any] = {
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "max_length": 4096,
    "max_samples": None,
    "seed": 42,
}

# LoRA 结构：保留常用 adapter 调参项。
LORA_DEFAULTS: dict[str, Any] = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"),
}

# 优化参数：保持训练执行逻辑不变，仅分组提升可读性。
OPTIM_DEFAULTS: dict[str, Any] = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 3.0,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "logging_steps": 20,
    "save_steps": 100,
    "max_steps": 0,
    "save_total_limit": 3,
}

# 验证与早停：按原样驱动 validation split 和 EarlyStoppingCallback。
EVAL_DEFAULTS: dict[str, Any] = {
    "eval_steps": 100,
    "eval_split_ratio": 0.05,
    "validation_split": True,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.0,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}

# 运行环境：launcher 只读取 gpu_indices；check_only 用于数据快速检查。
RUNTIME_DEFAULTS: dict[str, Any] = {
    "fp16": False,
    "bf16": False,
    "load_4bit": False,
    "disable_tqdm": False,
    "gpu_indices": "0,1",
    "check_only": False,
    "check_only_max_samples": 20,
    "train_test_seed": 777,
}

DEFAULT_CONFIG_PATH = "config/train_sft_lora.yml"
DEFAULT_ARG_VALUES: dict[str, Any] = {
    **PATH_DEFAULTS,
    **DATA_DEFAULTS,
    **LORA_DEFAULTS,
    **OPTIM_DEFAULTS,
    **EVAL_DEFAULTS,
    **RUNTIME_DEFAULTS,
}


BooleanOptionalAction = getattr(argparse, "BooleanOptionalAction", None)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to load training YAML config.") from exc

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid config format at {path}: expected a mapping.")

    config = {str(key).replace("-", "_"): value for key, value in payload.items()}
    allowed = {*DEFAULT_ARG_VALUES, "config"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise SystemExit(f"Unknown training config keys in {path}: {', '.join(unknown)}")
    return config


def _defaults_from_config(config_path: str, *, explicit_config: bool) -> dict[str, Any]:
    defaults = dict(DEFAULT_ARG_VALUES)
    path = Path(config_path)
    if explicit_config and not path.exists():
        raise SystemExit(f"Training config not found: {path}")
    if path.exists():
        defaults.update(_load_yaml_config(path))
    return defaults


def _build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-7B with LoRA on trajectory SFT data.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config file with training arguments.")
    parser.add_argument("--model-path", default=defaults["model_path"], help="Path to the base model.")
    parser.add_argument("--data-root", default=defaults["data_root"], help="SFT trajectory directory.")
    parser.add_argument("--output-root", default=defaults["output_root"], help="Output root for timestamped SFT runs.")
    parser.add_argument("--system-prompt", default=defaults["system_prompt"], help="System prompt for training examples.")
    parser.add_argument("--max-length", type=int, default=defaults["max_length"], help="Max input length after prompt+target tokenization.")
    parser.add_argument("--max-samples", type=int, default=defaults["max_samples"], help="Optional original-sample cap for smoke tests.")
    parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed.")

    parser.add_argument("--lora-r", type=int, default=defaults["lora_r"], help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=defaults["lora_alpha"], help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=defaults["lora_dropout"], help="LoRA dropout.")
    parser.add_argument("--target-modules", nargs="*", default=defaults["target_modules"])

    parser.add_argument("--per-device-train-batch-size", type=int, default=defaults["per_device_train_batch_size"], help="Per-device training batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults["gradient_accumulation_steps"], help="Gradient accumulation steps.")
    parser.add_argument("--num-train-epochs", type=float, default=defaults["num_train_epochs"], help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"], help="Learning rate.")
    parser.add_argument("--lr-scheduler-type", default=defaults["lr_scheduler_type"], help="Learning rate scheduler.")
    parser.add_argument("--warmup-ratio", type=float, default=defaults["warmup_ratio"], help="Warmup ratio.")
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"], help="Weight decay.")
    parser.add_argument("--logging-steps", type=int, default=defaults["logging_steps"], help="Logging interval.")
    parser.add_argument("--save-steps", type=int, default=defaults["save_steps"], help="Save interval.")
    parser.add_argument("--eval-steps", type=int, default=defaults["eval_steps"], help="Eval interval. 0 disables eval.")
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"], help="Optional max steps override.")
    parser.add_argument("--save-total-limit", type=int, default=defaults["save_total_limit"], help="Max checkpoints to keep.")
    parser.add_argument("--eval-split-ratio", type=float, default=defaults["eval_split_ratio"], help="Validation split ratio by original samples.")
    parser.add_argument("--validation-split", action=BooleanOptionalAction, default=defaults["validation_split"], help="Enable train/validation split.")
    parser.add_argument("--early-stopping-patience", type=int, default=defaults["early_stopping_patience"], help="Stop after this many evals without improvement. 0 disables early stopping.")
    parser.add_argument("--early-stopping-threshold", type=float, default=defaults["early_stopping_threshold"], help="Minimum metric improvement for early stopping.")
    parser.add_argument("--metric-for-best-model", default=defaults["metric_for_best_model"], help="Metric used for best checkpoint and early stopping.")
    parser.add_argument("--greater-is-better", action=BooleanOptionalAction, default=defaults["greater_is_better"], help="Whether the best-model metric should increase.")
    parser.add_argument("--fp16", action=BooleanOptionalAction, default=defaults["fp16"], help="Use fp16.")
    parser.add_argument("--bf16", action=BooleanOptionalAction, default=defaults["bf16"], help="Use bf16.")
    parser.add_argument("--load-4bit", action=BooleanOptionalAction, default=defaults["load_4bit"], help="Enable 4-bit quantized loading (requires bitsandbytes).")
    parser.add_argument("--disable-tqdm", action=BooleanOptionalAction, default=defaults["disable_tqdm"], help="Disable tqdm progress bars.")
    parser.add_argument("--gpu-indices", default=defaults["gpu_indices"], help="Comma-separated GPU indices exposed to the training process.")
    parser.add_argument("--check-only", action=BooleanOptionalAction, default=defaults["check_only"], help="Only validate data and print stats.")
    parser.add_argument("--check-only-max-samples", type=int, default=defaults["check_only_max_samples"], help="Max samples to display for checks.")
    parser.add_argument("--train-test-seed", type=int, default=defaults["train_test_seed"], help="Seed for optional original-sample split.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    raw_args = sys.argv[1:] if argv is None else argv
    explicit_config = "--config" in raw_args
    defaults = _defaults_from_config(config_args.config, explicit_config=explicit_config)
    parser = _build_parser(defaults)
    return parser.parse_args(argv)
