from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prompt_config import DEFAULT_SYSTEM_PROMPT

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
    "max_samples_by_dataset": {},
    "data_sampling_seed": 42,
    "seed": 42,
    "max_rounds": 4,
    "retrieval_top_k": 5,
    "prompt_config_path": "config/prompts.yml",
    "require_teacher_provenance": False,
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
    "resume_from_checkpoint": None,
}

# 验证与早停：按原样驱动 validation split 和 EarlyStoppingCallback。
EVAL_DEFAULTS: dict[str, Any] = {
    "eval_strategy": "epoch",
    "eval_steps": 100,
    "eval_split_ratio": 0.05,
    "validation_split": True,
    "early_stopping_enabled": False,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.0,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "restore_callback_states_from_checkpoint": True,
}

# 运行环境：launcher 只读取 gpu_indices；check_only 用于数据快速检查。
RUNTIME_DEFAULTS: dict[str, Any] = {
    "fp16": False,
    "bf16": False,
    "attn_implementation": "sdpa",
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
CANONICAL_DATASETS = ("2wiki", "hotpotqa", "musique")


def _parse_sample_limits(value: Any) -> dict[str, int]:
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise argparse.ArgumentTypeError(
                f"max_samples_by_dataset must be a JSON mapping: {exc.msg}"
            ) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("max_samples_by_dataset must be a mapping.")

    normalized: dict[str, int] = {}
    for raw_dataset, raw_limit in value.items():
        dataset = str(raw_dataset).strip().lower()
        if dataset not in CANONICAL_DATASETS:
            raise argparse.ArgumentTypeError(
                f"max_samples_by_dataset contains unknown dataset {raw_dataset!r}; "
                f"expected one of {', '.join(CANONICAL_DATASETS)}."
            )
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit <= 0:
            raise argparse.ArgumentTypeError(
                f"max_samples_by_dataset[{dataset!r}] must be a positive integer; got {raw_limit!r}."
            )
        normalized[dataset] = raw_limit
    return normalized


def _validate_args(args: argparse.Namespace) -> None:
    try:
        args.max_samples_by_dataset = _parse_sample_limits(args.max_samples_by_dataset)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.max_samples is not None and args.max_samples_by_dataset:
        raise SystemExit("max_samples and max_samples_by_dataset cannot both be active.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit(f"max_samples must be positive; got {args.max_samples}.")
    if not args.early_stopping_enabled:
        return
    if not args.validation_split:
        raise SystemExit("early_stopping_enabled requires validation_split=true.")
    if args.eval_strategy != "steps":
        raise SystemExit(
            f"early_stopping_enabled requires eval_strategy='steps'; got {args.eval_strategy!r}."
        )
    if args.eval_steps <= 0:
        raise SystemExit(f"early_stopping_enabled requires eval_steps > 0; got {args.eval_steps}.")
    if args.save_steps <= 0:
        raise SystemExit(f"early_stopping_enabled requires save_steps > 0; got {args.save_steps}.")
    if args.save_steps % args.eval_steps != 0:
        raise SystemExit(
            "early_stopping_enabled requires save_steps to be divisible by eval_steps; "
            f"got save_steps={args.save_steps}, eval_steps={args.eval_steps}."
        )
    if args.early_stopping_patience <= 0:
        raise SystemExit(
            "early_stopping_patience must be positive when early stopping is enabled; "
            f"got {args.early_stopping_patience}."
        )
    if args.early_stopping_threshold < 0:
        raise SystemExit(
            f"early_stopping_threshold must be non-negative; got {args.early_stopping_threshold}."
        )
    if args.metric_for_best_model != "eval_loss":
        raise SystemExit(
            "early stopping requires metric_for_best_model='eval_loss'; "
            f"got {args.metric_for_best_model!r}."
        )
    if args.greater_is_better:
        raise SystemExit("early stopping on eval_loss requires greater_is_better=false.")
    if not args.restore_callback_states_from_checkpoint:
        raise SystemExit(
            "early_stopping_enabled requires restore_callback_states_from_checkpoint=true."
        )


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
    parser.add_argument(
        "--max-samples-by-dataset",
        type=_parse_sample_limits,
        default=defaults["max_samples_by_dataset"],
        help="JSON mapping of canonical dataset names to pre-split trajectory quotas.",
    )
    parser.add_argument("--data-sampling-seed", type=int, default=defaults["data_sampling_seed"])
    parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed.")
    parser.add_argument("--max-rounds", type=int, default=defaults["max_rounds"])
    parser.add_argument("--retrieval-top-k", type=int, default=defaults["retrieval_top_k"])
    parser.add_argument("--prompt-config-path", default=defaults["prompt_config_path"])
    parser.add_argument(
        "--require-teacher-provenance",
        action=BooleanOptionalAction,
        default=defaults["require_teacher_provenance"],
    )

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
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=defaults["eval_steps"],
        help="Optimizer-step interval used only when eval_strategy=steps.",
    )
    parser.add_argument(
        "--eval-strategy",
        choices=("epoch", "steps"),
        default=defaults["eval_strategy"],
        help="Run validation once per epoch or at a fixed optimizer-step interval.",
    )
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"], help="Optional max steps override.")
    parser.add_argument("--save-total-limit", type=int, default=defaults["save_total_limit"], help="Max checkpoints to keep.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=defaults["resume_from_checkpoint"],
        help="Resume a full SFT Trainer checkpoint in its existing run directory.",
    )
    parser.add_argument("--eval-split-ratio", type=float, default=defaults["eval_split_ratio"], help="Validation split ratio by original samples.")
    parser.add_argument("--validation-split", action=BooleanOptionalAction, default=defaults["validation_split"], help="Enable train/validation split.")
    parser.add_argument(
        "--early-stopping-enabled",
        action=BooleanOptionalAction,
        default=defaults["early_stopping_enabled"],
        help="Enable eval-loss early stopping and best-checkpoint restoration.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=defaults["early_stopping_patience"], help="Stop after this many evaluations without a meaningful improvement.")
    parser.add_argument("--early-stopping-threshold", type=float, default=defaults["early_stopping_threshold"], help="Minimum metric improvement for early stopping.")
    parser.add_argument("--metric-for-best-model", default=defaults["metric_for_best_model"], help="Metric used for best checkpoint and early stopping.")
    parser.add_argument("--greater-is-better", action=BooleanOptionalAction, default=defaults["greater_is_better"], help="Whether the best-model metric should increase.")
    parser.add_argument(
        "--restore-callback-states-from-checkpoint",
        action=BooleanOptionalAction,
        default=defaults["restore_callback_states_from_checkpoint"],
        help="Restore stateful Trainer callbacks during full checkpoint resume.",
    )
    parser.add_argument("--fp16", action=BooleanOptionalAction, default=defaults["fp16"], help="Use fp16.")
    parser.add_argument("--bf16", action=BooleanOptionalAction, default=defaults["bf16"], help="Use bf16.")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=defaults["attn_implementation"],
    )
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
    args = parser.parse_args(argv)
    _validate_args(args)
    return args
