from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from prompt_config import DEFAULT_SYSTEM_PROMPT


DEFAULT_CONFIG_PATH = "config/train_grpo.yml"

# 基础路径：保持入口、数据、检索索引和输出目录由 YAML 统一管理。
PATH_DEFAULTS: dict[str, Any] = {
    "model_path": "model/Qwen2.5-7B-Instruct",
    "sft_adapter_path": "outputs/lora_qwen2.5-7b_trajectory_20260625_163149/adapter",
    "rl_data_root": "data/rl/trajectory_train",
    "rl_data_files": (),
    "retrieval_root": "data/trajectory_train_retrieval",
    "output_root": "outputs/grpo_qwen2.5-7b_trajectory",
}

# rollout 与采样：控制每个样本的 RAG 交互轮数、组内采样数和生成截断。
ROLLOUT_DEFAULTS: dict[str, Any] = {
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "max_samples": None,
    "max_total_samples": None,
    "seed": 42,
    "max_rounds": 3,
    "group_size": 4,
    "num_train_epochs": 1.0,
    "max_steps": 0,
    "max_prompt_length": 4096,
    "max_completion_length": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 5,
}

# GRPO 优化：只训练 LoRA adapter，损失、KL 和梯度累积语义不变。
OPTIMIZATION_DEFAULTS: dict[str, Any] = {
    "learning_rate": 1e-5,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "per_device_train_batch_size": 1,
    "reference_per_device_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "skip_zero_advantage_updates": True,
    "kl_beta": 0.02,
    "clip_epsilon": 0.2,
    "bf16": False,
    "fp16": False,
    "load_4bit": True,
    "gradient_checkpointing": True,
    "query_local_credit_weight": 0.75,
    "evidence_local_credit_weight": 0.70,
    "answer_local_credit_weight": 0.30,
}

# 运行环境：launcher 会优先读取 gpu_indices，gpu_index 仅作为兼容回退。
RUNTIME_DEFAULTS: dict[str, Any] = {
    "gpu_index": 0,
    "gpu_indices": "0,1",
    "check_only": False,
    "disable_tqdm": False,
}

# 日志与检查点：长样本训练依赖 JSONL 心跳，不改变原文件名默认值。
LOGGING_DEFAULTS: dict[str, Any] = {
    "save_steps": 100,
    "save_total_limit": 3,
    "logging_steps": 1,
    "log_all_group_rollouts": True,
}

# 检索环境：这些参数必须与预构建 LinearRAG 索引保持一致。
RETRIEVAL_DEFAULTS: dict[str, Any] = {
    "retrieval_embedding_model": "BAAI/bge-base-en-v1.5",
    "retrieval_spacy_model": None,
    "retrieval_top_k": 5,
    "retrieval_max_workers": 8,
    "retrieval_batch_size": 128,
    "use_vectorized_retrieval": False,
    "retrieval_query_cache_size": 4096,
}

# vLLM 生成与权重同步：当前支持 dense 与 LoRA 两种同步路径。
VLLM_DEFAULTS: dict[str, Any] = {
    "use_vllm_generation": False,
    "vllm_host": "127.0.0.1",
    "vllm_port": 8000,
    "vllm_gpu_indices": "0",
    "vllm_tensor_parallel_size": 1,
    "vllm_data_parallel_size": 1,
    "vllm_gpu_memory_utilization": 0.75,
    "vllm_max_model_len": 4608,
    "vllm_dtype": "auto",
    "vllm_sync_after_step": True,
    "vllm_sync_every_steps": 1,
    "vllm_sync_trainable_only": True,
    "vllm_timeout_seconds": 120.0,
    "vllm_sync_mode": "dense",
    "vllm_lora_name": "macorag_train",
    "vllm_lora_int_id": 1,
    "vllm_lora_adapter_path": "",
}

DEFAULT_ARG_VALUES: dict[str, Any] = {
    **PATH_DEFAULTS,
    **ROLLOUT_DEFAULTS,
    **OPTIMIZATION_DEFAULTS,
    **VLLM_DEFAULTS,
    **RETRIEVAL_DEFAULTS,
    **LOGGING_DEFAULTS,
    **RUNTIME_DEFAULTS,
}

BooleanOptionalAction = getattr(argparse, "BooleanOptionalAction", None)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to load GRPO YAML config.") from exc

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid config format at {path}: expected a mapping.")

    config = {str(key).replace("-", "_"): value for key, value in payload.items()}

    allowed = {*DEFAULT_ARG_VALUES, "config"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise SystemExit(f"Unknown GRPO config keys in {path}: {', '.join(unknown)}")
    return config


def _defaults_from_config(config_path: str, *, explicit_config: bool) -> dict[str, Any]:
    defaults = dict(DEFAULT_ARG_VALUES)
    path = Path(config_path)
    if explicit_config and not path.exists():
        raise SystemExit(f"GRPO config not found: {path}")
    if path.exists():
        defaults.update(_load_yaml_config(path))
    return defaults


def _build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MACORAG with online RAG-GRPO.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config file.")

    paths = parser.add_argument_group("基础路径")
    paths.add_argument("--model-path", default=defaults["model_path"])
    paths.add_argument("--sft-adapter-path", default=defaults["sft_adapter_path"])
    paths.add_argument("--rl-data-root", default=defaults["rl_data_root"])
    paths.add_argument("--rl-data-files", nargs="*", default=defaults["rl_data_files"])
    paths.add_argument("--retrieval-root", default=defaults["retrieval_root"])
    paths.add_argument("--output-root", default=defaults["output_root"])

    rollout = parser.add_argument_group("rollout 与采样")
    rollout.add_argument("--system-prompt", default=defaults["system_prompt"])
    rollout.add_argument("--max-samples", type=int, default=defaults["max_samples"])
    rollout.add_argument(
        "--max-total-samples",
        type=int,
        default=defaults["max_total_samples"],
    )
    rollout.add_argument("--seed", type=int, default=defaults["seed"])
    rollout.add_argument("--max-rounds", type=int, default=defaults["max_rounds"])
    rollout.add_argument("--group-size", type=int, default=defaults["group_size"])
    rollout.add_argument("--num-train-epochs", type=float, default=defaults["num_train_epochs"])
    rollout.add_argument("--max-steps", type=int, default=defaults["max_steps"])
    rollout.add_argument("--max-prompt-length", type=int, default=defaults["max_prompt_length"])
    rollout.add_argument("--max-completion-length", type=int, default=defaults["max_completion_length"])
    rollout.add_argument("--temperature", type=float, default=defaults["temperature"])
    rollout.add_argument("--top-p", type=float, default=defaults["top_p"])
    rollout.add_argument("--top-k", type=int, default=defaults["top_k"])

    optimization = parser.add_argument_group("GRPO 优化")
    optimization.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    optimization.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    optimization.add_argument("--warmup-ratio", type=float, default=defaults["warmup_ratio"])
    optimization.add_argument("--per-device-train-batch-size", type=int, default=defaults["per_device_train_batch_size"])
    optimization.add_argument(
        "--reference-per-device-batch-size",
        type=int,
        default=defaults["reference_per_device_batch_size"],
    )
    optimization.add_argument("--gradient-accumulation-steps", type=int, default=defaults["gradient_accumulation_steps"])
    optimization.add_argument(
        "--skip-zero-advantage-updates",
        action=BooleanOptionalAction,
        default=defaults["skip_zero_advantage_updates"],
    )
    optimization.add_argument("--kl-beta", type=float, default=defaults["kl_beta"])
    optimization.add_argument("--clip-epsilon", type=float, default=defaults["clip_epsilon"])
    optimization.add_argument("--bf16", action=BooleanOptionalAction, default=defaults["bf16"])
    optimization.add_argument("--fp16", action=BooleanOptionalAction, default=defaults["fp16"])
    optimization.add_argument("--load-4bit", action=BooleanOptionalAction, default=defaults["load_4bit"])
    optimization.add_argument(
        "--gradient-checkpointing",
        action=BooleanOptionalAction,
        default=defaults["gradient_checkpointing"],
    )
    optimization.add_argument(
        "--query-local-credit-weight",
        type=float,
        default=defaults["query_local_credit_weight"],
    )
    optimization.add_argument(
        "--evidence-local-credit-weight",
        type=float,
        default=defaults["evidence_local_credit_weight"],
    )
    optimization.add_argument(
        "--answer-local-credit-weight",
        type=float,
        default=defaults["answer_local_credit_weight"],
    )

    vllm = parser.add_argument_group("vLLM 生成与权重同步")
    vllm.add_argument("--use-vllm-generation", action=BooleanOptionalAction, default=defaults["use_vllm_generation"])
    vllm.add_argument("--vllm-host", default=defaults["vllm_host"])
    vllm.add_argument("--vllm-port", type=int, default=defaults["vllm_port"])
    vllm.add_argument("--vllm-gpu-indices", default=defaults["vllm_gpu_indices"])
    vllm.add_argument("--vllm-tensor-parallel-size", type=int, default=defaults["vllm_tensor_parallel_size"])
    vllm.add_argument("--vllm-data-parallel-size", type=int, default=defaults["vllm_data_parallel_size"])
    vllm.add_argument("--vllm-gpu-memory-utilization", type=float, default=defaults["vllm_gpu_memory_utilization"])
    vllm.add_argument("--vllm-max-model-len", type=int, default=defaults["vllm_max_model_len"])
    vllm.add_argument("--vllm-dtype", default=defaults["vllm_dtype"])
    vllm.add_argument("--vllm-sync-after-step", action=BooleanOptionalAction, default=defaults["vllm_sync_after_step"])
    vllm.add_argument("--vllm-sync-every-steps", type=int, default=defaults["vllm_sync_every_steps"])
    vllm.add_argument("--vllm-sync-trainable-only", action=BooleanOptionalAction, default=defaults["vllm_sync_trainable_only"])
    vllm.add_argument("--vllm-timeout-seconds", type=float, default=defaults["vllm_timeout_seconds"])
    vllm.add_argument("--vllm-sync-mode", choices=("dense", "lora"), default=defaults["vllm_sync_mode"])
    vllm.add_argument("--vllm-lora-name", default=defaults["vllm_lora_name"])
    vllm.add_argument("--vllm-lora-int-id", type=int, default=defaults["vllm_lora_int_id"])
    vllm.add_argument("--vllm-lora-adapter-path", default=defaults["vllm_lora_adapter_path"])

    retrieval = parser.add_argument_group("检索环境")
    retrieval.add_argument("--retrieval-embedding-model", default=defaults["retrieval_embedding_model"])
    retrieval.add_argument("--retrieval-spacy-model", default=defaults["retrieval_spacy_model"])
    retrieval.add_argument("--retrieval-top-k", type=int, default=defaults["retrieval_top_k"])
    retrieval.add_argument("--retrieval-max-workers", type=int, default=defaults["retrieval_max_workers"])
    retrieval.add_argument("--retrieval-batch-size", type=int, default=defaults["retrieval_batch_size"])
    retrieval.add_argument(
        "--retrieval-query-cache-size",
        type=int,
        default=defaults["retrieval_query_cache_size"],
    )
    retrieval.add_argument(
        "--use-vectorized-retrieval",
        action=BooleanOptionalAction,
        default=defaults["use_vectorized_retrieval"],
    )

    logging = parser.add_argument_group("日志与检查点")
    logging.add_argument("--save-steps", type=int, default=defaults["save_steps"])
    logging.add_argument("--save-total-limit", type=int, default=defaults["save_total_limit"])
    logging.add_argument("--logging-steps", type=int, default=defaults["logging_steps"])
    logging.add_argument(
        "--log-all-group-rollouts",
        action=BooleanOptionalAction,
        default=defaults["log_all_group_rollouts"],
    )

    runtime = parser.add_argument_group("运行环境")
    runtime.add_argument("--gpu-index", type=int, default=defaults["gpu_index"])
    runtime.add_argument("--gpu-indices", default=defaults["gpu_indices"])
    runtime.add_argument("--check-only", action=BooleanOptionalAction, default=defaults["check_only"])
    runtime.add_argument("--disable-tqdm", action=BooleanOptionalAction, default=defaults["disable_tqdm"])
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
    if not getattr(args, "vllm_lora_adapter_path", None):
        args.vllm_lora_adapter_path = args.sft_adapter_path
    return args
