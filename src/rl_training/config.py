from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "config/train_grpo.yml"

DEFAULT_ARG_VALUES: dict[str, Any] = {
    "model_path": "model/Qwen2.5-7B-Instruct",
    "sft_adapter_path": "outputs/lora_qwen2.5-7b_trajectory_20260625_163149/adapter",
    "rl_data_root": "data/rl/trajectory_train",
    "rl_data_files": (),
    "retrieval_root": "data/trajectory_train_retrieval",
    "output_dir": "outputs/grpo_qwen2.5-7b_trajectory",
    "system_prompt": "Follow the role-specific prompt. Output exactly the requested XML-style tag with valid JSON.",
    "max_samples": None,
    "seed": 42,
    "max_rounds": 3,
    "group_size": 4,
    "num_train_epochs": 1.0,
    "learning_rate": 1e-5,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_prompt_length": 4096,
    "max_completion_length": 256,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 5,
    "kl_beta": 0.02,
    "clip_epsilon": 0.2,
    "save_steps": 100,
    "save_total_limit": 3,
    "logging_steps": 1,
    "max_steps": 0,
    "bf16": False,
    "fp16": False,
    "load_4bit": True,
    "gradient_checkpointing": True,
    "gpu_index": 0,
    "gpu_indices": "0,1",
    "check_only": False,
    "disable_tqdm": False,
    "log_jsonl_path": None,
    "rollout_jsonl_path": None,
    "retrieval_embedding_model": "BAAI/bge-base-en-v1.5",
    "retrieval_spacy_model": None,
    "retrieval_top_k": 5,
    "retrieval_max_workers": 8,
    "retrieval_batch_size": 128,
    "use_vectorized_retrieval": False,
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
    "vllm_sync_trainable_only": True,
    "vllm_timeout_seconds": 120.0,
    "vllm_sync_mode": "dense",
    "vllm_lora_name": "macorag_train",
    "vllm_lora_int_id": 1,
    "vllm_lora_adapter_path": "",
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
    parser.add_argument("--model-path", default=defaults["model_path"])
    parser.add_argument("--sft-adapter-path", default=defaults["sft_adapter_path"])
    parser.add_argument("--rl-data-root", default=defaults["rl_data_root"])
    parser.add_argument("--rl-data-files", nargs="*", default=defaults["rl_data_files"])
    parser.add_argument("--retrieval-root", default=defaults["retrieval_root"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--system-prompt", default=defaults["system_prompt"])
    parser.add_argument("--max-samples", type=int, default=defaults["max_samples"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--max-rounds", type=int, default=defaults["max_rounds"])
    parser.add_argument("--group-size", type=int, default=defaults["group_size"])
    parser.add_argument("--num-train-epochs", type=float, default=defaults["num_train_epochs"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--warmup-ratio", type=float, default=defaults["warmup_ratio"])
    parser.add_argument("--per-device-train-batch-size", type=int, default=defaults["per_device_train_batch_size"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults["gradient_accumulation_steps"])
    parser.add_argument("--max-prompt-length", type=int, default=defaults["max_prompt_length"])
    parser.add_argument("--max-completion-length", type=int, default=defaults["max_completion_length"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--kl-beta", type=float, default=defaults["kl_beta"])
    parser.add_argument("--clip-epsilon", type=float, default=defaults["clip_epsilon"])
    parser.add_argument("--save-steps", type=int, default=defaults["save_steps"])
    parser.add_argument("--save-total-limit", type=int, default=defaults["save_total_limit"])
    parser.add_argument("--logging-steps", type=int, default=defaults["logging_steps"])
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"])
    parser.add_argument("--bf16", action=BooleanOptionalAction, default=defaults["bf16"])
    parser.add_argument("--fp16", action=BooleanOptionalAction, default=defaults["fp16"])
    parser.add_argument("--load-4bit", action=BooleanOptionalAction, default=defaults["load_4bit"])
    parser.add_argument("--gradient-checkpointing", action=BooleanOptionalAction, default=defaults["gradient_checkpointing"])
    parser.add_argument("--gpu-index", type=int, default=defaults["gpu_index"])
    parser.add_argument("--gpu-indices", default=defaults["gpu_indices"])
    parser.add_argument("--check-only", action=BooleanOptionalAction, default=defaults["check_only"])
    parser.add_argument("--disable-tqdm", action=BooleanOptionalAction, default=defaults["disable_tqdm"])
    parser.add_argument("--log-jsonl-path", default=defaults["log_jsonl_path"])
    parser.add_argument("--rollout-jsonl-path", default=defaults["rollout_jsonl_path"])
    parser.add_argument("--retrieval-embedding-model", default=defaults["retrieval_embedding_model"])
    parser.add_argument("--retrieval-spacy-model", default=defaults["retrieval_spacy_model"])
    parser.add_argument("--retrieval-top-k", type=int, default=defaults["retrieval_top_k"])
    parser.add_argument("--retrieval-max-workers", type=int, default=defaults["retrieval_max_workers"])
    parser.add_argument("--retrieval-batch-size", type=int, default=defaults["retrieval_batch_size"])
    parser.add_argument("--use-vectorized-retrieval", action=BooleanOptionalAction, default=defaults["use_vectorized_retrieval"])
    parser.add_argument("--use-vllm-generation", action=BooleanOptionalAction, default=defaults["use_vllm_generation"])
    parser.add_argument("--vllm-host", default=defaults["vllm_host"])
    parser.add_argument("--vllm-port", type=int, default=defaults["vllm_port"])
    parser.add_argument("--vllm-gpu-indices", default=defaults["vllm_gpu_indices"])
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=defaults["vllm_tensor_parallel_size"])
    parser.add_argument("--vllm-data-parallel-size", type=int, default=defaults["vllm_data_parallel_size"])
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=defaults["vllm_gpu_memory_utilization"])
    parser.add_argument("--vllm-max-model-len", type=int, default=defaults["vllm_max_model_len"])
    parser.add_argument("--vllm-dtype", default=defaults["vllm_dtype"])
    parser.add_argument("--vllm-sync-after-step", action=BooleanOptionalAction, default=defaults["vllm_sync_after_step"])
    parser.add_argument("--vllm-sync-trainable-only", action=BooleanOptionalAction, default=defaults["vllm_sync_trainable_only"])
    parser.add_argument("--vllm-timeout-seconds", type=float, default=defaults["vllm_timeout_seconds"])
    parser.add_argument("--vllm-sync-mode", choices=("dense", "lora"), default=defaults["vllm_sync_mode"])
    parser.add_argument("--vllm-lora-name", default=defaults["vllm_lora_name"])
    parser.add_argument("--vllm-lora-int-id", type=int, default=defaults["vllm_lora_int_id"])
    parser.add_argument("--vllm-lora-adapter-path", default=defaults["vllm_lora_adapter_path"])
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
