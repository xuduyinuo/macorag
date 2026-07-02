from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "config/evaluate_rag_model.yml"

DEFAULT_ARG_VALUES: dict[str, Any] = {
    "model_path": "model/Qwen2.5-3B-Instruct",
    "adapter_path": "outputs/grpo_qwen2.5-3b/adapter",
    "data_root": "data/eval_1000",
    "data_files": (),
    "retrieval_root": "data/eval_1000_retrieval",
    "output_dir": "outputs/eval_rag_model",
    "fixed_output_dir": False,
    "system_prompt": "Follow the role-specific prompt. Output exactly the requested XML-style tag with valid JSON.",
    "max_samples": None,
    "seed": 42,
    "max_rounds": 3,
    "max_prompt_length": 4096,
    "max_completion_length": 256,
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 5,
    "bf16": False,
    "fp16": False,
    "load_4bit": True,
    "gpu_index": 0,
    "gpu_indices": "1",
    "disable_tqdm": False,
    "retrieval_embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "retrieval_spacy_model": "en_core_web_trf",
    "retrieval_top_k": 5,
    "retrieval_max_workers": 4,
    "retrieval_batch_size": 32,
    "use_vectorized_retrieval": True,
    "skip_judge": False,
    "judge_model": "qwen-plus",
    "judge_endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "judge_api_key_env": "DASHSCOPE_API_KEY",
    "judge_temperature": 0.0,
    "judge_max_tokens": 8,
    "judge_timeout": 120,
    "judge_retries": 3,
    "judge_retry_sleep_seconds": 2.0,
    "judge_workers": 4,
    "inference_backend": "hf_local",
    "vllm_base_urls": (),
    "vllm_model": "",
    "vllm_api_key_env": "",
    "vllm_timeout": 120,
    "vllm_retries": 3,
    "vllm_retry_sleep_seconds": 1.0,
    "eval_request_workers": 1,
}

BooleanOptionalAction = getattr(argparse, "BooleanOptionalAction", None)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to load evaluation YAML config.") from exc

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid config format at {path}: expected a mapping.")

    config = {str(key).replace("-", "_"): value for key, value in payload.items()}
    allowed = {*DEFAULT_ARG_VALUES, "config"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise SystemExit(f"Unknown evaluation config keys in {path}: {', '.join(unknown)}")
    return config


def _defaults_from_config(config_path: str, *, explicit_config: bool) -> dict[str, Any]:
    defaults = dict(DEFAULT_ARG_VALUES)
    path = Path(config_path)
    if explicit_config and not path.exists():
        raise SystemExit(f"Evaluation config not found: {path}")
    if path.exists():
        defaults.update(_load_yaml_config(path))
    return defaults


def _build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MACORAG SFT/RL adapter with the configured RAG loop.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config file.")
    parser.add_argument("--model-path", default=defaults["model_path"])
    parser.add_argument("--adapter-path", default=defaults["adapter_path"])
    parser.add_argument("--data-root", default=defaults["data_root"])
    parser.add_argument("--data-files", nargs="*", default=defaults["data_files"])
    parser.add_argument("--retrieval-root", default=defaults["retrieval_root"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--fixed-output-dir", action=BooleanOptionalAction, default=defaults["fixed_output_dir"])
    parser.add_argument("--system-prompt", default=defaults["system_prompt"])
    parser.add_argument("--max-samples", type=int, default=defaults["max_samples"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--max-rounds", type=int, default=defaults["max_rounds"])
    parser.add_argument("--max-prompt-length", type=int, default=defaults["max_prompt_length"])
    parser.add_argument("--max-completion-length", type=int, default=defaults["max_completion_length"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--bf16", action=BooleanOptionalAction, default=defaults["bf16"])
    parser.add_argument("--fp16", action=BooleanOptionalAction, default=defaults["fp16"])
    parser.add_argument("--load-4bit", action=BooleanOptionalAction, default=defaults["load_4bit"])
    parser.add_argument("--gpu-index", type=int, default=defaults["gpu_index"])
    parser.add_argument("--gpu-indices", default=defaults["gpu_indices"])
    parser.add_argument("--disable-tqdm", action=BooleanOptionalAction, default=defaults["disable_tqdm"])
    parser.add_argument("--retrieval-embedding-model", default=defaults["retrieval_embedding_model"])
    parser.add_argument("--retrieval-spacy-model", default=defaults["retrieval_spacy_model"])
    parser.add_argument("--retrieval-top-k", type=int, default=defaults["retrieval_top_k"])
    parser.add_argument("--retrieval-max-workers", type=int, default=defaults["retrieval_max_workers"])
    parser.add_argument("--retrieval-batch-size", type=int, default=defaults["retrieval_batch_size"])
    parser.add_argument("--use-vectorized-retrieval", action=BooleanOptionalAction, default=defaults["use_vectorized_retrieval"])
    parser.add_argument("--skip-judge", action=BooleanOptionalAction, default=defaults["skip_judge"])
    parser.add_argument("--judge-model", default=defaults["judge_model"])
    parser.add_argument("--judge-endpoint", default=defaults["judge_endpoint"])
    parser.add_argument("--judge-api-key-env", default=defaults["judge_api_key_env"])
    parser.add_argument("--judge-temperature", type=float, default=defaults["judge_temperature"])
    parser.add_argument("--judge-max-tokens", type=int, default=defaults["judge_max_tokens"])
    parser.add_argument("--judge-timeout", type=int, default=defaults["judge_timeout"])
    parser.add_argument("--judge-retries", type=int, default=defaults["judge_retries"])
    parser.add_argument("--judge-retry-sleep-seconds", type=float, default=defaults["judge_retry_sleep_seconds"])
    parser.add_argument("--judge-workers", type=int, default=defaults["judge_workers"])
    parser.add_argument("--inference-backend", choices=("hf_local", "vllm_openai"), default=defaults["inference_backend"])
    parser.add_argument("--vllm-base-urls", nargs="*", default=defaults["vllm_base_urls"])
    parser.add_argument("--vllm-model", default=defaults["vllm_model"])
    parser.add_argument("--vllm-api-key-env", default=defaults["vllm_api_key_env"])
    parser.add_argument("--vllm-timeout", type=int, default=defaults["vllm_timeout"])
    parser.add_argument("--vllm-retries", type=int, default=defaults["vllm_retries"])
    parser.add_argument("--vllm-retry-sleep-seconds", type=float, default=defaults["vllm_retry_sleep_seconds"])
    parser.add_argument("--eval-request-workers", type=int, default=defaults["eval_request_workers"])
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    raw_args = sys.argv[1:] if argv is None else argv
    explicit_config = any(arg == "--config" or arg.startswith("--config=") for arg in raw_args)
    defaults = _defaults_from_config(config_args.config, explicit_config=explicit_config)
    parser = _build_parser(defaults)
    return parser.parse_args(argv)
