from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "config/evaluate_rag_model.yml"

DEFAULT_ARG_VALUES: dict[str, Any] = {
    # 基础路径：评估结果会写入 output_root/时间戳 目录。
    "data_root": "data/eval_1000",
    "data_files": (),
    "retrieval_root": "data/eval_1000_retrieval",
    "output_root": "outputs/eval_rag_model",
    "output_dir": "",
    "resume": False,
    "manifest_meta_path": "",
    "adapter_label": "",
    "adapter_identity_path": "",
    "model_path": "",
    "adapter_path": "",
    "prompt_config_path": "config/prompts.yml",
    # 推理采样。
    "max_samples": None,
    "max_rounds": 3,
    "max_prompt_length": 4096,
    "max_completion_length": 256,
    "temperature": 0.0,
    "top_p": 0.95,
    "gpu_indices": "1",
    # 检索配置：retrieval_top_k 才是每次检索返回的段落数量。
    "retrieval_backend": "e5_faiss",
    "retrieval_embedding_model": "intfloat/e5-base-v2",
    "retrieval_device": "cpu",
    "retrieval_max_length": 512,
    "retrieval_spacy_model": "en_core_web_trf",
    "retrieval_top_k": 5,
    "retrieval_max_workers": 4,
    "retrieval_batch_size": 32,
    "use_vectorized_retrieval": True,
    # 推理服务：评估端只调用 OpenAI-compatible vLLM 服务，模型加载由服务端配置负责。
    "vllm_base_urls": (),
    "vllm_transport": "openai",
    "vllm_model": "",
    "vllm_api_key_env": "",
    "vllm_timeout": 120,
    "vllm_retries": 3,
    "vllm_retry_sleep_seconds": 1.0,
    "eval_request_workers": 1,
    "eval_generate_batch_size": 1,
    "eval_generate_batch_wait_ms": 20.0,
}

# 统一 YAML 中供 vLLM 启动器使用的字段。评估客户端验证它们的名称，
# 但不把它们暴露为 evaluate_rag_model 的命令行参数。
VLLM_SERVER_CONFIG_KEYS = {
    "vllm_bin",
    "vllm_gpu_indices",
    "vllm_host",
    "vllm_dtype",
    "vllm_gpu_memory_utilization",
    "vllm_max_model_len",
    "vllm_trust_remote_code",
    "vllm_environment",
    "vllm_extra_args",
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
    allowed = {*DEFAULT_ARG_VALUES, *VLLM_SERVER_CONFIG_KEYS, "config"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise SystemExit(f"Unknown evaluation config keys in {path}: {', '.join(unknown)}")
    return {key: value for key, value in config.items() if key in DEFAULT_ARG_VALUES}


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
    parser.add_argument("--data-root", default=defaults["data_root"])
    parser.add_argument("--data-files", nargs="*", default=defaults["data_files"])
    parser.add_argument("--retrieval-root", default=defaults["retrieval_root"])
    parser.add_argument("--output-root", default=defaults["output_root"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--resume", action=BooleanOptionalAction, default=defaults["resume"])
    parser.add_argument("--manifest-meta-path", default=defaults["manifest_meta_path"])
    parser.add_argument("--adapter-label", default=defaults["adapter_label"])
    parser.add_argument("--adapter-identity-path", default=defaults["adapter_identity_path"])
    parser.add_argument(
        "--model-path",
        default=defaults["model_path"],
        help="Base model served by vLLM; recorded as evaluation provenance.",
    )
    parser.add_argument(
        "--adapter-path",
        default=defaults["adapter_path"],
        help="LoRA adapter served by vLLM; recorded as evaluation provenance.",
    )
    parser.add_argument("--prompt-config-path", default=defaults["prompt_config_path"])
    parser.add_argument("--max-samples", type=int, default=defaults["max_samples"])
    parser.add_argument("--max-rounds", type=int, default=defaults["max_rounds"])
    parser.add_argument("--max-prompt-length", type=int, default=defaults["max_prompt_length"])
    parser.add_argument("--max-completion-length", type=int, default=defaults["max_completion_length"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    parser.add_argument("--gpu-indices", default=defaults["gpu_indices"])
    parser.add_argument("--retrieval-backend", choices=("e5_faiss", "linear_rag"), default=defaults["retrieval_backend"])
    parser.add_argument("--retrieval-embedding-model", default=defaults["retrieval_embedding_model"])
    parser.add_argument("--retrieval-device", default=defaults["retrieval_device"])
    parser.add_argument("--retrieval-max-length", type=int, default=defaults["retrieval_max_length"])
    parser.add_argument("--retrieval-spacy-model", default=defaults["retrieval_spacy_model"])
    parser.add_argument("--retrieval-top-k", type=int, default=defaults["retrieval_top_k"])
    parser.add_argument("--retrieval-max-workers", type=int, default=defaults["retrieval_max_workers"])
    parser.add_argument("--retrieval-batch-size", type=int, default=defaults["retrieval_batch_size"])
    parser.add_argument("--use-vectorized-retrieval", action=BooleanOptionalAction, default=defaults["use_vectorized_retrieval"])
    parser.add_argument("--vllm-base-urls", nargs="*", default=defaults["vllm_base_urls"])
    parser.add_argument(
        "--vllm-transport",
        choices=("openai", "training_server"),
        default=defaults["vllm_transport"],
    )
    parser.add_argument("--vllm-model", default=defaults["vllm_model"])
    parser.add_argument("--vllm-api-key-env", default=defaults["vllm_api_key_env"])
    parser.add_argument("--vllm-timeout", type=int, default=defaults["vllm_timeout"])
    parser.add_argument("--vllm-retries", type=int, default=defaults["vllm_retries"])
    parser.add_argument("--vllm-retry-sleep-seconds", type=float, default=defaults["vllm_retry_sleep_seconds"])
    parser.add_argument("--eval-request-workers", type=int, default=defaults["eval_request_workers"])
    parser.add_argument("--eval-generate-batch-size", type=int, default=defaults["eval_generate_batch_size"])
    parser.add_argument("--eval-generate-batch-wait-ms", type=float, default=defaults["eval_generate_batch_wait_ms"])
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    raw_args = sys.argv[1:] if argv is None else argv
    explicit_config = any(arg == "--config" or arg.startswith("--config=") for arg in raw_args)
    defaults = _defaults_from_config(config_args.config, explicit_config=explicit_config)
    parser = _build_parser(defaults)
    args = parser.parse_args(argv)
    if not str(args.adapter_identity_path or "").strip():
        args.adapter_identity_path = str(args.adapter_path or "").strip()
    return args
