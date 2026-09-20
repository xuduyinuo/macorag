#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .protocol import validate_prompt_contract


def _yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to launch the MAPPO vLLM server") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid MAPPO YAML: {path}")
    return payload


def _absolute(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_command(config: dict[str, Any], *, root: Path, adapter_override: str = "") -> tuple[list[str], dict[str, str]]:
    model = _absolute(root, str(config.get("model_path") or ""))
    adapter_value = adapter_override or str(config.get("sft_adapter_path") or "")
    adapter = _absolute(root, adapter_value)
    for path, expected in ((model, "config.json"), (adapter, "adapter_config.json")):
        if not (path / expected).is_file():
            raise SystemExit(f"Invalid vLLM path {path}: missing {expected}")
    reference_adapter = _absolute(root, str(config.get("sft_adapter_path") or ""))
    validate_prompt_contract(
        _absolute(root, str(config.get("prompt_config_path") or "src/rl_v2/policy_prompts.yml")),
        reference_adapter,
        str(config.get("expected_prompt_contract_version") or "macorag-policy-v3"),
    )
    lora_name = str(config.get("vllm_lora_name", "rl_v2_policy"))
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model),
        "--served-model-name", str(config.get("vllm_served_model_name", "rl_v2_base")),
        "--host", str(config.get("vllm_host", "127.0.0.1")),
        "--port", str(int(config.get("vllm_port", 8003))),
        "--tensor-parallel-size", str(int(config.get("vllm_tensor_parallel_size", 1))),
        "--gpu-memory-utilization", str(float(config.get("vllm_gpu_memory_utilization", 0.85))),
        "--max-model-len", str(int(config.get("vllm_max_model_len", 4096))),
        "--max-num-seqs", str(int(config.get("vllm_max_num_seqs", 8))),
        "--dtype", str(config.get("vllm_dtype", "bfloat16")),
        # Ignore model-authored generation defaults (notably Qwen's
        # repetition_penalty=1.05). The request explicitly supplies the raw
        # softmax sampling parameters required by learner re-scoring.
        "--generation-config", "vllm",
        "--enable-lora",
        "--max-loras", "1",
        "--max-cpu-loras", "2",
        "--max-lora-rank", str(int(config.get("vllm_max_lora_rank", 64))),
        "--lora-modules", f"{lora_name}={adapter}",
        "--disable-log-requests",
    ]
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(config.get("vllm_gpu_indices", "1")),
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
    })
    for proxy in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(proxy, None)
    return command, environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the self-contained MAPPO vLLM service")
    parser.add_argument("--config", default=str(Path(__file__).with_name("train_mappo.yml")))
    parser.add_argument("--adapter-path", default="", help="Override initial LoRA, e.g. checkpoint-N/actor")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    command, environment = build_command(
        _yaml(Path(args.config)), root=root, adapter_override=args.adapter_path,
    )
    if args.dry_run:
        print(json.dumps({
            "command": command,
            "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": environment["VLLM_ALLOW_RUNTIME_LORA_UPDATING"],
        }, ensure_ascii=False, indent=2))
        return 0
    os.chdir(root)
    os.execvpe(command[0], command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
