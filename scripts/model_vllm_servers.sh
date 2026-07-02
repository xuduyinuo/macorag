#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/model_vllm_servers.yml}"

"${PYTHON:-python}" - "${CONFIG_PATH}" "$@" <<'PY'
import argparse
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml


DEFAULTS = {
    "vllm_bin": "/data/conda/envs/vllm/bin/vllm",
    "model_path": "model/Qwen2.5-7B-Instruct",
    "adapter_path": "outputs/lora_qwen2.5-7b_trajectory_20260627_203027/adapter",
    "vllm_model": "macorag-lora",
    "gpu_indices": "0",
    "vllm_base_urls": ["http://127.0.0.1:8000/v1"],
    "host": "127.0.0.1",
    "dtype": "auto",
    "gpu_memory_utilization": 0.9,
    "max_model_len": None,
    "trust_remote_code": True,
    "environment": {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    },
    "extra_args": [],
}


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"vLLM server config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"vLLM server config must be a mapping: {path}")
    config = dict(DEFAULTS)
    config.update(loaded)
    return config


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _is_present(value) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"none", "null"}


def _ports_from_urls(urls: list[str]) -> list[int]:
    ports = []
    for url in urls:
        parsed = urlparse(str(url))
        if parsed.port is None:
            raise SystemExit(f"vllm_base_urls entry must include a port: {url}")
        ports.append(parsed.port)
    if not ports:
        raise SystemExit("vllm_base_urls must contain at least one URL before starting vLLM servers.")
    return ports


def _as_env(value) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit("environment must be a mapping of KEY: VALUE entries.")
    return {str(key): str(env_value) for key, env_value in value.items()}


def _env_from_cli(values: list[str] | None) -> dict[str, str] | None:
    if values is None:
        return None
    result = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--env entries must use KEY=VALUE format: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--env entries must include a non-empty key: {item}")
        result[key] = value
    return result


def _parse_args(default_config_path: str, cli_args: list[str]) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=default_config_path)
    config_args, _ = config_parser.parse_known_args(cli_args)
    config = _load_config(config_args.config)

    parser = argparse.ArgumentParser(description="Start one or more vLLM OpenAI-compatible servers for MACORAG.")
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--vllm-bin", default=config["vllm_bin"])
    parser.add_argument("--model-path", default=config["model_path"])
    parser.add_argument("--adapter-path", default=config["adapter_path"])
    parser.add_argument("--vllm-model", default=config["vllm_model"])
    parser.add_argument("--gpu-indices", default=str(config["gpu_indices"]))
    parser.add_argument("--vllm-base-urls", nargs="+", default=_as_list(config["vllm_base_urls"]))
    parser.add_argument("--host", default=config["host"])
    parser.add_argument("--dtype", default=config["dtype"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=config["gpu_memory_utilization"])
    parser.add_argument("--max-model-len", default=config["max_model_len"])
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=bool(config["trust_remote_code"]))
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--env", action="append", default=None, help="Set an environment variable for vLLM, formatted as KEY=VALUE.")
    parser.add_argument("--extra-arg", action="append", default=None)
    args = parser.parse_args(cli_args)
    args.environment = _env_from_cli(args.env)
    if args.environment is None:
        args.environment = _as_env(config.get("environment"))
    args.extra_args = args.extra_arg if args.extra_arg is not None else _as_list(config.get("extra_args"))
    return args


def _build_commands(args: argparse.Namespace) -> list[tuple[str, dict[str, str], list[str]]]:
    # Starts vllm serve processes with matching ports from vllm_base_urls.
    gpus = [item.strip() for item in str(args.gpu_indices).split(",") if item.strip()]
    if not gpus:
        gpus = ["0"]
    ports = _ports_from_urls(_as_list(args.vllm_base_urls))
    commands = []
    for index, port in enumerate(ports):
        argv = [
            str(args.vllm_bin),
            "serve",
            str(args.model_path),
            "--host",
            str(args.host),
            "--port",
            str(port),
        ]
        if _is_present(args.adapter_path):
            argv.extend(
                [
                    "--enable-lora",
                    "--lora-modules",
                    f"{args.vllm_model}={args.adapter_path}",
                ]
            )
        elif _is_present(args.vllm_model):
            argv.extend(["--served-model-name", str(args.vllm_model)])
        if args.dtype:
            argv.extend(["--dtype", str(args.dtype)])
        if args.gpu_memory_utilization is not None:
            argv.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
        if args.max_model_len is not None:
            argv.extend(["--max-model-len", str(args.max_model_len)])
        if args.trust_remote_code:
            argv.append("--trust-remote-code")
        argv.extend(_as_list(args.extra_args))
        commands.append((gpus[index % len(gpus)], dict(args.environment), argv))
    return commands


def _print_dry_run(commands: list[tuple[str, dict[str, str], list[str]]]) -> None:
    for gpu_index, extra_env, argv in commands:
        env_parts = [f"CUDA_VISIBLE_DEVICES={gpu_index}", *[f"{key}={value}" for key, value in extra_env.items()]]
        print(f"{' '.join(env_parts)} {shlex.join(argv)}")


def _run(commands: list[tuple[str, dict[str, str], list[str]]]) -> None:
    processes = []

    def stop_processes(signum=None, frame=None) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop_processes)
    signal.signal(signal.SIGTERM, stop_processes)

    for gpu_index, extra_env, argv in commands:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_index
        env.update(extra_env)
        processes.append(subprocess.Popen(argv, env=env))

    exit_code = 0
    for process in processes:
        exit_code = max(exit_code, process.wait())
    raise SystemExit(exit_code)


def main() -> None:
    args = _parse_args(sys.argv[1], sys.argv[2:])
    commands = _build_commands(args)
    if os.environ.get("MACORAG_VLLM_DRY_RUN", "0") == "1":
        _print_dry_run(commands)
        return
    _run(commands)


if __name__ == "__main__":
    main()
PY
