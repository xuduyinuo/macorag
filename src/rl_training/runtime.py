from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_gpu_indices(value: str | int | None) -> set[str]:
    """解析 YAML/CLI 中的 GPU 列表，供训练进程和 vLLM 进程做冲突检查。"""
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def validate_vllm_gpu_placement(args: Any) -> None:
    """确保 trainer 和 vLLM generation 不抢同一张物理 GPU。"""
    if not getattr(args, "use_vllm_generation", False):
        return
    trainer_gpus = parse_gpu_indices(getattr(args, "gpu_indices", None))
    if not trainer_gpus:
        trainer_gpus = parse_gpu_indices(getattr(args, "gpu_index", None))
    vllm_gpus = parse_gpu_indices(getattr(args, "vllm_gpu_indices", None))
    overlap = trainer_gpus & vllm_gpus
    if overlap:
        raise SystemExit(
            "vLLM GPU overlap detected: trainer gpu_indices="
            f"{sorted(trainer_gpus)} and vllm_gpu_indices={sorted(vllm_gpus)} share {sorted(overlap)}. "
            "Use separate GPUs for trainer and vLLM generation."
        )


def is_local_host(host: str | None) -> bool:
    return str(host or "").strip() in {"", "127.0.0.1", "localhost", "0.0.0.0"}


def iter_proc_cmdlines() -> list[list[str]]:
    """读取本机进程命令行，用于发现是否连到了旧的本地 vLLM 服务。"""
    cmdlines: list[list[str]] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return cmdlines
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if parts:
            cmdlines.append(parts)
    return cmdlines


def extract_vllm_server_model_paths(cmdlines: list[list[str]]) -> list[str]:
    model_paths: list[str] = []
    for cmdline in cmdlines:
        if "vllm-serve" not in cmdline:
            continue
        if not any("trl" in Path(part).name for part in cmdline):
            continue
        for index, part in enumerate(cmdline):
            if part == "--model" and index + 1 < len(cmdline):
                model_paths.append(cmdline[index + 1])
            elif part.startswith("--model="):
                model_paths.append(part.split("=", 1)[1])
    return model_paths


def same_model_path(left: str, right: str) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    if left_path == right_path:
        return True
    try:
        return left_path.resolve() == right_path.resolve()
    except OSError:
        return False


def validate_local_vllm_server_model(args: Any, *, cmdlines: list[list[str]] | None = None) -> None:
    """本地 vLLM 服务存在时，先确认它加载的是当前训练配置中的基座模型。"""
    if not getattr(args, "use_vllm_generation", False):
        return
    if not is_local_host(getattr(args, "vllm_host", None)):
        return
    model_path = str(getattr(args, "model_path", "") or "")
    if not model_path:
        return
    server_models = extract_vllm_server_model_paths(cmdlines if cmdlines is not None else iter_proc_cmdlines())
    if not server_models:
        return
    if any(same_model_path(path, model_path) for path in server_models):
        return
    raise SystemExit(
        "vLLM server model mismatch: trainer model_path="
        f"{model_path!r}, but local trl vllm-serve process uses {server_models!r}. "
        "Stop the stale vLLM server and restart scripts/run_grpo_vllm_server.sh with the current config."
    )
