from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "train_sft.yml"


@dataclass
class SFTConfig:
    model_path: str
    train_file: str
    validation_file: str
    teacher_run_config: str
    prompt_config_path: str
    output_root: str
    expected_teacher_prompt_contract_version: str = "sft-v2-unified-wiki18-v2"
    expected_train_trajectories: int = 1000
    expected_validation_source_trajectories: int = 200
    validation_trajectory_limit: int = 100
    max_length: int = 8192
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"
    )
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 2.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 5
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    load_4bit: bool = True
    attn_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    gpu_indices: str = "0,1"
    resume_from_checkpoint: str | None = None


def _resolve_repo_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else REPO_ROOT / path)


def load_config(path: str | Path) -> SFTConfig:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    known = {item.name for item in fields(SFTConfig)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"Unknown SFT config keys: {', '.join(unknown)}")
    if "target_modules" in payload:
        payload["target_modules"] = tuple(payload["target_modules"])
    config = SFTConfig(**payload)
    for name in (
        "model_path", "train_file", "validation_file", "teacher_run_config",
        "prompt_config_path", "output_root",
    ):
        setattr(config, name, _resolve_repo_path(str(getattr(config, name))))
    if config.resume_from_checkpoint:
        config.resume_from_checkpoint = _resolve_repo_path(config.resume_from_checkpoint)
    if config.max_length <= 0:
        raise ValueError("max_length must be positive")
    if config.expected_train_trajectories <= 0 or config.expected_validation_source_trajectories <= 0:
        raise ValueError("expected split sizes must be positive")
    if not 0 < config.validation_trajectory_limit <= config.expected_validation_source_trajectories:
        raise ValueError(
            "validation_trajectory_limit must be positive and no larger than "
            "expected_validation_source_trajectories"
        )
    if config.eval_steps <= 0 or config.save_steps <= 0:
        raise ValueError("eval_steps and save_steps must be positive")
    if config.save_steps % config.eval_steps != 0:
        raise ValueError("save_steps must be divisible by eval_steps")
    if config.bf16 and config.fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    return config


def parse_args(argv: list[str] | None = None) -> tuple[SFTConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Train the MACORAG v2 shared SFT policy")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--check-only-max-samples", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = _resolve_repo_path(args.resume_from_checkpoint)
    return config, args
