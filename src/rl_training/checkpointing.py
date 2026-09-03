from __future__ import annotations

import json
import hashlib
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from answer_metrics import ANSWER_F1_CONTRACT


CHECKPOINT_SCHEMA_VERSION = 3
CHECKPOINT_COMPLETE_FILE = "COMPLETE"
CHECKPOINT_MANIFEST_FILE = "checkpoint_manifest.json"
OPTIMIZER_STATE_FILE = "optimizer.pt"
TRAINER_STATE_FILE = "trainer_state.pt"
SCHEDULER_STATE_FILE = "scheduler.pt"
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
_CRITICAL_CONFIG_FIELDS = (
    "model_path",
    "sft_adapter_path",
    "system_prompt",
    "seed",
    "group_size",
    "degenerate_bucket_fallback_weight",
    "max_rounds",
    "num_train_epochs",
    "max_steps",
    "max_prompt_length",
    "max_completion_length",
    "temperature",
    "top_p",
    "top_k",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "lr_scheduler_type",
    "min_lr_ratio",
    "max_grad_norm",
    "per_device_train_batch_size",
    "reference_per_device_batch_size",
    "gradient_accumulation_steps",
    "skip_zero_advantage_updates",
    "clip_epsilon",
    "kl_beta",
    "bf16",
    "fp16",
    "attn_implementation",
    "load_4bit",
    "gradient_checkpointing",
    "query_global_reward_weight",
    "evidence_global_reward_weight",
    "answer_global_reward_weight",
    "answer_local_reward_weight",
    "advantage_epsilon",
    "advantage_granularity",
    "use_vllm_generation",
    "vllm_dtype",
    "vllm_sync_mode",
    "vllm_sync_after_step",
    "vllm_sync_every_steps",
    "retrieval_backend",
    "retrieval_root",
    "retrieval_embedding_model",
    "retrieval_max_length",
    "retrieval_top_k",
)


def _local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        return 0


def _world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def _barrier(torch_module: Any) -> None:
    distributed = getattr(torch_module, "distributed", None)
    if (
        _world_size() > 1
        and distributed is not None
        and distributed.is_available()
        and distributed.is_initialized()
    ):
        distributed.barrier()


def _capture_rng_state(torch_module: Any) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    cuda_states = []
    if cuda is not None and cuda.is_available():
        cuda_states = cuda.get_rng_state_all()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fingerprint(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def fingerprint_dataset(samples: list[Any]) -> str:
    return _fingerprint(
        [
            {
                "index": index,
                "dataset": str(getattr(sample, "dataset", "")),
                "qid": str(getattr(sample, "qid", "")),
                "question": str(getattr(sample, "question", "")),
                "answer": str(getattr(sample, "answer", "")),
                "answer_aliases": getattr(sample, "answer_aliases", []),
                "supporting_facts": getattr(sample, "supporting_facts", []),
                "context_doc_ids": getattr(sample, "context_doc_ids", []),
                "metadata": getattr(sample, "metadata", {}),
            }
            for index, sample in enumerate(samples)
        ]
    )


def fingerprint_config(args: Any) -> str:
    return _fingerprint(
        {
            **{name: getattr(args, name, None) for name in _CRITICAL_CONFIG_FIELDS},
            "answer_f1_contract": ANSWER_F1_CONTRACT,
        }
    )


def _torch_load(torch_module: Any, path: Path) -> Any:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def load_full_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_dir():
        raise RuntimeError(f"Full checkpoint directory not found: {checkpoint}")
    if not (checkpoint / CHECKPOINT_COMPLETE_FILE).is_file():
        raise RuntimeError(f"Full checkpoint is not complete: {checkpoint}")
    manifest_path = checkpoint / CHECKPOINT_MANIFEST_FILE
    if not manifest_path.is_file():
        raise RuntimeError(f"Full checkpoint manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Full checkpoint manifest is invalid: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Full checkpoint manifest must be an object: {manifest_path}")
    if int(manifest.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported full checkpoint schema: "
            f"expected {CHECKPOINT_SCHEMA_VERSION}, got {manifest.get('schema_version')}"
        )
    expected_files = manifest.get("expected_files")
    if not isinstance(expected_files, list) or not all(
        isinstance(name, str) and name for name in expected_files
    ):
        raise RuntimeError(f"Full checkpoint expected_files is invalid: {manifest_path}")
    missing = [name for name in expected_files if not (checkpoint / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Full checkpoint is missing expected files at {checkpoint}: {', '.join(missing)}"
        )
    return manifest


def _restore_rng_state(state: dict[str, Any], torch_module: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda") or []
    if cuda_states:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is None or not cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG states but CUDA is unavailable")
        cuda.set_rng_state_all(cuda_states)


def restore_full_training_state(
    checkpoint_path: str | Path,
    *,
    optimizer: Any,
    scheduler: Any | None = None,
    torch_module: Any,
    expected_dataset_fingerprint: str,
    expected_config_fingerprint: str,
    expected_gradient_accumulation_steps: int,
    expected_world_size: int,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    manifest = validate_full_checkpoint_identity(
        checkpoint,
        expected_dataset_fingerprint=expected_dataset_fingerprint,
        expected_config_fingerprint=expected_config_fingerprint,
        expected_gradient_accumulation_steps=expected_gradient_accumulation_steps,
        expected_world_size=expected_world_size,
    )

    trainer_state = _torch_load(torch_module, checkpoint / TRAINER_STATE_FILE)
    if not isinstance(trainer_state, dict):
        raise RuntimeError(f"Full checkpoint trainer state is invalid: {checkpoint}")
    for key in (
        "schema_version",
        "epoch",
        "samples_consumed",
        "global_step",
        "generation_counter",
        "gradient_accumulation_steps",
        "world_size",
        "dataset_fingerprint",
        "config_fingerprint",
        "successful_optimizer_updates",
        "scheduler_total_updates",
        "scheduler_warmup_updates",
        "optimization_contract",
    ):
        if trainer_state.get(key) != manifest.get(key):
            raise RuntimeError(f"Full checkpoint trainer state disagrees with manifest for {key}")

    optimizer_state = _torch_load(torch_module, checkpoint / OPTIMIZER_STATE_FILE)
    optimizer.load_state_dict(optimizer_state)
    scheduler_path = checkpoint / SCHEDULER_STATE_FILE
    if not scheduler_path.is_file():
        raise RuntimeError(f"Full checkpoint scheduler state is missing: {scheduler_path}")
    if scheduler is None:
        raise RuntimeError("Full checkpoint requires a scheduler for state restoration")
    scheduler.load_state_dict(_torch_load(torch_module, scheduler_path))
    if int(getattr(scheduler, "last_epoch", -1)) != int(trainer_state["successful_optimizer_updates"]):
        raise RuntimeError(
            "Full checkpoint scheduler position disagrees with successful optimizer updates"
        )
    rank = _local_rank()
    rng_path = checkpoint / f"rng_state_rank{rank}.pt"
    rng_state = _torch_load(torch_module, rng_path)
    if not isinstance(rng_state, dict):
        raise RuntimeError(f"Full checkpoint RNG state is invalid: {rng_path}")
    _restore_rng_state(rng_state, torch_module)
    return trainer_state


def validate_full_checkpoint_identity(
    checkpoint_path: str | Path,
    *,
    expected_dataset_fingerprint: str,
    expected_config_fingerprint: str,
    expected_gradient_accumulation_steps: int,
    expected_world_size: int,
) -> dict[str, Any]:
    manifest = load_full_checkpoint_metadata(checkpoint_path)
    comparisons = (
        (
            "dataset fingerprint",
            str(manifest.get("dataset_fingerprint")),
            str(expected_dataset_fingerprint),
        ),
        (
            "config fingerprint",
            str(manifest.get("config_fingerprint")),
            str(expected_config_fingerprint),
        ),
        (
            "gradient accumulation",
            int(manifest.get("gradient_accumulation_steps", -1)),
            int(expected_gradient_accumulation_steps),
        ),
        (
            "world size",
            int(manifest.get("world_size", -1)),
            int(expected_world_size),
        ),
    )
    for label, actual, expected in comparisons:
        if actual != expected:
            raise RuntimeError(
                f"Full checkpoint {label} mismatch: expected {expected!r}, got {actual!r}"
            )
    return manifest


def save_full_checkpoint(
    *,
    raw_policy_model: Any,
    tokenizer: Any,
    optimizer: Any,
    output_dir: str | Path,
    step: int,
    epoch: int,
    samples_consumed: int,
    generation_counter: int,
    gradient_accumulation_steps: int,
    dataset_fingerprint: str,
    config_fingerprint: str,
    torch_module: Any,
    save_total_limit: int,
    milestone_steps: int,
    scheduler: Any | None = None,
    successful_optimizer_updates: int = 0,
    scheduler_total_updates: int = 0,
    scheduler_warmup_updates: int = 0,
    optimization_contract: dict[str, Any] | None = None,
    prompt_contract_metadata: dict[str, Any] | None = None,
) -> Path:
    if scheduler is None:
        raise ValueError("Schema-v2 full checkpoints require scheduler state")
    if int(scheduler_total_updates) <= 0:
        raise ValueError("Schema-v2 full checkpoints require a positive scheduler horizon")
    if not isinstance(optimization_contract, dict) or not optimization_contract:
        raise ValueError("Schema-v2 full checkpoints require an optimization contract")
    if int(getattr(scheduler, "last_epoch", -1)) != int(successful_optimizer_updates):
        raise ValueError("Scheduler position must match successful optimizer updates")
    output_path = Path(output_dir)
    checkpoint_path = output_path / f"checkpoint-{int(step)}"
    temporary_path = output_path / f".checkpoint-{int(step)}.tmp"
    rank = _local_rank()

    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        temporary_path.mkdir(parents=True)
    _barrier(torch_module)

    torch_module.save(
        _capture_rng_state(torch_module),
        temporary_path / f"rng_state_rank{rank}.pt",
    )
    _barrier(torch_module)

    if rank == 0:
        if callable(getattr(raw_policy_model, "set_adapter", None)):
            raw_policy_model.set_adapter("default")
        raw_policy_model.save_pretrained(
            temporary_path,
            selected_adapters=["default"],
        )
        tokenizer.save_pretrained(temporary_path)
        if prompt_contract_metadata is not None:
            _write_json(
                temporary_path / "prompt_contract.json",
                dict(prompt_contract_metadata),
            )
        torch_module.save(optimizer.state_dict(), temporary_path / OPTIMIZER_STATE_FILE)
        torch_module.save(scheduler.state_dict(), temporary_path / SCHEDULER_STATE_FILE)
        trainer_state = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "epoch": int(epoch),
            "samples_consumed": int(samples_consumed),
            "global_step": int(step),
            "generation_counter": int(generation_counter),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "world_size": _world_size(),
            "dataset_fingerprint": str(dataset_fingerprint),
            "config_fingerprint": str(config_fingerprint),
        }
        trainer_state.update(
            {
                "successful_optimizer_updates": int(successful_optimizer_updates),
                "scheduler_total_updates": int(scheduler_total_updates),
                "scheduler_warmup_updates": int(scheduler_warmup_updates),
                "optimization_contract": dict(optimization_contract),
            }
        )
        torch_module.save(trainer_state, temporary_path / TRAINER_STATE_FILE)
        expected_files = sorted(
            [path.name for path in temporary_path.iterdir()]
            + [CHECKPOINT_MANIFEST_FILE, CHECKPOINT_COMPLETE_FILE]
        )
        manifest = {
            **trainer_state,
            "expected_files": expected_files,
        }
        _write_json(temporary_path / CHECKPOINT_MANIFEST_FILE, manifest)
        (temporary_path / CHECKPOINT_COMPLETE_FILE).write_text(
            "complete\n",
            encoding="utf-8",
        )
        temporary_path.replace(checkpoint_path)
        prune_full_checkpoints(
            output_path,
            keep_last=save_total_limit,
            milestone_steps=milestone_steps,
        )
    _barrier(torch_module)
    return checkpoint_path


def prune_full_checkpoints(
    output_dir: str | Path,
    *,
    keep_last: int,
    milestone_steps: int,
) -> list[Path]:
    if keep_last < 0:
        raise ValueError("keep_last must be non-negative")
    if milestone_steps < 0:
        raise ValueError("milestone_steps must be non-negative")

    checkpoints: list[tuple[int, Path]] = []
    for path in Path(output_dir).glob("checkpoint-*"):
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if (
            match is None
            or not path.is_dir()
            or not (path / CHECKPOINT_COMPLETE_FILE).is_file()
        ):
            continue
        checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()

    newest = {step for step, _ in checkpoints[-keep_last:]} if keep_last else set()
    milestones = {
        step
        for step, _ in checkpoints
        if milestone_steps > 0 and step % milestone_steps == 0
    }
    keep = newest | milestones
    removed: list[Path] = []
    for step, path in checkpoints:
        if step in keep:
            continue
        shutil.rmtree(path)
        removed.append(path)
    return removed
