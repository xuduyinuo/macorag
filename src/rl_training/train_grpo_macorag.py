#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import random
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from answer_metrics import ANSWER_F1_CONTRACT
from rag import RAGLoopExecutor

from .config import parse_args
from .credit_assignment import compute_decision_returns, group_relative_normalization
from prompt_config import load_prompt_contract
from rag.protocol_metrics import ProtocolWindowMonitor, compute_protocol_metrics
from .batched_rollout import run_batched_rollouts
from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_COMPLETE_FILE,
    CHECKPOINT_MANIFEST_FILE,
    fingerprint_config,
    fingerprint_dataset,
    load_full_checkpoint_metadata,
    restore_full_training_state,
    save_full_checkpoint,
    validate_full_checkpoint_identity,
)
from .data import RLSample, epoch_sample_order, load_rl_samples, select_balanced_samples
from .logging_utils import append_jsonl as _append_jsonl
from .logging_utils import make_timestamped_run_dir
from .logging_utils import write_json as _write_json
from .policy import HFSharedPolicy, VLLMSharedPolicy, batched_sequence_logprobs
from .retrieval import create_retrieval_env
from .scheduling import build_cosine_scheduler
from .rewards import compute_action_rewards, compute_rl_rewards
from .runtime import extract_vllm_server_model_paths as _extract_vllm_server_model_paths
from .runtime import parse_gpu_indices as _parse_gpu_indices
from .runtime import validate_local_vllm_server_model as _validate_local_vllm_server_model
from .runtime import validate_vllm_gpu_placement as _validate_vllm_gpu_placement
from .trainer import compute_grpo_loss, normalize_group_advantages
from .vllm_client import VLLMGenerationClient


_POLICY_ADAPTER_NAME = "default"
_REFERENCE_ADAPTER_NAME = "reference"


@dataclass(frozen=True)
class ResumeState:
    checkpoint_path: Path | None
    epoch: int
    samples_consumed: int
    global_step: int
    generation_counter: int = 0
    is_full_checkpoint: bool = False
    optimizer_state_restored: bool = False
    rng_state_restored: bool = False


def _configure_visible_gpus(args: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    gpu_indices = str(getattr(args, "gpu_indices", "") or "").strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices or str(args.gpu_index)


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


def _is_main_process() -> bool:
    return _local_rank() == 0


def _load_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install transformers, peft, torch and optional bitsandbytes "
            "in the same environment used for MACORAG SFT training."
        ) from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "PeftModel": PeftModel,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
    }


def _torch_dtype(args: Any, torch: Any) -> Any:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float16


def _validate_acceleration_runtime(
    args: Any,
    torch: Any,
    find_spec=importlib.util.find_spec,
    import_module=importlib.import_module,
) -> None:
    if args.bf16 and args.fp16:
        raise SystemExit("bf16 and fp16 cannot both be enabled.")
    if args.attn_implementation == "flash_attention_2":
        if find_spec("flash_attn") is None:
            raise SystemExit(
                "FlashAttention 2 requested but flash_attn is not installed in the active environment."
            )
        try:
            flash_attn = import_module("flash_attn")
            flash_attn_func = getattr(flash_attn, "flash_attn_func")
            if not callable(flash_attn_func):
                raise ImportError("flash_attn.flash_attn_func is not callable")
        except (ImportError, OSError, AttributeError) as exc:
            raise SystemExit(
                "FlashAttention 2 requested but flash_attn could not be imported: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not torch.cuda.is_available():
            raise SystemExit("FlashAttention 2 requested but CUDA is unavailable.")
    if args.bf16 and _world_size() > 1 and torch.cuda.is_available():
        torch.cuda.set_device(_local_rank())
    if args.bf16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise SystemExit("bf16 requested but the selected CUDA device does not support bf16.")


def _model_kwargs(args: Any, torch: Any, local_rank: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype(args, torch),
        "attn_implementation": args.attn_implementation,
    }
    if not args.load_4bit:
        return kwargs
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # type: ignore  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(f"4-bit quantization requested but dependency missing: {exc.name}.") from exc
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=_torch_dtype(args, torch),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    kwargs["device_map"] = {"": local_rank} if torch.cuda.is_available() else None
    return kwargs


def _setup_distributed(torch: Any) -> None:
    if _world_size() <= 1:
        return
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(_local_rank())


def _cleanup_distributed(torch: Any) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{_local_rank()}")
    return torch.device("cpu")


def _set_adapter_trainability(model: Any, adapter_name: str, *, trainable: bool) -> None:
    marker = f".{adapter_name}."
    for name, parameter in model.named_parameters():
        if marker in f".{name}.":
            parameter.requires_grad_(trainable)


def _activate_policy_adapter(model: Any) -> None:
    set_adapter = getattr(model, "set_adapter", None)
    if not callable(set_adapter):
        raise SystemExit("Shared-base GRPO requires PEFT set_adapter() support.")
    set_adapter(_POLICY_ADAPTER_NAME)
    _set_adapter_trainability(model, _REFERENCE_ADAPTER_NAME, trainable=False)
    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if getattr(parameter, "requires_grad", False)
    ]
    policy_marker = f".{_POLICY_ADAPTER_NAME}."
    unexpected = [name for name in trainable_names if policy_marker not in f".{name}."]
    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise SystemExit(f"Shared-base GRPO found non-policy trainable parameters: {preview}")
    if not trainable_names:
        raise SystemExit("Shared-base GRPO policy adapter has no trainable parameters.")


def _set_deterministic_policy_train_mode(model: Any, *, torch: Any) -> None:
    """Enable gradient checkpointing's train path without stochastic dropout."""
    train = getattr(model, "train", None)
    if not callable(train):
        return
    train()
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return
    dropout_base = torch.nn.modules.dropout._DropoutNd
    for module in modules():
        if isinstance(module, dropout_base):
            module.eval()


@contextmanager
def _reference_adapter_context(reference_model: Any, policy_model: Any):
    set_adapter = getattr(reference_model, "set_adapter", None)
    if reference_model is not policy_model or not callable(set_adapter):
        yield reference_model
        return

    was_training = bool(getattr(reference_model, "training", False))
    set_adapter(_REFERENCE_ADAPTER_NAME)
    _set_adapter_trainability(reference_model, _REFERENCE_ADAPTER_NAME, trainable=False)
    reference_model.eval()
    try:
        yield reference_model
    finally:
        _activate_policy_adapter(reference_model)
        reference_model.train(was_training)


def _load_policy_and_reference(args: Any, deps: dict[str, Any], device: Any) -> tuple[Any, Any, Any]:
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    PeftModel = deps["PeftModel"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]

    resume_checkpoint = str(getattr(args, "resume_from_checkpoint", "") or "").strip()
    policy_adapter_path = resume_checkpoint or args.sft_adapter_path
    tokenizer = AutoTokenizer.from_pretrained(policy_adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **_model_kwargs(args, torch, _local_rank()),
    )
    base_model.config.use_cache = False
    if args.load_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    policy_model = PeftModel.from_pretrained(
        base_model,
        policy_adapter_path,
        adapter_name=_POLICY_ADAPTER_NAME,
        is_trainable=True,
    )
    load_adapter = getattr(policy_model, "load_adapter", None)
    if not callable(load_adapter):
        raise SystemExit("Shared-base GRPO requires PEFT load_adapter() support.")
    load_adapter(
        getattr(args, "reference_model_path", "") or args.sft_adapter_path,
        adapter_name=_REFERENCE_ADAPTER_NAME,
        is_trainable=False,
    )
    _activate_policy_adapter(policy_model)
    if args.gradient_checkpointing and hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not args.load_4bit:
        policy_model.to(device)
    return tokenizer, policy_model, policy_model


def _validate_sft_prompt_contract(args: Any) -> dict[str, Any]:
    contract = load_prompt_contract(args.prompt_config_path)
    metadata_path = Path(args.sft_adapter_path) / "prompt_contract.json"
    if not metadata_path.is_file():
        if args.require_sft_prompt_contract:
            raise SystemExit(f"SFT adapter prompt contract metadata is missing: {metadata_path}")
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("prompt_contract_version")) != contract.version:
        raise SystemExit("SFT adapter prompt contract version mismatch")
    if str(metadata.get("prompt_contract_fingerprint")) != contract.fingerprint:
        raise SystemExit("SFT adapter prompt contract fingerprint mismatch")
    if int(metadata.get("max_rounds", -1)) != int(args.max_rounds):
        raise SystemExit("SFT adapter max_rounds mismatch")
    if int(metadata.get("retrieval_top_k", -1)) != int(args.retrieval_top_k):
        raise SystemExit("SFT adapter retrieval_top_k mismatch")
    return metadata


def _validate_online_policy_sync(args: Any) -> None:
    if getattr(args, "use_vllm_generation", False) and (
        not getattr(args, "vllm_sync_after_step", True)
        or int(getattr(args, "vllm_sync_every_steps", 1)) != 1
    ):
        raise SystemExit(
            "Paper-faithful online GRPO requires vllm_sync_after_step=true and "
            "vllm_sync_every_steps=1 to prevent off-policy rollout weights."
        )


def _validate_paper_credit_config(args: Any) -> None:
    if str(getattr(args, "advantage_granularity", "role_only")) != "role_only":
        raise SystemExit(
            "Section 3.3 requires same-question, same-agent normalization across all rounds; "
            "advantage_granularity must be role_only."
        )
    if float(getattr(args, "degenerate_bucket_fallback_weight", 0.0)) != 0.0:
        raise SystemExit(
            "Section 3.3 keeps tied groups at zero; degenerate_bucket_fallback_weight must be 0."
        )


def _wrap_ddp(model: Any, torch: Any) -> Any:
    if _world_size() <= 1:
        return model
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(
        model,
        device_ids=[_local_rank()] if torch.cuda.is_available() else None,
        output_device=_local_rank() if torch.cuda.is_available() else None,
        find_unused_parameters=False,
    )


def _rank_samples(samples: list[RLSample]) -> list[tuple[int, RLSample]]:
    world_size = _world_size()
    rank = _local_rank()
    return [(index, sample) for index, sample in enumerate(samples) if index % world_size == rank]


def _validate_resume_checkpoint(checkpoint_path: Path) -> None:
    config_path = checkpoint_path / "adapter_config.json"
    if not config_path.is_file():
        raise SystemExit(f"GRPO resume checkpoint is missing adapter_config.json: {checkpoint_path}")
    try:
        from peft import PeftConfig

        PeftConfig.from_pretrained(str(checkpoint_path))
    except Exception as exc:
        raise SystemExit(f"Invalid GRPO resume adapter config at {config_path}: {exc}") from exc

    safetensors_path = checkpoint_path / "adapter_model.safetensors"
    binary_path = checkpoint_path / "adapter_model.bin"
    if not safetensors_path.is_file() and not binary_path.is_file():
        raise SystemExit(f"GRPO resume checkpoint is missing adapter weights: {checkpoint_path}")
    if safetensors_path.is_file():
        try:
            from safetensors import safe_open

            with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
                if not list(handle.keys()):
                    raise ValueError("no tensors found")
        except Exception as exc:
            raise SystemExit(f"Invalid GRPO resume adapter weights at {safetensors_path}: {exc}") from exc
    elif binary_path.stat().st_size <= 0:
        raise SystemExit(f"Invalid GRPO resume adapter weights at {binary_path}: empty file")


def _resolve_resume_state(
    args: Any,
    *,
    rank_epoch_size: int,
    total_epochs: int,
) -> ResumeState:
    checkpoint_value = str(getattr(args, "resume_from_checkpoint", "") or "").strip()
    resume_epoch = int(getattr(args, "resume_epoch", 1))
    samples_consumed = int(getattr(args, "resume_samples_consumed", 0))
    global_step = int(getattr(args, "resume_global_step", 0))
    if not checkpoint_value:
        if resume_epoch != 1 or samples_consumed != 0 or global_step != 0:
            raise SystemExit("Resume position arguments require --resume-from-checkpoint.")
        return ResumeState(checkpoint_path=None, epoch=1, samples_consumed=0, global_step=0)

    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_dir():
        raise SystemExit(f"GRPO resume checkpoint directory not found: {checkpoint_path}")
    _validate_resume_checkpoint(checkpoint_path)
    has_full_state_marker = (
        (checkpoint_path / CHECKPOINT_MANIFEST_FILE).exists()
        or (checkpoint_path / CHECKPOINT_COMPLETE_FILE).exists()
    )
    if has_full_state_marker:
        manifest = load_full_checkpoint_metadata(checkpoint_path)
        resume_epoch = int(manifest["epoch"])
        samples_consumed = int(manifest["samples_consumed"])
        global_step = int(manifest["global_step"])
        generation_counter = int(manifest["generation_counter"])
    else:
        generation_counter = 0
    if resume_epoch < 1 or resume_epoch > total_epochs:
        raise SystemExit(f"resume_epoch must be between 1 and {total_epochs}, got {resume_epoch}.")
    if samples_consumed < 0 or samples_consumed > rank_epoch_size:
        raise SystemExit(
            "resume_samples_consumed must be between 0 and the rank-local epoch size "
            f"{rank_epoch_size}, got {samples_consumed}."
        )
    if global_step < 0:
        raise SystemExit(f"resume_global_step must be non-negative, got {global_step}.")
    return ResumeState(
        checkpoint_path=checkpoint_path,
        epoch=resume_epoch,
        samples_consumed=samples_consumed,
        global_step=global_step,
        generation_counter=generation_counter,
        is_full_checkpoint=has_full_state_marker,
    )


def _rank_samples_for_epoch(
    samples: list[RLSample],
    *,
    seed: int,
    epoch: int,
    resume_state: ResumeState,
) -> list[tuple[int, RLSample]]:
    rank_samples = _rank_samples(epoch_sample_order(samples, seed=seed, epoch=epoch))
    if resume_state.checkpoint_path is not None and epoch == resume_state.epoch:
        return rank_samples[resume_state.samples_consumed :]
    return rank_samples


def _run_step_limit(args: Any) -> int:
    run_until = int(getattr(args, "run_until_step", 0) or 0)
    return run_until if run_until > 0 else int(getattr(args, "max_steps", 0) or 0)


def _resume_meta_payload(resume_state: ResumeState) -> dict[str, Any]:
    return {
        "resume_from_checkpoint": (
            str(resume_state.checkpoint_path) if resume_state.checkpoint_path is not None else None
        ),
        "resume_epoch": resume_state.epoch,
        "resume_samples_consumed": resume_state.samples_consumed,
        "resume_global_step": resume_state.global_step,
        "generation_counter": resume_state.generation_counter,
        "checkpoint_schema_version": (
            CHECKPOINT_SCHEMA_VERSION if resume_state.is_full_checkpoint else None
        ),
        "optimizer_state_restored": resume_state.optimizer_state_restored,
        "rng_state_restored": resume_state.rng_state_restored,
    }


def _build_retrieval_env(args: Any) -> Any:
    return create_retrieval_env(
        backend=getattr(args, "retrieval_backend", "linear_rag"),
        retrieval_root=args.retrieval_root,
        embedding_model=args.retrieval_embedding_model,
        device=getattr(args, "retrieval_device", "cpu"),
        max_length=getattr(args, "retrieval_max_length", 512),
        spacy_model=args.retrieval_spacy_model,
        top_k=args.retrieval_top_k,
        max_workers=args.retrieval_max_workers,
        batch_size=args.retrieval_batch_size,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        query_cache_size=getattr(args, "retrieval_query_cache_size", 0),
    )


def _build_policy(
    args: Any,
    raw_policy_model: Any,
    tokenizer: Any,
    *,
    generation_counter: int = 0,
) -> HFSharedPolicy:
    common = {
        "model": raw_policy_model,
        "tokenizer": tokenizer,
        "system_prompt": args.system_prompt,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if not args.use_vllm_generation:
        return HFSharedPolicy(**common)
    _validate_local_vllm_server_model(args)
    client = VLLMGenerationClient(
        host=args.vllm_host,
        port=args.vllm_port,
        timeout_seconds=args.vllm_timeout_seconds,
        max_generate_attempts=getattr(args, "vllm_generate_max_attempts", 3),
        retry_backoff_seconds=getattr(args, "vllm_generate_retry_backoff_seconds", 1.0),
    )
    if getattr(args, "vllm_sync_mode", "dense") == "lora":
        client.validate_lora_server(args)
    else:
        client.check_server()
    return VLLMSharedPolicy(
        vllm_client=client,
        generation_seed=int(getattr(args, "seed", 0)) + (_local_rank() * 1_000_000),
        generation_counter=int(generation_counter),
        **common,
    )


def _sync_vllm_after_optimizer_step(
    policy: Any,
    raw_policy_model: Any,
    args: Any,
    *,
    completed_step: int | None = None,
) -> float:
    if not getattr(args, "use_vllm_generation", False):
        return 0.0
    if not getattr(args, "vllm_sync_after_step", True):
        return 0.0
    if not _is_main_process():
        return 0.0
    sync_every_steps = max(1, int(getattr(args, "vllm_sync_every_steps", 1)))
    if completed_step is not None and completed_step % sync_every_steps != 0:
        return 0.0
    client = getattr(policy, "vllm_client", None)
    if client is None:
        raise SystemExit("vLLM generation is enabled but policy has no vLLM client.")
    if callable(getattr(raw_policy_model, "set_adapter", None)):
        _activate_policy_adapter(raw_policy_model)
    sync_mode = getattr(args, "vllm_sync_mode", "dense")
    if sync_mode == "lora":
        return float(client.sync_lora_parameters(raw_policy_model))
    if sync_mode == "dense":
        return float(client.sync_trainable_parameters(raw_policy_model))
    raise SystemExit(f"Unsupported vLLM sync mode: {sync_mode}")


def _sync_vllm_before_first_rollout(
    policy: Any,
    raw_policy_model: Any,
    args: Any,
) -> float:
    if not getattr(args, "use_vllm_generation", False) or not _is_main_process():
        return 0.0
    client = getattr(policy, "vllm_client", None)
    if client is None:
        raise SystemExit("vLLM generation is enabled but policy has no vLLM client.")
    if callable(getattr(raw_policy_model, "set_adapter", None)):
        _activate_policy_adapter(raw_policy_model)
    sync_mode = getattr(args, "vllm_sync_mode", "dense")
    if sync_mode == "lora":
        return float(client.sync_lora_parameters(raw_policy_model))
    if sync_mode == "dense":
        return float(client.sync_trainable_parameters(raw_policy_model))
    raise SystemExit(f"Unsupported vLLM sync mode: {sync_mode}")


def _gradient_health(model: Any, *, torch: Any) -> tuple[float | None, bool | None]:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None, None
    gradients = [
        parameter.grad.detach()
        for parameter in parameters()
        if getattr(parameter, "grad", None) is not None
    ]
    if not gradients:
        return None, None
    per_gradient_norms = torch.stack(
        [torch.linalg.vector_norm(gradient.float()) for gradient in gradients]
    )
    total_norm = torch.linalg.vector_norm(per_gradient_norms)
    return float(total_norm.item()), bool(torch.isfinite(total_norm).item())


def _flush_pending_gradients_if_finite(
    *,
    raw_policy_model: Any,
    optimizer: Any,
    torch: Any,
    sync_weights: Any,
    scheduler: Any | None = None,
    max_grad_norm: float = 1.0,
) -> bool:
    _, gradients_finite = _gradient_health(raw_policy_model, torch=torch)
    if gradients_finite is False:
        optimizer.zero_grad(set_to_none=True)
        return False
    trainable_parameters = [
        parameter
        for parameter in raw_policy_model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    torch.nn.utils.clip_grad_norm_(trainable_parameters, float(max_grad_norm))
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    sync_weights()
    return True


_ROLLOUT_TIMING_KEYS = (
    "time_rollout_seconds",
    "time_vllm_generate_seconds",
    "time_behavior_rescore_seconds",
    "time_reward_seconds",
    "time_retrieval_seconds",
    "retrieval_cache_hits",
    "retrieval_cache_misses",
)


def _empty_rollout_timing() -> dict[str, Any]:
    return {
        "time_rollout_seconds": 0.0,
        "time_vllm_generate_seconds": 0.0,
        "time_behavior_rescore_seconds": 0.0,
        "time_reward_seconds": 0.0,
        "time_retrieval_seconds": 0.0,
        "retrieval_cache_hits": 0,
        "retrieval_cache_misses": 0,
    }


def _accumulate_rollout_timing(total: dict[str, Any], update: dict[str, Any]) -> None:
    for key in _ROLLOUT_TIMING_KEYS:
        total[key] = total.get(key, 0) + update.get(key, 0)


def _generate_rollout_candidates(
    *,
    args: Any,
    sample: RLSample,
    policy: HFSharedPolicy,
    retrieval_env: CachedLinearRAGRetrievalEnv,
    candidate_count: int,
    group_index_offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    time_rollout_seconds = 0.0
    time_vllm_generate_seconds = 0.0
    time_behavior_rescore_seconds = 0.0
    retrieval_stats = getattr(retrieval_env, "stats", None)
    retrieval_before = retrieval_stats() if callable(retrieval_stats) else {}

    supports_batch = callable(getattr(policy, "generate_batch", None))
    if supports_batch:
        group_results = []
        rollout_batch_size = max(
            1, int(getattr(args, "rollout_batch_size", candidate_count))
        )
        for batch_offset in range(0, candidate_count, rollout_batch_size):
            batch_count = min(rollout_batch_size, candidate_count - batch_offset)
            policy.reset_trace()
            rollout_start = time.perf_counter()
            batch_results = run_batched_rollouts(
                question=sample.question,
                dataset=sample.dataset,
                group_size=batch_count,
                max_rounds=args.max_rounds,
                policy=policy,
                retrieval_env=retrieval_env,
            )
            time_rollout_seconds += time.perf_counter() - rollout_start
            time_vllm_generate_seconds += float(
                getattr(policy, "timing", {}).get("time_vllm_generate_seconds", 0.0)
            )
            time_behavior_rescore_seconds += float(
                getattr(policy, "timing", {}).get("time_behavior_rescore_seconds", 0.0)
            )
            group_results.extend(
                (
                    group_index_offset + batch_offset + group_index,
                    item.result,
                    item.trace,
                )
                for group_index, item in enumerate(batch_results)
            )
    else:
        group_results = []
        for group_index in range(candidate_count):
            policy.reset_trace()
            executor = RAGLoopExecutor(policy=policy, retrieval_env=retrieval_env, max_rounds=args.max_rounds)
            rollout_start = time.perf_counter()
            result = executor.run(question=sample.question, dataset=sample.dataset)
            time_rollout_seconds += time.perf_counter() - rollout_start
            time_vllm_generate_seconds += float(
                getattr(policy, "timing", {}).get("time_vllm_generate_seconds", 0.0)
            )
            time_behavior_rescore_seconds += float(
                getattr(policy, "timing", {}).get("time_behavior_rescore_seconds", 0.0)
            )
            group_results.append(
                (group_index_offset + group_index, result, policy.trace)
            )

    rollouts: list[dict[str, Any]] = []
    for group_index, result, trace in group_results:
        rollouts.append(
            {
                "group_index": group_index,
                "question_id": sample.qid,
                "result": result,
                "trajectory": result.trajectory,
                "parse_errors": result.parse_errors,
                "final_answer": result.final_answer,
                "actions": list(trace.actions),
            }
        )
    retrieval_after = retrieval_stats() if callable(retrieval_stats) else {}
    return rollouts, {
        "time_rollout_seconds": time_rollout_seconds,
        "time_vllm_generate_seconds": time_vllm_generate_seconds,
        "time_behavior_rescore_seconds": time_behavior_rescore_seconds,
        "time_reward_seconds": 0.0,
        "time_retrieval_seconds": float(
            retrieval_after.get("time_retrieval_seconds", 0.0)
            - retrieval_before.get("time_retrieval_seconds", 0.0)
        ),
        "retrieval_cache_hits": int(
            retrieval_after.get("cache_hits", 0) - retrieval_before.get("cache_hits", 0)
        ),
        "retrieval_cache_misses": int(
            retrieval_after.get("cache_misses", 0)
            - retrieval_before.get("cache_misses", 0)
        ),
    }


def _score_rollout_candidates(
    *,
    args: Any,
    sample: RLSample,
    rollouts: list[dict[str, Any]],
) -> dict[str, Any]:
    reward_start = time.perf_counter()
    reward_sample = sample.to_reward_sample()
    for rollout in rollouts:
        rewards = compute_rl_rewards(
            rollout=rollout,
            sample=reward_sample,
            eta_query=float(getattr(args, "eta_query", 0.2)),
            eta_evidence=float(getattr(args, "eta_evidence", 0.2)),
            omega_answer=float(getattr(args, "omega_answer", 1.0)),
            omega_evidence=float(getattr(args, "omega_evidence", 1.0)),
        )
        action_credit = compute_action_rewards(
            rollout=rollout,
            sample=reward_sample,
            eta_query=float(getattr(args, "eta_query", 0.2)),
            eta_evidence=float(getattr(args, "eta_evidence", 0.2)),
            omega_answer=float(getattr(args, "omega_answer", 1.0)),
            omega_evidence=float(getattr(args, "omega_evidence", 1.0)),
            answer_local_reward_weight=float(getattr(args, "answer_local_reward_weight", 1.0)),
        )
        rollout["rewards"] = rewards
        rollout["action_rewards"] = action_credit["action_rewards"]
        rollout["terminal_reward"] = action_credit["terminal_reward"]
        if not math.isfinite(float(rewards["total"])):
            raise RuntimeError("Rollout aggregate reward must be finite.")
        if not math.isfinite(float(action_credit["terminal_reward"])):
            raise RuntimeError("Rollout terminal reward must be finite.")
    advantages = normalize_group_advantages(
        [float(item["terminal_reward"]) for item in rollouts]
    )
    for rollout, advantage in zip(rollouts, advantages):
        rollout["advantage"] = advantage
    lambda_by_agent = {
        "query_retriever": float(
            getattr(args, "lambda_query", getattr(args, "query_global_reward_weight", 1.0 / 3.0))
        ),
        "evidence_updater": float(
            getattr(args, "lambda_evidence", getattr(args, "evidence_global_reward_weight", 3.0 / 7.0))
        ),
        "answer_generator": float(
            getattr(args, "lambda_answer", getattr(args, "answer_global_reward_weight", 7.0 / 3.0))
        ),
    }
    for rollout in rollouts:
        reward_by_key = {
            (str(item["role"]), int(item["round_index"])): item
            for item in rollout["action_rewards"]
        }
        for action in rollout.get("actions", []):
            role_name = getattr(action.role, "value", str(action.role))
            credit = reward_by_key.get((role_name, int(action.round_index)))
            if credit is None:
                continue
            action.local_reward = credit["local_reward"]
            action.is_valid = bool(credit.get("is_valid", True))
            action.local_reward_valid = bool(credit.get("local_reward_valid", True))
            action.forced_termination = bool(credit.get("forced_termination", False))
        compute_decision_returns(
            rollout.get("actions", []),
            global_reward=float(rollout["terminal_reward"]),
            lambda_by_agent=lambda_by_agent,
        )
    agent_credit_stats = group_relative_normalization(
        rollouts,
        advantage_eps=float(
            getattr(args, "advantage_eps", getattr(args, "advantage_epsilon", 1.0e-8))
        ),
    )
    for rollout in rollouts:
        for action in rollout.get("actions", []):
            if not math.isfinite(float(action.advantage)):
                raise RuntimeError("Assigned action advantage must be finite.")
    return {
        "time_reward_seconds": time.perf_counter() - reward_start,
        "agent_credit_stats": agent_credit_stats,
    }


def _completion_actions(rollouts: list[dict[str, Any]]) -> list[Any]:
    return [
        action
        for rollout in rollouts
        for action in rollout.get("actions", [])
        if action.completion_ids
    ]


def _nonzero_action_fraction(
    rollouts: list[dict[str, Any]],
    *,
    field: str,
    tolerance: float = 1.0e-12,
) -> float:
    actions = _completion_actions(rollouts)
    if not actions:
        return 0.0
    return sum(
        abs(float(getattr(action, field, 0.0))) > tolerance for action in actions
    ) / len(actions)


def _group_diversity_snapshot(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reward_unique_count": len(
            {float(rollout["rewards"]["total"]) for rollout in rollouts}
        ),
        "terminal_unique_count": len(
            {float(rollout["terminal_reward"]) for rollout in rollouts}
        ),
        "primary_nonzero_action_fraction": _nonzero_action_fraction(
            rollouts,
            field="primary_advantage",
        ),
        "fallback_nonzero_action_fraction": _nonzero_action_fraction(
            rollouts,
            field="fallback_advantage",
        ),
        "final_nonzero_action_fraction": _nonzero_action_fraction(
            rollouts,
            field="advantage",
        ),
    }


def _rollout_group(
    *,
    args: Any,
    sample: RLSample,
    policy: HFSharedPolicy,
    retrieval_env: CachedLinearRAGRetrievalEnv,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rollouts, initial_timing = _generate_rollout_candidates(
        args=args,
        sample=sample,
        policy=policy,
        retrieval_env=retrieval_env,
        candidate_count=int(args.group_size),
    )
    total_timing = _empty_rollout_timing()
    _accumulate_rollout_timing(total_timing, initial_timing)
    score_timing = _score_rollout_candidates(
        args=args,
        sample=sample,
        rollouts=rollouts,
    )
    _accumulate_rollout_timing(total_timing, score_timing)
    total_timing["agent_credit_stats"] = score_timing["agent_credit_stats"]
    snapshot = _group_diversity_snapshot(rollouts)
    total_timing.update(
        {
            "effective_group_size": len(rollouts),
            "reward_unique_count": snapshot["reward_unique_count"],
            "terminal_unique_count": snapshot["terminal_unique_count"],
            "primary_nonzero_action_fraction": snapshot[
                "primary_nonzero_action_fraction"
            ],
            "fallback_nonzero_action_fraction": snapshot[
                "fallback_nonzero_action_fraction"
            ],
            "final_nonzero_action_fraction": snapshot[
                "final_nonzero_action_fraction"
            ],
        }
    )
    return rollouts, total_timing


def _rescore_behavior_logprobs(
    *,
    actions: list[Any],
    model: Any,
    torch: Any,
    device: Any,
    pad_token_id: int,
    batch_size: int,
) -> dict[str, float]:
    if not actions:
        return {
            "server_hf_logprob_mae": 0.0,
            "server_hf_logprob_max_abs": 0.0,
        }
    was_training = bool(getattr(model, "training", False))
    if callable(getattr(model, "set_adapter", None)):
        _activate_policy_adapter(model)
    if callable(getattr(model, "eval", None)):
        model.eval()
    absolute_differences: list[Any] = []
    try:
        with torch.no_grad():
            for offset in range(0, len(actions), max(1, int(batch_size))):
                batch = actions[offset : offset + max(1, int(batch_size))]
                values, mask = batched_sequence_logprobs(
                    model=model,
                    prompt_id_batches=[action.prompt_ids for action in batch],
                    completion_id_batches=[action.completion_ids for action in batch],
                    device=device,
                    pad_token_id=pad_token_id,
                )
                for row, action in enumerate(batch):
                    length = len(action.completion_ids)
                    if not bool(mask[row, :length].all().item()):
                        raise RuntimeError("Behavior completion mask is missing valid tokens.")
                    rescored = values[row, :length].detach()
                    if not bool(torch.isfinite(rescored).all().item()):
                        raise RuntimeError("HF behavior rescore produced non-finite logprobs.")
                    server = getattr(action, "server_logprobs", None)
                    if server is not None:
                        server = server.to(device=device)
                        if server.numel() != length:
                            raise ValueError("Server logprobs must align with completion tokens.")
                        absolute_differences.append((server - rescored).abs().float())
                    action.old_logprobs = rescored.cpu()
    finally:
        if callable(getattr(model, "train", None)):
            model.train(was_training)
    if not absolute_differences:
        return {
            "server_hf_logprob_mae": 0.0,
            "server_hf_logprob_max_abs": 0.0,
        }
    differences = torch.cat(absolute_differences)
    return {
        "server_hf_logprob_mae": float(differences.mean().item()),
        "server_hf_logprob_max_abs": float(differences.max().item()),
    }


def _train_on_rollouts(
    *,
    rollouts: list[dict[str, Any]],
    train_model: Any,
    raw_policy_model: Any,
    ref_model: Any,
    optimizer: Any,
    args: Any,
    torch: Any,
    device: Any,
    should_step: bool,
    scheduler: Any | None = None,
    pad_token_id: int = 0,
) -> dict[str, Any]:
    trainable_actions = [
        (rollout, action)
        for rollout in rollouts
        for action in rollout["actions"]
        if action.completion_ids
    ]
    action_count = len(trainable_actions)
    total_token_count = sum(
        sum(int(value) for value in getattr(
            action, "decision_token_mask", [1] * len(action.completion_ids)
        ))
        for _, action in trainable_actions
    )
    base_metrics = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "kl": 0.0,
        "clip_fraction": 0.0,
        "server_hf_logprob_mae": 0.0,
        "server_hf_logprob_max_abs": 0.0,
        "preupdate_logratio_mean": 0.0,
        "preupdate_logratio_max_abs": 0.0,
        "ratio_mean": 1.0,
        "ratio_p95": 1.0,
        "trainable_action_count": action_count,
        "valid_completion_token_count": total_token_count,
        "policy_forward_batch_count": 0,
        "reference_forward_batch_count": 0,
        "did_backward": False,
        "did_optimizer_step": False,
        "did_clear_gradients": False,
        "skipped_update_reason": None,
        "gradient_norm": None,
        "gradient_norm_before_clip": None,
        "gradient_norm_after_clip": None,
        "gradient_was_clipped": False,
        "gradients_finite": None,
        "time_policy_forward_seconds": 0.0,
        "time_reference_forward_seconds": 0.0,
        "time_backward_seconds": 0.0,
        "time_optimizer_step_seconds": 0.0,
    }
    if not trainable_actions or total_token_count == 0:
        base_metrics["skipped_update_reason"] = "no_trainable_actions"
        return base_metrics

    gradient_accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    if (
        bool(getattr(args, "skip_zero_advantage_updates", False))
        and gradient_accumulation_steps == 1
        and all(abs(float(action.advantage)) <= 1e-12 for _, action in trainable_actions)
    ):
        base_metrics["skipped_update_reason"] = "zero_advantage"
        return base_metrics

    microbatch_size = max(1, int(getattr(args, "per_device_train_batch_size", 1)))
    reference_batch_size = max(
        1,
        int(getattr(args, "reference_per_device_batch_size", microbatch_size)),
    )
    behavior_diagnostics = _rescore_behavior_logprobs(
        actions=[action for _, action in trainable_actions],
        model=raw_policy_model,
        torch=torch,
        device=device,
        pad_token_id=pad_token_id,
        batch_size=microbatch_size,
    )
    loss_total = 0.0
    policy_loss_total = 0.0
    kl_total = 0.0
    clip_fraction_total = 0.0
    preupdate_logratio_mean_total = 0.0
    preupdate_logratio_max_abs = 0.0
    ratio_mean_total = 0.0
    ratio_p95_total = 0.0
    time_policy_forward_seconds = 0.0
    time_reference_forward_seconds = 0.0
    time_backward_seconds = 0.0
    time_optimizer_step_seconds = 0.0
    did_optimizer_step = False
    did_clear_gradients = False
    reference_forward_batch_count = 0
    reference_by_action: list[Any] = []
    reference_forward_start = time.perf_counter()
    with _reference_adapter_context(ref_model, raw_policy_model) as active_reference_model:
        with torch.no_grad():
            for offset in range(0, len(trainable_actions), reference_batch_size):
                reference_actions = [
                    action
                    for _, action in trainable_actions[offset : offset + reference_batch_size]
                ]
                reference_batch, reference_mask = batched_sequence_logprobs(
                    model=active_reference_model,
                    prompt_id_batches=[action.prompt_ids for action in reference_actions],
                    completion_id_batches=[action.completion_ids for action in reference_actions],
                    device=device,
                    pad_token_id=pad_token_id,
                )
                reference_forward_batch_count += 1
                for row, action in enumerate(reference_actions):
                    completion_length = len(action.completion_ids)
                    if not bool(reference_mask[row, :completion_length].all().item()):
                        raise RuntimeError("Reference completion mask is missing valid tokens.")
                    reference_by_action.append(reference_batch[row, :completion_length].detach())
    time_reference_forward_seconds += time.perf_counter() - reference_forward_start

    policy_mode_target = (
        train_model
        if callable(getattr(train_model, "train", None))
        else raw_policy_model
    )
    _set_deterministic_policy_train_mode(policy_mode_target, torch=torch)

    policy_forward_batch_count = 0
    for offset in range(0, len(trainable_actions), microbatch_size):
        microbatch = trainable_actions[offset : offset + microbatch_size]
        actions = [action for _, action in microbatch]
        prompt_id_batches = [action.prompt_ids for action in actions]
        completion_id_batches = [action.completion_ids for action in actions]
        advantage = torch.tensor(
            [float(action.advantage) for action in actions],
            dtype=torch.float32,
            device=device,
        )
        policy_forward_start = time.perf_counter()
        current, mask = batched_sequence_logprobs(
            model=train_model,
            prompt_id_batches=prompt_id_batches,
            completion_id_batches=completion_id_batches,
            device=device,
            pad_token_id=pad_token_id,
        )
        # ``batched_sequence_logprobs`` already excludes prompts. Intersect its
        # padding mask with the persisted per-decision output-token mask.
        for row, action in enumerate(actions):
            decision_mask = torch.tensor(
                getattr(action, "decision_token_mask", [1] * len(action.completion_ids)),
                dtype=torch.bool,
                device=device,
            )
            if decision_mask.numel() != len(action.completion_ids):
                raise ValueError("Decision token mask must align with completion tokens.")
            mask[row, : decision_mask.numel()] &= decision_mask
        if not bool(mask.any().item()):
            continue
        policy_forward_batch_count += 1
        time_policy_forward_seconds += time.perf_counter() - policy_forward_start
        reference = torch.zeros_like(current)
        old = torch.zeros_like(current)
        for row, action in enumerate(actions):
            reference_row = reference_by_action[offset + row]
            reference[row, : reference_row.numel()] = reference_row
            action_old = action.old_logprobs.to(device=device)
            if action_old.numel() != len(action.completion_ids):
                raise ValueError(
                    "Stored behavior logprobs must align with the action completion tokens."
                )
            old[row, : action_old.numel()] = action_old
        loss, metrics = compute_grpo_loss(
            current_logprobs=current,
            old_logprobs=old,
            ref_logprobs=reference,
            action_mask=mask,
            advantages=advantage,
            clip_epsilon=args.clip_epsilon,
            kl_beta=args.kl_beta,
        )
        # Aggregate microbatches as one masked token mean, independent of
        # variable decision lengths and padding.
        action_weight = int(mask.sum().item()) / total_token_count
        loss_total += metrics["loss"] * action_weight
        policy_loss_total += metrics["policy_loss"] * action_weight
        kl_total += metrics["kl"] * action_weight
        clip_fraction_total += metrics["clip_fraction"] * action_weight
        preupdate_logratio_mean_total += metrics.get("preupdate_logratio_mean", 0.0) * action_weight
        preupdate_logratio_max_abs = max(
            preupdate_logratio_max_abs,
            metrics.get("preupdate_logratio_max_abs", 0.0),
        )
        ratio_mean_total += metrics.get("ratio_mean", 1.0) * action_weight
        ratio_p95_total += metrics.get("ratio_p95", 1.0) * action_weight
        backward_start = time.perf_counter()
        (loss * action_weight / gradient_accumulation_steps).backward()
        time_backward_seconds += time.perf_counter() - backward_start
    gradient_norm, gradients_finite = _gradient_health(raw_policy_model, torch=torch)
    gradient_norm_after_clip = gradient_norm
    if should_step and gradients_finite is False:
        optimizer.zero_grad(set_to_none=True)
        did_clear_gradients = True
    elif should_step:
        parameter_source = getattr(raw_policy_model, "parameters", None)
        trainable_parameters = [
            parameter
            for parameter in (parameter_source() if callable(parameter_source) else [])
            if parameter.requires_grad and parameter.grad is not None
        ]
        max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
        gradient_norm_after_clip, _ = _gradient_health(raw_policy_model, torch=torch)
        optimizer_start = time.perf_counter()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        did_optimizer_step = True
        time_optimizer_step_seconds += time.perf_counter() - optimizer_start
    return {
        "loss": loss_total,
        "policy_loss": policy_loss_total,
        "kl": kl_total,
        "clip_fraction": clip_fraction_total,
        **behavior_diagnostics,
        "preupdate_logratio_mean": preupdate_logratio_mean_total,
        "preupdate_logratio_max_abs": preupdate_logratio_max_abs,
        "ratio_mean": ratio_mean_total,
        "ratio_p95": ratio_p95_total,
        "trainable_action_count": action_count,
        "valid_completion_token_count": total_token_count,
        "policy_forward_batch_count": policy_forward_batch_count,
        "reference_forward_batch_count": reference_forward_batch_count,
        "did_backward": True,
        "did_optimizer_step": did_optimizer_step,
        "did_clear_gradients": did_clear_gradients,
        "skipped_update_reason": (
            "nonfinite_gradients" if gradients_finite is False else None
        ),
        "gradient_norm": gradient_norm,
        "gradient_norm_before_clip": gradient_norm,
        "gradient_norm_after_clip": gradient_norm_after_clip,
        "gradient_was_clipped": bool(
            gradient_norm is not None and gradient_norm > float(getattr(args, "max_grad_norm", 1.0))
        ),
        "gradients_finite": gradients_finite,
        "time_policy_forward_seconds": time_policy_forward_seconds,
        "time_reference_forward_seconds": time_reference_forward_seconds,
        "time_backward_seconds": time_backward_seconds,
        "time_optimizer_step_seconds": time_optimizer_step_seconds,
    }


def _save_policy_adapter(raw_policy_model: Any, output_dir: Path) -> None:
    if callable(getattr(raw_policy_model, "set_adapter", None)):
        _activate_policy_adapter(raw_policy_model)
    raw_policy_model.save_pretrained(
        output_dir,
        selected_adapters=[_POLICY_ADAPTER_NAME],
    )


def _save_checkpoint(raw_policy_model: Any, tokenizer: Any, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _save_policy_adapter(raw_policy_model, checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def _checkpoint_save_decision(
    *,
    global_step: int,
    save_steps: int,
    optimizer_safe_boundary: bool,
    pending: bool,
) -> tuple[bool, bool]:
    due = bool(pending) or (
        int(save_steps) > 0 and int(global_step) % int(save_steps) == 0
    )
    should_save = due and bool(optimizer_safe_boundary)
    return should_save, False if should_save else due


def _pending_gradients_after_sample(
    *,
    has_pending_gradients: bool,
    metrics: dict[str, Any],
) -> bool:
    pending = bool(has_pending_gradients)
    if metrics.get("did_backward"):
        pending = True
    if metrics.get("did_optimizer_step") or metrics.get("did_clear_gradients"):
        pending = False
    return pending


def _run_final_checkpoint_actions(
    *,
    has_pending_gradients: bool,
    checkpoint_pending: bool,
    flush_pending_gradients: Any,
    save_pending_checkpoint: Any,
) -> None:
    if has_pending_gradients:
        flush_pending_gradients()
    if checkpoint_pending:
        save_pending_checkpoint()


def _safe_dataset_name(dataset: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(dataset).strip()) or "unknown"


def _dataset_rollout_path(output_dir: Path, dataset: str) -> Path:
    """按数据集拆分 rollout 样本，避免所有数据集混在一个 JSONL。"""
    return output_dir / "rollout_samples" / f"{_safe_dataset_name(dataset)}.jsonl"


def _json_serializable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_serializable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_serializable_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_serializable_value(item) for item in sorted(value, key=str)]
    return str(value)


def _resolved_args_payload(args: Any) -> dict[str, Any]:
    """Return the effective CLI/config values in a stable JSON representation."""
    return {
        key: _json_serializable_value(value)
        for key, value in sorted(vars(args).items())
    }


def _group_diversity_contract(args: Any) -> dict[str, Any]:
    return {
        "group_size": int(args.group_size),
        "degenerate_bucket_fallback_weight": float(
            args.degenerate_bucket_fallback_weight
        ),
    }


def _build_train_metrics_payload(
    *,
    epoch: int,
    sample_index: int,
    sample_total: int,
    sample: Any,
    global_step: int,
    metrics: dict[str, Any],
    rollouts: list[dict[str, Any]],
    best_rollout: dict[str, Any],
    learning_rate: float,
    rollout_timing: dict[str, Any],
    time_initial_weight_sync_seconds: float,
    time_weight_sync_seconds: float,
    time_total_seconds: float,
    successful_optimizer_updates: int = 0,
) -> dict[str, Any]:
    reward_totals = [float(item["terminal_reward"]) for item in rollouts]
    monitor_reward_totals = [float(item["rewards"]["total"]) for item in rollouts]
    action_advantages = [
        float(action.advantage)
        for rollout in rollouts
        for action in rollout.get("actions", [])
    ]
    rollout_advantages = [float(item["advantage"]) for item in rollouts]
    reward_mean = sum(reward_totals) / len(reward_totals)
    actions_by_role = {
        role: [
            action for rollout in rollouts for action in rollout.get("actions", [])
            if getattr(
                getattr(action, "role", None),
                "value",
                str(getattr(action, "role", "")),
            ) == role
        ]
        for role in ("query_retriever", "evidence_updater", "answer_generator")
    }

    def role_local_mean(role: str) -> float:
        values = [
            float(action.local_reward) for action in actions_by_role[role]
            if getattr(action, "local_reward", None) is not None
        ]
        return sum(values) / len(values) if values else 0.0

    def role_advantage_stats(role: str) -> tuple[float, float]:
        values = [float(action.advantage) for action in actions_by_role[role]]
        return (
            sum(values) / len(values) if values else 0.0,
            statistics.pstdev(values) if len(values) > 1 else 0.0,
        )

    behavior = {}
    for role in actions_by_role:
        actions = actions_by_role[role]
        behavior[role] = {
            "invalid_rate": (
                sum(not bool(getattr(action, "is_valid", True)) for action in actions) / len(actions)
                if actions else 0.0
            )
        }
    query_adv = role_advantage_stats("query_retriever")
    evidence_adv = role_advantage_stats("evidence_updater")
    answer_adv = role_advantage_stats("answer_generator")
    avg_rounds = sum(len(item.get("trajectory", [])) for item in rollouts) / len(rollouts)
    policy_stops = sum(
        any(
            bool((turn.get("answer") or {}).get("can_answer"))
            and not bool(turn.get("force_final_answer", False))
            for turn in rollout.get("trajectory", [])
        )
        for rollout in rollouts
    )
    forced_stops = sum(
        bool(
            rollout.get("trajectory")
            and rollout["trajectory"][-1].get("force_final_answer", False)
        )
        for rollout in rollouts
    )
    selection_counts = [
        len((turn.get("update_evidence") or {}).get("selected_passage_ids") or [])
        for rollout in rollouts for turn in rollout.get("trajectory", [])
    ]
    mean_answer_f1 = sum(
        float(item["rewards"].get("answer_f1", 0.0)) for item in rollouts
    ) / len(rollouts)
    mean_coverage = sum(
        float(item["rewards"].get("evidence_coverage", item["rewards"].get("support_coverage", 0.0)))
        for item in rollouts
    ) / len(rollouts)
    return {
        "epoch": epoch,
        "sample": sample_index + 1,
        "sample_total": sample_total,
        "qid": sample.qid,
        "dataset": sample.dataset,
        "step": global_step,
        "loss": metrics["loss"],
        "policy_loss": metrics["policy_loss"],
        "kl": metrics["kl"],
        "clip_fraction": metrics.get("clip_fraction", 0.0),
        "trainable_action_count": metrics.get("trainable_action_count", 0),
        "valid_completion_token_count": metrics.get("valid_completion_token_count", 0),
        "policy_forward_batch_count": metrics.get("policy_forward_batch_count", 0),
        "reference_forward_batch_count": metrics.get("reference_forward_batch_count", 0),
        "did_optimizer_step": bool(metrics.get("did_optimizer_step", False)),
        "successful_optimizer_updates": int(successful_optimizer_updates),
        "skipped_update_reason": metrics.get("skipped_update_reason"),
        "gradient_norm": metrics.get("gradient_norm"),
        "gradient_norm_before_clip": metrics.get("gradient_norm_before_clip"),
        "gradient_norm_after_clip": metrics.get("gradient_norm_after_clip"),
        "gradient_was_clipped": bool(metrics.get("gradient_was_clipped", False)),
        "gradients_finite": metrics.get("gradients_finite"),
        "server_hf_logprob_mae": metrics.get("server_hf_logprob_mae", 0.0),
        "server_hf_logprob_max_abs": metrics.get("server_hf_logprob_max_abs", 0.0),
        "preupdate_logratio_mean": metrics.get("preupdate_logratio_mean", 0.0),
        "preupdate_logratio_max_abs": metrics.get("preupdate_logratio_max_abs", 0.0),
        "ratio_mean": metrics.get("ratio_mean", 1.0),
        "ratio_p95": metrics.get("ratio_p95", 1.0),
        "total_loss": metrics.get("total_loss", metrics["loss"]),
        "kl_loss": metrics.get("kl_loss", metrics["kl"]),
        "mean_ratio": metrics.get("mean_ratio", metrics.get("ratio_mean", 1.0)),
        "approx_kl": metrics.get("approx_kl", 0.0),
        "reward_total": reward_mean,
        "reward_group": {
            "min": min(reward_totals),
            "max": max(reward_totals),
            "mean": reward_mean,
            "std": statistics.pstdev(reward_totals) if len(reward_totals) > 1 else 0.0,
        },
        "monitor_reward_total": sum(monitor_reward_totals) / len(monitor_reward_totals),
        "monitor_reward_group": {
            "min": min(monitor_reward_totals),
            "max": max(monitor_reward_totals),
            "mean": sum(monitor_reward_totals) / len(monitor_reward_totals),
            "std": (
                statistics.pstdev(monitor_reward_totals)
                if len(monitor_reward_totals) > 1
                else 0.0
            ),
        },
        "reward_query": best_rollout["rewards"]["query_reward"],
        "reward_evidence": best_rollout["rewards"]["evidence_reward"],
        "reward_answer_f1": best_rollout["rewards"]["answer_f1"],
        "reward/query": role_local_mean("query_retriever"),
        "reward/evidence": role_local_mean("evidence_updater"),
        "reward/answer": role_local_mean("answer_generator"),
        "reward/global": reward_mean,
        "reward/answer_f1": mean_answer_f1,
        "reward/evidence_coverage": mean_coverage,
        "advantage/query_mean": query_adv[0],
        "advantage/query_std": query_adv[1],
        "advantage/evidence_mean": evidence_adv[0],
        "advantage/evidence_std": evidence_adv[1],
        "advantage/answer_mean": answer_adv[0],
        "advantage/answer_std": answer_adv[1],
        "behavior/avg_rounds": avg_rounds,
        "behavior/stop_rate": policy_stops / len(rollouts),
        "behavior/forced_stop_rate": forced_stops / len(rollouts),
        "behavior/invalid_query_rate": behavior["query_retriever"]["invalid_rate"],
        "behavior/invalid_evidence_rate": behavior["evidence_updater"]["invalid_rate"],
        "behavior/invalid_answer_rate": behavior["answer_generator"]["invalid_rate"],
        "behavior/avg_selected_passages": (
            sum(selection_counts) / len(selection_counts) if selection_counts else 0.0
        ),
        "performance/final_answer_f1": mean_answer_f1,
        "performance/evidence_coverage": mean_coverage,
        "advantage_mean": sum(item["advantage"] for item in rollouts) / len(rollouts),
        "action_advantage_mean": (
            sum(action_advantages) / len(action_advantages)
            if action_advantages
            else 0.0
        ),
        "rollout_advantage_std": (
            statistics.pstdev(rollout_advantages) if len(rollout_advantages) > 1 else 0.0
        ),
        "action_advantage_std": (
            statistics.pstdev(action_advantages) if len(action_advantages) > 1 else 0.0
        ),
        "agent_credit_stats": rollout_timing.get("agent_credit_stats", {}),
        "effective_group_size": rollout_timing.get("effective_group_size", len(rollouts)),
        "reward_unique_count": rollout_timing.get(
            "reward_unique_count",
            len(set(reward_totals)),
        ),
        "terminal_unique_count": rollout_timing.get(
            "terminal_unique_count",
            len({float(item.get("terminal_reward", 0.0)) for item in rollouts}),
        ),
        "primary_nonzero_action_fraction": rollout_timing.get(
            "primary_nonzero_action_fraction",
            0.0,
        ),
        "fallback_nonzero_action_fraction": rollout_timing.get(
            "fallback_nonzero_action_fraction",
            0.0,
        ),
        "final_nonzero_action_fraction": rollout_timing.get(
            "final_nonzero_action_fraction",
            0.0,
        ),
        "gold_answer": sample.answer,
        "generated_answer": best_rollout["final_answer"],
        "retrieval_count": len(best_rollout["trajectory"]),
        "parse_errors": best_rollout["parse_errors"],
        "protocol_metrics": rollout_timing.get("protocol_metrics", {}),
        "learning_rate": learning_rate,
        "timing": {
            "rollout_seconds": rollout_timing["time_rollout_seconds"],
            "vllm_generate_seconds": rollout_timing.get("time_vllm_generate_seconds", 0.0),
            "behavior_rescore_seconds": rollout_timing.get(
                "time_behavior_rescore_seconds",
                0.0,
            ),
            "retrieval_seconds": rollout_timing.get("time_retrieval_seconds", 0.0),
            "retrieval_cache_hits": rollout_timing.get("retrieval_cache_hits", 0),
            "retrieval_cache_misses": rollout_timing.get("retrieval_cache_misses", 0),
            "reward_seconds": rollout_timing["time_reward_seconds"],
            "policy_forward_seconds": metrics.get("time_policy_forward_seconds", 0.0),
            "reference_forward_seconds": metrics.get(
                "time_reference_forward_seconds",
                0.0,
            ),
            "backward_seconds": metrics["time_backward_seconds"],
            "optimizer_step_seconds": metrics["time_optimizer_step_seconds"],
            "initial_weight_sync_seconds": time_initial_weight_sync_seconds,
            "weight_sync_seconds": time_weight_sync_seconds,
            "total_seconds": time_total_seconds,
        },
    }


def _action_credit_payload(action: Any) -> dict[str, Any]:
    return {
        "role": getattr(action.role, "value", str(action.role)),
        "round_index": action.round_index,
        "local_reward": action.local_reward,
        "terminal_reward": action.terminal_reward,
        "decision_return": getattr(action, "decision_return", 0.0),
        "primary_advantage": float(
            getattr(action, "primary_advantage", action.advantage)
        ),
        "fallback_advantage": float(getattr(action, "fallback_advantage", 0.0)),
        "advantage": action.advantage,
    }


def _rollout_log_payload(
    *,
    epoch: int,
    sample_index: int,
    sample: Any,
    rollouts: list[dict[str, Any]],
    best_rollout: dict[str, Any],
    log_all_group_rollouts: bool,
) -> dict[str, Any]:
    payload = {
        "epoch": epoch,
        "sample": sample_index + 1,
        "qid": sample.qid,
        "dataset": sample.dataset,
        "question": sample.question,
        "gold_answer": sample.answer,
        "best_reward": best_rollout["rewards"],
        "terminal_reward": best_rollout["terminal_reward"],
        "action_credit": [
            _action_credit_payload(action) for action in best_rollout["actions"]
        ],
        "trajectory": best_rollout["trajectory"],
    }
    if log_all_group_rollouts:
        payload["group_rollouts"] = [
            {
                "group_index": rollout["group_index"],
                "rewards": rollout["rewards"],
                "terminal_reward": rollout["terminal_reward"],
                "parse_errors": rollout["parse_errors"],
                "final_answer": rollout["final_answer"],
                "generated_action_count": len(rollout["actions"]),
                "action_credit": [
                    _action_credit_payload(action) for action in rollout["actions"]
                ],
                "trajectory": rollout["trajectory"],
            }
            for rollout in rollouts
        ]
    return payload


def _print_debug_rollout(sample: RLSample, rollout: dict[str, Any]) -> None:
    """Human-readable Section 3.3 audit trace; enabled only by config."""

    credit = {
        (item["role"], int(item["round_index"])): item
        for item in rollout.get("action_rewards", [])
    }
    actions = {
        (getattr(action.role, "value", str(action.role)), int(action.round_index)): action
        for action in rollout.get("actions", [])
    }
    print(f"\n[debug_rollout] Question: {sample.question}")
    for fallback_round, turn in enumerate(rollout.get("trajectory", [])):
        round_index = int(turn.get("round", fallback_round))
        print(f"Round {round_index + 1}")
        for role, label in (
            ("query_retriever", "Query"),
            ("evidence_updater", "Evidence Selection"),
            ("answer_generator", "Answer Decision"),
        ):
            action = actions.get((role, round_index))
            reward = credit.get((role, round_index), {})
            if action is not None:
                print(f"  {label}: {action.response}")
                print(
                    "  local_reward="
                    f"{reward.get('local_reward')} return={action.decision_return:.6f} "
                    f"advantage={action.advantage:.6f} valid={getattr(action, 'is_valid', True)}"
                )
        passages = (turn.get("observation") or {}).get("passages") or []
        selected = (turn.get("update_evidence") or {}).get("selected_passage_ids") or []
        print(f"  retrieved={len(passages)} selected={selected}")
    rewards = rollout.get("rewards", {})
    print(f"Final Answer: {rollout.get('final_answer')}")
    print(
        f"Answer F1={rewards.get('answer_f1', 0.0):.6f} "
        f"Evidence Coverage={rewards.get('evidence_coverage', 0.0):.6f} "
        f"Global Reward={rollout.get('terminal_reward', 0.0):.6f}\n"
    )


def _protocol_warning_event(
    *,
    step: int,
    protocol_status: dict[str, Any],
) -> dict[str, Any] | None:
    if not protocol_status.get("should_warn"):
        return None
    return {
        "event": "protocol_warning",
        "step": step,
        "protocol_metrics": protocol_status,
    }


def _make_progress_bar(args: Any, total: int) -> Any:
    if not _is_main_process() or args.disable_tqdm:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(
        total=total,
        desc="rl train samples",
        unit="sample",
        dynamic_ncols=True,
        leave=True,
    )


def main() -> None:
    args = parse_args()
    _validate_online_policy_sync(args)
    _validate_paper_credit_config(args)
    sft_prompt_metadata = _validate_sft_prompt_contract(args)
    active_prompt_contract = load_prompt_contract(args.prompt_config_path)
    checkpoint_prompt_metadata = {
        "prompt_contract_version": active_prompt_contract.version,
        "prompt_contract_fingerprint": active_prompt_contract.fingerprint,
        "max_rounds": int(args.max_rounds),
        "retrieval_top_k": int(args.retrieval_top_k),
        "sft_adapter_path": str(args.sft_adapter_path),
    }
    _validate_vllm_gpu_placement(args)
    _configure_visible_gpus(args)
    deps = _load_training_dependencies()
    torch = deps["torch"]
    _setup_distributed(torch)
    _validate_acceleration_runtime(args, torch)
    device = _device(torch)
    random.seed(args.seed + _local_rank())
    np.random.seed(args.seed + _local_rank())
    torch.manual_seed(args.seed + _local_rank())

    samples, data_summary = load_rl_samples(
        data_root=args.rl_data_root,
        data_files=list(args.rl_data_files or []),
        max_samples=args.max_samples,
        data_sampling_strategy=args.data_sampling_strategy,
        data_sampling_seed=args.data_sampling_seed,
    )
    loaded_sample_count = len(samples)
    samples = select_balanced_samples(
        samples,
        max_total_samples=args.max_total_samples,
        seed=args.seed,
    )
    data_summary["loaded_samples_before_total_limit"] = loaded_sample_count
    data_summary["loaded_samples"] = len(samples)
    data_summary["max_total_samples"] = args.max_total_samples
    data_summary["counts_by_dataset"] = {
        dataset: sum(sample.dataset == dataset for sample in samples)
        for dataset in sorted({sample.dataset for sample in samples})
    }
    dataset_fingerprint = fingerprint_dataset(samples)
    config_fingerprint = fingerprint_config(args)
    total_epochs = max(1, int(math.ceil(args.num_train_epochs)))
    resume_state = _resolve_resume_state(
        args,
        rank_epoch_size=len(_rank_samples(samples)),
        total_epochs=total_epochs,
    )
    if resume_state.is_full_checkpoint:
        try:
            validate_full_checkpoint_identity(
                resume_state.checkpoint_path,
                expected_dataset_fingerprint=dataset_fingerprint,
                expected_config_fingerprint=config_fingerprint,
                expected_gradient_accumulation_steps=int(args.gradient_accumulation_steps),
                expected_world_size=_world_size(),
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    if _is_main_process():
        print(f"Loaded {len(samples)} RL samples from {args.rl_data_root}")
        print(f"Counts by dataset: {data_summary['counts_by_dataset']}")
        if resume_state.checkpoint_path is not None:
            suffix = (
                "full optimizer and RNG state will be restored."
                if resume_state.is_full_checkpoint
                else "optimizer state will be reinitialized (legacy warm start)."
            )
            print(
                "Resuming policy from "
                f"{resume_state.checkpoint_path} at epoch {resume_state.epoch}, "
                f"after {resume_state.samples_consumed} samples, global step {resume_state.global_step}; "
                f"{suffix}"
            )
    if args.check_only:
        if _is_main_process():
            print("Check-only complete. No model training started.")
        _cleanup_distributed(torch)
        return

    tokenizer, raw_policy_model, ref_model = _load_policy_and_reference(args, deps, device)
    train_model = _wrap_ddp(raw_policy_model, torch)
    raw_policy_model.train()
    _activate_policy_adapter(raw_policy_model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in raw_policy_model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    training_step_horizon = (
        int(args.max_steps)
        if int(args.max_steps) > 0
        else len(_rank_samples(samples)) * total_epochs
    )
    scheduler_total_updates = max(
        1,
        math.ceil(training_step_horizon / max(1, int(args.gradient_accumulation_steps))),
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        total_updates=scheduler_total_updates,
        warmup_ratio=float(args.warmup_ratio),
        min_lr_ratio=float(args.min_lr_ratio),
    )
    scheduler_warmup_updates = int(math.ceil(scheduler_total_updates * float(args.warmup_ratio)))
    successful_optimizer_updates = 0
    optimization_contract = {
        "answer_f1_contract": ANSWER_F1_CONTRACT,
        "answer_local_reward_weight": args.answer_local_reward_weight,
        "max_grad_norm": float(args.max_grad_norm),
        "logprob_source": "hf_rescore_v1",
        "load_4bit": bool(args.load_4bit),
        "bf16": bool(args.bf16),
        "vllm_dtype": str(args.vllm_dtype),
        "vllm_sync_mode": str(args.vllm_sync_mode),
        "vllm_sync_every_steps": int(args.vllm_sync_every_steps),
        "group_diversity": _group_diversity_contract(args),
    }
    retrieval_env = _build_retrieval_env(args)
    retrieval_env.prewarm(sorted({sample.dataset for sample in samples}))
    policy = _build_policy(
        args,
        raw_policy_model,
        tokenizer,
        generation_counter=resume_state.generation_counter,
    )
    time_initial_weight_sync_seconds = _sync_vllm_before_first_rollout(
        policy,
        raw_policy_model,
        args,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if resume_state.is_full_checkpoint:
        try:
            restored_state = restore_full_training_state(
                resume_state.checkpoint_path,
                optimizer=optimizer,
                scheduler=scheduler,
                torch_module=torch,
                expected_dataset_fingerprint=dataset_fingerprint,
                expected_config_fingerprint=config_fingerprint,
                expected_gradient_accumulation_steps=int(args.gradient_accumulation_steps),
                expected_world_size=_world_size(),
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if int(restored_state["generation_counter"]) != int(
            getattr(policy, "generation_counter", 0)
        ):
            raise SystemExit("Restored vLLM generation counter does not match the policy state.")
        resume_state = replace(
            resume_state,
            optimizer_state_restored=True,
            rng_state_restored=True,
        )
        successful_optimizer_updates = int(
            restored_state.get("successful_optimizer_updates", 0)
        )

    base_output_dir = Path(args.output_root)
    output_dir = make_timestamped_run_dir(base_output_dir)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        shared_output = [str(output_dir) if _is_main_process() else None]
        torch.distributed.broadcast_object_list(shared_output, src=0)
        output_dir = Path(shared_output[0])
    log_path = output_dir / "train_metrics.jsonl"
    rollout_dir = output_dir / "rollout_samples"
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "rl_dataset_summary.json", data_summary)
        _write_json(output_dir / "resume_meta.json", _resume_meta_payload(resume_state))
        print(f"Run output directory: {output_dir}")

    global_step = resume_state.global_step
    progress_total = sum(
        len(
            _rank_samples_for_epoch(
                samples,
                seed=args.seed,
                epoch=epoch,
                resume_state=resume_state,
            )
        )
        for epoch in range(resume_state.epoch, total_epochs + 1)
    )
    run_step_limit = _run_step_limit(args)
    if run_step_limit > 0:
        progress_total = min(progress_total, max(0, run_step_limit - global_step))
    progress_bar = _make_progress_bar(args, progress_total)
    checkpoint_pending = False
    has_pending_gradients = False
    protocol_monitor = ProtocolWindowMonitor(
        window_size=100,
        max_parse_failure_rate=0.02,
        bad_windows_to_warn=2,
    )
    latest_protocol_status: dict[str, Any] = {"checkpoint_eligible": False}
    last_epoch = resume_state.epoch
    last_samples_consumed = resume_state.samples_consumed
    try:
        for epoch in range(resume_state.epoch, total_epochs + 1):
            rank_samples = _rank_samples_for_epoch(
                samples,
                seed=args.seed,
                epoch=epoch,
                resume_state=resume_state,
            )
            consumed_before_epoch = (
                resume_state.samples_consumed
                if resume_state.checkpoint_path is not None and epoch == resume_state.epoch
                else 0
            )
            for consumed_offset, (sample_index, sample) in enumerate(rank_samples, start=1):
                if run_step_limit > 0 and global_step >= run_step_limit:
                    break
                sample_start_time = time.perf_counter()
                rollouts, rollout_timing = _rollout_group(
                    args=args,
                    sample=sample,
                    policy=policy,
                    retrieval_env=retrieval_env,
                )
                if (
                    _is_main_process()
                    and bool(getattr(args, "debug_rollout", False))
                    and global_step < int(getattr(args, "debug_rollout_samples", 2))
                ):
                    _print_debug_rollout(sample, random.choice(rollouts))
                rollout_timing["protocol_metrics"] = compute_protocol_metrics(rollouts)
                latest_protocol_status = protocol_monitor.add(rollouts)
                protocol_warning = _protocol_warning_event(
                    step=global_step,
                    protocol_status=latest_protocol_status,
                )
                if protocol_warning is not None and _is_main_process():
                    _append_jsonl(log_path, protocol_warning)
                should_step = (
                    (global_step + 1)
                    % max(1, int(args.gradient_accumulation_steps))
                    == 0
                )
                metrics = _train_on_rollouts(
                    rollouts=rollouts,
                    train_model=train_model,
                    raw_policy_model=raw_policy_model,
                    ref_model=ref_model,
                    optimizer=optimizer,
                    args=args,
                    torch=torch,
                    device=device,
                    should_step=should_step,
                    scheduler=scheduler,
                    pad_token_id=int(tokenizer.pad_token_id or 0),
                )
                has_pending_gradients = _pending_gradients_after_sample(
                    has_pending_gradients=has_pending_gradients,
                    metrics=metrics,
                )
                time_weight_sync_seconds = 0.0
                if metrics.get("did_optimizer_step"):
                    successful_optimizer_updates += 1
                    time_weight_sync_seconds = _sync_vllm_after_optimizer_step(
                        policy,
                        raw_policy_model,
                        args,
                        completed_step=global_step + 1,
                    )
                time_total_seconds = time.perf_counter() - sample_start_time
                global_step += 1
                last_epoch = epoch
                last_samples_consumed = consumed_before_epoch + consumed_offset
                reward_totals = [float(item["terminal_reward"]) for item in rollouts]
                best_rollout = max(
                    rollouts,
                    key=lambda item: (
                        float(item["terminal_reward"]),
                        float(item["rewards"]["total"]),
                    ),
                )
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        {
                            "epoch": epoch,
                            "loss": f"{metrics['loss']:.4f}",
                            "reward": f"{sum(reward_totals) / len(reward_totals):.3f}",
                            "kl": f"{metrics['kl']:.4f}",
                        }
                    )
                    progress_bar.update(1)
                if _is_main_process() and (global_step % args.logging_steps == 0):
                    payload = _build_train_metrics_payload(
                        epoch=epoch,
                        sample_index=sample_index,
                        sample_total=len(samples),
                        sample=sample,
                        global_step=global_step,
                        metrics=metrics,
                        rollouts=rollouts,
                        best_rollout=best_rollout,
                        learning_rate=float(optimizer.param_groups[0]["lr"]),
                        rollout_timing=rollout_timing,
                        time_initial_weight_sync_seconds=time_initial_weight_sync_seconds,
                        time_weight_sync_seconds=time_weight_sync_seconds,
                        time_total_seconds=time_total_seconds,
                        successful_optimizer_updates=successful_optimizer_updates,
                    )
                    _append_jsonl(log_path, payload)
                    _append_jsonl(
                        _dataset_rollout_path(output_dir, sample.dataset),
                        _rollout_log_payload(
                            epoch=epoch,
                            sample_index=sample_index,
                            sample=sample,
                            rollouts=rollouts,
                            best_rollout=best_rollout,
                            log_all_group_rollouts=bool(args.log_all_group_rollouts),
                        ),
                    )
                should_save, checkpoint_pending = _checkpoint_save_decision(
                    global_step=global_step,
                    save_steps=int(args.save_steps),
                    optimizer_safe_boundary=not has_pending_gradients,
                    pending=checkpoint_pending,
                )
                if should_save:
                    save_full_checkpoint(
                        raw_policy_model=raw_policy_model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        successful_optimizer_updates=successful_optimizer_updates,
                        scheduler_total_updates=scheduler_total_updates,
                        scheduler_warmup_updates=scheduler_warmup_updates,
                        optimization_contract=optimization_contract,
                        prompt_contract_metadata=checkpoint_prompt_metadata,
                        output_dir=output_dir,
                        step=global_step,
                        epoch=epoch,
                        samples_consumed=last_samples_consumed,
                        generation_counter=int(getattr(policy, "generation_counter", 0)),
                        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
                        dataset_fingerprint=dataset_fingerprint,
                        config_fingerprint=config_fingerprint,
                        torch_module=torch,
                        save_total_limit=int(args.save_total_limit),
                        milestone_steps=int(args.save_milestone_steps),
                    )
            if run_step_limit > 0 and global_step >= run_step_limit:
                break
    finally:
        if progress_bar is not None:
            progress_bar.close()

    def flush_pending_gradients() -> None:
        nonlocal successful_optimizer_updates
        stepped = _flush_pending_gradients_if_finite(
            raw_policy_model=raw_policy_model,
            optimizer=optimizer,
            torch=torch,
            scheduler=scheduler,
            max_grad_norm=float(args.max_grad_norm),
            sync_weights=lambda: _sync_vllm_after_optimizer_step(
                policy,
                raw_policy_model,
                args,
                completed_step=global_step,
            ),
        )
        if stepped:
            successful_optimizer_updates += 1

    def save_pending_checkpoint() -> None:
        save_full_checkpoint(
            raw_policy_model=raw_policy_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            successful_optimizer_updates=successful_optimizer_updates,
            scheduler_total_updates=scheduler_total_updates,
            scheduler_warmup_updates=scheduler_warmup_updates,
            optimization_contract=optimization_contract,
            prompt_contract_metadata=checkpoint_prompt_metadata,
            output_dir=output_dir,
            step=global_step,
            epoch=last_epoch,
            samples_consumed=last_samples_consumed,
            generation_counter=int(getattr(policy, "generation_counter", 0)),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            dataset_fingerprint=dataset_fingerprint,
            config_fingerprint=config_fingerprint,
            torch_module=torch,
            save_total_limit=int(args.save_total_limit),
            milestone_steps=int(args.save_milestone_steps),
        )

    _run_final_checkpoint_actions(
        has_pending_gradients=has_pending_gradients,
        checkpoint_pending=checkpoint_pending,
        flush_pending_gradients=flush_pending_gradients,
        save_pending_checkpoint=save_pending_checkpoint,
    )

    if _is_main_process():
        _save_policy_adapter(raw_policy_model, output_dir / "adapter")
        tokenizer.save_pretrained(output_dir / "adapter")
        _write_json(
            output_dir / "adapter" / "prompt_contract.json",
            {
                "prompt_contract_version": active_prompt_contract.version,
                "prompt_contract_fingerprint": active_prompt_contract.fingerprint,
                "max_rounds": args.max_rounds,
                "retrieval_top_k": args.retrieval_top_k,
                "sft_adapter_path": args.sft_adapter_path,
            },
        )
        _write_json(
            output_dir / "train_meta.json",
            {
                "model_path": args.model_path,
                "sft_adapter_path": args.sft_adapter_path,
                "prompt_contract_version": active_prompt_contract.version,
                "prompt_contract_fingerprint": active_prompt_contract.fingerprint,
                "prompt_config_path": str(active_prompt_contract.source_path),
                "sft_prompt_contract": sft_prompt_metadata,
                "output_root": str(base_output_dir),
                "output_dir": str(output_dir / "adapter"),
                "rl_data_root": args.rl_data_root,
                "retrieval_root": args.retrieval_root,
                "num_rl_samples": len(samples),
                "group_size": args.group_size,
                "group_diversity": _group_diversity_contract(args),
                "max_rounds": args.max_rounds,
                "kl_beta": args.kl_beta,
                "clip_epsilon": args.clip_epsilon,
                "query_global_reward_weight": args.query_global_reward_weight,
                "evidence_global_reward_weight": args.evidence_global_reward_weight,
                "answer_global_reward_weight": args.answer_global_reward_weight,
                "answer_local_reward_weight": args.answer_local_reward_weight,
                "answer_f1_contract": ANSWER_F1_CONTRACT,
                "advantage_epsilon": args.advantage_epsilon,
                "world_size": _world_size(),
                "global_step": global_step,
                "log_jsonl_path": str(log_path),
                "rollout_jsonl_dir": str(rollout_dir),
                "use_vllm_generation": args.use_vllm_generation,
                "vllm_host": args.vllm_host,
                "vllm_port": args.vllm_port,
                "vllm_gpu_indices": args.vllm_gpu_indices,
                "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
                "vllm_max_model_len": args.vllm_max_model_len,
                "vllm_sync_mode": args.vllm_sync_mode,
                "vllm_sync_every_steps": args.vllm_sync_every_steps,
                "time_initial_weight_sync_seconds": time_initial_weight_sync_seconds,
                "resume": _resume_meta_payload(resume_state),
                "resolved_args": _resolved_args_payload(args),
                "selected_qids": [sample.qid for sample in samples],
                "selected_counts_by_dataset": {
                    dataset: sum(sample.dataset == dataset for sample in samples)
                    for dataset in sorted({sample.dataset for sample in samples})
                },
            },
        )
        print(f"GRPO training complete. Adapter saved to {output_dir / 'adapter'}.")
    _cleanup_distributed(torch)


if __name__ == "__main__":
    main()
