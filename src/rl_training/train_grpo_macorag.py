#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any

from rag import RAGLoopExecutor

from .config import parse_args
from .batched_rollout import run_batched_rollouts
from .data import RLSample, epoch_sample_order, load_rl_samples, select_balanced_samples
from .logging_utils import append_jsonl as _append_jsonl
from .logging_utils import make_timestamped_run_dir
from .logging_utils import write_json as _write_json
from .policy import HFSharedPolicy, VLLMSharedPolicy, batched_sequence_logprobs
from .retrieval import CachedLinearRAGRetrievalEnv
from .rewards import compute_action_rewards, compute_rl_rewards
from .runtime import extract_vllm_server_model_paths as _extract_vllm_server_model_paths
from .runtime import parse_gpu_indices as _parse_gpu_indices
from .runtime import validate_local_vllm_server_model as _validate_local_vllm_server_model
from .runtime import validate_vllm_gpu_placement as _validate_vllm_gpu_placement
from .trainer import assign_action_advantages, compute_grpo_loss, normalize_group_advantages
from .vllm_client import VLLMGenerationClient


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


def _model_kwargs(args: Any, torch: Any, local_rank: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"torch_dtype": _torch_dtype(args, torch)}
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


def _load_policy_and_reference(args: Any, deps: dict[str, Any], device: Any) -> tuple[Any, Any, Any]:
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    PeftModel = deps["PeftModel"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]

    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter_path, trust_remote_code=True)
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
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    policy_model = PeftModel.from_pretrained(base_model, args.sft_adapter_path, is_trainable=True)
    if args.gradient_checkpointing and hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    ref_base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **_model_kwargs(args, torch, _local_rank()),
    )
    ref_model = PeftModel.from_pretrained(ref_base, args.sft_adapter_path, is_trainable=False)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    if not args.load_4bit:
        policy_model.to(device)
        ref_model.to(device)
    return tokenizer, policy_model, ref_model


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


def _build_retrieval_env(args: Any) -> CachedLinearRAGRetrievalEnv:
    return CachedLinearRAGRetrievalEnv(
        retrieval_root=args.retrieval_root,
        embedding_model=args.retrieval_embedding_model,
        spacy_model=args.retrieval_spacy_model,
        top_k=args.retrieval_top_k,
        max_workers=args.retrieval_max_workers,
        batch_size=args.retrieval_batch_size,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        query_cache_size=getattr(args, "retrieval_query_cache_size", 0),
    )


def _build_policy(args: Any, raw_policy_model: Any, tokenizer: Any) -> HFSharedPolicy:
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
    )
    if getattr(args, "vllm_sync_mode", "dense") == "lora":
        client.validate_lora_server(args)
    else:
        client.check_server()
    return VLLMSharedPolicy(vllm_client=client, **common)


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
    sync_mode = getattr(args, "vllm_sync_mode", "dense")
    if sync_mode == "lora":
        return float(client.sync_lora_parameters(raw_policy_model))
    if sync_mode == "dense":
        return float(client.sync_trainable_parameters(raw_policy_model))
    raise SystemExit(f"Unsupported vLLM sync mode: {sync_mode}")


def _rollout_group(
    *,
    args: Any,
    sample: RLSample,
    policy: HFSharedPolicy,
    retrieval_env: CachedLinearRAGRetrievalEnv,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rollouts: list[dict[str, Any]] = []
    time_rollout_seconds = 0.0
    time_vllm_generate_seconds = 0.0
    time_behavior_rescore_seconds = 0.0
    time_reward_seconds = 0.0
    retrieval_stats = getattr(retrieval_env, "stats", None)
    retrieval_before = retrieval_stats() if callable(retrieval_stats) else {}

    supports_batch = callable(getattr(policy, "generate_batch", None))
    if supports_batch:
        policy.reset_trace()
        rollout_start = time.perf_counter()
        batch_results = run_batched_rollouts(
            question=sample.question,
            dataset=sample.dataset,
            group_size=args.group_size,
            max_rounds=args.max_rounds,
            policy=policy,
            retrieval_env=retrieval_env,
        )
        time_rollout_seconds = time.perf_counter() - rollout_start
        time_vllm_generate_seconds += float(
            getattr(policy, "timing", {}).get("time_vllm_generate_seconds", 0.0)
        )
        time_behavior_rescore_seconds += float(
            getattr(policy, "timing", {}).get("time_behavior_rescore_seconds", 0.0)
        )
        group_results = [
            (group_index, item.result, item.trace)
            for group_index, item in enumerate(batch_results)
        ]
    else:
        group_results = []
        for group_index in range(args.group_size):
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
            group_results.append((group_index, result, policy.trace))

    for group_index, result, trace in group_results:
        rollout = {
            "group_index": group_index,
            "result": result,
            "trajectory": result.trajectory,
            "parse_errors": result.parse_errors,
            "final_answer": result.final_answer,
            "actions": list(trace.actions),
        }
        reward_start = time.perf_counter()
        rewards = compute_rl_rewards(rollout=rollout, sample=sample.to_reward_sample())
        action_credit = compute_action_rewards(rollout=rollout, sample=sample.to_reward_sample())
        time_reward_seconds += time.perf_counter() - reward_start
        rollout["rewards"] = rewards
        rollout["action_rewards"] = action_credit["action_rewards"]
        rollout["terminal_reward"] = action_credit["terminal_reward"]
        rollouts.append(rollout)
    reward_start = time.perf_counter()
    advantages = normalize_group_advantages([item["rewards"]["total"] for item in rollouts])
    for rollout, advantage in zip(rollouts, advantages):
        rollout["advantage"] = advantage
    assign_action_advantages(
        rollouts,
        local_weights={
            "query_retriever": float(getattr(args, "query_local_credit_weight", 0.75)),
            "evidence_updater": float(getattr(args, "evidence_local_credit_weight", 0.70)),
            "answer_generator": float(getattr(args, "answer_local_credit_weight", 0.30)),
        },
    )
    time_reward_seconds += time.perf_counter() - reward_start
    retrieval_after = retrieval_stats() if callable(retrieval_stats) else {}
    return rollouts, {
        "time_rollout_seconds": time_rollout_seconds,
        "time_vllm_generate_seconds": time_vllm_generate_seconds,
        "time_behavior_rescore_seconds": time_behavior_rescore_seconds,
        "time_reward_seconds": time_reward_seconds,
        "time_retrieval_seconds": float(
            retrieval_after.get("time_retrieval_seconds", 0.0)
            - retrieval_before.get("time_retrieval_seconds", 0.0)
        ),
        "retrieval_cache_hits": int(
            retrieval_after.get("cache_hits", 0) - retrieval_before.get("cache_hits", 0)
        ),
        "retrieval_cache_misses": int(
            retrieval_after.get("cache_misses", 0) - retrieval_before.get("cache_misses", 0)
        ),
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
    pad_token_id: int = 0,
) -> dict[str, Any]:
    del raw_policy_model
    trainable_actions = [
        (rollout, action)
        for rollout in rollouts
        for action in rollout["actions"]
        if action.completion_ids
    ]
    action_count = len(trainable_actions)
    total_token_count = sum(len(action.completion_ids) for _, action in trainable_actions)
    base_metrics = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "kl": 0.0,
        "clip_fraction": 0.0,
        "trainable_action_count": action_count,
        "valid_completion_token_count": total_token_count,
        "policy_forward_batch_count": 0,
        "reference_forward_batch_count": 0,
        "did_optimizer_step": False,
        "skipped_update_reason": None,
        "time_policy_forward_seconds": 0.0,
        "time_reference_forward_seconds": 0.0,
        "time_backward_seconds": 0.0,
        "time_optimizer_step_seconds": 0.0,
    }
    if not trainable_actions:
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
    loss_total = 0.0
    policy_loss_total = 0.0
    kl_total = 0.0
    clip_fraction_total = 0.0
    time_policy_forward_seconds = 0.0
    time_reference_forward_seconds = 0.0
    time_backward_seconds = 0.0
    time_optimizer_step_seconds = 0.0
    did_optimizer_step = False
    reference_forward_batch_count = 0
    reference_by_action: list[Any] = []
    reference_forward_start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(trainable_actions), reference_batch_size):
            reference_actions = [
                action
                for _, action in trainable_actions[offset : offset + reference_batch_size]
            ]
            reference_batch, reference_mask = batched_sequence_logprobs(
                model=ref_model,
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
        microbatch_token_count = int(mask.sum().item())
        token_weight = microbatch_token_count / total_token_count
        loss_total += metrics["loss"] * token_weight
        policy_loss_total += metrics["policy_loss"] * token_weight
        kl_total += metrics["kl"] * token_weight
        clip_fraction_total += metrics["clip_fraction"] * token_weight
        backward_start = time.perf_counter()
        (loss * token_weight / gradient_accumulation_steps).backward()
        time_backward_seconds += time.perf_counter() - backward_start
    if should_step:
        optimizer_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        did_optimizer_step = True
        time_optimizer_step_seconds += time.perf_counter() - optimizer_start
    return {
        "loss": loss_total,
        "policy_loss": policy_loss_total,
        "kl": kl_total,
        "clip_fraction": clip_fraction_total,
        "trainable_action_count": action_count,
        "valid_completion_token_count": total_token_count,
        "policy_forward_batch_count": policy_forward_batch_count,
        "reference_forward_batch_count": reference_forward_batch_count,
        "did_optimizer_step": did_optimizer_step,
        "skipped_update_reason": None,
        "time_policy_forward_seconds": time_policy_forward_seconds,
        "time_reference_forward_seconds": time_reference_forward_seconds,
        "time_backward_seconds": time_backward_seconds,
        "time_optimizer_step_seconds": time_optimizer_step_seconds,
    }


def _save_checkpoint(raw_policy_model: Any, tokenizer: Any, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_policy_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def _safe_dataset_name(dataset: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(dataset).strip()) or "unknown"


def _dataset_rollout_path(output_dir: Path, dataset: str) -> Path:
    """按数据集拆分 rollout 样本，避免所有数据集混在一个 JSONL。"""
    return output_dir / "rollout_samples" / f"{_safe_dataset_name(dataset)}.jsonl"


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
    time_weight_sync_seconds: float,
    time_total_seconds: float,
) -> dict[str, Any]:
    reward_totals = [item["rewards"]["total"] for item in rollouts]
    action_advantages = [
        float(action.advantage)
        for rollout in rollouts
        for action in rollout.get("actions", [])
    ]
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
        "reward_total": sum(reward_totals) / len(reward_totals),
        "reward_query": best_rollout["rewards"]["query_reward"],
        "reward_evidence": best_rollout["rewards"]["evidence_reward"],
        "reward_answer_f1": best_rollout["rewards"]["answer_f1"],
        "advantage_mean": sum(item["advantage"] for item in rollouts) / len(rollouts),
        "action_advantage_mean": (
            sum(action_advantages) / len(action_advantages)
            if action_advantages
            else 0.0
        ),
        "gold_answer": sample.answer,
        "generated_answer": best_rollout["final_answer"],
        "retrieval_count": len(best_rollout["trajectory"]),
        "parse_errors": best_rollout["parse_errors"],
        "learning_rate": learning_rate,
        "timing": {
            "rollout_seconds": rollout_timing["time_rollout_seconds"],
            "vllm_generate_seconds": rollout_timing.get("time_vllm_generate_seconds", 0.0),
            "behavior_rescore_seconds": rollout_timing.get(
                "time_behavior_rescore_seconds",
                0.0,
            ),
            "reward_seconds": rollout_timing["time_reward_seconds"],
            "policy_forward_seconds": metrics.get("time_policy_forward_seconds", 0.0),
            "reference_forward_seconds": metrics.get(
                "time_reference_forward_seconds",
                0.0,
            ),
            "backward_seconds": metrics["time_backward_seconds"],
            "optimizer_step_seconds": metrics["time_optimizer_step_seconds"],
            "weight_sync_seconds": time_weight_sync_seconds,
            "total_seconds": time_total_seconds,
        },
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
    _validate_vllm_gpu_placement(args)
    _configure_visible_gpus(args)
    deps = _load_training_dependencies()
    torch = deps["torch"]
    _setup_distributed(torch)
    device = _device(torch)
    random.seed(args.seed + _local_rank())
    torch.manual_seed(args.seed + _local_rank())

    samples, data_summary = load_rl_samples(
        data_root=args.rl_data_root,
        data_files=list(args.rl_data_files or []),
        max_samples=args.max_samples,
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
    if _is_main_process():
        print(f"Loaded {len(samples)} RL samples from {args.rl_data_root}")
        print(f"Counts by dataset: {data_summary['counts_by_dataset']}")
    if args.check_only:
        if _is_main_process():
            print("Check-only complete. No model training started.")
        _cleanup_distributed(torch)
        return

    tokenizer, raw_policy_model, ref_model = _load_policy_and_reference(args, deps, device)
    train_model = _wrap_ddp(raw_policy_model, torch)
    raw_policy_model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in raw_policy_model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    retrieval_env = _build_retrieval_env(args)
    retrieval_env.prewarm(sorted({sample.dataset for sample in samples}))
    policy = _build_policy(args, raw_policy_model, tokenizer)
    time_initial_weight_sync_seconds = _sync_vllm_before_first_rollout(
        policy,
        raw_policy_model,
        args,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    base_output_dir = Path(args.output_root)
    output_dir = make_timestamped_run_dir(base_output_dir)
    log_path = output_dir / "train_metrics.jsonl"
    rollout_dir = output_dir / "rollout_samples"
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "rl_dataset_summary.json", data_summary)
        print(f"Run output directory: {output_dir}")

    total_epochs = max(1, int(math.ceil(args.num_train_epochs)))
    global_step = 0
    progress_total = len(_rank_samples(samples)) * total_epochs
    if args.max_steps > 0:
        progress_total = min(progress_total, args.max_steps)
    progress_bar = _make_progress_bar(args, progress_total)
    try:
        for epoch in range(1, total_epochs + 1):
            rank_samples = _rank_samples(
                epoch_sample_order(samples, seed=args.seed, epoch=epoch)
            )
            for sample_index, sample in rank_samples:
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
                sample_start_time = time.perf_counter()
                rollouts, rollout_timing = _rollout_group(
                    args=args,
                    sample=sample,
                    policy=policy,
                    retrieval_env=retrieval_env,
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
                    should_step=(global_step + 1) % max(1, int(args.gradient_accumulation_steps)) == 0,
                    pad_token_id=int(tokenizer.pad_token_id or 0),
                )
                time_weight_sync_seconds = 0.0
                if metrics.get("did_optimizer_step"):
                    time_weight_sync_seconds = _sync_vllm_after_optimizer_step(
                        policy,
                        raw_policy_model,
                        args,
                        completed_step=global_step + 1,
                    )
                time_total_seconds = time.perf_counter() - sample_start_time
                global_step += 1
                reward_totals = [item["rewards"]["total"] for item in rollouts]
                best_rollout = max(rollouts, key=lambda item: item["rewards"]["total"])
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
                        learning_rate=args.learning_rate,
                        rollout_timing=rollout_timing,
                        time_weight_sync_seconds=time_weight_sync_seconds,
                        time_total_seconds=time_total_seconds,
                    )
                    _append_jsonl(log_path, payload)
                    _append_jsonl(
                        _dataset_rollout_path(output_dir, sample.dataset),
                        {
                            "epoch": epoch,
                            "sample": sample_index + 1,
                            "qid": sample.qid,
                            "dataset": sample.dataset,
                            "question": sample.question,
                            "gold_answer": sample.answer,
                            "best_reward": best_rollout["rewards"],
                            "terminal_reward": best_rollout["terminal_reward"],
                            "action_credit": [
                                {
                                    "role": getattr(action.role, "value", str(action.role)),
                                    "round_index": action.round_index,
                                    "local_reward": action.local_reward,
                                    "terminal_reward": action.terminal_reward,
                                    "advantage": action.advantage,
                                }
                                for action in best_rollout["actions"]
                            ],
                            "trajectory": best_rollout["trajectory"],
                        },
                    )
                if _is_main_process() and args.save_steps > 0 and global_step % args.save_steps == 0:
                    _save_checkpoint(raw_policy_model, tokenizer, output_dir, global_step)
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if global_step % max(1, int(args.gradient_accumulation_steps)) != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _sync_vllm_after_optimizer_step(policy, raw_policy_model, args, completed_step=global_step)

    if _is_main_process():
        raw_policy_model.save_pretrained(output_dir / "adapter")
        tokenizer.save_pretrained(output_dir / "adapter")
        _write_json(
            output_dir / "train_meta.json",
            {
                "model_path": args.model_path,
                "sft_adapter_path": args.sft_adapter_path,
                "output_root": str(base_output_dir),
                "output_dir": str(output_dir / "adapter"),
                "rl_data_root": args.rl_data_root,
                "retrieval_root": args.retrieval_root,
                "num_rl_samples": len(samples),
                "group_size": args.group_size,
                "max_rounds": args.max_rounds,
                "kl_beta": args.kl_beta,
                "clip_epsilon": args.clip_epsilon,
                "query_local_credit_weight": args.query_local_credit_weight,
                "evidence_local_credit_weight": args.evidence_local_credit_weight,
                "answer_local_credit_weight": args.answer_local_credit_weight,
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
            },
        )
        print(f"GRPO training complete. Adapter saved to {output_dir / 'adapter'}.")
    _cleanup_distributed(torch)


if __name__ == "__main__":
    main()
