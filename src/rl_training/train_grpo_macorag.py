#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from rag import RAGLoopExecutor
from sft_training.callbacks import _make_timestamped_output_dir

from .config import parse_args
from .data import RLSample, load_rl_samples
from .policy import HFSharedPolicy, sequence_logprobs
from .retrieval import CachedLinearRAGRetrievalEnv
from .rewards import compute_rl_rewards
from .trainer import compute_grpo_loss, normalize_group_advantages


def _configure_visible_gpus(args: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    gpu_indices = str(getattr(args, "gpu_indices", "") or "").strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices or str(args.gpu_index)


def _parse_gpu_indices(value: str | int | None) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def _validate_vllm_gpu_placement(args: Any) -> None:
    if not getattr(args, "use_vllm_generation", False):
        return
    trainer_gpus = _parse_gpu_indices(getattr(args, "gpu_indices", None))
    if not trainer_gpus:
        trainer_gpus = _parse_gpu_indices(getattr(args, "gpu_index", None))
    vllm_gpus = _parse_gpu_indices(getattr(args, "vllm_gpu_indices", None))
    overlap = trainer_gpus & vllm_gpus
    if overlap:
        raise SystemExit(
            "vLLM GPU overlap detected: trainer gpu_indices="
            f"{sorted(trainer_gpus)} and vllm_gpu_indices={sorted(vllm_gpus)} share {sorted(overlap)}. "
            "Use separate GPUs for trainer and vLLM generation."
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


def _is_main_process() -> bool:
    return _local_rank() == 0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_train_event(
    path: Path,
    *,
    event: str,
    epoch: int,
    sample_index: int,
    sample_total: int,
    sample_qid: str,
    sample_dataset: str,
    step: int,
    **extra: Any,
) -> None:
    payload = {
        "event": event,
        "epoch": epoch,
        "sample": sample_index + 1,
        "sample_total": sample_total,
        "qid": sample_qid,
        "dataset": sample_dataset,
        "step": step,
    }
    payload.update(extra)
    _append_jsonl(path, payload)


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
    )


def _rollout_group(
    *,
    args: Any,
    sample: RLSample,
    policy: HFSharedPolicy,
    retrieval_env: CachedLinearRAGRetrievalEnv,
    event_path: Path | None = None,
    epoch: int = 0,
    sample_index: int = 0,
    sample_total: int = 0,
    step: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rollouts: list[dict[str, Any]] = []
    time_rollout_seconds = 0.0
    time_reward_seconds = 0.0
    for group_index in range(args.group_size):
        policy.reset_trace()
        executor = RAGLoopExecutor(policy=policy, retrieval_env=retrieval_env, max_rounds=args.max_rounds)
        rollout_start = time.perf_counter()
        result = executor.run(question=sample.question, dataset=sample.dataset)
        time_rollout_seconds += time.perf_counter() - rollout_start
        rollout = {
            "group_index": group_index,
            "result": result,
            "trajectory": result.trajectory,
            "parse_errors": result.parse_errors,
            "final_answer": result.final_answer,
            "actions": list(policy.trace.actions),
        }
        reward_start = time.perf_counter()
        rewards = compute_rl_rewards(rollout=rollout, sample=sample.to_reward_sample())
        time_reward_seconds += time.perf_counter() - reward_start
        rollout["rewards"] = rewards
        rollouts.append(rollout)
        if event_path is not None and _is_main_process():
            _write_train_event(
                event_path,
                event="rollout_complete",
                epoch=epoch,
                sample_index=sample_index,
                sample_total=sample_total,
                sample_qid=sample.qid,
                sample_dataset=sample.dataset,
                step=step,
                group_index=group_index,
                reward_total=rewards["total"],
                reward_query=rewards["query_reward"],
                reward_evidence=rewards["evidence_reward"],
                reward_answer_f1=rewards["answer_f1"],
                retrieval_count=len(result.trajectory),
                parse_errors=result.parse_errors,
            )
    reward_start = time.perf_counter()
    advantages = normalize_group_advantages([item["rewards"]["total"] for item in rollouts])
    for rollout, advantage in zip(rollouts, advantages):
        rollout["advantage"] = advantage
    time_reward_seconds += time.perf_counter() - reward_start
    return rollouts, {
        "time_rollout_seconds": time_rollout_seconds,
        "time_reward_seconds": time_reward_seconds,
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
) -> dict[str, float]:
    del raw_policy_model
    trainable_actions = [
        (rollout, action)
        for rollout in rollouts
        for action in rollout["actions"]
        if action.completion_ids
    ]
    if not trainable_actions:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "kl": 0.0,
            "time_backward_seconds": 0.0,
            "time_optimizer_step_seconds": 0.0,
        }

    loss_scale = len(trainable_actions) * max(1, int(args.gradient_accumulation_steps))
    loss_total = 0.0
    time_backward_seconds = 0.0
    time_optimizer_step_seconds = 0.0
    metrics_list: list[dict[str, float]] = []
    for rollout, action in trainable_actions:
        advantage = torch.tensor([float(rollout["advantage"])], dtype=torch.float32, device=device)
        current = sequence_logprobs(
            model=train_model,
            prompt_ids=action.prompt_ids,
            completion_ids=action.completion_ids,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            reference = sequence_logprobs(
                model=ref_model,
                prompt_ids=action.prompt_ids,
                completion_ids=action.completion_ids,
                device=device,
            ).unsqueeze(0)
        old = action.old_logprobs.to(device=device).unsqueeze(0)
        mask = torch.ones_like(current)
        loss, metrics = compute_grpo_loss(
            current_logprobs=current,
            old_logprobs=old,
            ref_logprobs=reference,
            action_mask=mask,
            advantages=advantage,
            clip_epsilon=args.clip_epsilon,
            kl_beta=args.kl_beta,
        )
        loss_total += float(loss.detach().item())
        metrics_list.append(metrics)
        backward_start = time.perf_counter()
        (loss / loss_scale).backward()
        time_backward_seconds += time.perf_counter() - backward_start
    if should_step:
        optimizer_start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        time_optimizer_step_seconds += time.perf_counter() - optimizer_start
    return {
        "loss": loss_total / len(metrics_list),
        "policy_loss": sum(item["policy_loss"] for item in metrics_list) / len(metrics_list),
        "kl": sum(item["kl"] for item in metrics_list) / len(metrics_list),
        "time_backward_seconds": time_backward_seconds,
        "time_optimizer_step_seconds": time_optimizer_step_seconds,
    }


def _save_checkpoint(raw_policy_model: Any, tokenizer: Any, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_policy_model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


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
    policy = HFSharedPolicy(
        model=raw_policy_model,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    base_output_dir = Path(args.output_dir)
    output_dir = _make_timestamped_output_dir(base_output_dir)
    log_path = Path(args.log_jsonl_path) if args.log_jsonl_path else output_dir / "train_metrics.jsonl"
    rollout_path = Path(args.rollout_jsonl_path) if args.rollout_jsonl_path else output_dir / "rollout_samples.jsonl"
    event_path = output_dir / "train_events.jsonl"
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "rl_dataset_summary.json", data_summary)
        print(f"Run output directory: {output_dir}")

    rank_samples = _rank_samples(samples)
    total_epochs = max(1, int(math.ceil(args.num_train_epochs)))
    global_step = 0
    progress_total = len(rank_samples) * total_epochs
    if args.max_steps > 0:
        progress_total = min(progress_total, args.max_steps)
    progress_bar = _make_progress_bar(args, progress_total)
    try:
        for epoch in range(1, total_epochs + 1):
            for sample_index, sample in rank_samples:
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
                sample_start_time = time.perf_counter()
                if _is_main_process():
                    _write_train_event(
                        event_path,
                        event="sample_start",
                        epoch=epoch,
                        sample_index=sample_index,
                        sample_total=len(samples),
                        sample_qid=sample.qid,
                        sample_dataset=sample.dataset,
                        step=global_step,
                    )
                try:
                    rollouts, rollout_timing = _rollout_group(
                        args=args,
                        sample=sample,
                        policy=policy,
                        retrieval_env=retrieval_env,
                        event_path=event_path,
                        epoch=epoch,
                        sample_index=sample_index,
                        sample_total=len(samples),
                        step=global_step,
                    )
                except Exception as exc:
                    if _is_main_process():
                        _write_train_event(
                            event_path,
                            event="sample_error",
                            epoch=epoch,
                            sample_index=sample_index,
                            sample_total=len(samples),
                            sample_qid=sample.qid,
                            sample_dataset=sample.dataset,
                            step=global_step,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    raise
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
                    payload = {
                        "epoch": epoch,
                        "sample": sample_index + 1,
                        "sample_total": len(samples),
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
                        "generated_answer": best_rollout["final_answer"],
                        "retrieval_count": len(best_rollout["trajectory"]),
                        "parse_errors": best_rollout["parse_errors"],
                        "learning_rate": args.learning_rate,
                        "time_rollout_seconds": rollout_timing["time_rollout_seconds"],
                        "time_reward_seconds": rollout_timing["time_reward_seconds"],
                        "time_backward_seconds": metrics["time_backward_seconds"],
                        "time_optimizer_step_seconds": metrics["time_optimizer_step_seconds"],
                        "time_total_seconds": time_total_seconds,
                    }
                    _append_jsonl(log_path, payload)
                    _write_train_event(
                        event_path,
                        event="sample_complete",
                        epoch=epoch,
                        sample_index=sample_index,
                        sample_total=len(samples),
                        sample_qid=sample.qid,
                        sample_dataset=sample.dataset,
                        step=global_step,
                        loss=metrics["loss"],
                        reward_total=payload["reward_total"],
                        kl=metrics["kl"],
                        time_rollout_seconds=payload["time_rollout_seconds"],
                        time_reward_seconds=payload["time_reward_seconds"],
                        time_backward_seconds=payload["time_backward_seconds"],
                        time_optimizer_step_seconds=payload["time_optimizer_step_seconds"],
                        time_total_seconds=payload["time_total_seconds"],
                    )
                    _append_jsonl(
                        rollout_path,
                        {
                            "epoch": epoch,
                            "sample": sample_index + 1,
                            "qid": sample.qid,
                            "dataset": sample.dataset,
                            "question": sample.question,
                            "gold_answer": sample.answer,
                            "best_reward": best_rollout["rewards"],
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

    if _is_main_process():
        raw_policy_model.save_pretrained(output_dir / "adapter")
        tokenizer.save_pretrained(output_dir / "adapter")
        _write_json(
            output_dir / "train_meta.json",
            {
                "model_path": args.model_path,
                "sft_adapter_path": args.sft_adapter_path,
                "output_dir": str(output_dir / "adapter"),
                "rl_data_root": args.rl_data_root,
                "retrieval_root": args.retrieval_root,
                "num_rl_samples": len(samples),
                "group_size": args.group_size,
                "max_rounds": args.max_rounds,
                "kl_beta": args.kl_beta,
                "clip_epsilon": args.clip_epsilon,
                "world_size": _world_size(),
                "global_step": global_step,
                "log_jsonl_path": str(log_path),
                "rollout_jsonl_path": str(rollout_path),
                "event_jsonl_path": str(event_path),
            },
        )
        print(f"GRPO training complete. Adapter saved to {output_dir / 'adapter'}.")
    _cleanup_distributed(torch)


if __name__ == "__main__":
    main()
