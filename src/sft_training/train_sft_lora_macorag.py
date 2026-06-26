#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from .callbacks import (
    _make_eval_metrics_callback,
    _make_jsonl_logging_callback,
    _make_sample_progress_callback,
    _make_timestamped_output_dir,
    _resolve_run_log_path,
)
from .config import DEFAULT_ARG_VALUES, DEFAULT_CONFIG_PATH, DEFAULT_SYSTEM_PROMPT, parse_args
from .data import (
    TrajectoryRecord,
    TrainingData,
    TrainingSample,
    _resolve_dataset_paths,
    build_train_records,
    build_training_data,
    flatten_training_samples,
    split_records as _split_records,
    split_training_samples,
    trajectory_to_sft_records,
)
from .dataset import _build_dataset, _pad_batch, _tokenize_records
from .trainer import _make_target_only_trainer_cls


def _configure_visible_gpus(args: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    gpu_indices = str(getattr(args, "gpu_indices", "") or "").strip()
    if gpu_indices:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for item in rows:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def _print_check_only(args: Any, training_data: TrainingData) -> None:
    records = list(training_data.records)
    random.seed(args.seed)
    random.shuffle(records)
    sample = records[: args.check_only_max_samples]
    print("Sample records after masking state/observation:")
    for item in sample:
        print(f"{item.qid} [{item.dataset}/{item.action_type}]: {item.question[:120]!r}")
        print("INPUT:")
        print(item.prompt_text[:600])
        print("LABEL:")
        print(item.target_text[:600])
        leaked_tokens = ("<state>", "<observation>", '"evidence"', '"text"', '"score"', '"source_query"')
        if any(token in item.target_text for token in leaked_tokens):
            raise SystemExit("Masked fields leaked into target text")
    dataset_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for item in training_data.records:
        dataset_counts[item.dataset] = dataset_counts.get(item.dataset, 0) + 1
        action_counts[item.action_type] = action_counts.get(item.action_type, 0) + 1
    print("Original sample counts:", training_data.source_sample_counts_by_dataset)
    print("Record counts:", dataset_counts)
    print("Action counts:", action_counts)


def _load_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            EarlyStoppingCallback,
            PrinterCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install transformers and peft (and optionally bitsandbytes), "
            "then rerun with the same command."
        ) from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "LoraConfig": LoraConfig,
        "PrinterCallback": PrinterCallback,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "get_peft_model": get_peft_model,
        "TaskType": TaskType,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
    }


def _torch_dtype(args: Any, torch: Any) -> Any:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float16


def _model_kwargs(args: Any, torch_dtype: Any) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if not args.load_4bit:
        return model_kwargs
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # type: ignore  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(f"4-bit quantization requested but dependency missing: {exc.name}.") from exc

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs["device_map"] = {"": _local_rank()} if _world_size() > 1 else "auto"
    return model_kwargs


def _split_train_eval_samples(
    args: Any,
    samples: list[TrainingSample],
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    if not args.validation_split:
        return samples, []
    if args.eval_split_ratio <= 0.0:
        raise SystemExit("validation_split requires eval_split_ratio > 0.")
    train_samples, val_samples = split_training_samples(samples, args.eval_split_ratio, args.train_test_seed)
    if not train_samples:
        raise SystemExit("Validation split left no training samples. Lower eval_split_ratio.")
    if not val_samples:
        raise SystemExit("Validation split produced no validation samples. Increase eval_split_ratio.")
    return train_samples, val_samples


def _training_arguments(args: Any, output_dir: Path, has_eval: bool, TrainingArguments: Any) -> Any:
    if has_eval and args.eval_steps <= 0:
        raise SystemExit("validation_split requires eval_steps > 0.")
    if has_eval and args.early_stopping_patience > 0 and args.save_steps % args.eval_steps != 0:
        raise SystemExit("early stopping requires save_steps to be a multiple of eval_steps.")

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "logging_strategy": "steps",
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16 and not args.bf16,
        "dataloader_num_workers": 0,
        "eval_steps": args.eval_steps if has_eval else None,
        "max_steps": args.max_steps if args.max_steps > 0 else -1,
        "remove_unused_columns": False,
        "report_to": [],
        "disable_tqdm": True,
        "load_best_model_at_end": has_eval and args.early_stopping_patience > 0,
        "metric_for_best_model": args.metric_for_best_model if has_eval else None,
        "greater_is_better": args.greater_is_better if has_eval else None,
    }
    if _world_size() > 1:
        training_kwargs["ddp_find_unused_parameters"] = False

    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_key] = "steps" if has_eval else "no"
    return TrainingArguments(**training_kwargs)


def main() -> None:
    args = parse_args()
    _configure_visible_gpus(args)

    data_root = Path(args.data_root)
    resolved_paths = _resolve_dataset_paths(str(data_root))
    if not resolved_paths or not all(path.exists() for path in resolved_paths):
        missing = [str(path) for path in resolved_paths if path and not path.exists()]
        raise SystemExit(f"Missing trajectory files: {missing} or invalid root: {data_root}")

    training_data = build_training_data(data_root, max_samples=args.max_samples)
    records = training_data.records
    source_sample_count = training_data.source_sample_count
    print(f"Loaded {len(records)} SFT action records from {data_root}")
    print(f"Loaded {source_sample_count} original trajectory samples from {data_root}")
    if len(records) == 0:
        raise SystemExit("No usable trajectory samples found.")
    if args.check_only:
        _print_check_only(args, training_data)
        return

    deps = _load_training_dependencies()
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    EarlyStoppingCallback = deps["EarlyStoppingCallback"]
    LoraConfig = deps["LoraConfig"]
    PrinterCallback = deps["PrinterCallback"]
    Trainer = deps["Trainer"]
    TrainerCallback = deps["TrainerCallback"]
    TrainingArguments = deps["TrainingArguments"]
    TaskType = deps["TaskType"]
    get_peft_model = deps["get_peft_model"]
    prepare_model_for_kbit_training = deps["prepare_model_for_kbit_training"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_samples, val_samples = _split_train_eval_samples(args, training_data.samples)
    train_records = flatten_training_samples(train_samples)
    val_records = flatten_training_samples(val_samples)
    train_source_sample_count = len(train_samples)

    base_output_dir = Path(args.output_dir)
    output_dir = _make_timestamped_output_dir(base_output_dir)
    log_jsonl_path = _resolve_run_log_path(args.log_jsonl_path, base_output_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _is_main_process():
        print(f"Run output directory: {output_dir}")

    skipped_train_records: list[dict[str, Any]] = []
    train_dataset = _build_dataset(
        tokenizer,
        train_records,
        args.max_length,
        args.system_prompt,
        skipped_records=skipped_train_records,
    )
    if _is_main_process() and skipped_train_records:
        skipped_path = output_dir / "skipped_overlength_records.jsonl"
        _write_jsonl(skipped_path, skipped_train_records)
        print(
            f"Skipped {len(skipped_train_records)} overlength SFT action records "
            f"with token_length > max_length ({args.max_length}). Details: {skipped_path}"
        )

    if val_records:
        skipped_eval_records: list[dict[str, Any]] = []
        eval_dataset = _build_dataset(
            tokenizer,
            val_records,
            args.max_length,
            args.system_prompt,
            skipped_records=skipped_eval_records,
        )
        if _is_main_process() and skipped_eval_records:
            skipped_eval_path = output_dir / "skipped_eval_overlength_records.jsonl"
            _write_jsonl(skipped_eval_path, skipped_eval_records)
            print(
                f"Skipped {len(skipped_eval_records)} overlength eval action records "
                f"with token_length > max_length ({args.max_length}). Details: {skipped_eval_path}"
            )
    else:
        eval_dataset = None
        skipped_eval_records = []
    if len(train_dataset) == 0:
        raise SystemExit("No trainable samples remain after max_length filtering.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **_model_kwargs(args, _torch_dtype(args, torch)),
    )
    model.config.use_cache = False
    if args.load_4bit:
        model = prepare_model_for_kbit_training(
            model,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    if _is_main_process():
        model.print_trainable_parameters()

    effective_devices = max(1, _world_size())
    action_steps_per_epoch = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * effective_devices))
    optimizer_steps_per_epoch = math.ceil(action_steps_per_epoch / args.gradient_accumulation_steps)
    total_optimizer_steps = (
        args.max_steps if args.max_steps > 0 else math.ceil(optimizer_steps_per_epoch * args.num_train_epochs)
    )
    progress_epochs = args.num_train_epochs
    if args.max_steps > 0:
        progress_epochs = min(args.num_train_epochs, args.max_steps / optimizer_steps_per_epoch)
    total_source_sample_visits = int(math.ceil(train_source_sample_count * progress_epochs))
    if _is_main_process():
        print(f"Training original samples per epoch: {train_source_sample_count}")
        print(f"Validation original samples per eval: {len(val_samples)}")
        print(f"Training SFT action records per epoch: {len(train_dataset)}")
        print(f"Validation SFT action records per eval: {len(eval_dataset) if eval_dataset is not None else 0}")
        print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * effective_devices}")
        print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")
        print(f"Total optimizer steps: {total_optimizer_steps}")
        print(f"Total original sample visits: {total_source_sample_visits}")

    train_args = _training_arguments(args, output_dir, eval_dataset is not None, TrainingArguments)

    callbacks = [
        _make_eval_metrics_callback(
            output_dir / "eval_metrics.jsonl",
            TrainerCallback,
        ),
        _make_jsonl_logging_callback(
            log_jsonl_path,
            TrainerCallback,
            train_source_sample_count,
            args.num_train_epochs,
        )
    ]
    if eval_dataset is not None and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    if not args.disable_tqdm:
        callbacks.append(_make_sample_progress_callback(TrainerCallback, train_source_sample_count, args.num_train_epochs))

    trainer_cls = _make_target_only_trainer_cls(Trainer)
    trainer = trainer_cls(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda features: _pad_batch(features, tokenizer.pad_token_id, args.max_length),
        callbacks=callbacks,
    )
    trainer.remove_callback(PrinterCallback)

    trainer.train()
    if _is_main_process():
        model.save_pretrained(output_dir / "adapter")
        tokenizer.save_pretrained(output_dir / "adapter")
        train_args_dict = {
            "num_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "num_records": len(records),
            "num_original_samples": source_sample_count,
            "num_train_original_samples_per_epoch": train_source_sample_count,
            "num_eval_original_samples": len(val_samples),
            "num_train_action_records_per_epoch": len(train_dataset),
            "num_eval_action_records": len(eval_dataset) if eval_dataset is not None else 0,
            "num_skipped_overlength_records": len(skipped_train_records),
            "num_skipped_eval_overlength_records": len(skipped_eval_records),
            "validation_split": bool(eval_dataset is not None),
            "eval_split_ratio": args.eval_split_ratio,
            "early_stopping_patience": args.early_stopping_patience,
            "metric_for_best_model": args.metric_for_best_model,
            "gpu_indices": args.gpu_indices,
            "world_size": _world_size(),
            "output_dir": str(output_dir / "adapter"),
            "base_output_dir": str(base_output_dir),
            "log_jsonl_path": str(log_jsonl_path),
            "max_length": args.max_length,
            "seed": args.seed,
        }
        with (output_dir / "train_meta.json").open("w", encoding="utf-8") as file:
            json.dump(train_args_dict, file, ensure_ascii=False, indent=2)
        print(f"Training complete. Adapter saved to {output_dir/'adapter'}.")


if __name__ == "__main__":
    main()
