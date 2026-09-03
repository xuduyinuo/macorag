#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib
import importlib.util
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
    _make_phase_metrics_callback,
    _make_sample_progress_callback,
    _prepare_resume_logs,
    make_run_dir,
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
    sample_counts_by_dataset,
    split_records as _split_records,
    split_training_samples,
    split_training_samples_by_dataset,
    trajectory_to_sft_records,
    validate_teacher_dataset_contract,
)
from prompt_config import load_prompt_contract
from .dataset import _build_dataset, _dataset_fingerprint, _pad_batch, _tokenize_records
from .trainer import _make_target_only_trainer_cls


def _configure_visible_gpus(args: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    gpu_indices = str(getattr(args, "gpu_indices", "") or "").strip()
    if gpu_indices:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except ImportError:
        pass
    return _local_rank() == 0


def _synchronize_resume_logs(
    output_dir: Path,
    checkpoint: Path,
    *,
    samples_per_epoch: int,
) -> int:
    import torch

    if (
        _world_size() > 1
        and torch.distributed.is_available()
        and not torch.distributed.is_initialized()
    ):
        if torch.cuda.is_available():
            torch.cuda.set_device(_local_rank())
            backend = "nccl"
        else:
            backend = "gloo"
        torch.distributed.init_process_group(backend=backend)

    resume_segment = 0
    if _is_main_process():
        resume_segment = _prepare_resume_logs(
            output_dir,
            checkpoint,
            samples_per_epoch=samples_per_epoch,
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        values = [resume_segment]
        torch.distributed.broadcast_object_list(values, src=0)
        torch.distributed.barrier()
        resume_segment = int(values[0])
    return resume_segment


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for item in rows:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def _resolve_resume_checkpoint(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    checkpoint = Path(value)
    required_files = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
    )
    missing = [name for name in required_files if not (checkpoint / name).is_file()]
    if not (checkpoint / "rng_state.pth").is_file() and not any(checkpoint.glob("rng_state_*.pth")):
        missing.append("rng_state.pth or rng_state_<rank>.pth")
    if missing:
        raise SystemExit(
            f"Incomplete SFT resume checkpoint {checkpoint}: missing {', '.join(missing)}"
        )
    return checkpoint


def _resume_uses_random_sampler(checkpoint: Path) -> bool:
    manifest_path = checkpoint.parent / "sft_run_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid SFT run manifest {manifest_path}: {exc}") from exc
    sampler = str(payload.get("train_sampler") or "")
    if sampler not in {"random", "sequential"}:
        raise SystemExit(f"Invalid train_sampler in {manifest_path}: {sampler!r}")
    return sampler == "random"


def _validate_resume_runtime_files(checkpoint: Path, *, world_size: int, fp16: bool) -> None:
    missing: list[str] = []
    if world_size <= 1:
        if not (checkpoint / "rng_state.pth").is_file() and not (checkpoint / "rng_state_0.pth").is_file():
            missing.append("rng_state.pth")
    else:
        missing.extend(
            f"rng_state_{rank}.pth"
            for rank in range(world_size)
            if not (checkpoint / f"rng_state_{rank}.pth").is_file()
        )
    if fp16 and not (checkpoint / "scaler.pt").is_file():
        missing.append("scaler.pt")
    if missing:
        raise SystemExit(
            f"Incomplete SFT runtime state in {checkpoint}: missing {', '.join(missing)}"
        )


def _validate_resume_compatibility(checkpoint: Path, expected: dict[str, Any]) -> None:
    manifest_path = checkpoint.parent / "sft_run_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid SFT run manifest {manifest_path}: {exc}") from exc
    mismatches = [
        key
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    if mismatches:
        raise SystemExit(
            f"SFT resume contract mismatch in {manifest_path}: {', '.join(mismatches)}"
        )


def _run_trainer(trainer: Any, resume_checkpoint: Path | None) -> Any:
    if resume_checkpoint is None:
        return trainer.train()
    return trainer.train(resume_from_checkpoint=str(resume_checkpoint))


def _build_early_stopping_callback(
    args: Any,
    *,
    has_eval: bool,
    callback_cls: Any,
) -> Any | None:
    if not has_eval or not args.early_stopping_enabled:
        return None
    return callback_cls(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )


def _trainer_completion_metadata(trainer: Any, total_optimizer_steps: int) -> dict[str, Any]:
    state = trainer.state
    stopped_early = bool(
        trainer.control.should_training_stop and state.global_step < total_optimizer_steps
    )
    return {
        "stopped_early": stopped_early,
        "best_metric": state.best_metric,
        "best_model_checkpoint": state.best_model_checkpoint,
        "stopped_epoch": state.epoch,
        "global_step": state.global_step,
    }


def _sample_qid_fingerprint(samples: list[TrainingSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.dataset.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.qid.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _print_check_only(args: Any, training_data: TrainingData) -> None:
    train_samples, val_samples = _split_train_eval_samples(args, training_data.samples)
    check_contract = {
        "source_sample_counts_by_dataset": training_data.source_sample_counts_by_dataset,
        "selected_sample_counts_by_dataset": training_data.selected_sample_counts_by_dataset,
        "train_sample_counts_by_dataset": sample_counts_by_dataset(train_samples),
        "eval_sample_counts_by_dataset": sample_counts_by_dataset(val_samples),
        "early_stopping": {
            "enabled": args.early_stopping_enabled,
            "eval_strategy": args.eval_strategy,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "patience": args.early_stopping_patience,
            "threshold": args.early_stopping_threshold,
            "metric": args.metric_for_best_model,
            "greater_is_better": args.greater_is_better,
            "restore_callback_states_from_checkpoint": args.restore_callback_states_from_checkpoint,
        },
    }
    print("SFT check contract:", json.dumps(check_contract, ensure_ascii=False, sort_keys=True))
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


def _validate_acceleration_runtime(
    args: Any,
    torch: Any,
    find_spec=importlib.util.find_spec,
    import_module=importlib.import_module,
) -> None:
    if args.bf16 and args.fp16:
        raise SystemExit("bf16 and fp16 cannot both be enabled.")
    if args.attn_implementation == "flash_attention_2":
        if not torch.cuda.is_available():
            raise SystemExit("FlashAttention 2 requested but CUDA is unavailable.")
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
    if args.bf16 and _world_size() > 1 and torch.cuda.is_available():
        torch.cuda.set_device(_local_rank())
    if args.bf16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise SystemExit("bf16 requested but the selected CUDA device does not support bf16.")


def _model_kwargs(args: Any, torch_dtype: Any) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "attn_implementation": args.attn_implementation,
    }
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
    try:
        train_samples, val_samples = split_training_samples_by_dataset(
            samples, args.eval_split_ratio, args.train_test_seed
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not train_samples:
        raise SystemExit("Validation split left no training samples. Lower eval_split_ratio.")
    if not val_samples:
        raise SystemExit("Validation split produced no validation samples. Increase eval_split_ratio.")
    return train_samples, val_samples


def _training_arguments(args: Any, output_dir: Path, has_eval: bool, TrainingArguments: Any) -> Any:
    eval_strategy = args.eval_strategy if has_eval else "no"
    if eval_strategy == "steps" and args.eval_steps <= 0:
        raise SystemExit("step-based validation requires eval_steps > 0.")
    if (
        eval_strategy == "steps"
        and args.early_stopping_enabled
        and args.save_steps % args.eval_steps != 0
    ):
        raise SystemExit("early stopping requires save_steps to be a multiple of eval_steps.")
    load_best_model = bool(has_eval and args.early_stopping_enabled)
    save_strategy = eval_strategy if load_best_model else "steps"

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
        "save_strategy": save_strategy,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16 and not args.bf16,
        "dataloader_num_workers": 0,
        "eval_steps": args.eval_steps if eval_strategy == "steps" else None,
        "max_steps": args.max_steps if args.max_steps > 0 else -1,
        "remove_unused_columns": False,
        "report_to": [],
        "disable_tqdm": True,
        "load_best_model_at_end": load_best_model,
        "metric_for_best_model": args.metric_for_best_model if has_eval else None,
        "greater_is_better": args.greater_is_better if has_eval else None,
        "restore_callback_states_from_checkpoint": args.restore_callback_states_from_checkpoint,
    }
    if _world_size() > 1:
        training_kwargs["ddp_find_unused_parameters"] = False

    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_kwargs[strategy_key] = eval_strategy
    return TrainingArguments(**training_kwargs)


def main() -> None:
    args = parse_args()
    _configure_visible_gpus(args)

    data_root = Path(args.data_root)
    resolved_paths = _resolve_dataset_paths(str(data_root))
    if not resolved_paths or not all(path.exists() for path in resolved_paths):
        missing = [str(path) for path in resolved_paths if path and not path.exists()]
        raise SystemExit(f"Missing trajectory files: {missing} or invalid root: {data_root}")

    prompt_contract = load_prompt_contract(args.prompt_config_path)
    teacher_metadata: dict[str, Any] = {}
    if args.require_teacher_provenance:
        teacher_metadata = validate_teacher_dataset_contract(
            data_root,
            expected_contract=prompt_contract,
            max_rounds=args.max_rounds,
            retrieval_top_k=args.retrieval_top_k,
        )

    try:
        training_data = build_training_data(
            data_root,
            max_samples=args.max_samples,
            max_samples_by_dataset=args.max_samples_by_dataset,
            data_sampling_seed=args.data_sampling_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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

    _validate_acceleration_runtime(args, torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_samples, val_samples = _split_train_eval_samples(args, training_data.samples)
    train_records = flatten_training_samples(train_samples)
    val_records = flatten_training_samples(val_samples)
    train_source_sample_count = len(train_samples)

    base_output_dir = Path(args.output_root)
    resume_checkpoint = _resolve_resume_checkpoint(args.resume_from_checkpoint)
    if resume_checkpoint is not None:
        _validate_resume_runtime_files(
            resume_checkpoint,
            world_size=_world_size(),
            fp16=bool(args.fp16 and not args.bf16),
        )
    output_dir = resume_checkpoint.parent if resume_checkpoint is not None else make_run_dir(base_output_dir)
    train_shuffle = resume_checkpoint is None or _resume_uses_random_sampler(resume_checkpoint)
    log_jsonl_path = output_dir / "train_metrics.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_segment = 0
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
    run_manifest = {
        "schema_version": 2,
        "model_path": args.model_path,
        "data_root": args.data_root,
        "prompt_contract_fingerprint": prompt_contract.fingerprint,
        "max_length": args.max_length,
        "seed": args.seed,
        "train_action_records": len(train_dataset),
        "eval_action_records": len(eval_dataset) if eval_dataset is not None else 0,
        "train_dataset_fingerprint": _dataset_fingerprint(train_dataset),
        "eval_dataset_fingerprint": _dataset_fingerprint(eval_dataset) if eval_dataset is not None else None,
        "max_samples": args.max_samples,
        "max_samples_by_dataset": args.max_samples_by_dataset,
        "data_sampling_seed": args.data_sampling_seed,
        "source_sample_counts_by_dataset": training_data.source_sample_counts_by_dataset,
        "selected_sample_counts_by_dataset": training_data.selected_sample_counts_by_dataset,
        "selected_qid_fingerprint": _sample_qid_fingerprint(training_data.samples),
        "train_sample_counts_by_dataset": sample_counts_by_dataset(train_samples),
        "eval_sample_counts_by_dataset": sample_counts_by_dataset(val_samples),
        "train_qid_fingerprint": _sample_qid_fingerprint(train_samples),
        "eval_qid_fingerprint": _sample_qid_fingerprint(val_samples),
        "train_test_seed": args.train_test_seed,
        "eval_split_ratio": args.eval_split_ratio,
        "eval_strategy": args.eval_strategy,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "early_stopping_enabled": args.early_stopping_enabled,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": args.greater_is_better,
        "restore_callback_states_from_checkpoint": args.restore_callback_states_from_checkpoint,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": list(args.target_modules),
        "load_4bit": args.load_4bit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "attn_implementation": args.attn_implementation,
        "world_size": _world_size(),
        "train_sampler": "random" if train_shuffle else "sequential",
        "eval_loss_semantics": "macro_mean_of_per_action_target_token_mean",
    }
    if resume_checkpoint is not None:
        _validate_resume_compatibility(resume_checkpoint, run_manifest)
        resume_segment = _synchronize_resume_logs(
            output_dir,
            resume_checkpoint,
            samples_per_epoch=train_source_sample_count,
        )
    if _is_main_process():
        print(f"Training original samples per epoch: {train_source_sample_count}")
        print(f"Validation original samples per eval: {len(val_samples)}")
        print(f"Training SFT action records per epoch: {len(train_dataset)}")
        print(f"Validation SFT action records per eval: {len(eval_dataset) if eval_dataset is not None else 0}")
        print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * effective_devices}")
        print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")
        print(f"Total optimizer steps: {total_optimizer_steps}")
        print(f"Total original sample visits: {total_source_sample_visits}")
        _write_json_atomic(output_dir / "sft_run_manifest.json", run_manifest)

    train_args = _training_arguments(args, output_dir, eval_dataset is not None, TrainingArguments)

    callbacks = [
        _make_eval_metrics_callback(
            output_dir / "eval_metrics.jsonl",
            TrainerCallback,
            resume_segment=resume_segment,
        ),
        _make_jsonl_logging_callback(
            log_jsonl_path,
            TrainerCallback,
            train_source_sample_count,
            args.num_train_epochs,
            resume_segment=resume_segment,
        ),
        _make_phase_metrics_callback(
            output_dir / "phase_metrics.jsonl",
            TrainerCallback,
            eval_token_count=sum(
                len(eval_dataset[index]["input_ids"])
                for index in range(len(eval_dataset))
            ) if eval_dataset is not None else 0,
            resume_segment=resume_segment,
        ),
    ]
    early_stopping_callback = _build_early_stopping_callback(
        args,
        has_eval=eval_dataset is not None,
        callback_cls=EarlyStoppingCallback,
    )
    if early_stopping_callback is not None:
        callbacks.append(early_stopping_callback)
    if not args.disable_tqdm:
        callbacks.append(_make_sample_progress_callback(TrainerCallback, train_source_sample_count, args.num_train_epochs))

    trainer_cls = _make_target_only_trainer_cls(Trainer, train_shuffle=train_shuffle)
    trainer = trainer_cls(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda features: _pad_batch(features, tokenizer.pad_token_id, args.max_length),
        callbacks=callbacks,
    )
    trainer.remove_callback(PrinterCallback)

    _run_trainer(trainer, resume_checkpoint)
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
            "early_stopping_enabled": args.early_stopping_enabled,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
            "restore_callback_states_from_checkpoint": args.restore_callback_states_from_checkpoint,
            "metric_for_best_model": args.metric_for_best_model,
            "gpu_indices": args.gpu_indices,
            "world_size": _world_size(),
            "output_dir": str(output_dir / "adapter"),
            "output_root": str(base_output_dir),
            "log_jsonl_path": str(log_jsonl_path),
            "phase_metrics_path": str(output_dir / "phase_metrics.jsonl"),
            "eval_strategy": args.eval_strategy if eval_dataset is not None else "no",
            "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "train_sampler": "random" if train_shuffle else "sequential",
            "eval_loss_semantics": "macro_mean_of_per_action_target_token_mean",
            "max_length": args.max_length,
            "seed": args.seed,
            "prompt_contract_version": prompt_contract.version,
            "prompt_contract_fingerprint": prompt_contract.fingerprint,
            "prompt_config_path": str(prompt_contract.source_path),
            "max_rounds": args.max_rounds,
            "retrieval_top_k": args.retrieval_top_k,
            "teacher_run_config": teacher_metadata,
            **_trainer_completion_metadata(trainer, total_optimizer_steps),
        }
        with (output_dir / "train_meta.json").open("w", encoding="utf-8") as file:
            json.dump(train_args_dict, file, ensure_ascii=False, indent=2)
        with (output_dir / "adapter" / "prompt_contract.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "prompt_contract_version": prompt_contract.version,
                    "prompt_contract_fingerprint": prompt_contract.fingerprint,
                    "max_rounds": args.max_rounds,
                    "retrieval_top_k": args.retrieval_top_k,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Training complete. Adapter saved to {output_dir/'adapter'}.")


if __name__ == "__main__":
    main()
