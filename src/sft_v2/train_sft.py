#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .policy_prompts import load_policy_prompt_contract
from .sft_config import SFTConfig, parse_args
from .sft_data import LoadedSplit, SFTDecision, load_split, select_trajectory_subset


def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _configure_cuda(config: SFTConfig) -> None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_indices


def _load_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Missing training dependency: {exc.name}") from exc
    return locals()


def _render_ids(tokenizer: Any, decision: SFTDecision, max_length: int) -> dict[str, Any] | None:
    prompt_ids = tokenizer.apply_chat_template(
        decision.messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [*decision.messages, {"role": "assistant", "content": decision.target}],
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        common = 0
        for left, right in zip(prompt_ids, full_ids):
            if left != right:
                break
            common += 1
        raise ValueError(
            f"Chat template prompt is not a prefix for {decision.sample_id}; "
            f"common_prefix={common}, prompt_tokens={len(prompt_ids)}"
        )
    if len(full_ids) > max_length:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(value != -100 for value in labels):
        raise ValueError(f"Empty target token span for {decision.sample_id}")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "sample_id": decision.sample_id,
        "role": decision.role,
    }


def _tokenize_split(
    tokenizer: Any,
    split: LoadedSplit,
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features: list[dict[str, Any]] = []
    skipped = Counter()
    lengths: list[int] = []
    target_lengths: list[int] = []
    for decision in split.decisions:
        feature = _render_ids(tokenizer, decision, max_length)
        if feature is None:
            skipped[decision.role] += 1
            continue
        lengths.append(len(feature["input_ids"]))
        target_lengths.append(sum(value != -100 for value in feature["labels"]))
        features.append(feature)
    if not features:
        raise ValueError(f"No tokenized decisions remain for {split.path}")
    stats = {
        "decision_count_before_tokenization": len(split.decisions),
        "decision_count_after_tokenization": len(features),
        "skipped_overlength_by_role": dict(sorted(skipped.items())),
        "max_sequence_tokens": max(lengths),
        "mean_sequence_tokens": sum(lengths) / len(lengths),
        "max_target_tokens": max(target_lengths),
        "mean_target_tokens": sum(target_lengths) / len(target_lengths),
    }
    return features, stats


class DecisionDataset:
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


class TargetOnlyCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(item["input_ids"]) for item in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _split_summary(split: LoadedSplit) -> dict[str, Any]:
    return {
        "path": str(split.path),
        "trajectory_count": split.trajectory_count,
        "trajectory_counts_by_dataset": split.trajectory_counts_by_dataset,
        "decision_count": len(split.decisions),
        "decision_counts_by_role": split.decision_counts_by_role,
        "answer_counts": split.answer_counts,
    }


def _validate_splits(config: SFTConfig, train: LoadedSplit, validation: LoadedSplit) -> None:
    if train.trajectory_count != config.expected_train_trajectories:
        raise SystemExit(
            f"Train split count mismatch: expected {config.expected_train_trajectories}, "
            f"got {train.trajectory_count}"
        )
    if validation.trajectory_count != config.expected_validation_source_trajectories:
        raise SystemExit(
            "Validation source count mismatch: "
            f"expected {config.expected_validation_source_trajectories}, "
            f"got {validation.trajectory_count}"
        )
    train_ids = {item.trajectory_id for item in train.decisions}
    validation_ids = {item.trajectory_id for item in validation.decisions}
    overlap = train_ids & validation_ids
    if overlap:
        raise SystemExit(f"Train/validation qid overlap: {sorted(overlap)[:5]}")
    for name, split in (("train", train), ("validation", validation)):
        counts = split.answer_counts
        if counts.get("can_answer_true", 0) == 0 or counts.get("can_answer_false", 0) == 0:
            raise SystemExit(f"{name} split must contain both answer outcomes: {counts}")


def _validate_teacher_provenance(config: SFTConfig) -> dict[str, Any]:
    path = Path(config.teacher_run_config)
    if not path.is_file():
        raise SystemExit(f"Teacher run config is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid teacher run config {path}: {exc}") from exc
    expected = {
        "prompt_contract_version": config.expected_teacher_prompt_contract_version,
        "train_target_total": config.expected_train_trajectories,
        "validation_target_total": config.expected_validation_source_trajectories,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise SystemExit(
            f"Teacher provenance mismatch in {path}: {', '.join(mismatches)}"
        )
    return payload


def _fingerprint_features(features: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in features:
        digest.update(str(item["sample_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_directory(config: SFTConfig) -> Path:
    run_id = os.environ.get("SFT_V2_RUN_ID") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(config.output_root) / run_id


def _print_examples(tokenizer: Any, split: LoadedSplit, count: int, seed: int) -> None:
    examples = list(split.decisions)
    random.Random(seed).shuffle(examples)
    for decision in examples[:count]:
        rendered = tokenizer.apply_chat_template(
            decision.messages, tokenize=False, add_generation_prompt=True
        )
        print("=" * 80)
        print(f"{decision.sample_id} final_round={decision.final_round}")
        print(rendered[-3000:])
        print("TARGET:", decision.target)


def main() -> None:
    config, cli = parse_args()
    _configure_cuda(config)
    contract = load_policy_prompt_contract(config.prompt_config_path)
    train_split = load_split(config.train_file, contract)
    validation_source = load_split(config.validation_file, contract)
    _validate_splits(config, train_split, validation_source)
    validation_split = select_trajectory_subset(
        validation_source,
        limit=config.validation_trajectory_limit,
        seed=config.seed,
    )
    if validation_split.trajectory_count != config.validation_trajectory_limit:
        raise SystemExit(
            "Selected validation count mismatch: "
            f"expected {config.validation_trajectory_limit}, "
            f"got {validation_split.trajectory_count}"
        )
    if (
        validation_split.answer_counts.get("can_answer_true", 0) == 0
        or validation_split.answer_counts.get("can_answer_false", 0) == 0
    ):
        raise SystemExit(
            "Selected validation subset must contain both answer outcomes: "
            f"{validation_split.answer_counts}"
        )
    teacher_provenance = _validate_teacher_provenance(config)
    print("Train split:", json.dumps(_split_summary(train_split), ensure_ascii=False, sort_keys=True))
    print("Validation source:", json.dumps(_split_summary(validation_source), ensure_ascii=False, sort_keys=True))
    print("Validation selected:", json.dumps(_split_summary(validation_split), ensure_ascii=False, sort_keys=True))
    deps = _load_dependencies()
    tokenizer = deps["AutoTokenizer"].from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_features, train_token_stats = _tokenize_split(tokenizer, train_split, config.max_length)
    validation_features, validation_token_stats = _tokenize_split(
        tokenizer, validation_split, config.max_length
    )
    print("Train tokenization:", json.dumps(train_token_stats, ensure_ascii=False, sort_keys=True))
    print("Validation tokenization:", json.dumps(validation_token_stats, ensure_ascii=False, sort_keys=True))
    if cli.check_only:
        _print_examples(tokenizer, train_split, cli.check_only_max_samples, config.seed)
        print("SFT_V2_CHECK_OK")
        return

    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("SFT training requires CUDA")
    if config.bf16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("Selected GPU does not support bf16")
    deps["set_seed"](config.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dtype = torch.bfloat16 if config.bf16 else torch.float16
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": config.attn_implementation,
    }
    if config.load_4bit:
        model_kwargs["quantization_config"] = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = {"": local_rank}
    model = deps["AutoModelForCausalLM"].from_pretrained(
        config.model_path, trust_remote_code=True, **model_kwargs
    )
    if config.load_4bit:
        model = deps["prepare_model_for_kbit_training"](
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    lora_config = deps["LoraConfig"](
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type=deps["TaskType"].CAUSAL_LM,
    )
    model = deps["get_peft_model"](model, lora_config)
    if _is_main_process():
        model.print_trainable_parameters()

    output_dir = _run_directory(config)
    training_args = deps["TrainingArguments"](
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=config.bf16,
        fp16=config.fp16 and not config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=DecisionDataset(train_features),
        eval_dataset=DecisionDataset(validation_features),
        data_collator=TargetOnlyCollator(tokenizer),
        callbacks=[deps["EarlyStoppingCallback"](early_stopping_patience=3)],
    )
    if _is_main_process():
        contract_payload = {
            "prompt_contract_version": contract.version,
            "prompt_contract_fingerprint": contract.fingerprint,
            "prompt_config_path": str(contract.source_path),
        }
        manifest = {
            "config": asdict(config),
            "prompt_contract": contract_payload,
            "teacher_provenance": {
                "path": config.teacher_run_config,
                "prompt_contract_version": teacher_provenance["prompt_contract_version"],
                "prompt_contract_fingerprint": teacher_provenance.get("prompt_contract_fingerprint"),
                "teacher_model": teacher_provenance.get("teacher_model"),
                "index_manifest_fingerprint": teacher_provenance.get("index_manifest_fingerprint"),
            },
            "train": _split_summary(train_split),
            "validation_source": _split_summary(validation_source),
            "validation": _split_summary(validation_split),
            "train_file_sha256": _sha256_file(train_split.path),
            "validation_file_sha256": _sha256_file(validation_split.path),
            "train_tokenization": train_token_stats,
            "validation_tokenization": validation_token_stats,
            "train_decision_fingerprint": _fingerprint_features(train_features),
            "validation_decision_fingerprint": _fingerprint_features(validation_features),
        }
        _write_json(output_dir / "sft_run_manifest.json", manifest)
        _write_json(output_dir / "prompt_contract.json", contract_payload)
    result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "adapter"))
    if _is_main_process():
        tokenizer.save_pretrained(str(output_dir / "adapter"))
        contract_payload = {
            "prompt_contract_version": contract.version,
            "prompt_contract_fingerprint": contract.fingerprint,
            "prompt_config_path": str(contract.source_path),
        }
        _write_json(output_dir / "adapter" / "prompt_contract.json", contract_payload)
        _write_json(
            output_dir / "training_summary.json",
            {
                "global_step": trainer.state.global_step,
                "best_metric": trainer.state.best_metric,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "train_metrics": result.metrics,
            },
        )


if __name__ == "__main__":
    main()
