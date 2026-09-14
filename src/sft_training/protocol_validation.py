from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from rag import RAGLoopExecutor
from rag.protocol_metrics import compute_protocol_metrics
from rl_training.policy import HFSharedPolicy
from rl_training.retrieval import create_retrieval_env

from .data import TrainingSample


def select_fixed_protocol_samples(
    validation_samples: list[TrainingSample],
    *,
    size: int,
    seed: int,
) -> list[TrainingSample]:
    """Select a deterministic, approximately balanced subset without teacher decisions."""

    if size <= 0 or size > len(validation_samples):
        raise ValueError(
            f"protocol validation size must be in [1, {len(validation_samples)}]; got {size}"
        )
    grouped: dict[str, list[TrainingSample]] = defaultdict(list)
    for sample in validation_samples:
        if not sample.records or not str(sample.records[0].question).strip():
            raise ValueError(f"validation sample {sample.dataset}:{sample.qid} has no question")
        grouped[sample.dataset].append(sample)
    datasets = sorted(grouped)
    if not datasets:
        raise ValueError("protocol validation requires at least one dataset")

    base, remainder = divmod(size, len(datasets))
    quotas = {dataset: base + (index < remainder) for index, dataset in enumerate(datasets)}
    selected_keys: set[tuple[str, str]] = set()
    for index, dataset in enumerate(datasets):
        candidates = list(grouped[dataset])
        quota = quotas[dataset]
        if quota > len(candidates):
            raise ValueError(
                f"protocol validation dataset {dataset} has {len(candidates)} samples, needs {quota}"
            )
        random.Random(seed + index).shuffle(candidates)
        selected_keys.update((item.dataset, item.qid) for item in candidates[:quota])
    return [
        sample
        for sample in validation_samples
        if (sample.dataset, sample.qid) in selected_keys
    ]


def protocol_sample_manifest(samples: list[TrainingSample]) -> list[dict[str, Any]]:
    return [
        {
            "qid": sample.qid,
            "dataset": sample.dataset,
            "question": sample.records[0].question,
        }
        for sample in samples
    ]


def protocol_selection_score(*, eligible: bool, eval_loss: float, metrics: dict[str, Any]) -> float:
    if eligible:
        # Every eligible checkpoint outranks every ineligible checkpoint; among
        # eligible checkpoints, ordinary held-out loss remains the tie-breaker.
        return 1_000_000.0 - float(eval_loss)
    return (
        -1_000_000.0
        + float(metrics.get("final_compliance_rate", 0.0))
        - float(metrics.get("parse_failure_rate", 1.0))
        - float(metrics.get("missing_answer_tag_rate", 1.0))
    )


def select_top_eval_checkpoints(output_dir: Path, *, count: int) -> list[dict[str, Any]]:
    metrics_path = output_dir / "eval_metrics.jsonl"
    if not metrics_path.is_file():
        raise SystemExit(f"Missing eval metrics for final checkpoint selection: {metrics_path}")
    latest_by_step: dict[int, dict[str, Any]] = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "step" not in row or "eval_loss" not in row:
            continue
        step = int(row["step"])
        checkpoint_path = output_dir / f"checkpoint-{step}"
        if checkpoint_path.is_dir():
            latest_by_step[step] = {
                "step": step,
                "epoch": row.get("epoch"),
                "eval_loss": float(row["eval_loss"]),
                "checkpoint_path": str(checkpoint_path),
            }
    candidates = sorted(
        latest_by_step.values(),
        key=lambda item: (item["eval_loss"], item["step"]),
    )[:count]
    if not candidates:
        raise SystemExit("No saved checkpoint has a matching eval_loss record.")
    _atomic_json(
        output_dir / "protocol_validation" / "candidate_checkpoints.json",
        {"requested_count": count, "actual_count": len(candidates), "candidates": candidates},
    )
    return candidates


def prune_non_candidate_checkpoints(
    output_dir: Path,
    candidates: list[dict[str, Any]],
) -> list[str]:
    keep = {Path(str(item["checkpoint_path"])).resolve() for item in candidates}
    removed: list[str] = []
    for checkpoint in sorted(output_dir.glob("checkpoint-*")):
        if not checkpoint.is_dir() or not checkpoint.name.removeprefix("checkpoint-").isdigit():
            continue
        if checkpoint.resolve() in keep:
            continue
        # The glob and numeric suffix constrain deletion to Trainer checkpoint
        # children of this run directory.
        shutil.rmtree(checkpoint)
        removed.append(str(checkpoint))
    _atomic_json(
        output_dir / "protocol_validation" / "checkpoint_retention.json",
        {
            "kept": [str(item["checkpoint_path"]) for item in candidates],
            "removed": removed,
        },
    )
    return removed


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def _prediction(sample: TrainingSample, result: Any) -> dict[str, Any]:
    return {
        "qid": sample.qid,
        "dataset": sample.dataset,
        "question": sample.records[0].question,
        "pred_answer": str(result.final_answer or ""),
        "trajectory": list(result.trajectory),
        "parse_errors": list(result.parse_errors),
        "retrieval_count": int(result.state.retrieval_count),
    }


def make_protocol_validation_callback(
    trainer_callback_cls: Any,
    *,
    output_dir: Path,
    samples: list[TrainingSample],
    tokenizer: Any,
    prompt_contract: Any,
    max_rounds: int,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    retrieval_top_k: int,
    retrieval_backend: str,
    retrieval_root: str,
    retrieval_embedding_model: str,
    retrieval_device: str,
    retrieval_max_length: int,
    retrieval_batch_size: int,
    max_parse_failure_rate: float,
    max_missing_answer_tag_rate: float,
    min_final_compliance_rate: float,
    disable_tqdm: bool = False,
    run_every_steps: int | None = None,
    metric_prefix: str = "eval_protocol",
    artifact_dir_name: str = "protocol_validation",
    checkpoint_root: Path | None = None,
    load_existing_best: bool = True,
    distributed_generation: bool = True,
    retrieval_env_factory: Callable[..., Any] = create_retrieval_env,
) -> Any:
    class ProtocolValidationCallback(trainer_callback_cls):
        def __init__(self) -> None:
            self.retrieval_env: Any | None = None
            self.best_eligible: dict[str, Any] | None = None
            selection_path = output_dir / artifact_dir_name / "best_checkpoint.json"
            if load_existing_best and selection_path.is_file():
                payload = json.loads(selection_path.read_text(encoding="utf-8"))
                if isinstance(payload.get("best"), dict):
                    self.best_eligible = payload["best"]

        def _get_retrieval_env(self) -> Any:
            if self.retrieval_env is None:
                self.retrieval_env = retrieval_env_factory(
                    backend=retrieval_backend,
                    retrieval_root=retrieval_root,
                    embedding_model=retrieval_embedding_model,
                    device=retrieval_device,
                    top_k=retrieval_top_k,
                    max_length=retrieval_max_length,
                    batch_size=retrieval_batch_size,
                    query_cache_size=4096,
                )
            return self.retrieval_env

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any] | None = None,
            model: Any = None,
            **kwargs: Any,
        ) -> None:
            if metrics is None or model is None:
                raise RuntimeError("protocol validation requires evaluation metrics and model")
            step = int(getattr(state, "global_step", 0) or 0)
            if run_every_steps is not None and step % run_every_steps != 0:
                return
            eval_loss = float(metrics.get("eval_loss", math.inf))
            if not math.isfinite(eval_loss):
                raise RuntimeError("protocol validation requires a finite eval_loss")

            import torch

            distributed = bool(
                distributed_generation
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            rank = torch.distributed.get_rank() if distributed else 0
            world_size = torch.distributed.get_world_size() if distributed else 1
            if not distributed_generation and int(getattr(args, "process_index", 0)) != 0:
                return
            generation_model = model.module if distributed and hasattr(model, "module") else model

            policy = HFSharedPolicy(
                model=generation_model,
                tokenizer=tokenizer,
                system_prompt=None,
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
                temperature=temperature,
                top_p=top_p,
                top_k=0,
                prompt_contract=prompt_contract,
                score_completions=False,
                generation_use_cache=True,
            )
            executor = RAGLoopExecutor(
                policy=policy,
                retrieval_env=self._get_retrieval_env(),
                max_rounds=max_rounds,
            )
            result_dir = output_dir / artifact_dir_name / f"checkpoint-{step}"
            predictions_path = result_dir / "predictions.jsonl"
            local_predictions_path = (
                result_dir / f"predictions.rank-{rank}.jsonl"
                if distributed
                else predictions_path
            )
            local_predictions_path.parent.mkdir(parents=True, exist_ok=True)
            local_predictions_path.unlink(missing_ok=True)
            from tqdm.auto import tqdm

            predictions: list[dict[str, Any]] = []
            protocol_started_at = time.monotonic()
            iterator = tqdm(
                samples[rank::world_size],
                desc=f"{artifact_dir_name} checkpoint-{step} rank-{rank}",
                unit="question",
                dynamic_ncols=True,
                leave=True,
                disable=disable_tqdm or rank != 0,
            )
            for sample in iterator:
                policy.reset_trace()
                result = executor.run(
                    question=sample.records[0].question,
                    dataset=sample.dataset,
                )
                prediction = _prediction(sample, result)
                predictions.append(prediction)
                _append_jsonl(local_predictions_path, prediction)
            local_runtime = time.monotonic() - protocol_started_at

            if distributed:
                gathered: list[Any] = [None] * world_size
                torch.distributed.all_gather_object(
                    gathered,
                    {"predictions": predictions, "runtime": local_runtime},
                )
                if rank != 0:
                    return
                predictions = [
                    prediction
                    for payload in gathered
                    for prediction in payload["predictions"]
                ]
                sample_order = {
                    (sample.dataset, sample.qid): index for index, sample in enumerate(samples)
                }
                predictions.sort(
                    key=lambda item: sample_order[(str(item["dataset"]), str(item["qid"]))]
                )
                protocol_runtime = max(float(payload["runtime"]) for payload in gathered)
            else:
                protocol_runtime = local_runtime

            protocol = compute_protocol_metrics(
                predictions,
                max_parse_failure_rate=max_parse_failure_rate,
                max_missing_answer_tag_rate=max_missing_answer_tag_rate,
                min_final_compliance_rate=min_final_compliance_rate,
            )
            eligible = bool(protocol["checkpoint_eligible"])
            score = protocol_selection_score(
                eligible=eligible,
                eval_loss=eval_loss,
                metrics=protocol,
            )
            metrics.update(
                {
                    f"{metric_prefix}_parse_failure_rate": protocol["parse_failure_rate"],
                    f"{metric_prefix}_missing_answer_tag_rate": protocol["missing_answer_tag_rate"],
                    f"{metric_prefix}_final_compliance_rate": protocol["final_compliance_rate"],
                    f"{metric_prefix}_checkpoint_eligible": float(eligible),
                    f"{metric_prefix}_selection_score": score,
                    f"{metric_prefix}_runtime": protocol_runtime,
                }
            )

            checkpoint_path = (checkpoint_root or output_dir) / f"checkpoint-{step}"
            result_payload = {
                **protocol,
                "step": step,
                "epoch": getattr(state, "epoch", None),
                "eval_loss": eval_loss,
                "selection_score": score,
                "runtime": protocol_runtime,
                "checkpoint_path": str(checkpoint_path),
                "thresholds": {
                    "max_parse_failure_rate": max_parse_failure_rate,
                    "max_missing_answer_tag_rate": max_missing_answer_tag_rate,
                    "min_final_compliance_rate": min_final_compliance_rate,
                },
            }
            _write_jsonl(predictions_path, predictions)
            _atomic_json(result_dir / "protocol_metrics.json", result_payload)

            if eligible and (
                self.best_eligible is None
                or eval_loss < float(self.best_eligible["eval_loss"])
            ):
                self.best_eligible = result_payload
            _atomic_json(
                output_dir / artifact_dir_name / "best_checkpoint.json",
                {
                    "status": "eligible" if self.best_eligible is not None else "no_eligible_checkpoint",
                    "best": self.best_eligible,
                    "latest": result_payload,
                },
            )

    return ProtocolValidationCallback()


def require_eligible_best_checkpoint(callback: Any, trainer: Any | None = None) -> dict[str, Any]:
    best = callback.best_eligible
    if best is None:
        raise SystemExit(
            "Training finished without a checkpoint satisfying all protocol thresholds; "
            "the final adapter was not exported."
        )
    expected = Path(str(best["checkpoint_path"]))
    trainer_best = (
        Path(str(trainer.state.best_model_checkpoint or ""))
        if trainer is not None
        else expected
    )
    if trainer_best != expected:
        raise SystemExit(
            "Trainer best-checkpoint mismatch: "
            f"protocol={expected}, trainer={trainer_best}. Final adapter was not exported."
        )
    if not expected.is_dir():
        raise SystemExit(f"Eligible best checkpoint is missing: {expected}")
    return best
